"""Stdlib tests for the microstructure feature layer — no pytest, no GPU.

Run:  python ml/test_micro.py     (exits non-zero on the first failed assertion)

Covers the book-free microstructure features derived from raw OrderFilled events
(effective/realized spread, fill-size distribution, maker/taker Herfindahl,
Kyle's-lambda price impact, fee intensity), the wei-string parsing, and the
leakage-safe per-token sliding build. Fast enough to run on every commit.
"""

from __future__ import annotations

import math

from features_micro import (
    MICRO_FEATURES,
    Bucket,
    _hhi,
    _large_fill_ratio,
    _synth_market,
    _wei_to_units,
    build,
    market_to_rich,
    normalize_fill,
    window_features_micro,
)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _fill_event(price, usdc, buy=True, token_id="TKN", ts=0,
                maker="0xm", taker="0xt", fee=0.0):
    """Build a raw OrderFilled event (wei strings) at a given price/size/side."""
    token = usdc / price
    usdc_wei = str(int(round(usdc * 1_000_000)))
    token_wei = str(int(round(token * 1_000_000)))
    fee_wei = str(int(round(fee * 1_000_000)))
    if buy:  # taker gives USDC, receives token
        ev = {"maker_asset_id": token_id, "taker_asset_id": "0",
              "maker_amount_filled": token_wei, "taker_amount_filled": usdc_wei}
    else:    # maker gives USDC, taker gives token
        ev = {"maker_asset_id": "0", "taker_asset_id": token_id,
              "maker_amount_filled": usdc_wei, "taker_amount_filled": token_wei}
    ev.update({"timestamp": ts, "maker": maker, "taker": taker,
               "protocol_fee": fee_wei, "taker_fee": "0", "maker_fee": "0"})
    return ev


def _bucket_from(fills):
    vol = sum(f.usdc for f in fills)
    price = sum(f.price * f.usdc for f in fills) / vol if vol > 0 else fills[-1].price
    return Bucket(0, price, fills)


# --------------------------------------------------------------------------- #
# wei-string parsing


def test_wei_parsing():
    assert _wei_to_units("1500000") == 1.5          # 6 decimals
    assert _wei_to_units("1000000") == 1.0
    assert _wei_to_units("0") == 0.0
    assert _wei_to_units("") == 0.0
    assert _wei_to_units(None) == 0.0
    assert _wei_to_units(" 2500000 ") == 2.5        # whitespace tolerant
    # big uint256 string well past float precision is parsed as an exact int
    assert _wei_to_units("123456789000000") == 123456789.0


def test_normalize_fill_price_and_direction():
    # BUY: taker pays 30 USDC for 100 tokens → price 0.30, direction +1.
    buy = normalize_fill(_fill_event(0.30, 30.0, buy=True))
    assert buy is not None
    assert approx(buy.price, 0.30)
    assert buy.direction == 1
    assert approx(buy.usdc, 30.0)
    assert buy.token_id == "TKN"
    # SELL: maker pays USDC, taker delivers token → direction -1.
    sell = normalize_fill(_fill_event(0.60, 60.0, buy=False))
    assert sell is not None
    assert approx(sell.price, 0.60)
    assert sell.direction == -1
    # signed notional carries the taker side.
    assert approx(buy.signed, 30.0)
    assert approx(sell.signed, -60.0)


def test_normalize_fill_rejects_non_pairs():
    # neither leg is collateral → rejected
    ev = {"maker_asset_id": "AAA", "taker_asset_id": "BBB",
          "maker_amount_filled": "1000000", "taker_amount_filled": "1000000"}
    assert normalize_fill(ev) is None
    # both legs collateral → rejected
    ev2 = {"maker_asset_id": "0", "taker_asset_id": "0",
           "maker_amount_filled": "1000000", "taker_amount_filled": "1000000"}
    assert normalize_fill(ev2) is None
    # zero token amount → rejected (no valid price)
    ev3 = _fill_event(0.5, 10.0, buy=True)
    ev3["maker_amount_filled"] = "0"
    assert normalize_fill(ev3) is None


# --------------------------------------------------------------------------- #
# feature vector shape / finiteness / bounds


def _window(n=16):
    """A window of buckets with a mild up-drift and buy-leaning fills."""
    buckets = []
    price = 0.4
    ts = 0
    for _ in range(n):
        fills = []
        for j in range(4):
            price = min(0.95, price + 0.002)
            fills.append(normalize_fill(
                _fill_event(price, 100.0 + 10 * j, buy=(j % 4 != 3), ts=ts, fee=0.1)))
            ts += 1
        buckets.append(_bucket_from(fills))
    return buckets


def test_feature_vector_shape_and_finite():
    v = window_features_micro(_window(16))
    assert len(v) == len(MICRO_FEATURES) == 14, len(v)
    assert all(math.isfinite(x) for x in v)


def test_bounded_features_in_unit_interval():
    v = window_features_micro(_window(16))
    for name in ("large_fill_ratio", "maker_concentration", "taker_concentration"):
        val = v[MICRO_FEATURES.index(name)]
        assert 0.0 <= val <= 1.0, (name, val)
    # non-negative-by-construction features
    for name in ("effective_spread", "fill_size_cv", "fee_intensity"):
        assert v[MICRO_FEATURES.index(name)] >= 0.0, name


def test_hhi_bounds_and_concentration():
    assert approx(_hhi([100.0]), 1.0)                 # single participant
    assert approx(_hhi([1.0, 1.0, 1.0, 1.0]), 0.25)   # 4 equal → 1/4
    assert _hhi([100.0, 1.0, 1.0]) > _hhi([1.0, 1.0, 1.0])
    assert _hhi([]) == 0.0
    assert _hhi([0.0, 0.0]) == 0.0
    for w in ([100.0], [1.0] * 5, [3.0, 1.0], []):
        assert 0.0 <= _hhi(w) <= 1.0


def test_large_fill_ratio_bounds():
    assert 0.0 <= _large_fill_ratio([100.0] * 10) <= 1.0
    # one whale dominating → higher share than a flat book
    assert _large_fill_ratio([1000.0] + [1.0] * 19) > _large_fill_ratio([50.0] * 20)
    assert _large_fill_ratio([]) == 0.0
    assert _large_fill_ratio([0.0, 0.0]) == 0.0


def test_price_impact_positive_when_signed_size_drives_price():
    """Construct a tape where each buy's size pushes price up proportionally →
    Kyle's-lambda regression must recover a positive, finite slope."""
    buckets = []
    price = 0.30
    ts = 0
    for _ in range(16):
        fills = []
        for _ in range(6):
            size = 50.0 + (ts % 5) * 80.0        # varied signed sizes
            price = min(0.95, price + 1e-4 * size)   # impact: bigger buy, bigger jump
            fills.append(normalize_fill(_fill_event(price, size, buy=True, ts=ts)))
            ts += 1
        buckets.append(_bucket_from(fills))
    pi = window_features_micro(buckets)[MICRO_FEATURES.index("price_impact")]
    assert math.isfinite(pi)
    assert pi > 0.0, pi


def test_degenerate_windows_are_finite():
    # a single fill per bucket, flat price → everything still finite
    buckets = []
    for _ in range(16):
        f = normalize_fill(_fill_event(0.5, 100.0, buy=True))
        buckets.append(_bucket_from([f]))
    v = window_features_micro(buckets)
    assert all(math.isfinite(x) for x in v)


# --------------------------------------------------------------------------- #
# end-to-end build


def test_build_labels_and_finiteness():
    groups = build([_synth_market(80, 7), _synth_market(90, 19)])
    n = sum(len(g) for g in groups)
    assert n > 0
    lr = MICRO_FEATURES.index("large_fill_ratio")
    mc = MICRO_FEATURES.index("maker_concentration")
    tc = MICRO_FEATURES.index("taker_concentration")
    for g in groups:
        for s in g:
            assert len(s.feat) == len(MICRO_FEATURES)
            assert all(math.isfinite(x) for x in s.feat)
            assert s.label in (0, 1)
            assert 0.0 <= s.feat[lr] <= 1.0
            assert 0.0 <= s.feat[mc] <= 1.0
            assert 0.0 <= s.feat[tc] <= 1.0
            assert math.isfinite(s.fwd)


def test_build_recovers_positive_impact_on_synthetic():
    groups = build([_synth_market(80, 7), _synth_market(90, 19)])
    pi = MICRO_FEATURES.index("price_impact")
    vals = [s.feat[pi] for g in groups for s in g]
    from features_micro import _mean
    assert _mean(vals) > 0.0


def test_market_to_rich_matches_sample_shape():
    rows = market_to_rich(_synth_market(70, 3))
    assert rows
    s = rows[0]
    assert len(s.seq) == 15                    # WINDOW - 1 increments
    assert len(s.feat) == len(MICRO_FEATURES)
    assert math.isfinite(s.fwd)


def test_build_groups_are_per_token():
    """Two distinct tokens in one market must yield separate honest groups."""
    a = _synth_market(70, 5, token_id="TKA")
    b = _synth_market(70, 11, token_id="TKB")
    merged = {"fills": a["fills"] + b["fills"], "collateral_ids": ["0"]}
    groups = build([merged])
    assert len(groups) == 2                     # one group per token


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
