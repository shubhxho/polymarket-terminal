"""Stdlib tests for the order-flow feature layer — no pytest, no GPU.

Run:  python ml/test_flow.py     (exits non-zero on the first failed assertion)

Covers the true-aggressor order-flow features (signed imbalance, VPIN, toxicity,
whale ratio, buy pressure) and the leakage-safe sliding build. Fast enough to
run on every commit.
"""

from __future__ import annotations

import math

from features_flow import (
    FLOW_FEATURES,
    _buy_pressure,
    _signed_imbalance,
    _synth_market,
    _toxicity,
    _vpin,
    _whale_ratio,
    build,
    market_to_rich,
    window_features_flow,
)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _bucket(close: float, volume: float, signed_flow: float, trades: int = 3) -> dict:
    return {
        "open": close, "high": close + 0.01, "low": close - 0.01,
        "close": close, "volume": volume, "trades": trades, "signed_flow": signed_flow,
    }


def _window(n: int = 16) -> list:
    """A simple up-drifting window with buy-leaning flow."""
    out = []
    p = 0.4
    for _ in range(n):
        p += 0.01
        out.append(_bucket(p, 100.0, 40.0))
    return out


def test_feature_vector_shape_and_finite():
    v = window_features_flow(_window(16))
    assert len(v) == len(FLOW_FEATURES) == 11, len(v)
    assert all(math.isfinite(x) for x in v)


def test_signed_imbalance_bounds_and_sign():
    idx = FLOW_FEATURES.index("signed_imbalance")
    buy = [_bucket(0.5, 100.0, 100.0) for _ in range(16)]   # all buy-aggressor
    sell = [_bucket(0.5, 100.0, -100.0) for _ in range(16)]  # all sell-aggressor
    bal = [_bucket(0.5, 100.0, 0.0) for _ in range(16)]      # balanced
    assert approx(window_features_flow(buy)[idx], 1.0)
    assert approx(window_features_flow(sell)[idx], -1.0)
    assert approx(window_features_flow(bal)[idx], 0.0)
    # Always in bounds even on mixed flow.
    for w in (buy, sell, bal, _window(16)):
        val = window_features_flow(w)[idx]
        assert -1.0 <= val <= 1.0


def test_buy_pressure_bounds_and_sign():
    idx = FLOW_FEATURES.index("buy_pressure")
    buy = [_bucket(0.5, 100.0, 100.0) for _ in range(16)]
    sell = [_bucket(0.5, 100.0, -100.0) for _ in range(16)]
    bal = [_bucket(0.5, 100.0, 0.0) for _ in range(16)]
    assert approx(window_features_flow(buy)[idx], 1.0)
    assert approx(window_features_flow(sell)[idx], 0.0)
    assert approx(window_features_flow(bal)[idx], 0.5)
    for w in (buy, sell, bal, _window(16)):
        val = window_features_flow(w)[idx]
        assert 0.0 <= val <= 1.0


def test_vpin_and_toxicity_range():
    flows = [80.0, -60.0, 100.0, -20.0]
    vols = [100.0, 100.0, 100.0, 100.0]
    v = _vpin(flows, vols)
    t = _toxicity(flows, vols)
    assert 0.0 <= v <= 1.0
    assert 0.0 <= t <= 1.0
    # Perfectly one-sided flow pins both to 1; balanced flow to 0.
    assert approx(_vpin([100.0] * 4, [100.0] * 4), 1.0)
    assert approx(_vpin([0.0] * 4, [100.0] * 4), 0.0)
    assert approx(_toxicity([100.0] * 4, [100.0] * 4), 1.0)


def test_whale_ratio_range_and_concentration():
    uniform = [100.0] * 10
    concentrated = [100.0] + [1.0] * 9
    assert 0.0 <= _whale_ratio(uniform) <= 1.0
    assert 0.0 <= _whale_ratio(concentrated) <= 1.0
    # One dominant bucket → higher share than a flat book.
    assert _whale_ratio(concentrated) > _whale_ratio(uniform)
    assert _whale_ratio([]) == 0.0
    assert _whale_ratio([0.0] * 5) == 0.0


def test_signed_imbalance_helper_degenerate():
    assert _signed_imbalance([], []) == 0.0
    assert _signed_imbalance([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert _buy_pressure([], []) == 0.5


def test_degenerate_windows_are_finite():
    flat = [_bucket(0.5, 0.0, 0.0) for _ in range(16)]       # zero volume
    same = [_bucket(0.5, 100.0, 0.0) for _ in range(16)]     # flat price
    for w in (flat, same):
        assert all(math.isfinite(x) for x in window_features_flow(w))


def test_build_labels_and_finiteness():
    groups = build([_synth_market(80, 7), _synth_market(90, 19)])
    n = sum(len(g) for g in groups)
    assert n > 0
    for g in groups:
        for s in g:
            assert len(s.feat) == len(FLOW_FEATURES)
            assert all(math.isfinite(x) for x in s.feat)
            assert s.label in (0, 1)
            # every emitted feature vector respects the documented bounds
            si = FLOW_FEATURES.index("signed_imbalance")
            bp = FLOW_FEATURES.index("buy_pressure")
            assert -1.0 <= s.feat[si] <= 1.0
            assert 0.0 <= s.feat[bp] <= 1.0


def test_market_to_rich_matches_sample_shape():
    rows = market_to_rich(_synth_market(70, 3))
    assert rows
    s = rows[0]
    # Sample carries a return sequence + feature vector + label + forward return.
    assert len(s.seq) == 15  # WINDOW - 1 increments
    assert len(s.feat) == len(FLOW_FEATURES)
    assert math.isfinite(s.fwd)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
