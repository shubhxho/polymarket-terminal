"""Stdlib tests for the smart-money / behavioral feature layer.

Run:  python ml/test_smart.py     (exits non-zero on the first failed assertion)

Covers wallet PnL scoring, deterministic smart-cohort tagging, the per-market
behavioral features (bounds + finiteness), and the Sample labelling. No pytest,
no GPU, no numpy — same discipline as test_ml.py.
"""

from __future__ import annotations

import math

from features_smart import (
    EPS,
    SMART_FEATURES,
    build,
    build_leaderboard,
    market_to_smart,
    synthesize,
    tag_smart,
    wallet_pnl,
    window_features_smart,
)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ── wallet PnL / leaderboard ─────────────────────────────────────────────────

def test_wallet_pnl_mark_to_final():
    """Buy low then resolve YES → positive PnL; the opposite side loses it."""
    m = {
        "final": 1.0,
        "timestamp": [1, 2],
        "user": ["buyer", "seller"],
        "role": ["taker", "maker"],
        "price": [0.4, 0.4],
        "usd_amount": [40.0, 40.0],
        "token_amount": [100.0, -100.0],  # buyer +100 shares, seller −100
    }
    pnl = wallet_pnl(m)
    assert approx(pnl["buyer"], 100.0 * (1.0 - 0.4))   # +60
    assert approx(pnl["seller"], -100.0 * (1.0 - 0.4))  # −60
    # zero-sum between the two counterparties
    assert approx(pnl["buyer"] + pnl["seller"], 0.0)


def test_final_falls_back_to_last_price():
    m = {
        "timestamp": [1], "user": ["w"], "role": ["taker"],
        "price": [0.3], "usd_amount": [30.0], "token_amount": [100.0],
    }
    # no "final" → uses last price 0.3 → PnL = 100·(0.3−0.3) = 0
    assert approx(wallet_pnl(m)["w"], 0.0)


# ── cohort tagging ───────────────────────────────────────────────────────────

def test_tag_smart_is_deterministic_and_sized():
    series = synthesize()
    board1 = build_leaderboard(series)
    board2 = build_leaderboard(series)
    smart1 = tag_smart(board1, top_frac=0.2)
    smart2 = tag_smart(board2, top_frac=0.2)
    assert smart1 == smart2                       # deterministic
    assert len(smart1) == math.ceil(0.2 * len(board1))
    # ties broken by wallet address, never by dict order
    tied = {"a": 5.0, "b": 5.0, "c": 5.0, "d": 1.0}
    assert tag_smart(tied, top_frac=0.5) == {"a", "b"}


def test_smart_cohort_captures_skilled_wallets():
    """The informed sub-population (w0..w7) should dominate the PnL leaderboard."""
    series = synthesize()
    smart = tag_smart(build_leaderboard(series))
    skilled = {f"w{i}" for i in range(8)}
    assert len(smart & skilled) >= 4              # majority of skilled captured


# ── per-window feature vector ────────────────────────────────────────────────

def _one_window():
    users = [f"u{i % 5}" for i in range(16)]
    roles = ["maker" if i % 2 else "taker" for i in range(16)]
    prices = [0.4 + 0.01 * i for i in range(16)]
    usds = [100.0 + i for i in range(16)]
    tokens = [(1.0 if i % 3 else -1.0) * (100.0 + i) for i in range(16)]
    return users, roles, prices, usds, tokens


def test_feature_vector_shape_and_finite():
    users, roles, prices, usds, tokens = _one_window()
    v = window_features_smart(users, roles, prices, usds, tokens,
                              smart_set={"u0", "u1"}, seen_before=set())
    assert len(v) == len(SMART_FEATURES) == 10
    assert all(math.isfinite(x) for x in v)


def test_bounded_features_in_range():
    users, roles, prices, usds, tokens = _one_window()
    v = dict(zip(SMART_FEATURES, window_features_smart(
        users, roles, prices, usds, tokens,
        smart_set={"u0", "u2"}, seen_before={"u0"})))
    for name in ("smart_share", "herding", "crowding", "maker_taker_ratio",
                 "whale_concentration", "last", "extremeness"):
        assert 0.0 <= v[name] <= 1.0, (name, v[name])
    for name in ("smart_net_flow", "flow_imbalance"):
        assert -1.0 <= v[name] <= 1.0, (name, v[name])
    assert -2.0 <= v["smart_minus_dumb"] <= 2.0


def test_smart_share_reflects_cohort_membership():
    users, roles, prices, usds, tokens = _one_window()
    none = dict(zip(SMART_FEATURES, window_features_smart(
        users, roles, prices, usds, tokens, smart_set=set(), seen_before=set())))
    allw = dict(zip(SMART_FEATURES, window_features_smart(
        users, roles, prices, usds, tokens,
        smart_set=set(users), seen_before=set())))
    assert approx(none["smart_share"], 0.0, tol=1e-6)
    assert allw["smart_share"] > 0.99             # everyone smart → ~all volume


def test_herding_one_sided_flow():
    """All-buy flow ⇒ herding ≈ 1 and flow_imbalance ≈ +1."""
    users = [f"u{i}" for i in range(16)]
    roles = ["taker"] * 16
    prices = [0.5 + 0.001 * i for i in range(16)]
    usds = [100.0] * 16
    tokens = [200.0] * 16                          # every fill a buy
    v = dict(zip(SMART_FEATURES, window_features_smart(
        users, roles, prices, usds, tokens, smart_set=set(), seen_before=set())))
    assert v["herding"] > 0.99
    assert v["flow_imbalance"] > 0.99


def test_crowding_new_wallet_rate():
    users, roles, prices, usds, tokens = _one_window()   # wallets u0..u4
    # all previously seen → crowding 0
    seen_all = window_features_smart(users, roles, prices, usds, tokens,
                                     smart_set=set(), seen_before=set(users))
    # none seen → every unique wallet is new → crowding 1
    seen_none = window_features_smart(users, roles, prices, usds, tokens,
                                      smart_set=set(), seen_before=set())
    ci = SMART_FEATURES.index("crowding")
    assert approx(seen_all[ci], 0.0)
    assert approx(seen_none[ci], 1.0)


def test_whale_concentration_single_wallet():
    """One wallet doing all the volume ⇒ whale_concentration ≈ 1."""
    users = ["whale"] * 16
    roles = ["taker"] * 16
    prices = [0.5 + 0.001 * i for i in range(16)]
    usds = [100.0] * 16
    tokens = [(1.0 if i % 2 else -1.0) * 100.0 for i in range(16)]
    wi = SMART_FEATURES.index("whale_concentration")
    v = window_features_smart(users, roles, prices, usds, tokens,
                              smart_set=set(), seen_before=set())
    assert v[wi] > 0.99


# ── end-to-end build ─────────────────────────────────────────────────────────

def test_build_emits_valid_labeled_samples():
    series = synthesize()
    groups = build(series)
    assert len(groups) == len(series)
    total = 0
    for g in groups:
        for s in g:
            total += 1
            assert len(s.feat) == len(SMART_FEATURES)
            assert all(math.isfinite(x) for x in s.feat)
            assert s.label in (0, 1)
            assert math.isfinite(s.fwd)
            v = dict(zip(SMART_FEATURES, s.feat))
            assert 0.0 <= v["smart_share"] <= 1.0
            assert 0.0 <= v["herding"] <= 1.0
            assert -1.0 <= v["smart_net_flow"] <= 1.0
    assert total > 0


def test_market_to_smart_time_orders_tape():
    """Shuffled timestamps must not change the emitted windows — build sorts."""
    m = synthesize(n_markets=1, trades_per_market=60)[0]
    smart = tag_smart(build_leaderboard([m]))
    ordered = market_to_smart(m, smart)
    # reverse every parallel column; sort inside must undo it
    rev = {k: (list(reversed(v)) if isinstance(v, list) else v) for k, v in m.items()}
    shuffled = market_to_smart(rev, smart)
    assert len(ordered) == len(shuffled)
    assert [s.label for s in ordered] == [s.label for s in shuffled]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
