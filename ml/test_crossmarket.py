"""Stdlib tests for the cross-market / neg-risk basket feature layer — no pytest.

Run:  python ml/test_crossmarket.py   (exits non-zero on the first failed assert)

Covers `market_features` (shape / finiteness / per-feature bounds), the basket
arb signal (residual ≈ 0 for a balanced basket, ≠ 0 for an imbalanced one), the
graceful degradation for non-neg-risk / singleton events, `outcome_prices` label
parsing (list-repr via ast, not json), and the `build` grouping pipeline. Fast
enough to run on every commit.
"""

from __future__ import annotations

import math

from features_crossmarket import (
    CROSSMARKET_FEATURES,
    Sample,
    _synthetic_event,
    build,
    label_from_market,
    market_features,
    parse_outcome_prices,
)

IDX = {name: i for i, name in enumerate(CROSSMARKET_FEATURES)}


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _balanced_basket():
    """Three-leg neg-risk basket whose YES prices sum to exactly 1."""
    return [
        {"market_id": "e-0", "event_id": "e", "neg_risk": True, "price": 0.5, "outcome_prices": "['1', '0']"},
        {"market_id": "e-1", "event_id": "e", "neg_risk": True, "price": 0.3, "outcome_prices": "['0', '1']"},
        {"market_id": "e-2", "event_id": "e", "neg_risk": True, "price": 0.2, "outcome_prices": "['0', '1']"},
    ]


# ── shape / finiteness / bounds ────────────────────────────────────────────────

def test_vector_shape_and_finite():
    legs = _balanced_basket()
    v = market_features(legs[0], legs)
    assert len(v) == len(CROSSMARKET_FEATURES) == 8, len(v)
    assert all(math.isfinite(x) for x in v)


def test_share_and_rank_bounds_over_all_legs():
    legs = _balanced_basket()
    for m in legs:
        v = market_features(m, legs)
        assert 0.0 <= v[IDX["share_of_basket"]] <= 1.0
        assert 0.0 <= v[IDX["rank_in_basket"]] <= 1.0


def test_extreme_prices_finite_and_bounded():
    legs = [
        {"event_id": "x", "neg_risk": True, "price": 0.0, "outcome_prices": "['0','1']"},
        {"event_id": "x", "neg_risk": True, "price": 1.0, "outcome_prices": "['1','0']"},
        {"event_id": "x", "neg_risk": True, "price": 2.0, "outcome_prices": "['0','1']"},   # out-of-range → clipped
    ]
    for m in legs:
        v = market_features(m, legs)
        assert all(math.isfinite(x) for x in v)
        assert 0.0 <= v[IDX["share_of_basket"]] <= 1.0
        assert 0.0 <= v[IDX["rank_in_basket"]] <= 1.0


# ── basket arithmetic ──────────────────────────────────────────────────────────

def test_basket_sum_and_residual_balanced():
    legs = _balanced_basket()
    v = market_features(legs[0], legs)
    assert approx(v[IDX["basket_sum"]], 1.0)
    assert approx(v[IDX["basket_residual"]], 0.0)          # fair basket → no arb
    assert approx(v[IDX["sibling_count"]], 2.0)            # two other legs
    assert v[IDX["is_neg_risk"]] == 1.0


def test_basket_residual_imbalanced():
    legs = _balanced_basket()
    legs[0]["price"] = 0.8                                  # sum now 1.3 → overpriced basket
    v = market_features(legs[0], legs)
    assert approx(v[IDX["basket_sum"]], 1.3)
    assert approx(v[IDX["basket_residual"]], 0.3)
    assert v[IDX["basket_residual"]] != 0.0

    legs[0]["price"] = 0.1                                  # sum now 0.6 → underpriced basket
    v = market_features(legs[0], legs)
    assert approx(v[IDX["basket_residual"]], -0.4)


def test_share_of_basket_sums_to_one_across_legs():
    legs = _balanced_basket()
    shares = [market_features(m, legs)[IDX["share_of_basket"]] for m in legs]
    assert approx(sum(shares), 1.0)                         # shares partition the basket
    assert approx(shares[0], 0.5)                           # favourite's price / sum(=1)


def test_rank_and_favorite_gap():
    legs = _balanced_basket()                              # prices 0.5, 0.3, 0.2
    fav = market_features(legs[0], legs)
    mid = market_features(legs[1], legs)
    low = market_features(legs[2], legs)
    # rank: favourite 0, longshot 1.
    assert approx(fav[IDX["rank_in_basket"]], 0.0)
    assert approx(low[IDX["rank_in_basket"]], 1.0)
    assert 0.0 < mid[IDX["rank_in_basket"]] < 1.0
    # favorite_gap: only the favourite has a positive lead over its top sibling.
    assert fav[IDX["favorite_gap"]] > 0.0                  # 0.5 − 0.3 = 0.2
    assert approx(fav[IDX["favorite_gap"]], 0.2)
    assert mid[IDX["favorite_gap"]] < 0.0                  # 0.3 − 0.5 = -0.2
    assert low[IDX["favorite_gap"]] < 0.0


def test_dispersion_reflects_spread():
    tight = [
        {"event_id": "t", "neg_risk": True, "price": 0.34, "outcome_prices": "['1','0']"},
        {"event_id": "t", "neg_risk": True, "price": 0.33, "outcome_prices": "['0','1']"},
        {"event_id": "t", "neg_risk": True, "price": 0.33, "outcome_prices": "['0','1']"},
    ]
    wide = [
        {"event_id": "w", "neg_risk": True, "price": 0.9, "outcome_prices": "['1','0']"},
        {"event_id": "w", "neg_risk": True, "price": 0.07, "outcome_prices": "['0','1']"},
        {"event_id": "w", "neg_risk": True, "price": 0.03, "outcome_prices": "['0','1']"},
    ]
    d_tight = market_features(tight[0], tight)[IDX["dispersion"]]
    d_wide = market_features(wide[0], wide)[IDX["dispersion"]]
    assert d_wide > d_tight >= 0.0


# ── graceful degradation ───────────────────────────────────────────────────────

def test_singleton_event_graceful():
    m = {"event_id": "solo", "neg_risk": True, "price": 0.42, "outcome_prices": "['1','0']"}
    v = market_features(m, [m])
    assert approx(v[IDX["basket_residual"]], 0.0)          # no siblings → no arb
    assert approx(v[IDX["share_of_basket"]], 1.0)          # its own whole basket
    assert approx(v[IDX["basket_sum"]], 0.42)
    assert approx(v[IDX["sibling_count"]], 0.0)
    assert approx(v[IDX["favorite_gap"]], 0.0)
    assert approx(v[IDX["dispersion"]], 0.0)


def test_non_neg_risk_multi_market_graceful():
    legs = [
        {"event_id": "p", "neg_risk": False, "price": 0.7, "outcome_prices": "['1','0']"},
        {"event_id": "p", "neg_risk": False, "price": 0.6, "outcome_prices": "['0','1']"},
    ]
    v = market_features(legs[0], legs)
    # prices sum to 1.3 but it's NOT a neg-risk basket → no arb signal claimed.
    assert approx(v[IDX["basket_residual"]], 0.0)
    assert approx(v[IDX["share_of_basket"]], 1.0)
    assert approx(v[IDX["basket_sum"]], 0.7)
    assert v[IDX["is_neg_risk"]] == 0.0
    assert approx(v[IDX["sibling_count"]], 1.0)            # sibling count stays factual


# ── label parsing ──────────────────────────────────────────────────────────────

def test_parse_outcome_prices_list_repr():
    # Single-quoted Python-list repr — json.loads would fail; ast.literal_eval works.
    assert parse_outcome_prices("['1', '0']") == 1
    assert parse_outcome_prices("['0', '1']") == 0
    assert parse_outcome_prices("['0.99', '0.01']") == 1
    assert parse_outcome_prices("['0.02', '0.98']") == 0


def test_parse_outcome_prices_json_and_edge():
    assert parse_outcome_prices('["0.99","0.01"]') == 1   # double-quoted also parses
    assert parse_outcome_prices(["1", "0"]) == 1          # already a list
    assert parse_outcome_prices(None) is None
    assert parse_outcome_prices("not a list") is None
    assert parse_outcome_prices("[]") is None


def test_parse_is_ast_not_json():
    import json
    op = "['1', '0']"
    raised = False
    try:
        json.loads(op)
    except ValueError:
        raised = True
    assert raised, "sanity: single-quoted repr must be unparseable by json"
    assert parse_outcome_prices(op) == 1


def test_label_from_market_prefers_explicit():
    assert label_from_market({"label": 0, "outcome_prices": "['1','0']"}) == 0
    assert label_from_market({"outcome_prices": "['1','0']"}) == 1
    assert label_from_market({}) is None


# ── build pipeline ─────────────────────────────────────────────────────────────

def test_build_groups_by_event_and_labels():
    markets = []
    for e in range(6):
        markets += _synthetic_event(e, n_legs=3, neg_risk=True,
                                    imbalance=0.0 if e % 2 == 0 else 0.2)
    samples = build(markets)
    assert samples and len(samples) == len(markets)
    for s in samples:
        assert isinstance(s, Sample)
        assert len(s.feat) == len(CROSSMARKET_FEATURES)
        assert all(math.isfinite(x) for x in s.feat)
        assert s.label in (0, 1)
    # Each 3-leg event has exactly one YES leg (the favourite).
    assert sum(s.label for s in samples) == 6


def test_build_features_computed_from_siblings():
    """A market's basket_sum must reflect its event siblings, not just itself."""
    markets = _synthetic_event(42, n_legs=4, neg_risk=True, imbalance=0.0)
    samples = build(markets)
    assert len(samples) == 4
    for s in samples:
        assert approx(s.feat[IDX["basket_sum"]], 1.0)      # balanced 4-leg basket
        assert approx(s.feat[IDX["sibling_count"]], 3.0)


def test_build_skips_unlabeled():
    markets = [
        {"event_id": "a", "neg_risk": True, "price": 0.5},                          # no label
        {"event_id": "a", "neg_risk": True, "price": 0.5, "label": 1},
    ]
    samples = build(markets)
    assert len(samples) == 1 and samples[0].label == 1


def test_build_separates_distinct_events():
    """Markets in different events must not pool into one basket."""
    markets = (
        _synthetic_event(1, n_legs=2, neg_risk=True, imbalance=0.0)
        + _synthetic_event(2, n_legs=2, neg_risk=True, imbalance=0.0)
    )
    samples = build(markets)
    assert len(samples) == 4
    for s in samples:
        assert approx(s.feat[IDX["sibling_count"]], 1.0)   # one sibling per 2-leg event


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
