"""Stdlib tests for the resolution-snapshot feature layer — no pytest, no GPU.

Run:  python ml/test_resolve.py   (exits non-zero on the first failed assertion)

Covers `snapshot_features` (shape / finiteness / per-feature bounds), the
`outcome_prices` label parsing, and the `build` pipeline. Fast enough to run on
every commit.
"""

from __future__ import annotations

import math

from features_resolve import (
    RESOLVE_FEATURES,
    Sample,
    _synthetic_snapshot,
    build,
    label_from_snapshot,
    parse_outcome_prices,
    snapshot_features,
)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _base_snapshot() -> dict:
    return {
        "price": 0.62,
        "t_snapshot": 1_700_000_000 + 10 * 24 * 3600,
        "t_created": 1_700_000_000,
        "t_end": 1_700_000_000 + 30 * 24 * 3600,
        "recent": [0.55, 0.57, 0.58, 0.60, 0.61, 0.62],
        "volume": 2000.0,
        "volume_total": 5000.0,
        "signed_flow": 40.0,
        "flow_total": 100.0,
        "trade_count": 60,
        "outcome_prices": '["0.99","0.01"]',
    }


def test_vector_shape_and_finite():
    v = snapshot_features(_base_snapshot())
    assert len(v) == len(RESOLVE_FEATURES) == 12, len(v)
    assert all(math.isfinite(x) for x in v)


def test_feature_bounds():
    v = snapshot_features(_base_snapshot())
    idx = {name: i for i, name in enumerate(RESOLVE_FEATURES)}
    assert 0.0 <= v[idx["price"]] <= 1.0
    assert 0.0 <= v[idx["dist_from_half"]] <= 0.5
    assert -1.0 <= v[idx["flow_imbalance"]] <= 1.0
    assert 0.0 <= v[idx["volume_maturity"]] <= 1.0
    # extra bounded features
    assert -1.0 <= v[idx["logit_price"]] <= 1.0
    assert 0.0 <= v[idx["time_to_resolution"]] <= 1.0
    assert 0.0 <= v[idx["age_fraction"]] <= 1.0
    assert -1.0 <= v[idx["momentum"]] <= 1.0
    assert 0.0 <= v[idx["realized_vol"]] <= 1.0
    assert 0.0 <= v[idx["calib_bucket"]] <= 1.0


def test_feature_values():
    v = snapshot_features(_base_snapshot())
    idx = {name: i for i, name in enumerate(RESOLVE_FEATURES)}
    assert approx(v[idx["price"]], 0.62)
    assert approx(v[idx["dist_from_half"]], 0.12)
    # 10 of 30 days elapsed → 1/3 elapsed, 2/3 remaining.
    assert approx(v[idx["age_fraction"]], 10.0 / 30.0, tol=1e-9)
    assert approx(v[idx["time_to_resolution"]], 20.0 / 30.0, tol=1e-9)
    # signed 40 / total 100 → +0.4 buy imbalance.
    assert approx(v[idx["flow_imbalance"]], 0.4)
    # 2000 / 5000 → 0.4 matured.
    assert approx(v[idx["volume_maturity"]], 0.4)
    # interaction == dist_from_half * time_to_resolution.
    assert approx(v[idx["price_x_time"]], v[idx["dist_from_half"]] * v[idx["time_to_resolution"]])
    # price > 0.5 ⇒ positive logit.
    assert v[idx["logit_price"]] > 0.0


def test_price_half_is_neutral():
    snap = _base_snapshot()
    snap["price"] = 0.5
    snap["recent"] = [0.5] * 6
    v = snapshot_features(snap)
    idx = {name: i for i, name in enumerate(RESOLVE_FEATURES)}
    assert approx(v[idx["dist_from_half"]], 0.0)
    assert approx(v[idx["logit_price"]], 0.0)
    assert approx(v[idx["momentum"]], 0.0)


def test_flow_imbalance_sign():
    snap = _base_snapshot()
    idx = RESOLVE_FEATURES.index("flow_imbalance")
    snap["signed_flow"] = -80.0
    snap["flow_total"] = 100.0
    assert snapshot_features(snap)[idx] < 0
    snap["signed_flow"] = 80.0
    assert snapshot_features(snap)[idx] > 0


def test_extreme_prices_finite_and_clipped():
    idx = RESOLVE_FEATURES.index("logit_price")
    for p in (0.0, 1.0, 1e-12, 1.0 - 1e-12):
        snap = _base_snapshot()
        snap["price"] = p
        v = snapshot_features(snap)
        assert all(math.isfinite(x) for x in v)
        assert -1.0 <= v[idx] <= 1.0


def test_missing_keys_default_finite():
    v = snapshot_features({})
    assert len(v) == len(RESOLVE_FEATURES)
    assert all(math.isfinite(x) for x in v)


def test_degenerate_life_finite():
    snap = _base_snapshot()
    snap["t_end"] = snap["t_created"]           # zero-length life
    v = snapshot_features(snap)
    assert all(math.isfinite(x) for x in v)


def test_zero_flow_and_volume():
    snap = _base_snapshot()
    snap["signed_flow"] = 0.0
    snap["flow_total"] = 0.0
    snap["volume"] = 0.0
    snap["volume_total"] = 0.0
    idx = {name: i for i, name in enumerate(RESOLVE_FEATURES)}
    v = snapshot_features(snap)
    assert approx(v[idx["flow_imbalance"]], 0.0)
    assert approx(v[idx["volume_maturity"]], 0.0)


def test_parse_outcome_prices():
    assert parse_outcome_prices('["0.99","0.01"]') == 1
    assert parse_outcome_prices('["0.02","0.98"]') == 0
    assert parse_outcome_prices('["1","0"]') == 1
    assert parse_outcome_prices('["0","1"]') == 0


def test_label_from_snapshot_prefers_explicit():
    assert label_from_snapshot({"label": 1, "outcome_prices": '["0.02","0.98"]'}) == 1
    assert label_from_snapshot({"outcome_prices": '["0.99","0.01"]'}) == 1
    assert label_from_snapshot({}) is None


def test_build_produces_labeled_samples():
    snaps = [_synthetic_snapshot(i, yes_won=(i % 2 == 0)) for i in range(40)]
    samples = build(snaps)
    assert samples and len(samples) == len(snaps)
    for s in samples:
        assert isinstance(s, Sample)
        assert len(s.feat) == len(RESOLVE_FEATURES)
        assert all(math.isfinite(x) for x in s.feat)
        assert s.label in (0, 1)
    # yes_won on even i ⇒ roughly half the labels are 1.
    assert 0 < sum(s.label for s in samples) < len(samples)


def test_build_skips_unlabeled():
    snaps = [{"price": 0.5}, {"price": 0.7, "label": 1}]
    samples = build(snaps)
    assert len(samples) == 1 and samples[0].label == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
