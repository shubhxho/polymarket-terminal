"""Exact-parity test for the stdlib GBDT predictor — needs numpy + lightgbm.

The flagship derivative signal is served by walking LightGBM's `dump_model()`
tree JSON in pure stdlib (`features_deriv._walk_tree` / `deriv_signal_gbdt`), so
it can ship in the dependency-free TS bundle. This test proves that walk
reproduces LightGBM's own `booster.predict` to machine precision — if the two
ever diverge, serving would silently disagree with training.

Run:  ml/.venv/bin/python test_deriv_gbdt.py     (skips cleanly if lgb absent)
"""

from __future__ import annotations

import math

try:
    import numpy as np
    import lightgbm as lgb
except ImportError:  # pragma: no cover - stdlib-only checkout
    print("skip test_deriv_gbdt: numpy/lightgbm not installed")
    raise SystemExit(0)

from features_deriv import _sigmoid, _walk_tree


def test_walker_matches_booster_predict():
    rng = np.random.RandomState(0)
    X = rng.randn(2000, 8)
    # A learnable target so the trees actually branch on several features.
    y = ((X[:, 0] + 0.6 * X[:, 3] - 0.4 * X[:, 5] + 0.3 * rng.randn(2000)) > 0).astype(float)
    booster = lgb.train(
        dict(objective="binary", num_leaves=31, min_data_in_leaf=20,
             learning_rate=0.1, verbose=-1, feature_pre_filter=False),
        lgb.Dataset(X, y), num_boost_round=60,
    )
    dump = booster.dump_model()
    ref = booster.predict(X[:500])
    trees = dump["tree_info"]
    mine = np.array([
        _sigmoid(sum(_walk_tree(t["tree_structure"], list(X[i])) for t in trees))
        for i in range(500)
    ])
    max_diff = float(np.max(np.abs(mine - ref)))
    assert max_diff < 1e-9, f"walker diverges from booster.predict: {max_diff}"
    print(f"ok  test_walker_matches_booster_predict (max abs diff {max_diff:.2e})")


def test_default_left_on_missing():
    # A NaN feature must follow the tree's default_left, matching LightGBM.
    rng = np.random.RandomState(1)
    X = rng.randn(500, 3)
    y = (X[:, 0] > 0).astype(float)
    booster = lgb.train(
        dict(objective="binary", num_leaves=7, learning_rate=0.2, verbose=-1),
        lgb.Dataset(X, y), num_boost_round=10,
    )
    dump = booster.dump_model()
    xn = [float("nan"), 0.0, 0.0]
    ref = float(booster.predict(np.array([[np.nan, 0.0, 0.0]]))[0])
    mine = _sigmoid(sum(_walk_tree(t["tree_structure"], xn) for t in dump["tree_info"]))
    assert abs(mine - ref) < 1e-9, (mine, ref)
    print(f"ok  test_default_left_on_missing (diff {abs(mine-ref):.2e})")


if __name__ == "__main__":
    test_walker_matches_booster_predict()
    test_default_left_on_missing()
    print("\n2 parity tests passed")
