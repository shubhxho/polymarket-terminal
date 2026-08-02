"""Stdlib tests for the price-derivative feature family — no pytest, no GPU.

Run:  python ml/test_features_deriv.py   (exits non-zero on the first failure)

Covers the causal Savitzky–Golay endpoint-derivative estimator (does it recover
the analytic derivatives of a known polynomial?), the flagship
`trend_consistency` signal, and the finiteness / flat-window guards.
"""

from __future__ import annotations

import math

from features_deriv import (
    DERIV_NAMES,
    _poly_endpoint,
    deriv_features,
    trend_consistency,
)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def test_sg_recovers_polynomial_derivatives():
    # p(s) = 0.2 + 0.01 s + 0.001 s²  (on the endpoint axis s = -(L-1)..0).
    # At the endpoint s=0: value 0.2, 1st deriv 0.01, 2nd deriv 0.002.
    win = [0.2 + 0.01 * (i - 15) + 0.001 * (i - 15) ** 2 for i in range(16)]
    f, d1, d2, d3 = _poly_endpoint(win, 3)
    assert approx(f, 0.2, 1e-6), f
    assert approx(d1, 0.01, 1e-6), d1
    assert approx(d2, 0.002, 1e-6), d2
    assert approx(d3, 0.0, 1e-6), d3          # no cubic term


def test_sg_linear_slope_is_exact():
    ramp = [0.3 + 0.02 * i for i in range(8)]
    assert approx(_poly_endpoint(ramp, 1)[1], 0.02, 1e-9)


def test_trend_consistency_bounds_and_sign():
    up = [0.1 * i for i in range(16)]
    dn = [1.6 - 0.1 * i for i in range(16)]
    assert approx(trend_consistency(up), 1.0)       # every step agrees → +1
    assert approx(trend_consistency(dn), -1.0)      # every step agrees, down → -1
    # Chop cannot reach a clean trend's magnitude.
    zig = [0.5 + 0.02 * (i % 2) for i in range(16)]
    assert abs(trend_consistency(zig)) < abs(trend_consistency(up))
    # Result always lives in [-1, 1].
    for w in (up, dn, zig, [0.5] * 16):
        assert -1.0 <= trend_consistency(w) <= 1.0


def test_flagship_beats_raw_velocity_direction():
    # A clean rise that only nudged up on average should still read as a strong,
    # confident trend via consistency even though its per-step velocity is tiny.
    clean = [0.40 + 0.005 * i for i in range(16)]
    assert trend_consistency(clean) > 0.9


def test_flat_window_is_all_zero():
    feats = deriv_features([0.5] * 16)
    assert set(feats) == set(DERIV_NAMES)
    assert all(v == 0.0 for v in feats.values()), feats


def test_all_features_finite():
    import random

    rng = random.Random(3)
    for _ in range(50):
        base = rng.uniform(0.05, 0.95)
        win = [min(0.99, max(0.01, base + rng.gauss(0, 0.05))) for _ in range(16)]
        feats = deriv_features(win)
        assert set(feats) == set(DERIV_NAMES)
        for k, v in feats.items():
            assert math.isfinite(v), (k, v)


def test_trend_consistency_is_the_flagship_entry():
    # The dict value and the standalone function must agree.
    win = [0.4 + 0.01 * i + 0.003 * math.sin(i) for i in range(16)]
    assert approx(deriv_features(win)["trend_consistency"], trend_consistency(win))
    assert DERIV_NAMES[0] == "trend_consistency"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
