"""Stdlib tests for the causal derivative estimators — no numpy, no GPU.

Run:  python ml/test_deriv_estimators.py

Each estimator must recover the right sign of the derivative on clean ramps, read
zero on a flat window, and give the exact slope on a pure line (for the linear
ones). Guards the calculus the whole signal stands on.
"""

from __future__ import annotations

from deriv_estimators import ESTIMATORS, holt, kalman, sg1, sg2, tvdiff


def test_all_estimators_sign_and_flat():
    up = [0.2 + 0.01 * i + 0.001 * i * i for i in range(16)]     # accelerating rise
    dn = [0.9 - 0.012 * i for i in range(16)]                    # steady fall
    for name, est in ESTIMATORS.items():
        assert est(up) > 0, name
        assert est(dn) < 0, name
        assert abs(est([0.5] * 16)) < 1e-6, name                 # flat → zero


def test_linear_estimators_recover_slope():
    line = [0.3 + 0.02 * i for i in range(10)]
    assert abs(sg1(line) - 0.02) < 1e-9
    # SG2 also lands on the slope for a pure line (no curvature to chase).
    assert abs(sg2(line) - 0.02) < 1e-6


def test_velocity_estimators_bounded_finite():
    import math

    win = [0.4 + 0.01 * i + 0.02 * math.sin(i) for i in range(16)]
    # holt / kalman are velocity-SCALED — bounded by roughly the move size.
    for est in (holt, kalman):
        v = est(win)
        assert math.isfinite(v) and abs(v) < 0.5
    # tvdiff is a strong DIRECTION estimator but not velocity-scaled (the bake-off
    # uses its sign only) — just require finiteness.
    assert math.isfinite(tvdiff(win))


def test_short_window_is_safe():
    for name, est in ESTIMATORS.items():
        assert est([0.5]) == 0.0, name          # one point → no derivative
        assert isinstance(est([0.5, 0.6]), float), name


def test_estimators_are_deterministic():
    import math

    win = [0.4 + 0.008 * i + 0.015 * math.sin(1.3 * i) for i in range(16)]
    for name, est in ESTIMATORS.items():
        assert est(list(win)) == est(list(win)), name


def test_all_agree_on_a_clear_trend():
    # On an unambiguous rise every estimator must call the direction up; on a fall,
    # down. (Magnitudes differ; the sign is what the signal actually uses.)
    up = [0.3 + 0.015 * i for i in range(16)]
    dn = [0.8 - 0.015 * i for i in range(16)]
    assert all(est(up) > 0 for est in ESTIMATORS.values())
    assert all(est(dn) < 0 for est in ESTIMATORS.values())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
