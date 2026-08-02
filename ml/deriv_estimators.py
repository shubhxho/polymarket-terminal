"""Causal derivative estimators for a noisy sampled signal — and which is best.

Differentiating a noisy series is the whole game: a naive finite difference
roughly doubles the noise power. This module collects the principled *causal*
(endpoint, no-lookahead) estimators of the first derivative so they can be raced
head-to-head on the actual task, rather than assumed:

  * `sg1`     — least-squares degree-1 slope over the window (Savitzky–Golay);
  * `sg2`     — degree-2 SG, endpoint first derivative (curvature-aware);
  * `holt`    — Holt double-exponential smoothing; its trend term is a denoised,
                one-sided velocity (an EWMA derivative);
  * `kalman`  — constant-velocity Kalman filter; the optimal *linear* causal
                velocity estimate under a random-acceleration model;
  * `tvdiff`  — a light total-variation-regularised derivative (the noise-robust
                differentiation idea — penalise a jerky derivative).

Empirical verdict (see `rank_estimators`, out-of-time on data/series.json, used
as the trend-direction reference that drives `trend_consistency`): total-variation
differentiation (`tvdiff`) is **marginally the best and the most walk-forward
stable** (AUC ~0.588, wf-min strength ~0.068) — the noise-robust idea pays off at
the margin. But it sits **within noise of `sg1`** (AUC 0.5878) at ~10× the compute,
while the heavier smoothers (Kalman, Holt) over-smooth: they trade endpoint
responsiveness for smoothness and *lag* the turn, which hurts a direction read.
So production (`features_deriv`) keeps the cheap `sg1` OLS slope — a cost/robustness
call now backed by the bake-off, with `tvdiff` documented as the marginal-best
alternative if the stability is ever worth the cost. Pure stdlib + math.

Run:  ml/.venv/bin/python deriv_estimators.py           # selfcheck
      ml/.venv/bin/python deriv_estimators.py --rank    # OOT estimator bake-off
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence

from features_deriv import _poly_endpoint, _increments


def sg1(window: Sequence[float]) -> float:
    """Least-squares degree-1 endpoint slope (Savitzky–Golay, order 1)."""
    return _poly_endpoint(window, 1)[1]


def sg2(window: Sequence[float]) -> float:
    """Degree-2 SG endpoint first derivative — follows curvature into the edge."""
    return _poly_endpoint(window, 2)[1]


def holt(window: Sequence[float], alpha: float = 0.4, beta: float = 0.3) -> float:
    """Holt double-exponential smoothing; return the final trend term b_t, a
    causal EWMA estimate of the derivative. alpha smooths the level, beta the
    trend."""
    if len(window) < 2:
        return 0.0
    level = window[0]
    trend = window[1] - window[0]
    for x in window[1:]:
        prev = level
        level = alpha * x + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return trend


def kalman(window: Sequence[float], q: float = 1e-4, r: float = 1e-3) -> float:
    """Constant-velocity Kalman filter; return the endpoint velocity estimate.

    State [position, velocity], transition [[1,1],[0,1]], scalar position
    measurement. `q` is the process (random-acceleration) variance, `r` the
    measurement variance — their ratio sets how hard the filter smooths.
    """
    if len(window) < 2:
        return 0.0
    x0, x1 = window[0], 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0
    for z in window[1:]:
        # Predict (F = [[1,1],[0,1]]).
        x0, x1 = x0 + x1, x1
        np00 = p00 + p10 + p01 + p11 + q
        np01 = p01 + p11
        np10 = p10 + p11
        np11 = p11 + q
        # Update (H = [1,0], R = r).
        s = np00 + r
        k0, k1 = np00 / s, np10 / s
        y = z - x0
        x0, x1 = x0 + k0 * y, x1 + k1 * y
        p00 = (1 - k0) * np00
        p01 = (1 - k0) * np01
        p10 = np10 - k1 * np00
        p11 = np11 - k1 * np01
    return x1


def tvdiff(window: Sequence[float], lam: float = 0.02, iters: int = 12,
           step: float = 0.05) -> float:
    """Light total-variation-regularised derivative (noise-robust differentiation):
    gradient descent on ½‖cumsum(u) − x‖² + λ·TV(u), returning the endpoint u.
    A few iterations suffice for a short window; returns a smoothed increment."""
    n = len(window)
    if n < 2:
        return 0.0
    u = [0.0] + [window[i] - window[i - 1] for i in range(1, n)]
    for _ in range(iters):
        integ = [window[0]]
        for i in range(1, n):
            integ.append(integ[-1] + u[i])
        res = [integ[i] - window[i] for i in range(n)]
        grad = [sum(res[i:]) for i in range(n)]          # A^T residual (A = cumsum)
        for i in range(1, n - 1):
            tv = (u[i] - u[i - 1]) - (u[i + 1] - u[i])
            u[i] -= step * (grad[i] + lam * tv)
        u[-1] -= step * grad[-1]
    return u[-1]


ESTIMATORS: Dict[str, Callable[[Sequence[float]], float]] = {
    "sg1": sg1, "sg2": sg2, "holt": holt, "kalman": kalman, "tvdiff": tvdiff,
}


def _trend_consistency_with(window: Sequence[float],
                            estimator: Callable[[Sequence[float]], float]) -> float:
    """`trend_consistency` but using `estimator` for the trend-direction reference,
    so the ranking measures the derivative estimator, not the consistency idea."""
    incs = _increments(window)
    v = estimator(window)
    d = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
    if not incs or d == 0.0:
        return 0.0
    return d * (sum(1 for x in incs if (x > 0) == (d > 0)) / len(incs))


def rank_estimators(data_path: str | None = None):
    """Out-of-time bake-off: each estimator as the trend-direction reference,
    ranked by |AUC−0.5| on the held-out slice (mirrors eval_deriv). Returns a
    list of (name, auc, wf_min_strength), best first. Import-heavy deps stay local
    so importing this module for the estimators alone costs nothing."""
    import json
    import os
    from features import WINDOW, HORIZON, MIN_STD, _std
    from eval_deriv import _auc, _oot_split, _walk_forward, Row

    path = data_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "data", "series.json")
    series = json.load(open(path, encoding="utf-8"))
    out = []
    for name, est in ESTIMATORS.items():
        rbs = []
        for p in series:
            rows = []
            n = len(p)
            for i in range(WINDOW, n - HORIZON):
                w = p[i - WINDOW : i]
                incs = [w[k] - w[k - 1] for k in range(1, len(w))]
                if _std(incs) < MIN_STD:
                    continue
                rows.append(Row({"f": _trend_consistency_with(w, est)},
                                1 if p[i + HORIZON] - p[i] > 0 else 0,
                                p[i + HORIZON] - p[i]))
            if rows:
                rbs.append(rows)
        _, va = _oot_split(rbs)
        auc = _auc([r.feats["f"] for r in va], [r.label for r in va])
        wf = [abs(_auc([r.feats["f"] for r in f], [r.label for r in f]) - 0.5)
              for _, f in _walk_forward(rbs)]
        out.append((name, auc, min(wf) if wf else 0.0))
    out.sort(key=lambda t: abs(t[1] - 0.5), reverse=True)
    return out


def _selfcheck() -> None:
    # On a clean quadratic ramp every estimator recovers a positive velocity, and
    # on a falling ramp a negative one — a sanity floor, not a precision test.
    up = [0.2 + 0.01 * i + 0.001 * i * i for i in range(16)]
    dn = [0.9 - 0.01 * i for i in range(16)]
    for name, est in ESTIMATORS.items():
        assert est(up) > 0, name
        assert est(dn) < 0, name
    # SG1 on a pure line is exactly the slope.
    assert abs(sg1([0.3 + 0.02 * i for i in range(8)]) - 0.02) < 1e-9
    # Flat window → zero velocity for all.
    for name, est in ESTIMATORS.items():
        assert abs(est([0.5] * 16)) < 1e-6, name
    print(f"deriv_estimators selfcheck ok — {len(ESTIMATORS)} causal estimators")


if __name__ == "__main__":
    import sys

    if "--rank" in sys.argv[1:]:
        ranked = rank_estimators()
        print("Causal derivative estimator bake-off (out-of-time, as the "
              "trend_consistency direction reference):")
        for name, auc, wf in ranked:
            print(f"  {name:<8} AUC {auc:.4f}  strength {abs(auc-0.5):.4f}  wf_min {wf:.4f}")
        best = ranked[0]
        sg1_auc = next(a for n, a, _ in ranked if n == "sg1")
        print(f"\nBest: {best[0]} (AUC {best[1]:.4f}, wf_min {best[2]:.4f}). "
              f"Production keeps sg1 (AUC {sg1_auc:.4f}) — within noise at a "
              f"fraction of the compute; {best[0]} is the marginal-best alternative.")
    else:
        _selfcheck()
