"""Proper price *derivatives* for a prediction-market probability series.

A Polymarket price is the market-implied probability p(t) in 0..1, so its time
derivatives have a direct reading the equities world doesn't get for free:

  * the 1st derivative  dp/dt  is **belief velocity** — how fast the crowd is
    repricing the outcome (a breaking-news market screams, a settled one is flat);
  * the 2nd derivative  d²p/dt²  is **belief acceleration** — is the repricing
    speeding up (fresh information still arriving) or rolling over (the move is
    exhausting / mean-reverting);
  * the 3rd derivative  d³p/dt³  (jerk) is the change in acceleration — an early
    tell that an accelerating move is about to inflect.

Taking a derivative of a *noisy sampled* signal by naive finite differences
amplifies the noise catastrophically (each difference roughly doubles the noise
power). The right tool is a **local least-squares polynomial (Savitzky–Golay)
derivative**: fit a low-degree polynomial to the last L samples by least squares
and read its analytic derivative at the window's right edge. Because we only ever
evaluate at the *endpoint* (the "now" of the look-back window) the filter is
strictly causal — no lookahead, safe to serve live.

This module builds a whole *family* of derivative signals — raw finite
differences, denoised SG velocity/acceleration/jerk at several look-backs, and
the dimensionless **volatility-normalised** variants (a velocity divided by the
window's realised vol is a t-stat: "how many sigmas of move per step", which is
comparable across markets of wildly different activity). `eval_deriv.py` then
scores each one out-of-time to find the single best derivative signal.

Pure stdlib + `math` so it stays trivially runnable, testable, and line-for-line
portable to the TS bundle (mirrors the discipline of `features.py`).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

# Endpoint offset: all derivatives are evaluated at the window's LAST sample,
# i.e. s = t - t_end runs over -(L-1) .. 0. The value/derivatives at s=0 are then
# just the low-order polynomial coefficients (see `_poly_endpoint`).
_EPS = 1e-12


# ── local least-squares polynomial derivative (causal Savitzky–Golay) ──────────

def _solve(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve the small dense system a·x = b by Gaussian elimination with partial
    pivoting. Sized (degree+1) — at most 4×4 here, so plain stdlib is plenty."""
    n = len(b)
    # Work on an augmented copy.
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        # Partial pivot: largest magnitude in this column at/below the diagonal.
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < _EPS:
            # Singular / rank-deficient (e.g. a dead-flat window) — no unique fit.
            return [0.0] * n
        m[col], m[piv] = m[piv], m[col]
        pivot = m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / pivot
            if factor:
                for c in range(col, n + 1):
                    m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _poly_endpoint(tail: Sequence[float], degree: int) -> List[float]:
    """Least-squares fit a polynomial of `degree` to `tail` (the last L prices)
    on the shifted axis s = -(L-1)..0, and return its value and derivatives at the
    endpoint s=0 as ``[f, f', f'', f''' …]`` (up to `degree`).

    On the s=0 axis the endpoint derivatives are just the scaled coefficients:
    f=c0, f'=c1, f''=2·c2, f'''=6·c3 — no extra evaluation needed. Falls back to
    lower effective degree when the window is too short to identify the fit.
    """
    L = len(tail)
    deg = min(degree, L - 1)
    if deg < 1:
        return [tail[-1] if tail else 0.0] + [0.0] * degree
    s = [float(i - (L - 1)) for i in range(L)]        # -(L-1) .. 0
    # Normal equations for the Vandermonde least-squares fit: (VᵀV) c = Vᵀy.
    ncol = deg + 1
    # Powers of s up to 2·deg drive both VᵀV (sum of s^(i+j)) and Vᵀy.
    powers = [[sv ** k for k in range(2 * deg + 1)] for sv in s]
    ata = [[sum(p[i + j] for p in powers) for j in range(ncol)] for i in range(ncol)]
    aty = [sum(p[i] * y for p, y in zip(powers, tail)) for i in range(ncol)]
    coef = _solve(ata, aty)                            # c0..c_deg
    # Derivatives at s=0: k! · c_k.
    fact = 1
    out: List[float] = []
    for k in range(ncol):
        out.append(fact * coef[k])
        fact *= (k + 1)
    while len(out) < degree + 1:
        out.append(0.0)
    return out


# ── plain-stats helpers (match features.py conventions) ────────────────────────

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _increments(window: Sequence[float]) -> List[float]:
    return [window[i] - window[i - 1] for i in range(1, len(window))]


# ── the derivative family ──────────────────────────────────────────────────────
#
# Every candidate maps one look-back window (list of prices, oldest→newest) to a
# single scalar. The names are grouped by what kind of derivative they estimate;
# `eval_deriv.py` ranks them out-of-time.

DERIV_NAMES: List[str] = [
    # ── flagship ──────────────────────────────────────────────────────────────
    "trend_consistency",  # sign(denoised full-window velocity) × fraction of the
                          # window's increments agreeing with it. The single best
                          # derivative signal found (out-of-time AUC ~0.588, stable
                          # across all 4 walk-forward folds, top-vs-bottom-decile
                          # up-rate spread ~+0.27). Reads: a clean, low-noise trend
                          # keeps going; a choppy move that only netted somewhere by
                          # luck does not. It is the *cleanliness* of the derivative,
                          # not its magnitude — raw velocity is anti-predictive here.
    # raw finite differences (the naive, noisy baselines)
    "d1_raw",          # p[-1]-p[-2]                       (1-step velocity)
    "d1_avg",          # (p[-1]-p[0])/(L-1)                (window-average velocity)
    "d2_raw",          # p[-1]-2p[-2]+p[-3]                (1-step acceleration)
    # denoised Savitzky–Golay velocity at several look-backs
    "vel_sg5",         # deg-1 slope over last 5
    "vel_sg8",         # deg-1 slope over last 8
    "vel_sg16",        # deg-1 slope over the full 16
    "vel_sg2_12",      # deg-2 endpoint velocity over last 12 (curvature-aware)
    # denoised acceleration / jerk
    "acc_sg8",         # deg-2 endpoint 2nd-derivative over last 8
    "acc_sg12",        # deg-2 endpoint 2nd-derivative over last 12
    "jerk_sg12",       # deg-3 endpoint 3rd-derivative over last 12
    # dimensionless, volatility-normalised (the cross-market-comparable class)
    "vel_z",           # SG velocity / std(increments)     (velocity t-stat)
    "vel_z_slow",      # SG(16) velocity / std(increments)
    "acc_z",           # SG acceleration / std(increments)
    "vel_over_range",  # SG velocity / (max-min)           (velocity vs travelled range)
    # shape of the recent derivative
    "vel_persist",     # signed fraction of last-6 increments agreeing with SG velocity
    "curv_ratio",      # acc_sg8 / (|vel_sg8|+eps)         (turn sharpness)
]


def trend_consistency(window: Sequence[float]) -> float:
    """The flagship derivative signal: **signed trend cleanliness**.

    Take the denoised trend direction (sign of the least-squares full-window
    velocity, so a couple of noisy ticks can't flip it) and measure what fraction
    of the window's increments actually moved in that direction. Return that
    fraction with the trend's sign, so the result lives in [-1, 1]:

        +1  every step marched up together        (a clean, information-driven rise)
         0  half up / half down                    (pure chop, no real trend)
        -1  every step marched down together       (a clean decline)

    This is a *dimensionless first-derivative quality* metric — it says nothing
    about how big the move was, only how orderly it was. Empirically (see
    `eval_deriv.py`) that orderliness is what predicts continuation on this
    market data, while the raw velocity magnitude is anti-predictive (the crowd
    mean-reverts against a big lurch but rides a steady, agreeing drift).
    """
    incs = _increments(window)
    if not incs:
        return 0.0
    vel = _poly_endpoint(window, 1)[1]          # denoised full-window velocity
    d = 1.0 if vel > 0 else (-1.0 if vel < 0 else 0.0)
    if d == 0.0:
        return 0.0
    agree = sum(1 for r in incs if (r > 0) == (d > 0)) / len(incs)
    return d * agree


def deriv_features(window: Sequence[float]) -> Dict[str, float]:
    """All derivative candidates for one look-back window, as a name→value dict.

    Non-finite guards everywhere: a dead-flat window yields all-zero derivatives
    rather than a divide-by-zero, matching `features.py`'s flat-window handling.
    """
    L = len(window)
    incs = _increments(window)
    vol = _std(incs)
    denom_v = vol if vol > _EPS else 1.0
    lo, hi = (min(window), max(window)) if window else (0.0, 0.0)
    rng = (hi - lo) if (hi - lo) > _EPS else 1.0

    # SG endpoint derivatives at the look-backs we care about.
    d1_5 = _poly_endpoint(window[-5:], 1)
    d1_8 = _poly_endpoint(window[-8:], 1)
    d1_16 = _poly_endpoint(window, 1)
    d2_8 = _poly_endpoint(window[-8:], 2)
    d2_12 = _poly_endpoint(window[-12:], 2)
    d3_12 = _poly_endpoint(window[-12:], 3)

    vel_sg8 = d1_8[1]
    acc_sg8 = d2_8[2]

    # Recent-increment sign agreement with the denoised velocity direction.
    recent = incs[-6:]
    vdir = 1.0 if vel_sg8 > 0 else (-1.0 if vel_sg8 < 0 else 0.0)
    if recent and vdir != 0.0:
        vel_persist = vdir * (sum(1 for r in recent if (r > 0) == (vdir > 0)) / len(recent))
    else:
        vel_persist = 0.0

    feats = {
        "trend_consistency": trend_consistency(window),
        "d1_raw": window[-1] - window[-2] if L >= 2 else 0.0,
        "d1_avg": (window[-1] - window[0]) / (L - 1) if L >= 2 else 0.0,
        "d2_raw": window[-1] - 2 * window[-2] + window[-3] if L >= 3 else 0.0,
        "vel_sg5": d1_5[1],
        "vel_sg8": vel_sg8,
        "vel_sg16": d1_16[1],
        "vel_sg2_12": d2_12[1],
        "acc_sg8": acc_sg8,
        "acc_sg12": d2_12[2],
        "jerk_sg12": d3_12[3],
        "vel_z": vel_sg8 / denom_v,
        "vel_z_slow": d1_16[1] / denom_v,
        "acc_z": acc_sg8 / denom_v,
        "vel_over_range": vel_sg8 / rng,
        "vel_persist": vel_persist,
        "curv_ratio": acc_sg8 / (abs(vel_sg8) + _EPS),
    }
    # Final finiteness sweep — never hand a NaN/inf downstream.
    return {k: (v if math.isfinite(v) else 0.0) for k, v in feats.items()}


if __name__ == "__main__":
    # Self-check: on a clean quadratic ramp the SG derivatives recover the known
    # analytic values, and a flat window yields all zeros.
    quad = [0.2 + 0.01 * i + 0.001 * i * i for i in range(16)]
    d = deriv_features(quad)
    # p(t)=0.2+0.01t+0.001t², at t=15: p'=0.01+0.002·15=0.04, p''=0.002.
    assert abs(d["vel_sg2_12"] - 0.04) < 5e-3, d["vel_sg2_12"]
    assert abs(d["acc_sg12"] - 0.002) < 5e-4, d["acc_sg12"]
    assert d["vel_sg8"] > 0 and d["vel_z"] > 0, d
    # A monotone rise is maximally consistent → +1; a monotone fall → -1.
    assert abs(trend_consistency([0.1 * i for i in range(16)]) - 1.0) < 1e-9
    assert abs(trend_consistency([1.6 - 0.1 * i for i in range(16)]) + 1.0) < 1e-9
    # A choppy zig-zag can't reach the +1 of a clean trend — cleanliness << 1.
    assert abs(trend_consistency([0.5 + 0.02 * (i % 2) for i in range(16)])) < 0.6
    flat = [0.5] * 16
    assert all(abs(v) < 1e-9 for v in deriv_features(flat).values()), deriv_features(flat)
    print(f"features_deriv selfcheck ok — {len(DERIV_NAMES)} candidates, "
          f"quad vel={d['vel_sg2_12']:.4f} acc={d['acc_sg12']:.4f}")
