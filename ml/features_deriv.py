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

import json
import math
import os
from typing import Dict, List, Optional, Sequence

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


def _ema(xs: Sequence[float], span: int) -> List[float]:
    a = 2.0 / (span + 1.0)
    out = [xs[0]]
    for x in xs[1:]:
        out.append(a * x + (1 - a) * out[-1])
    return out


def _consistency(window: Sequence[float], k: int, ref_dir: float) -> float:
    """Signed fraction of the last-`k` increments that agree with `ref_dir`.
    `ref_dir` is a denoised trend direction (±1); 0 → no trend → 0."""
    incs = _increments(window)
    rec = incs[-k:] if 0 < k < len(incs) else incs
    if not rec or ref_dir == 0.0:
        return 0.0
    up = ref_dir > 0
    return ref_dir * (sum(1 for r in rec if (r > 0) == up) / len(rec))


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
    # consistency variants + regime (the combined-signal inputs)
    "cons_ema",        # trend consistency on an EMA(3)-denoised series
    "cons_rec",        # recency-weighted trend consistency (recent ticks count more)
    "cons8",           # trend consistency over the last 8 (short-scale)
    "extremeness",     # min(p, 1-p) — room to move; pinned markets don't trend
    "last",            # the raw price level — a *directional* regime term. Unfolded
                       # `extremeness`: near 0 the crowd drifts down, near 1 up, so
                       # the signed level out-predicts the folded room (AUC .596 vs
                       # .582) and is the combiner's strongest single input.
    "tc_x_ext",        # trend_consistency × 2·extremeness (clean trend AND room)
    "tc_x_amp",        # trend_consistency × vel_z         (clean trend AND size)
    "cons_rec_x_ext",  # cons_rec × 2·extremeness (recency-clean trend AND room)
]

# Ordered inputs to the combined derivative signal (`deriv_signal`). Chosen by
# out-of-time greedy forward selection (see RESEARCH note in `train_deriv.py`) —
# the leanest set that reaches the combiner's plateau AUC ~0.62, well above the
# best single derivative (0.588). Order is FROZEN: the trained weights in
# `data/deriv_signal.json` are positional, so never reorder without retraining.
DERIV_SIGNAL_FEATURES: List[str] = [
    "last", "cons_rec_x_ext", "d1_avg", "tc_x_ext", "acc_sg12", "d1_raw",
    "trend_consistency", "cons_rec",
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
    vel_z = vel_sg8 / denom_v

    # Consistency / regime block (the combined-signal inputs).
    tc = trend_consistency(window)
    slow_dir = 1.0 if d1_16[1] > 0 else (-1.0 if d1_16[1] < 0 else 0.0)
    smooth = _ema(window, 3)
    sm_inc = _increments(smooth)
    sm_dir = (1.0 if _poly_endpoint(smooth, 1)[1] > 0
              else -1.0 if _poly_endpoint(smooth, 1)[1] < 0 else 0.0)
    if sm_inc and sm_dir != 0.0:
        cons_ema = sm_dir * (sum(1 for r in sm_inc if (r > 0) == (sm_dir > 0)) / len(sm_inc))
    else:
        cons_ema = 0.0
    if incs and slow_dir != 0.0:
        wts = [math.exp(0.15 * i) for i in range(len(incs))]
        sw = sum(wts) or 1.0
        cons_rec = slow_dir * sum(w for w, r in zip(wts, incs)
                                  if (r > 0) == (slow_dir > 0)) / sw
    else:
        cons_rec = 0.0
    d8 = 1.0 if d1_8[1] > 0 else (-1.0 if d1_8[1] < 0 else 0.0)
    cons8 = _consistency(window, 8, d8)
    extremeness = min(window[-1], 1.0 - window[-1]) if window else 0.0

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
        "vel_z": vel_z,
        "vel_z_slow": d1_16[1] / denom_v,
        "acc_z": acc_sg8 / denom_v,
        "vel_over_range": vel_sg8 / rng,
        "vel_persist": vel_persist,
        "curv_ratio": acc_sg8 / (abs(vel_sg8) + _EPS),
        "cons_ema": cons_ema,
        "cons_rec": cons_rec,
        "cons8": cons8,
        "extremeness": extremeness,
        "last": window[-1] if window else 0.0,
        "tc_x_ext": tc * (2.0 * extremeness),
        "tc_x_amp": tc * vel_z,
        "cons_rec_x_ext": cons_rec * (2.0 * extremeness),
    }
    # Final finiteness sweep — never hand a NaN/inf downstream.
    return {k: (v if math.isfinite(v) else 0.0) for k, v in feats.items()}


# ── the combined derivative signal ─────────────────────────────────────────────
#
# One probability from the whole derivative family, in two flavours, both fit
# strictly out-of-time and both served by a pure-stdlib forward pass (no numpy /
# lightgbm at serve time) so they drop straight into the terminal bundle like
# `mlSignal.ts`:
#   • deriv_signal_gbdt — the FLAGSHIP gradient-boosted trees (out-of-time AUC
#     ~0.649). Frozen as `data/deriv_gbdt.json` by `train_deriv_gbdt.py`.
#   • deriv_signal(…, kind="logistic") — the linear baseline (~0.603). Frozen as
#     `data/deriv_signal.json` by `train_deriv.py`.
# `deriv_signal` with the default kind="auto" serves the GBDT when present.

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_SIGNAL_PATH = os.path.join(_DATA_DIR, "deriv_signal.json")
_GBDT_PATH = os.path.join(_DATA_DIR, "deriv_gbdt.json")
_SIGNAL_CACHE: Optional[dict] = None
_GBDT_CACHE: Optional[dict] = None


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def load_signal_model(path: Optional[str] = None) -> Optional[dict]:
    """Load (and cache) the frozen combiner. Returns None if never trained yet."""
    global _SIGNAL_CACHE
    if path is None and _SIGNAL_CACHE is not None:
        return _SIGNAL_CACHE
    p = path or _SIGNAL_PATH
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        model = json.load(fh)
    if path is None:
        _SIGNAL_CACHE = model
    return model


def _logistic_signal(window: Sequence[float], m: dict) -> float:
    feats = deriv_features(window)
    mean, std, w = m["mean"], m["std"], m["weights"]
    z = float(m["bias"])
    for i, name in enumerate(m["features"]):
        s = std[i] if std[i] > 1e-12 else 1.0
        z += w[i] * ((feats[name] - mean[i]) / s)
    return _sigmoid(z)


# The flagship GBDT is served by walking LightGBM's own `dump_model()` tree JSON.
# A leaf node carries `leaf_value`; an internal node carries `split_feature`,
# `threshold`, `decision_type` ("<=") and left/right child subtrees. Summing the
# leaf values across trees and squashing reproduces `booster.predict` exactly
# (verified to 1e-16), with zero third-party deps — so it ports to the TS bundle.

def load_gbdt_model(path: Optional[str] = None) -> Optional[dict]:
    """Load (and cache) the frozen GBDT. None if never trained."""
    global _GBDT_CACHE
    if path is None and _GBDT_CACHE is not None:
        return _GBDT_CACHE
    p = path or _GBDT_PATH
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        model = json.load(fh)
    if path is None:
        _GBDT_CACHE = model
    return model


def _walk_tree(node: dict, x: List[float]) -> float:
    n = node
    while "leaf_value" not in n:
        xv = x[n["split_feature"]]
        thr = n["threshold"]
        if xv is None or (isinstance(xv, float) and math.isnan(xv)):
            go_left = n.get("default_left", True)
        elif n.get("decision_type", "<=") == "<=":
            go_left = xv <= thr
        else:
            go_left = xv < thr
        n = n["left_child"] if go_left else n["right_child"]
    return n["leaf_value"]


def deriv_signal_gbdt(window: Sequence[float], model: Optional[dict] = None) -> Optional[float]:
    """Flagship combined derivative signal via the frozen GBDT: P(up) in 0..1.

    Out-of-time AUC ~0.649 — well above the linear combiner (0.603) and the best
    single derivative (0.588), because the trees capture the interactions between
    trend cleanliness, room-to-move and move size that a linear model cannot.
    Returns None when the GBDT has not been trained.
    """
    m = model if model is not None else load_gbdt_model()
    if not m:
        return None
    feats = deriv_features(window)
    x = [feats[name] for name in m["features"]]
    raw = sum(_walk_tree(t["tree_structure"], x) for t in m["model"]["tree_info"])
    return _sigmoid(raw)


def deriv_signal(window: Sequence[float], kind: str = "auto",
                 model: Optional[dict] = None) -> Optional[float]:
    """Combined derivative signal: P(up) in 0..1 from the whole family.

    ``kind="auto"`` (default) serves the flagship GBDT when it is present and
    falls back to the linear logistic combiner, then to None. ``kind="gbdt"`` /
    ``kind="logistic"`` force one path. The GBDT is the stronger model
    (out-of-time AUC ~0.649 vs ~0.603); the logistic is the minimal linear
    baseline and the single-flagship fallback for callers that want it.
    """
    if kind == "logistic":
        m = model if model is not None else load_signal_model()
        return _logistic_signal(window, m) if m else None
    if kind == "gbdt":
        return deriv_signal_gbdt(window, model)
    # auto
    g = deriv_signal_gbdt(window)
    if g is not None:
        return g
    m = load_signal_model()
    return _logistic_signal(window, m) if m else None


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
    ff = deriv_features(flat)
    # Every *derivative* is zero on a dead-flat window; `extremeness` and `last`
    # are price *levels* (0.5 here), not derivatives, so they are exempt.
    _levels = {"extremeness", "last"}
    assert all(abs(v) < 1e-9 for k, v in ff.items() if k not in _levels), ff
    assert abs(ff["extremeness"] - 0.5) < 1e-9 and abs(ff["last"] - 0.5) < 1e-9, ff
    print(f"features_deriv selfcheck ok — {len(DERIV_NAMES)} candidates, "
          f"quad vel={d['vel_sg2_12']:.4f} acc={d['acc_sg12']:.4f}")
