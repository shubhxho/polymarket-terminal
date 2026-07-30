"""Meta-ensemble combiner — one blended, calibrated signal from the base models.

The suite trains three independent base models, each answering a different
question about a market:

  - **resolve** (`features_resolve` / `train_resolve`): the calibrated
    probability that YES ultimately *resolves* true, from a mid-life snapshot.
  - **flow** (`features_flow` / `modal_flow`): short-horizon *direction* from the
    true aggressor order-flow tape (taker_direction).
  - **smart** (`features_smart` / `modal_smart`): the *smart-money* lean — which
    way the profitable-wallet cohort is positioned.

Nothing combined them: each shipped its own probability and a caller had to pick.
That is exactly the setting where stacking pays off — the three views are
partly independent, so a blend of them is a strictly better estimator than any
one alone, *and* their (dis)agreement is itself signal: when all three lean the
same way the call is trustworthy, when they fight it is not.

This module is the combiner. `blend()` folds the per-model probabilities into a
single calibrated signal plus a `confidence` and an `agreement` score, with
**agreement gating** (down-weight confidence when the models disagree, boost it
when they concur) and a graceful path when some base models are missing —
weights renormalise over whichever models are present. `StackWeights` is a small
logistic stacker fit by in-repo gradient descent (no sklearn); unfit it falls
back to sane equal weights.

Pure stdlib (`math`, `dataclasses`) — trivially runnable and testable, no numpy /
pandas / sklearn / GPU. Matches the `features_ohlcv.py` module shape: a named
list of expected inputs (`MODELS`), a result dataclass (`EnsembleSignal`), and a
`__main__` selfcheck.

Run:  python ml/ensemble.py      (selfcheck; exits non-zero on any failure)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# The base models this ensemble expects, in a stable order. A prediction dict may
# carry any subset of these — missing models are dropped and the remaining
# weights renormalise over what is present (see `blend`).
MODELS: List[str] = ["resolve", "flow", "smart"]

# Human-readable gloss, handy for reporting / MCP surfaces.
MODEL_LABELS: Dict[str, str] = {
    "resolve": "resolution-probability",
    "flow": "order-flow direction",
    "smart": "smart-money lean",
}

_EPS = 1e-6
# The largest a population std of probabilities in [0, 1] can be: half the mass at
# 0 and half at 1. Used to normalise disagreement into a 0..1 agreement score.
_MAX_PROB_STD = 0.5

# A per-model prediction is either a bare probability, a (prob, confidence) pair,
# or a mapping carrying "prob"/"p" and optional "confidence"/"conf".
Pred = Union[float, int, Tuple[float, float], Mapping[str, float]]


# ── small numeric helpers (stdlib only) ──────────────────────────────────────

def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _sigmoid(z: float) -> float:
    # Numerically stable logistic — never overflows for large |z|.
    if z >= 0.0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    """Log-odds of a probability, clamped off {0,1} so it stays finite."""
    q = min(1.0 - _EPS, max(_EPS, p))
    return math.log(q / (1.0 - q))


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pstdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _is_prob(x: float) -> bool:
    return math.isfinite(x) and 0.0 <= x <= 1.0


# ── prediction parsing ────────────────────────────────────────────────────────

def _default_conf(p: float) -> float:
    """A model's self-confidence when it reports none: how decisive its call is,
    i.e. distance from the 0.5 coin-flip, scaled to 0..1 (0.5→0, 0/1→1)."""
    return _clip01(abs(p - 0.5) * 2.0)


def _parse_one(v: Pred) -> Optional[Tuple[float, float]]:
    """Normalise one model's output to (prob, confidence) or None if unusable."""
    prob: Optional[float] = None
    conf: Optional[float] = None
    if isinstance(v, Mapping):
        raw = v.get("prob", v.get("p"))
        if raw is not None:
            prob = float(raw)
        c = v.get("confidence", v.get("conf"))
        if c is not None:
            conf = float(c)
    elif isinstance(v, (tuple, list)):
        if len(v) >= 1:
            prob = float(v[0])
        if len(v) >= 2 and v[1] is not None:
            conf = float(v[1])
    elif isinstance(v, bool):
        return None  # guard: bool is an int subclass, but not a probability here
    elif isinstance(v, (int, float)):
        prob = float(v)
    if prob is None or not _is_prob(prob):
        return None
    if conf is None or not math.isfinite(conf):
        conf = _default_conf(prob)
    return prob, _clip01(conf)


def _parse_preds(preds: Mapping[str, Pred]) -> Dict[str, Tuple[float, float]]:
    """Keep only expected models with a valid probability → {model: (prob, conf)}.
    Unknown keys and unusable values are dropped silently (graceful missing)."""
    out: Dict[str, Tuple[float, float]] = {}
    for m in MODELS:
        if m in preds:
            parsed = _parse_one(preds[m])
            if parsed is not None:
                out[m] = parsed
    return out


# ── agreement ─────────────────────────────────────────────────────────────────

def agreement_score(probs: Sequence[float]) -> float:
    """How much the base models concur, in 0..1.

    1.0 = identical calls, 0.0 = maximally split (half certain-YES, half
    certain-NO). Derived from the population std of the probabilities, normalised
    by its theoretical maximum (0.5). A lone model trivially "agrees" (1.0); the
    coverage penalty in `blend` keeps a single-model signal from overclaiming."""
    if len(probs) <= 1:
        return 1.0
    return _clip01(1.0 - _pstdev(probs) / _MAX_PROB_STD)


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleSignal:
    """The blended signal: a calibrated probability with its trust metadata.

    prob         final YES probability in 0..1
    confidence   how much to trust it in 0..1 (agreement × decisiveness × coverage)
    agreement    base-model concurrence in 0..1 (see `agreement_score`)
    contributing base models that actually fed this signal (present & valid)
    """

    prob: float
    confidence: float
    agreement: float
    contributing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "prob": round(self.prob, 6),
            "confidence": round(self.confidence, 6),
            "agreement": round(self.agreement, 6),
            "contributing": list(self.contributing),
        }


# ── learned stacker ───────────────────────────────────────────────────────────

@dataclass
class StackWeights:
    """A logistic stacker over the base models' log-odds.

    The combined log-odds is `bias + Σ_j w_j · logit(p_j)`, so the blend is a
    calibrated logistic function of the base probabilities. Weights are fit by
    plain batch gradient descent on binary cross-entropy (with a little L2) — no
    sklearn. Unfit, it defaults to unit weights over every model (a sensible
    "trust each equally" prior)."""

    weights: Dict[str, float]
    bias: float = 0.0
    fitted: bool = False

    @classmethod
    def default(cls) -> "StackWeights":
        return cls({m: 1.0 for m in MODELS}, 0.0, False)

    def predict_one(self, row: Mapping[str, float]) -> float:
        """Pure logistic-stacker probability for one row of base predictions.

        Missing models contribute logit(0.5)=0 (a neutral, weight-free vote)."""
        z = self.bias
        for m, w in self.weights.items():
            if m in row and _is_prob(float(row[m])):
                z += w * _logit(float(row[m]))
        return _sigmoid(z)

    def logloss(self, rows: Sequence[Mapping[str, float]], labels: Sequence[float]) -> float:
        """Mean binary cross-entropy of `predict_one` against 0/1 labels."""
        if not rows:
            return 0.0
        tot = 0.0
        for r, y in zip(rows, labels):
            p = min(1.0 - _EPS, max(_EPS, self.predict_one(r)))
            yy = float(y)
            tot += -(yy * math.log(p) + (1.0 - yy) * math.log(1.0 - p))
        return tot / len(rows)

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, float]],
        labels: Sequence[float],
        *,
        models: Optional[Sequence[str]] = None,
        lr: float = 0.2,
        epochs: int = 800,
        l2: float = 1e-4,
    ) -> "StackWeights":
        """Fit the stacker by batch gradient descent on logloss.

        `rows` are dicts of {model: probability}; `labels` are 0/1 outcomes.
        Missing entries impute logit(0.5)=0, so partially-observed rows are fine."""
        ms = list(models) if models is not None else list(MODELS)
        n = len(rows)
        if n == 0:
            return cls.default()
        # Design matrix in log-odds space (0 for missing = neutral).
        X = [[_logit(float(r[m])) if (m in r and _is_prob(float(r[m]))) else 0.0
              for m in ms] for r in rows]
        y = [float(v) for v in labels]
        w = [0.0] * len(ms)
        b = 0.0
        for _ in range(epochs):
            gw = [0.0] * len(ms)
            gb = 0.0
            for i in range(n):
                z = b + sum(w[j] * X[i][j] for j in range(len(ms)))
                err = _sigmoid(z) - y[i]
                for j in range(len(ms)):
                    gw[j] += err * X[i][j]
                gb += err
            for j in range(len(ms)):
                w[j] -= lr * (gw[j] / n + l2 * w[j])
            b -= lr * (gb / n)
        return cls({ms[j]: w[j] for j in range(len(ms))}, b, True)


# ── the combiner ──────────────────────────────────────────────────────────────

def blend(
    preds: Mapping[str, Pred],
    weights: Optional[StackWeights] = None,
    *,
    min_models: int = 1,
) -> EnsembleSignal:
    """Fuse per-model predictions into one calibrated `EnsembleSignal`.

    `preds` maps model name → prediction (a bare probability, a (prob, conf)
    pair, or a mapping with "prob"/"confidence"). Only models in `MODELS` count;
    missing/invalid ones are dropped and everything renormalises over the rest.

    The blended probability is a weighted mix of the base log-odds, where each
    model's pull is its (renormalised, non-negative) stack weight times its own
    confidence — so a decisive model moves the signal more than a wishy-washy
    one, and a model sitting at 0.5 stays out of the way. Here the stack weights
    act only as *relative* importance (a fitted stacker teaches the blend to
    lean less on a noisier model); the fully-calibrated absolute-scale logistic —
    learned `bias` and unnormalised weights included — is `StackWeights.predict_one`.
    `confidence` is gated
    by agreement: it is `agreement × mean_decisiveness × coverage`, so scattered
    votes, wishy-washy votes, or missing models all pull trust down; unanimous
    decisive votes across all models push it up.
    """
    if weights is None:
        weights = StackWeights.default()
    parsed = _parse_preds(preds)
    present = [m for m in MODELS if m in parsed]
    if len(present) < max(1, min_models):
        # Nothing usable to combine — return an explicit no-signal.
        return EnsembleSignal(prob=0.5, confidence=0.0, agreement=0.0, contributing=[])

    probs = [parsed[m][0] for m in present]
    confs = [parsed[m][1] for m in present]

    # Stack weights renormalised over the present models (graceful-missing path).
    # Negatives are clamped to 0 for the blend's convex mix; the pure logistic
    # `StackWeights.predict_one` is where signed weights live.
    w = [max(0.0, weights.weights.get(m, 0.0)) for m in present]
    if sum(w) <= _EPS:
        w = [1.0] * len(present)
    wsum = sum(w)
    w = [x / wsum for x in w]

    # Fold per-model confidence into the effective weight.
    eff = [w[i] * confs[i] for i in range(len(present))]
    esum = sum(eff)
    if esum <= _EPS:
        # Every present model is maximally wishy-washy (conf≈0): fall back to the
        # stack weights alone so we still return a sensible (near-0.5) prob.
        eff = w[:]
        esum = sum(eff)
    eff = [x / esum for x in eff]

    # Relative-importance convex blend of log-odds. The learned bias is an
    # absolute-scale offset tuned for `predict_one`'s unnormalised weights, so it
    # is deliberately NOT applied here (it would shift every blended signal).
    logit = sum(eff[i] * _logit(probs[i]) for i in range(len(present)))
    prob = _clip01(_sigmoid(logit))

    agreement = agreement_score(probs)
    coverage = len(present) / len(MODELS)
    mean_conf = _mean(confs)
    # Coverage never zeroes confidence on its own — a single present model still
    # gets a floor of 0.5× — but full coverage is required to reach the ceiling.
    confidence = _clip01(agreement * mean_conf * (0.5 + 0.5 * coverage))

    return EnsembleSignal(
        prob=prob,
        confidence=confidence,
        agreement=agreement,
        contributing=present,
    )


# ── selfcheck ─────────────────────────────────────────────────────────────────

def _selfcheck() -> None:
    # 1. Finite & bounded over a grid of synthetic base-model outputs, including
    #    the degenerate corners.
    grid = [0.0, 0.02, 0.25, 0.5, 0.75, 0.98, 1.0]
    checked = 0
    for a in grid:
        for b in grid:
            for c in grid:
                sig = blend({"resolve": a, "flow": b, "smart": c})
                for name, val in (("prob", sig.prob),
                                  ("confidence", sig.confidence),
                                  ("agreement", sig.agreement)):
                    assert math.isfinite(val), (name, a, b, c)
                    assert 0.0 <= val <= 1.0, (name, val, a, b, c)
                assert sig.contributing == MODELS, sig.contributing
                checked += 1
    assert checked == len(grid) ** 3

    # 2. Unanimous decisive calls → high agreement AND high confidence.
    strong = blend({"resolve": 0.92, "flow": 0.90, "smart": 0.94})
    assert strong.prob > 0.8, strong.prob
    assert strong.agreement > 0.9, strong.agreement
    assert strong.confidence > 0.7, strong.confidence

    # 3. A dead split → low agreement, and confidence crushed by the gate.
    split = blend({"resolve": 0.95, "flow": 0.05, "smart": 0.5})
    assert split.agreement < 0.3, split.agreement
    assert split.confidence < strong.confidence, (split.confidence, strong.confidence)

    # 4. Agreement → confidence monotonicity. Hold mean decisiveness and coverage
    #    fixed (explicit equal confidences, all three models present) and only
    #    tighten the probability spread; confidence must rise monotonically.
    conf_by_spread = []
    for spread in (0.45, 0.30, 0.15, 0.0):
        trio = {
            "resolve": {"prob": _clip01(0.8 + spread), "confidence": 0.6},
            "flow": {"prob": 0.8, "confidence": 0.6},
            "smart": {"prob": _clip01(0.8 - spread), "confidence": 0.6},
        }
        s = blend(trio)
        conf_by_spread.append((s.agreement, s.confidence))
    for i in range(1, len(conf_by_spread)):
        assert conf_by_spread[i][0] >= conf_by_spread[i - 1][0] - 1e-12, conf_by_spread
        assert conf_by_spread[i][1] >= conf_by_spread[i - 1][1] - 1e-12, conf_by_spread
    assert conf_by_spread[-1][1] > conf_by_spread[0][1], conf_by_spread

    # 5. Graceful missing: fewer models → lower coverage → lower confidence,
    #    same decisive call. Signal still finite and sensible.
    full = blend({"resolve": 0.9, "flow": 0.9, "smart": 0.9})
    partial = blend({"resolve": 0.9, "flow": 0.9})
    solo = blend({"resolve": 0.9})
    assert partial.contributing == ["resolve", "flow"]
    assert solo.contributing == ["resolve"]
    assert full.confidence > partial.confidence > solo.confidence, (
        full.confidence, partial.confidence, solo.confidence)
    assert 0.0 <= solo.prob <= 1.0

    # 6. Empty / unusable input → explicit no-signal, not a crash.
    none_sig = blend({})
    assert none_sig.contributing == [] and none_sig.confidence == 0.0
    junk = blend({"resolve": float("nan"), "flow": 2.0, "smart": None,
                  "unknown_model": 0.9})
    assert junk.contributing == [] and junk.prob == 0.5

    # 7. The learned stacker beats the unfit default on separable synthetic data.
    rows: List[Dict[str, float]] = []
    labels: List[float] = []
    rng = _lcg(12345)
    for _ in range(600):
        y = 1 if next(rng) > 0.5 else 0
        base = 0.65 if y else 0.35
        row = {
            "resolve": _clip01(base + 0.25 * (next(rng) - 0.5)),
            "flow": _clip01(base + 0.25 * (next(rng) - 0.5)),
            # a deliberately noisy / weak model the stacker should learn to trust less
            "smart": _clip01(0.5 + 0.6 * (next(rng) - 0.5)),
        }
        rows.append(row)
        labels.append(float(y))
    fitted = StackWeights.fit(rows, labels)
    assert fitted.fitted
    ll_fit = fitted.logloss(rows, labels)
    ll_def = StackWeights.default().logloss(rows, labels)
    assert ll_fit < ll_def, (ll_fit, ll_def)
    assert all(math.isfinite(v) for v in fitted.weights.values())

    # 8. blend honours a fitted stacker (bias/weights flow through) and stays bounded.
    bs = blend({"resolve": 0.7, "flow": 0.6, "smart": 0.55}, weights=fitted)
    assert 0.0 <= bs.prob <= 1.0 and 0.0 <= bs.confidence <= 1.0

    print("ensemble selfcheck ok "
          f"(grid={checked}, strong.conf={strong.confidence:.3f}, "
          f"split.conf={split.confidence:.3f}, "
          f"logloss fit={ll_fit:.4f} < default={ll_def:.4f})")


def _lcg(seed: int) -> Iterable[float]:
    """Tiny deterministic uniform(0,1) generator — keeps the selfcheck stdlib-only
    and reproducible without seeding the global `random`."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


if __name__ == "__main__":
    _selfcheck()
