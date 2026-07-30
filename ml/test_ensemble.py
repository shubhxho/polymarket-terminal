"""Stdlib tests for the meta-ensemble combiner — no pytest, no GPU.

Run:  python ml/test_ensemble.py   (exits non-zero on the first failed assertion)

Covers the blend contract (bounded prob/confidence/agreement), agreement gating
and its monotonic effect on confidence, graceful renormalisation when base
models are missing, and that the learned `StackWeights` stacker reduces logloss
versus the unfit default on separable synthetic data. Fast enough for every commit.
"""

from __future__ import annotations

import math

from ensemble import (
    MODELS,
    EnsembleSignal,
    StackWeights,
    agreement_score,
    blend,
)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _lcg(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


# ── blend contract ────────────────────────────────────────────────────────────

def test_blend_returns_bounded_signal():
    grid = [0.0, 0.1, 0.5, 0.9, 1.0]
    for a in grid:
        for b in grid:
            for c in grid:
                s = blend({"resolve": a, "flow": b, "smart": c})
                assert isinstance(s, EnsembleSignal)
                for v in (s.prob, s.confidence, s.agreement):
                    assert math.isfinite(v)
                    assert 0.0 <= v <= 1.0, v
                assert s.contributing == MODELS


def test_blend_prob_tracks_consensus_direction():
    up = blend({"resolve": 0.85, "flow": 0.8, "smart": 0.9})
    down = blend({"resolve": 0.15, "flow": 0.2, "smart": 0.1})
    assert 0.7 < up.prob <= 1.0, up.prob
    assert 0.0 <= down.prob < 0.3, down.prob


def test_neutral_models_keep_prob_near_half():
    s = blend({"resolve": 0.5, "flow": 0.5, "smart": 0.5})
    assert approx(s.prob, 0.5, tol=1e-6), s.prob
    assert s.confidence == 0.0, s.confidence   # zero decisiveness → zero trust


# ── agreement gating ──────────────────────────────────────────────────────────

def test_agreement_score_bounds_and_extremes():
    assert approx(agreement_score([0.7, 0.7, 0.7]), 1.0)
    assert approx(agreement_score([1.0, 0.0]), 0.0)          # maximal split
    assert agreement_score([0.9]) == 1.0                      # lone model
    mid = agreement_score([0.8, 0.6, 0.4])
    assert 0.0 < mid < 1.0, mid


def test_agreement_boosts_confidence_vs_disagreement():
    agree = blend({"resolve": 0.9, "flow": 0.9, "smart": 0.9})
    disagree = blend({"resolve": 0.95, "flow": 0.05, "smart": 0.5})
    assert agree.agreement > disagree.agreement
    assert agree.confidence > disagree.confidence


def test_confidence_monotonic_in_agreement():
    """Holding mean decisiveness and coverage fixed (explicit equal confidences,
    all three models present), tightening the spread must not lower — and overall
    must raise — both agreement and confidence."""
    seq = []
    for spread in (0.45, 0.30, 0.15, 0.0):
        s = blend({
            "resolve": {"prob": _clip01(0.8 + spread), "confidence": 0.6},
            "flow": {"prob": 0.8, "confidence": 0.6},
            "smart": {"prob": _clip01(0.8 - spread), "confidence": 0.6},
        })
        seq.append((s.agreement, s.confidence))
    for i in range(1, len(seq)):
        assert seq[i][0] >= seq[i - 1][0] - 1e-12, seq
        assert seq[i][1] >= seq[i - 1][1] - 1e-12, seq
    assert seq[-1][1] > seq[0][1], seq


# ── graceful missing / renormalisation ────────────────────────────────────────

def test_missing_models_renormalise_and_reduce_coverage():
    full = blend({"resolve": 0.9, "flow": 0.9, "smart": 0.9})
    partial = blend({"resolve": 0.9, "flow": 0.9})
    solo = blend({"resolve": 0.9})
    assert full.contributing == ["resolve", "flow", "smart"]
    assert partial.contributing == ["resolve", "flow"]
    assert solo.contributing == ["resolve"]
    # Fewer models present → lower coverage → strictly lower confidence.
    assert full.confidence > partial.confidence > solo.confidence
    for s in (full, partial, solo):
        assert 0.0 <= s.prob <= 1.0


def test_unknown_and_invalid_inputs_are_dropped():
    empty = blend({})
    assert empty.contributing == [] and empty.confidence == 0.0 and empty.prob == 0.5
    junk = blend({
        "resolve": float("nan"),
        "flow": 2.0,          # out of range
        "smart": None,
        "mystery": 0.9,        # not an expected model
    })
    assert junk.contributing == [] and junk.prob == 0.5
    # One good model among junk still yields a usable signal.
    mixed = blend({"resolve": 0.8, "flow": -1.0, "bogus": 0.99})
    assert mixed.contributing == ["resolve"] and mixed.prob > 0.5


def test_pred_input_forms_are_equivalent():
    p_float = blend({"resolve": 0.8, "flow": 0.7, "smart": 0.6})
    p_tuple = blend({"resolve": (0.8,), "flow": (0.7,), "smart": (0.6,)})
    p_dict = blend({"resolve": {"prob": 0.8}, "flow": {"p": 0.7}, "smart": {"prob": 0.6}})
    assert approx(p_float.prob, p_tuple.prob, tol=1e-9)
    assert approx(p_float.prob, p_dict.prob, tol=1e-9)


# ── learned stacker ───────────────────────────────────────────────────────────

def test_default_stackweights_unfit_equal():
    d = StackWeights.default()
    assert not d.fitted
    assert set(d.weights) == set(MODELS)
    assert all(w == 1.0 for w in d.weights.values())
    assert d.bias == 0.0


def _synthetic(n=600, seed=999):
    rng = _lcg(seed)
    rows, labels = [], []
    for _ in range(n):
        y = 1 if next(rng) > 0.5 else 0
        base = 0.65 if y else 0.35
        rows.append({
            "resolve": _clip01(base + 0.25 * (next(rng) - 0.5)),
            "flow": _clip01(base + 0.25 * (next(rng) - 0.5)),
            "smart": _clip01(0.5 + 0.6 * (next(rng) - 0.5)),   # weak/noisy
        })
        labels.append(float(y))
    return rows, labels


def test_stacker_fit_reduces_logloss():
    rows, labels = _synthetic()
    fitted = StackWeights.fit(rows, labels)
    assert fitted.fitted
    ll_fit = fitted.logloss(rows, labels)
    ll_def = StackWeights.default().logloss(rows, labels)
    assert math.isfinite(ll_fit) and ll_fit < ll_def, (ll_fit, ll_def)
    assert all(math.isfinite(w) for w in fitted.weights.values())


def test_stacker_learns_to_downweight_noise():
    rows, labels = _synthetic()
    fitted = StackWeights.fit(rows, labels)
    # The two informative models should carry more weight than the noise model.
    assert fitted.weights["resolve"] > fitted.weights["smart"]
    assert fitted.weights["flow"] > fitted.weights["smart"]


def test_blend_uses_fitted_weights_and_stays_bounded():
    rows, labels = _synthetic()
    fitted = StackWeights.fit(rows, labels)
    s = blend({"resolve": 0.7, "flow": 0.65, "smart": 0.55}, weights=fitted)
    assert 0.0 <= s.prob <= 1.0 and 0.0 <= s.confidence <= 1.0
    # A fitted, positively-weighted consensus above 0.5 should read bullish.
    assert s.prob > 0.5


def test_stacker_handles_missing_entries_in_fit():
    rows, labels = _synthetic(n=300)
    # Drop 'smart' from half the rows — fit must still converge finitely.
    for i, r in enumerate(rows):
        if i % 2 == 0:
            r.pop("smart", None)
    fitted = StackWeights.fit(rows, labels)
    assert fitted.fitted and all(math.isfinite(w) for w in fitted.weights.values())
    assert math.isfinite(fitted.logloss(rows, labels))


def test_selfcheck_runs_clean():
    import ensemble
    ensemble._selfcheck()   # must not raise


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
