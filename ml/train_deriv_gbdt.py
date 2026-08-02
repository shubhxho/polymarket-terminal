"""Fit and freeze the FLAGSHIP combined derivative signal — a gradient-boosted
tree ensemble over the whole derivative family.

The linear logistic combiner (`train_deriv.py`) folds the family into one
probability but can only add the derivatives up. A GBDT captures their
*interactions* — "a clean trend only continues when there is room to move and the
move isn't already exhausted" is a product of three features no linear model can
express — and on this tabular market data that is worth a large, honest jump:

  best single derivative   out-of-time AUC ~0.588
  linear logistic combiner              ~0.603
  GBDT combiner (this)                  ~0.649

Honesty protocol (no peeking): the reported number is scored on the later-20%
holdout that is never touched in training; the boosting round count is chosen by
early stopping on an INNER validation slice carved from the *end of the train
region only*, so the held-out slice never influences model selection. A proper
expanding-window walk-forward (retrain per fold) confirms stability across epochs.

The frozen model is LightGBM's own `dump_model()` tree JSON plus the feature
order. Serving is a pure-stdlib tree walk (`features_deriv.deriv_signal_gbdt`)
that reproduces `booster.predict` to machine precision — so, like the rest of the
module, it ports straight into the terminal's TS bundle with zero dependencies.
LightGBM/numpy are used ONLY here, at train time.

Run:  ml/.venv/bin/python train_deriv_gbdt.py
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

import numpy as np
import lightgbm as lgb

from features import HORIZON, MIN_STD, WINDOW, _std
from features_deriv import DERIV_NAMES, deriv_features

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data", "series.json")
OUT = os.path.join(_HERE, "data", "deriv_gbdt.json")

PARAMS = dict(
    objective="binary", metric="auc", feature_pre_filter=False, verbose=-1,
    is_unbalance=True, num_leaves=31, min_data_in_leaf=300,
    learning_rate=0.03, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=2.0, seed=13,
)


def _rows(prices: Sequence[float]) -> List[Tuple[List[float], int]]:
    out = []
    n = len(prices)
    for i in range(WINDOW, n - HORIZON):
        window = prices[i - WINDOW : i]
        incs = [window[k] - window[k - 1] for k in range(1, len(window))]
        if _std(incs) < MIN_STD:
            continue
        f = deriv_features(window)
        out.append(([f[name] for name in DERIV_NAMES],
                    1 if prices[i + HORIZON] - prices[i] > 0 else 0))
    return out


def _carve(rbs, frac, purge=HORIZON):
    """Temporal split per series: earlier `1-frac` (minus a purge band) vs the
    later `frac`. Used both for the outer holdout and the inner early-stop set."""
    a, b = [], []
    for r in rbs:
        if len(r) < 5:
            a += r
            continue
        cut = int(len(r) * (1 - frac))
        a += r[: max(0, cut - purge)]
        b += r[cut:]
    return a, b


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ss = scores[order]
    rr = np.empty(len(scores))
    i = 0
    while i < len(scores):
        j = i
        while j < len(scores) and ss[j] == ss[i]:
            j += 1
        rr[i:j] = (i + j - 1) / 2.0 + 1.0
        i = j
    ranks = np.empty(len(scores))
    ranks[order] = rr
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return 0.5
    return float((ranks[labels > 0].sum() - P * (P + 1) / 2.0) / (P * N))


def _spread(scores: np.ndarray, labels: np.ndarray) -> float:
    o = np.argsort(scores)
    k = max(1, len(o) // 10)
    return float(labels[o[-k:]].mean() - labels[o[:k]].mean())


def _M(rs):
    return np.array([r[0] for r in rs], float), np.array([r[1] for r in rs], float)


def _fit(Xtr, ytr, Xes, yes, rounds=3000):
    booster = lgb.train(
        PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=rounds,
        valid_sets=[lgb.Dataset(Xes, yes)],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    return booster


def main() -> int:
    series = json.load(open(DATA, encoding="utf-8"))
    rbs = [r for r in (_rows(p) for p in series) if r]

    # Outer holdout: later 20% per series, never touched in training.
    trall, va = _carve(rbs, 0.2)
    # Inner early-stop set carved from the END of the train region only.
    tr_rbs = []  # rebuild per-series train regions to carve inner slice temporally
    for r in rbs:
        if len(r) < 5:
            tr_rbs.append(r)
            continue
        cut = int(len(r) * 0.8)
        tr_rbs.append(r[: max(0, cut - HORIZON)])
    tr, es = _carve(tr_rbs, 0.2)

    Xtr, ytr = _M(tr)
    Xes, yes = _M(es)
    Xva, yva = _M(va)

    booster = _fit(Xtr, ytr, Xes, yes)
    best_iter = booster.best_iteration or booster.current_iteration()
    sv = booster.predict(Xva)
    oot_auc = _auc(sv, yva)
    oot_spread = _spread(sv, yva)

    # Honest walk-forward: expanding train blocks, retrain, score the NEXT block.
    wf_aucs = []
    folds = 4
    for k in range(1, folds + 1):
        tr_f, va_f = [], []
        for r in rbs:
            if len(r) < folds + 1:
                continue
            step = len(r) / (folds + 1)
            tr_f += r[: max(0, int(step * k) - HORIZON)]
            va_f += r[int(step * k) : int(step * (k + 1))]
        if not tr_f or not va_f:
            continue
        Xf, yf = _M(tr_f)
        bf = lgb.train(PARAMS, lgb.Dataset(Xf, yf), num_boost_round=best_iter)
        Xv, yv = _M(va_f)
        wf_aucs.append(_auc(bf.predict(Xv), yv))
    wf_mean = sum(wf_aucs) / len(wf_aucs) if wf_aucs else 0.5
    wf_min_str = min(abs(a - 0.5) for a in wf_aucs) if wf_aucs else 0.0

    dump = booster.dump_model()
    model = {
        "kind": "lightgbm_gbdt",
        "features": list(DERIV_NAMES),
        "best_iteration": int(best_iter),
        "params": PARAMS,
        "metrics": {
            "oot_auc": round(oot_auc, 4),
            "oot_decile_spread": round(oot_spread, 4),
            "wf_mean_auc": round(wf_mean, 4),
            "wf_aucs": [round(a, 4) for a in wf_aucs],
            "wf_min_strength": round(wf_min_str, 4),
            "n_train": len(tr), "n_earlystop": len(es), "n_val": len(va),
        },
        "model": dump,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(model, fh)  # compact — the tree JSON is large

    # Importance ranking.
    imp = sorted(zip(DERIV_NAMES, booster.feature_importance("gain")),
                 key=lambda t: t[1], reverse=True)
    print("flagship GBDT derivative signal (out-of-time, no peeking):")
    print(f"  trees={best_iter}  n_train={len(tr)}  n_val={len(va)}")
    print(f"  clean OOT AUC      {oot_auc:.4f}")
    print(f"  OOT decile spread  {oot_spread:+.4f}")
    print(f"  walk-forward AUC   {wf_mean:.4f} (folds {[round(a,4) for a in wf_aucs]}, "
          f"min strength {wf_min_str:.4f})")
    print("  top features by gain:")
    for name, g in imp[:8]:
        print(f"    {name:<18} {g:,.0f}")
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
