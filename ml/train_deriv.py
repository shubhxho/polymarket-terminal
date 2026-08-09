"""Fit and freeze the combined derivative signal (`features_deriv.deriv_signal`).

A standardised logistic regression over `DERIV_SIGNAL_FEATURES`, trained STRICTLY
out-of-time — on each market's earlier windows only, exactly the split
`eval_deriv`/`train_seq` validate on — so the frozen weights never touch the
validation slice they are then scored against. The forward pass shipped in
`features_deriv.deriv_signal` is pure stdlib; this trainer is the only place numpy
is used, and it writes everything the forward pass needs (feature order, per-
feature train mean/std, weights, bias) to `data/deriv_signal.json`.

Ranking context: the best *single* derivative (`trend_consistency`) reaches
out-of-time AUC ~0.588; this combined signal reaches ~0.607 — the lift from
folding in the consistency variants and the extremeness regime term.

Run:  ml/.venv/bin/python train_deriv.py         # fits, freezes, prints report
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

import numpy as np

from features import HORIZON, MIN_STD, WINDOW, _std
from features_deriv import DERIV_SIGNAL_FEATURES, deriv_features

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data", "series.json")
OUT = os.path.join(_HERE, "data", "deriv_signal.json")


def _rows(prices: Sequence[float]) -> List[Tuple[List[float], int]]:
    """(feature-vector-in-signal-order, up-label) rows for one series, using the
    module's own `deriv_features` so training and serving compute identically."""
    out = []
    n = len(prices)
    for i in range(WINDOW, n - HORIZON):
        window = prices[i - WINDOW : i]
        incs = [window[k] - window[k - 1] for k in range(1, len(window))]
        if _std(incs) < MIN_STD:
            continue
        f = deriv_features(window)
        out.append(([f[name] for name in DERIV_SIGNAL_FEATURES],
                    1 if prices[i + HORIZON] - prices[i] > 0 else 0))
    return out


def _split(rows_by_series, val_frac=0.2):
    tr, va = [], []
    for rows in rows_by_series:
        if len(rows) < 5:
            tr.extend(rows)
            continue
        cut = int(len(rows) * (1 - val_frac))
        tr.extend(rows[: max(0, cut - HORIZON)])
        va.extend(rows[cut:])
    return tr, va


def _walk_forward(rows_by_series, folds=4):
    blocks = folds + 1
    for k in range(1, blocks):
        va = []
        for rows in rows_by_series:
            if len(rows) < blocks:
                continue
            step = len(rows) / blocks
            va.extend(rows[int(step * k) : int(step * (k + 1))])
        if va:
            yield va


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    rr = np.empty(len(scores))
    while i < len(scores):
        j = i
        while j < len(scores) and s[j] == s[i]:
            j += 1
        rr[i:j] = (i + j - 1) / 2.0 + 1.0
        i = j
    ranks[order] = rr
    npos = labels.sum()
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    return float((ranks[labels > 0].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _spread(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    k = max(1, len(order) // 10)
    return float(labels[order[-k:]].mean() - labels[order[:k]].mean())


def _fit(X: np.ndarray, y: np.ndarray, l2=1e-3, lr=0.5, epochs=600):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    n, d = Z.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        e = p - y
        w -= lr * (Z.T @ e / n + l2 * w)
        b -= lr * e.mean()
    return w, b, mu, sd


def main() -> int:
    series = json.load(open(DATA, encoding="utf-8"))
    rbs = [_rows(p) for p in series]
    rbs = [r for r in rbs if r]
    tr, va = _split(rbs)

    Xtr = np.array([r[0] for r in tr], float)
    ytr = np.array([r[1] for r in tr], float)
    Xva = np.array([r[0] for r in va], float)
    yva = np.array([r[1] for r in va], float)

    w, b, mu, sd = _fit(Xtr, ytr)

    def score(X):
        return 1.0 / (1.0 + np.exp(-(((X - mu) / sd) @ w + b)))

    oot_auc = _auc(score(Xva), yva)
    oot_spread = _spread(score(Xva), yva)
    wf_aucs = []
    for fold in _walk_forward(rbs):
        Xf = np.array([r[0] for r in fold], float)
        yf = np.array([r[1] for r in fold], float)
        wf_aucs.append(_auc(score(Xf), yf))
    wf_min_str = min(abs(a - 0.5) for a in wf_aucs)

    model = {
        "kind": "standardised_logistic",
        "features": list(DERIV_SIGNAL_FEATURES),
        "mean": [round(float(x), 8) for x in mu],
        "std": [round(float(x), 8) for x in sd],
        "weights": [round(float(x), 8) for x in w],
        "bias": round(float(b), 8),
        "train": {
            "n_train": len(tr),
            "n_val": len(va),
            "l2": 1e-3, "lr": 0.5, "epochs": 600,
        },
        "metrics": {
            "oot_auc": round(oot_auc, 4),
            "oot_decile_spread": round(oot_spread, 4),
            "wf_mean_auc": round(sum(wf_aucs) / len(wf_aucs), 4),
            "wf_aucs": [round(a, 4) for a in wf_aucs],
            "wf_min_strength": round(wf_min_str, 4),
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)

    print("combined derivative signal trained (out-of-time):")
    print(f"  features ({len(DERIV_SIGNAL_FEATURES)}): {', '.join(DERIV_SIGNAL_FEATURES)}")
    print(f"  OOT AUC          {oot_auc:.4f}")
    print(f"  OOT decile spread {oot_spread:+.4f}")
    print(f"  walk-forward AUC  {model['metrics']['wf_mean_auc']:.4f} "
          f"(folds {model['metrics']['wf_aucs']}, min strength {wf_min_str:.4f})")
    # Weight ranking — which derivative drives the signal most.
    order = sorted(zip(DERIV_SIGNAL_FEATURES, w), key=lambda t: abs(t[1]), reverse=True)
    print("  weight magnitude (standardised):")
    for name, wi in order:
        print(f"    {name:<18} {wi:+.3f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
