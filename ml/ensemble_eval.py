"""Does the derivative signal ADD anything to the FeatureMLP, or just echo it?

The MLP (data/seq_model.safetensors, temperature-calibrated) scores ~0.65 AUC;
the derivative logistic (data/deriv_signal.json) scores ~0.60. Weaker alone — so
it is only worth wiring into the terminal if it is *decorrelated* enough that an
ensemble beats the MLP by itself. This measures exactly that, on the same
strictly out-of-time validation split the models were validated on:

  * AUC of each model alone on val,
  * Pearson correlation of the two probability streams (low ⇒ independent info),
  * AUC of a plain average,
  * AUC of a 2-input logistic combiner FIT ON TRAIN and scored on val (honest —
    the combiner never sees the numbers it is judged on).

Run:  ml/.venv/bin/python ml/ensemble_eval.py
"""

from __future__ import annotations

import json
import math
import os

import mlx.core as mx
from mlx.utils import tree_unflatten

from features import FEATURE_NAMES, HORIZON, MIN_STD, WINDOW, _std, window_features
from features_deriv import deriv_signal, load_signal_model
from train_seq import FeatureMLP

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")


def _rows(prices):
    """(window, up-label) rows, dropping flat windows exactly as training does."""
    out = []
    for i in range(WINDOW, len(prices) - HORIZON):
        w = prices[i - WINDOW : i]
        incs = [w[k] - w[k - 1] for k in range(1, len(w))]
        if _std(incs) < MIN_STD:
            continue
        out.append((w, 1 if prices[i + HORIZON] - prices[i] > 0 else 0))
    return out


def _split(series, val_frac=0.2):
    tr, va = [], []
    for prices in series:
        r = _rows(prices)
        if len(r) < 5:
            tr.extend(r)
            continue
        cut = int(len(r) * (1 - val_frac))
        tr.extend(r[: max(0, cut - HORIZON)])
        va.extend(r[cut:])
    return tr, va


def _auc(scores, labels):
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            if labels[order[k]] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def _pearson(xs, ys):
    n = len(xs)
    mx_, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx_) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx_) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx > 1e-12 and dy > 1e-12 else 0.0


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _mlp_probs(windows):
    """Temperature-calibrated MLP P(up) for every window, in one batched pass."""
    norm = json.load(open(os.path.join(DATA, "seq_normalizer.json")))
    fmean, fstd = mx.array(norm["fmean"]), mx.array(norm["fstd"])
    temp = json.load(open(os.path.join(HERE, "..", "src", "lib", "mlModel.json"))).get(
        "temperature", 1.0
    )
    model = FeatureMLP(len(FEATURE_NAMES))
    model.update(tree_unflatten(list(mx.load(os.path.join(DATA, "seq_model.safetensors")).items())))
    model.eval()
    feats = mx.array([window_features(w) for w in windows])
    logits = model(None, (feats - fmean) / fstd).reshape(-1) / temp
    return mx.sigmoid(logits).tolist()


def main():
    series = json.load(open(os.path.join(DATA, "series.json")))
    tr, va = _split(series)
    dm = load_signal_model()

    tr_w, tr_y = [w for w, _ in tr], [y for _, y in tr]
    va_w, va_y = [w for w, _ in va], [y for _, y in va]

    tr_mlp, va_mlp = _mlp_probs(tr_w), _mlp_probs(va_w)
    tr_dv = [deriv_signal(w, dm) for w in tr_w]
    va_dv = [deriv_signal(w, dm) for w in va_w]

    # Honest combiner: fit a 2-input logistic on the TRAIN logits, score on val.
    a, b, c = 1.0, 0.0, 0.0  # w_mlp, w_dv, bias
    X = [(_logit(p), _logit(q)) for p, q in zip(tr_mlp, tr_dv)]
    lr, n = 0.1, len(X)
    for _ in range(400):
        ga = gb = gc = 0.0
        for (x1, x2), y in zip(X, tr_y):
            pred = 1 / (1 + math.exp(-(a * x1 + b * x2 + c)))
            e = pred - y
            ga += e * x1
            gb += e * x2
            gc += e
        a -= lr * ga / n
        b -= lr * gb / n
        c -= lr * gc / n
    va_comb = [
        1 / (1 + math.exp(-(a * _logit(p) + b * _logit(q) + c)))
        for p, q in zip(va_mlp, va_dv)
    ]

    report = {
        "val_n": len(va_y),
        "auc_mlp": round(_auc(va_mlp, va_y), 4),
        "auc_deriv": round(_auc(va_dv, va_y), 4),
        "auc_average": round(_auc([(p + q) / 2 for p, q in zip(va_mlp, va_dv)], va_y), 4),
        "auc_logistic_combiner": round(_auc(va_comb, va_y), 4),
        "prob_correlation": round(_pearson(va_mlp, va_dv), 4),
        "combiner_weights": {"mlp": round(a, 4), "deriv": round(b, 4), "bias": round(c, 4)},
    }
    print(json.dumps(report, indent=2))
    lift = report["auc_logistic_combiner"] - report["auc_mlp"]
    print(f"\nensemble lift over MLP alone: {lift:+.4f} AUC")
    print("VERDICT:", "worth wiring" if lift >= 0.003 else "not worth the complexity")


if __name__ == "__main__":
    main()
