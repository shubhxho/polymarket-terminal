"""Train the short-horizon direction classifier in MLX.

A small, regularised MLP over the microstructure features in `features.py`,
predicting whether a market's price is higher HORIZON steps ahead.

Done properly:
- market-disjoint train/val split (windows from one market never straddle the
  split, so val isn't inflated by leakage);
- features z-scored by TRAIN statistics only;
- class-weighted BCE, because the up/down base rate is skewed by drift and a
  model must not be able to win by always shouting "up";
- AdamW weight decay + dropout to fix the naive net's overfit;
- best-val-AUC early stopping;
- honest reporting: AUC and balanced accuracy (the right metrics for a skewed
  binary target) next to raw accuracy and the majority baseline.

Run:  python ml/train.py           (expects ml/data/series.json from fetch_data.py)
"""

from __future__ import annotations

import json
import math
import os
from typing import List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from features import FEATURE_NAMES, build_dataset

HERE = os.path.dirname(__file__)
SERIES = os.path.join(HERE, "data", "series.json")
WEIGHTS = os.path.join(HERE, "data", "model.safetensors")
METRICS = os.path.join(HERE, "data", "metrics.json")

HIDDEN = 24
DROPOUT = 0.2
EPOCHS = 120
BATCH = 256
LR = 2e-3
WEIGHT_DECAY = 2e-4
SEED = 7


class DirectionMLP(nn.Module):
    """in → hidden → hidden → 1 logit, with dropout between layers."""

    def __init__(self, n_in: int, hidden: int = HIDDEN, p: float = DROPOUT):
        super().__init__()
        self.l1 = nn.Linear(n_in, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(p)

    def __call__(self, x):
        x = self.drop(nn.relu(self.l1(x)))
        x = self.drop(nn.relu(self.l2(x)))
        return self.out(x)


def _split_by_market(series: List[List[float]], val_frac: float = 0.2):
    n_val = max(1, int(len(series) * val_frac))
    Xtr, ytr = build_dataset(series[n_val:])
    Xva, yva = build_dataset(series[:n_val])
    return (Xtr, ytr), (Xva, yva)


def _standardize(Xtr: List[List[float]], Xva: List[List[float]]):
    a = mx.array(Xtr, dtype=mx.float32)
    mean = a.mean(axis=0)
    std = a.std(axis=0) + 1e-6
    norm = lambda M: (mx.array(M, dtype=mx.float32) - mean) / std  # noqa: E731
    return norm(Xtr), norm(Xva), mean, std


def _auc(probs: List[float], labels: List[float]) -> float:
    """Rank-based ROC AUC (Mann–Whitney U). 0.5 == no discrimination."""
    pairs = sorted(zip(probs, labels), key=lambda t: t[0])
    n_pos = sum(1 for _, y in pairs if y > 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    r = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (r + (r + (j - i) - 1)) / 2.0  # tie-averaged rank
        for k in range(i, j):
            if pairs[k][1] > 0.5:
                rank_sum += avg_rank
        r += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _probs(model, X) -> List[float]:
    was_training = model.training
    model.eval()
    out = mx.sigmoid(model(X).reshape(-1)).tolist()
    if was_training:
        model.train()
    return out


def _balanced_acc(p: List[float], yl: List[float], thr: float) -> float:
    tp = sum(1 for v, b in zip(p, yl) if v > thr and b == 1)
    tn = sum(1 for v, b in zip(p, yl) if v <= thr and b == 0)
    n_pos = sum(1 for b in yl if b == 1)
    n_neg = len(yl) - n_pos
    return 0.5 * ((tp / n_pos if n_pos else 0) + (tn / n_neg if n_neg else 0))


def _best_threshold(p: List[float], yl: List[float]) -> float:
    """Threshold maximising balanced accuracy — chosen on TRAIN, applied to val."""
    best_t, best_b = 0.5, -1.0
    for i in range(20, 81):
        t = i / 100.0
        b = _balanced_acc(p, yl, t)
        if b > best_b:
            best_b, best_t = b, t
    return best_t


def _evaluate(model, X, y, thr: float = 0.5) -> dict:
    p = _probs(model, X)
    yl = y.tolist()
    acc = sum(1 for v, b in zip(p, yl) if (v > thr) == (b > 0.5)) / len(yl)
    eps = 1e-6
    bce = -sum(
        b * math.log(min(max(v, eps), 1 - eps)) + (1 - b) * math.log(min(max(1 - v, eps), 1 - eps))
        for v, b in zip(p, yl)
    ) / len(yl)
    return {
        "acc": acc,
        "balanced_acc": _balanced_acc(p, yl, thr),
        "auc": _auc(p, yl),
        "bce": bce,
    }


def main() -> None:
    mx.random.seed(SEED)
    if not os.path.exists(SERIES):
        raise SystemExit(f"no dataset at {SERIES} — run: python ml/fetch_data.py")

    with open(SERIES) as f:
        series = json.load(f)
    print(f"loaded {len(series)} market series")

    (Xtr_raw, ytr_l), (Xva_raw, yva_l) = _split_by_market(series)
    if not Xtr_raw or not Xva_raw:
        raise SystemExit("not enough data — raise N_MARKETS in fetch_data.py")

    Xtr, Xva, mean, std = _standardize(Xtr_raw, Xva_raw)
    ytr = mx.array(ytr_l, dtype=mx.float32)
    yva = mx.array(yva_l, dtype=mx.float32)

    # Class weights invert the train base rate, so up and down contribute
    # equally to the loss and the model can't coast on the majority class.
    pos_rate = ytr.mean().item()
    w_pos = 0.5 / max(pos_rate, 1e-3)
    w_neg = 0.5 / max(1 - pos_rate, 1e-3)
    print(
        f"train {Xtr.shape[0]} rows (up {pos_rate:.2f}) · val {Xva.shape[0]} rows "
        f"(up {yva.mean().item():.2f}) · class weights +{w_pos:.2f}/-{w_neg:.2f}"
    )

    model = DirectionMLP(len(FEATURE_NAMES))
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    def loss_fn(model, X, y):
        logits = model(X).reshape(-1)
        w = y * w_pos + (1 - y) * w_neg
        return mx.mean(nn.losses.binary_cross_entropy(logits, y, with_logits=True) * w)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    n = Xtr.shape[0]
    best_auc = 0.0
    best_flat = None
    best_report = None
    for epoch in range(EPOCHS):
        model.train()
        perm = mx.random.permutation(n)
        for s in range(0, n, BATCH):
            idx = perm[s : s + BATCH]
            _, grads = loss_and_grad(model, Xtr[idx], ytr[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)

        va = _evaluate(model, Xva, yva)
        if va["auc"] > best_auc:
            best_auc = va["auc"]
            best_flat = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
            best_report = va
        if (epoch + 1) % 20 == 0 or epoch == EPOCHS - 1:
            tr = _evaluate(model, Xtr, ytr)
            print(
                f"epoch {epoch + 1:3d}  train auc {tr['auc']:.3f}  "
                f"val auc {va['auc']:.3f} bal-acc {va['balanced_acc']:.3f} acc {va['acc']:.3f}"
            )

    # Restore the best-val-AUC snapshot before saving/eval.
    if best_flat is not None:
        model.update(tree_unflatten(list(best_flat.items())))
    # Pick the operating threshold on TRAIN, then report val at it.
    thr = _best_threshold(_probs(model, Xtr), ytr.tolist())
    tr = _evaluate(model, Xtr, ytr, thr)
    va = _evaluate(model, Xva, yva, thr)
    base = max(yva.mean().item(), 1 - yva.mean().item())

    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    mx.save_safetensors(WEIGHTS, dict(tree_flatten(model.parameters())))
    with open(os.path.join(HERE, "data", "normalizer.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "features": FEATURE_NAMES}, f, indent=2)

    report = {
        "train_rows": int(Xtr.shape[0]),
        "val_rows": int(Xva.shape[0]),
        "val_auc": round(va["auc"], 4),
        "val_balanced_acc": round(va["balanced_acc"], 4),
        "val_acc": round(va["acc"], 4),
        "val_bce": round(va["bce"], 4),
        "train_auc": round(tr["auc"], 4),
        "threshold": round(thr, 3),
        "majority_baseline_acc": round(base, 4),
        "auc_edge": round(va["auc"] - 0.5, 4),
        "features": FEATURE_NAMES,
        "hidden": HIDDEN,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
    }
    with open(METRICS, "w") as f:
        json.dump(report, f, indent=2)
    print("\n== result (best val AUC snapshot) ==")
    print(json.dumps(report, indent=2))
    print(f"weights → {WEIGHTS}")


if __name__ == "__main__":
    main()
