"""Best-signal model: an MLX GRU over the price window, vs the feature-MLP.

The feature-MLP (train.py) throws away order — it sees summary statistics of the
window. A market's *path* carries more: a GRU reads the return sequence step by
step and fuses it with the same hand features, which is the natural way to
squeeze more signal out of the same data. This trains both, picks the better by
validation AUC, and — the part that actually matters for trading — runs a
decile backtest: does ranking markets by the model's score separate the ones
that go up next from the ones that go down?

Run:  python ml/train_seq.py     (expects ml/data/series.json)
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

from features import FEATURE_NAMES, WINDOW, series_to_rich, Sample

HERE = os.path.dirname(__file__)
SERIES = os.path.join(HERE, "data", "series.json")
WEIGHTS = os.path.join(HERE, "data", "seq_model.safetensors")
METRICS = os.path.join(HERE, "data", "seq_metrics.json")

SEQ_LEN = WINDOW - 1        # number of returns in a window
HIDDEN = 32
DROPOUT = 0.25
EPOCHS = 40
BATCH = 128
LR = 2e-3
WEIGHT_DECAY = 3e-4
SEED = 11


class FeatureMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = HIDDEN, p: float = DROPOUT):
        super().__init__()
        self.l1 = nn.Linear(n_in, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(p)

    def __call__(self, seq, feat):  # seq unused; same signature as the GRU
        x = self.drop(nn.relu(self.l1(feat)))
        x = self.drop(nn.relu(self.l2(x)))
        return self.out(x)


class SeqGRU(nn.Module):
    """GRU over the return sequence, its final state fused with the hand features."""

    def __init__(self, n_feat: int, hidden: int = HIDDEN, p: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(1, hidden)
        self.fproj = nn.Linear(n_feat, hidden)
        self.head1 = nn.Linear(hidden * 2, hidden)
        self.head2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(p)

    def __call__(self, seq, feat):
        h = self.gru(seq)               # (batch, seq_len, hidden)
        last = h[:, -1, :]              # final hidden state
        f = nn.relu(self.fproj(feat))
        x = mx.concatenate([last, f], axis=-1)
        x = self.drop(nn.relu(self.head1(x)))
        return self.head2(x)


def _auc(probs: List[float], labels: List[float]) -> float:
    pairs = sorted(zip(probs, labels), key=lambda t: t[0])
    n_pos = sum(1 for _, y in pairs if y > 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i, r = 0, 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (r + r + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] > 0.5:
                rank_sum += avg
        r += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _decile_backtest(probs: List[float], fwd: List[float], q: float = 0.2) -> dict:
    """Signal quality: take the top and bottom `q` of markets by model score and
    compare how they actually moved next. `up_rate` (fraction that rose) is the
    robust headline — it tracks AUC and shrugs off outliers; the mean forward
    return in points is reported too, but medians make it trade-relevant."""
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    k = max(1, int(len(order) * q))
    top = [fwd[i] for i in order[-k:]]
    bottom = [fwd[i] for i in order[:k]]

    def _median(xs):
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    up = lambda xs: sum(1 for v in xs if v > 0) / len(xs)  # noqa: E731
    return {
        "top_up_rate": round(up(top), 3),
        "bottom_up_rate": round(up(bottom), 3),
        "up_rate_spread": round(up(top) - up(bottom), 3),
        "top_median_pts": round(_median(top) * 100, 3),
        "bottom_median_pts": round(_median(bottom) * 100, 3),
        "slice": k,
    }


def _split(series):
    n_val = max(1, int(len(series) * 0.2))
    tr = [s for prices in series[n_val:] for s in series_to_rich(prices)]
    va = [s for prices in series[:n_val] for s in series_to_rich(prices)]
    return tr, va


def _tensors(samples: List[Sample], fmean, fstd, rstd):
    seq = mx.array([[[r] for r in s.seq] for s in samples], dtype=mx.float32) / rstd
    feat = (mx.array([s.feat for s in samples], dtype=mx.float32) - fmean) / fstd
    y = mx.array([float(s.label) for s in samples], dtype=mx.float32)
    return seq, feat, y


def _probs(model, seq, feat) -> List[float]:
    was = model.training
    model.eval()
    out = mx.sigmoid(model(seq, feat).reshape(-1)).tolist()
    if was:
        model.train()
    return out


def _train_one(name, model, tr, va, dims):
    seq_tr, feat_tr, y_tr = tr
    seq_va, feat_va, y_va = va
    pos = y_tr.mean().item()
    w_pos, w_neg = 0.5 / max(pos, 1e-3), 0.5 / max(1 - pos, 1e-3)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    def loss_fn(model, s, f, y):
        logit = model(s, f).reshape(-1)
        w = y * w_pos + (1 - y) * w_neg
        return mx.mean(nn.losses.binary_cross_entropy(logit, y, with_logits=True) * w)

    lg = nn.value_and_grad(model, loss_fn)
    n = seq_tr.shape[0]
    best_auc, best_flat = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        perm = mx.random.permutation(n)
        for s in range(0, n, BATCH):
            idx = perm[s : s + BATCH]
            _, g = lg(model, seq_tr[idx], feat_tr[idx], y_tr[idx])
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
        auc = _auc(_probs(model, seq_va, feat_va), y_va.tolist())
        if auc > best_auc:
            best_auc = auc
            best_flat = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  [{name}] epoch {epoch + 1:3d}  val auc {auc:.4f}  (best {best_auc:.4f})")
    if best_flat is not None:
        model.update(tree_unflatten(list(best_flat.items())))
    return best_auc


def main() -> None:
    mx.random.seed(SEED)
    if not os.path.exists(SERIES):
        raise SystemExit(f"no dataset at {SERIES} — run: python ml/fetch_data.py")
    with open(SERIES) as f:
        series = json.load(f)

    tr_s, va_s = _split(series)
    if not tr_s or not va_s:
        raise SystemExit("not enough data — raise N_MARKETS in fetch_data.py")
    print(f"loaded {len(series)} series → train {len(tr_s)} / val {len(va_s)} windows")

    # Normalisers from TRAIN only.
    feat_all = mx.array([s.feat for s in tr_s], dtype=mx.float32)
    fmean, fstd = feat_all.mean(axis=0), feat_all.std(axis=0) + 1e-6
    rstd = float(mx.array([r for s in tr_s for r in s.seq]).std().item()) + 1e-6

    tr = _tensors(tr_s, fmean, fstd, rstd)
    va = _tensors(va_s, fmean, fstd, rstd)
    fwd_va = [s.fwd for s in va_s]
    base = max(tr[2].mean().item(), 1 - tr[2].mean().item())

    results = {}
    models = {"feature_mlp": FeatureMLP(len(FEATURE_NAMES)), "seq_gru": SeqGRU(len(FEATURE_NAMES))}
    for name, model in models.items():
        print(f"training {name} …")
        auc = _train_one(name, model, tr, va, len(FEATURE_NAMES))
        p = _probs(model, va[0], va[1])
        results[name] = {
            "val_auc": round(auc, 4),
            "backtest": _decile_backtest(p, fwd_va),
        }

    winner = max(results, key=lambda k: results[k]["val_auc"])
    best_model = models[winner]
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    mx.save_safetensors(WEIGHTS, dict(tree_flatten(best_model.parameters())))

    report = {
        "series": len(series),
        "train_windows": tr[0].shape[0],
        "val_windows": va[0].shape[0],
        "majority_baseline_acc": round(base, 4),
        "models": results,
        "winner": winner,
        "seq_len": SEQ_LEN,
        "features": FEATURE_NAMES,
    }
    with open(METRICS, "w") as f:
        json.dump(report, f, indent=2)
    print("\n== result ==")
    print(json.dumps(report, indent=2))
    print(f"best model ({winner}) → {WEIGHTS}")


if __name__ == "__main__":
    main()
