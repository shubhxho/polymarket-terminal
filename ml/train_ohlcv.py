"""Train the signal model on real OHLCV candles with the richer feature set.

Same honest methodology as `train_seq.py` — GRU + feature-MLP, ensemble,
train-only standardisation, temporal (out-of-time) split with a purge, and a
walk-forward check — but over the 20 OHLCV features (proper CCI/stochastic/ATR
+ order-flow imbalance) built by `features_ohlcv.py` from the `ImpliedData`
OHLCV dataset. Writes `data/ohlcv_model.safetensors`, `ohlcv_normalizer.json`
and `ohlcv_metrics.json`.

    python ml/train_ohlcv.py     (expects ml/data/ohlcv.json — run fetch_hf.py)
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from features import HORIZON
from features_ohlcv import OHLCV_FEATURES, build
from train_seq import (
    FeatureMLP,
    SeqGRU,
    _auc,
    _decile_backtest,
    _normalizers,
    _probs,
    _tensors,
    _train_one,
)

HERE = os.path.dirname(__file__)
OHLCV = os.path.join(HERE, "data", "ohlcv.json")
WEIGHTS = os.path.join(HERE, "data", "ohlcv_model.safetensors")
NORM = os.path.join(HERE, "data", "ohlcv_normalizer.json")
METRICS = os.path.join(HERE, "data", "ohlcv_metrics.json")
SEED = 11


def _brier(probs, labels) -> float:
    """Brier score = mean squared error of the probability (lower = better
    calibrated). Deep research flags calibration as a first-class metric."""
    return round(float(np.mean([(p - y) ** 2 for p, y in zip(probs, labels)])), 4)


def _train_gbdt(tr_s, va_s):
    """LightGBM on the feature vectors — the model class the deep-research pass
    found beats small MLP/GRU on skewed, heavy-tailed tabular market data.
    Trees are scale-invariant, so they take the raw (un-standardised) features.
    Returns (val_probs, feature_importance_gain)."""
    import lightgbm as lgb  # native API — no scikit-learn dependency

    Xtr = np.array([s.feat for s in tr_s], dtype=np.float32)
    ytr = np.array([s.label for s in tr_s], dtype=np.float32)
    Xva = np.array([s.feat for s in va_s], dtype=np.float32)
    yva = np.array([s.label for s in va_s], dtype=np.float32)
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=list(OHLCV_FEATURES))
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 31,
        "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "is_unbalance": True, "seed": 11, "verbose": -1,
    }
    bst = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    probs = bst.predict(Xva).tolist()   # uses best_iteration automatically
    gain = bst.feature_importance(importance_type="gain")
    return probs, gain


def _split_groups(groups, val_frac: float = 0.2):
    """Time-ordered split within each market: earlier windows train, later ones
    validate, with a HORIZON purge so no forward label crosses the boundary."""
    tr, va = [], []
    for g in groups:
        if len(g) < 5:
            tr.extend(g)
            continue
        cut = int(len(g) * (1 - val_frac))
        tr.extend(g[: max(0, cut - HORIZON)])
        va.extend(g[cut:])
    return tr, va


def _walk_forward(groups, folds: int = 4):
    blocks = folds + 1
    for k in range(1, blocks):
        tr, va = [], []
        for g in groups:
            if len(g) < blocks:
                continue
            step = len(g) / blocks
            tr_end, va_end = int(step * k), int(step * (k + 1))
            tr.extend(g[: max(0, tr_end - HORIZON)])
            va.extend(g[tr_end:va_end])
        if tr and va:
            yield k, tr, va


def main() -> None:
    mx.random.seed(SEED)
    if not os.path.exists(OHLCV):
        raise SystemExit(f"no dataset at {OHLCV} — run: python ml/fetch_hf.py")
    series = json.load(open(OHLCV))
    groups = build(series)
    tr_s, va_s = _split_groups(groups)
    if not tr_s or not va_s:
        raise SystemExit("not enough data")
    print(f"{len(series)} markets → train {len(tr_s)} / val {len(va_s)} windows, {len(OHLCV_FEATURES)} features")

    fmean, fstd, rstd = _normalizers(tr_s)
    tr = _tensors(tr_s, fmean, fstd, rstd)
    va = _tensors(va_s, fmean, fstd, rstd)
    fwd_va = [s.fwd for s in va_s]
    base = max(tr[2].mean().item(), 1 - tr[2].mean().item())

    yv = va[2].tolist()
    results, probs = {}, {}
    models = {"feature_mlp": FeatureMLP(len(OHLCV_FEATURES)), "seq_gru": SeqGRU(len(OHLCV_FEATURES))}
    for name, model in models.items():
        print(f"training {name} …")
        auc = _train_one(name, model, tr, va, len(OHLCV_FEATURES))
        p = _probs(model, va[0], va[1])
        probs[name] = p
        results[name] = {"val_auc": round(auc, 4), "brier": _brier(p, yv), "backtest": _decile_backtest(p, fwd_va)}

    # GBDT — the deep-research-recommended primary for tabular market data.
    print("training gbdt (lightgbm) …")
    gbdt_p, gain = _train_gbdt(tr_s, va_s)
    probs["gbdt"] = gbdt_p
    results["gbdt"] = {"val_auc": round(_auc(gbdt_p, yv), 4), "brier": _brier(gbdt_p, yv),
                       "backtest": _decile_backtest(gbdt_p, fwd_va)}
    importance = sorted(zip(OHLCV_FEATURES, [round(float(g), 1) for g in gain]),
                        key=lambda t: t[1], reverse=True)

    # Ensemble across all three model families.
    ens = [sum(t) / 3 for t in zip(probs["feature_mlp"], probs["seq_gru"], probs["gbdt"])]
    results["ensemble"] = {"val_auc": round(_auc(ens, yv), 4), "brier": _brier(ens, yv),
                           "backtest": _decile_backtest(ens, fwd_va)}

    # Overall best (incl. GBDT) reported; served MLX weights use the best NN.
    overall_best = max(results, key=lambda k: results[k]["val_auc"])
    winner = max(("feature_mlp", "seq_gru"), key=lambda k: results[k]["val_auc"])
    best_model = models[winner]

    print("walk-forward verification over time …")
    wf, win_cls = [], SeqGRU if winner == "seq_gru" else FeatureMLP
    for k, tr_k, va_k in _walk_forward(groups, folds=4):
        fm, fs, rs = _normalizers(tr_k)
        m = win_cls(len(OHLCV_FEATURES))
        auc_k = _train_one(f"wf{k}", m, _tensors(tr_k, fm, fs, rs), _tensors(va_k, fm, fs, rs), len(OHLCV_FEATURES), verbose=False)
        bt = _decile_backtest(_probs(m, *_tensors(va_k, fm, fs, rs)[:2]), [s.fwd for s in va_k])
        wf.append({"fold": k, "val_auc": round(auc_k, 4), "up_rate_spread": bt["up_rate_spread"]})
        print(f"  fold {k}: auc {auc_k:.4f}  spread {bt['up_rate_spread']:+.3f}")
    aucs = [f["val_auc"] for f in wf]
    walk_forward = {"folds": wf, "mean_auc": round(sum(aucs) / len(aucs), 4) if aucs else None,
                    "min_auc": round(min(aucs), 4) if aucs else None}

    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    mx.save_safetensors(WEIGHTS, dict(tree_flatten(best_model.parameters())))
    with open(NORM, "w") as f:
        json.dump({"winner": winner, "fmean": fmean.tolist(), "fstd": fstd.tolist(),
                   "rstd": rstd, "features": OHLCV_FEATURES, "hidden": 32}, f, indent=2)
    report = {
        "dataset": "ImpliedData/prediction-markets OHLCV (polymarket+manifold)",
        "markets": len(series), "train_windows": tr[0].shape[0], "val_windows": va[0].shape[0],
        "majority_baseline_acc": round(base, 4), "models": results,
        "overall_best": overall_best, "served_nn_winner": winner,
        "feature_importance_gbdt": importance,
        "walk_forward": walk_forward, "features": OHLCV_FEATURES,
    }
    with open(METRICS, "w") as f:
        json.dump(report, f, indent=2)
    print("\n== result ==")
    for k, v in results.items():
        b = v["backtest"]
        print(f"  {k:12s} auc {v['val_auc']}  brier {v['brier']}  spread {b['up_rate_spread']:+.3f}")
    print(f"overall best: {overall_best} | served NN: {winner}")
    print("top features (gbdt gain):", [f for f, _ in importance[:6]])


if __name__ == "__main__":
    main()
