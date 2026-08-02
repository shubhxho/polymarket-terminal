"""Retrain the FeatureMLP with `trend_consistency` as a 14th feature.

The post-hoc ensemble (ensemble_eval.py) showed a linear combiner can't extract
anything the MLP doesn't already have. A *retrain* is the fair test: given the
raw feature inside the network, can it learn an interaction the combiner couldn't?

Trains the 13-feature model and the 14-feature model under the SAME seed, split,
hyperparameters and normaliser discipline, and reports both validation AUCs. If
the 14-feature model wins by a real margin it is saved (weights + normaliser) for
the calibration + TS-port cascade; otherwise nothing is written and the deployed
model stands.

Run:  ml/.venv/bin/python ml/retrain_mlp14.py
"""

from __future__ import annotations

import json
import os

import mlx.core as mx

from features import HORIZON, MIN_STD, WINDOW, _std, window_features
from features_deriv import trend_consistency
from train_seq import SEED, FeatureMLP, Sample, _normalizers, _tensors, _train_one

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")


def _rich(prices, extra):
    """Rows carrying 13 or 14 features. `extra=True` appends trend_consistency."""
    out = []
    for i in range(WINDOW, len(prices) - HORIZON):
        w = prices[i - WINDOW : i]
        rets = [w[k] - w[k - 1] for k in range(1, len(w))]
        if _std(rets) < MIN_STD:
            continue
        feat = window_features(w)
        if extra:
            feat = feat + [trend_consistency(w)]
        fwd = prices[i + HORIZON] - prices[i]
        out.append(Sample(rets, feat, 1 if fwd > 0 else 0, fwd))
    return out


def _split(series, extra, val_frac=0.2):
    tr, va = [], []
    for prices in series:
        r = _rich(prices, extra)
        if len(r) < 5:
            tr.extend(r)
            continue
        cut = int(len(r) * (1 - val_frac))
        tr.extend(r[: max(0, cut - HORIZON)])
        va.extend(r[cut:])
    return tr, va


def _train(series, extra):
    mx.random.seed(SEED)  # identical init/shuffle for a fair 13-vs-14 comparison
    tr, va = _split(series, extra)
    fmean, fstd, rstd = _normalizers(tr)
    dims = len(tr[0].feat)
    tr_t = _tensors(tr, fmean, fstd, rstd)
    va_t = _tensors(va, fmean, fstd, rstd)
    model = FeatureMLP(dims)
    auc = _train_one(f"mlp{dims}", model, tr_t, va_t, dims, verbose=False)
    return auc, model, (fmean, fstd, rstd), dims


def main():
    series = json.load(open(os.path.join(DATA, "series.json")))

    auc13, _, _, _ = _train(series, extra=False)
    auc14, model14, (fmean, fstd, rstd), dims = _train(series, extra=True)

    lift = auc14 - auc13
    print(json.dumps({"auc_13feat": round(auc13, 4), "auc_14feat": round(auc14, 4), "lift": round(lift, 4)}, indent=2))
    verdict = lift >= 0.003
    print("VERDICT:", "14-feature model wins — saving for re-port" if verdict else "no real lift — deployed model stands")

    if verdict:
        from mlx.utils import tree_flatten

        flat = dict(tree_flatten(model14.parameters()))
        mx.save_safetensors(os.path.join(DATA, "seq_model14.safetensors"), flat)
        json.dump(
            {
                "winner": "feature_mlp",
                "features": [*json.load(open(os.path.join(DATA, "seq_normalizer.json")))["features"], "trend_consistency"],
                "fmean": fmean.tolist(),
                "fstd": fstd.tolist(),
                "rstd": rstd,
                "val_auc": round(auc14, 4),
            },
            open(os.path.join(DATA, "seq_normalizer14.json"), "w"),
        )
        print("wrote seq_model14.safetensors + seq_normalizer14.json")


if __name__ == "__main__":
    main()
