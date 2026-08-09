"""Capacity/epoch sweep on the deployed 13 features.

trend_consistency didn't help; this asks the other question — is 0.65 a limit of
the *network* or of the *data*? Trains the FeatureMLP at several hidden widths
for longer, same split and seed, and reports validation AUC. A width that clears
the deployed 0.6508 by a real margin would be worth adopting; a flat sweep means
0.65 is the data ceiling and further tuning is noise.

Run:  ml/.venv/bin/python ml/sweep_mlp.py
"""

from __future__ import annotations

import json
import os

import mlx.core as mx

import train_seq as ts
from retrain_mlp14 import _split
from train_seq import SEED, FeatureMLP, _normalizers, _tensors, _train_one

DATA = os.path.join(os.path.dirname(__file__), "data")


def main():
    series = json.load(open(os.path.join(DATA, "series.json")))
    ts.EPOCHS = 80  # longer than the deployed 40, to expose any headroom

    tr, va = _split(series, extra=False)
    fmean, fstd, rstd = _normalizers(tr)
    dims = len(tr[0].feat)
    tr_t = _tensors(tr, fmean, fstd, rstd)
    va_t = _tensors(va, fmean, fstd, rstd)

    res = {}
    for h in (32, 48, 64, 96):
        mx.random.seed(SEED)
        auc = _train_one(f"h{h}", FeatureMLP(dims, hidden=h), tr_t, va_t, dims, verbose=False)
        res[f"hidden_{h}"] = round(auc, 4)
        print(f"hidden={h:3d}  val_auc={auc:.4f}", flush=True)

    best = max(res.values())
    print(json.dumps({**res, "best": best, "deployed_baseline": 0.6508}, indent=2))
    print("VERDICT:", "headroom found" if best >= 0.6538 else "0.65 is the data ceiling — tuning is noise")


if __name__ == "__main__":
    main()
