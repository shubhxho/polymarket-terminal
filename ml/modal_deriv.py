"""Run the derivative trading pipeline on Modal — one container per horizon.

The horizon backtest (`backtest_horizons.py`) is embarrassingly parallel across
horizons: each fits an independent GBDT and backtests it. This app fans those out
across Modal CPU containers and trains the frozen trader, so the whole
find-the-profitable-regime + freeze-the-policy pipeline runs in the cloud instead
of on one laptop core.

CPU only (LightGBM on tabular data — no GPU needed), so it is cheap and quick.
The heavy sources are mounted with `add_local_python_source`; `series.json` is
passed to the workers as an argument (it is ~1.4 MB), so no file-path juggling
inside the container.

Deploy/run (Modal authed as in the pipeline memory):
  modal run ml/modal_deriv.py                 # fan out horizons + train trader
  modal run ml/modal_deriv.py --horizons 8,16 # subset
"""

from __future__ import annotations

import modal

app = modal.App("pmt-deriv")

_SRC = ("features", "features_deriv", "backtest_horizons", "train_deriv_trader")

image = (
    modal.Image.debian_slim()
    .pip_install("numpy>=1.26", "lightgbm>=4.0")
    .add_local_python_source(*_SRC)
)

# GPU image: XGBoost with the CUDA build for a real device="cuda" training path.
gpu_image = (
    modal.Image.debian_slim()
    # lightgbm is imported by train_deriv_trader at module load (we reuse its
    # _rows/_sharpe helpers); xgboost is the actual CUDA training backend here.
    .pip_install("numpy>=1.26", "xgboost>=2.0", "lightgbm>=4.0")
    .add_local_python_source(*_SRC)
)


@app.function(image=image, cpu=4.0, timeout=1800)
def run_horizon(payload):
    """One horizon: train the GBDT + backtest its strategies. payload=(H, series)."""
    import backtest_horizons as bh

    return bh._run_horizon(payload)


@app.function(image=image, cpu=4.0, timeout=1800)
def train_trader(series):
    """Train the frozen H=16 selective-short trader and return its policy metrics
    (the model JSON itself is large; return only the numbers here)."""
    import numpy as np
    import lightgbm as lgb
    import train_deriv_trader as T

    rbs = [r for r in (T._rows(p) for p in series) if len(r) >= 5]

    def carve(rbs, frac):
        a, b = [], []
        for r in rbs:
            c = int(len(r) * (1 - frac))
            a += r[: max(0, c - T.HORIZON)]
            b += r[c:]
        return a, b

    trall, va = carve(rbs, 0.2)
    trreg = [r[: max(0, int(len(r) * 0.8) - T.HORIZON)] for r in rbs]
    tr, es = carve([r for r in trreg if len(r) >= 5], 0.2)

    def M(rs):
        return (np.array([x[0] for x in rs]), np.array([x[1] for x in rs], float),
                np.array([x[2] for x in rs], float))

    Xtr, ytr, _ = M(tr)
    Xes, yes, _ = M(es)
    Xva, _, fva = M(va)
    b = lgb.train(T.PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=3000,
                  valid_sets=[lgb.Dataset(Xes, yes)],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
    thr = float(np.quantile(b.predict(M(trall)[0]), 1.0 - T.TOP_Q))
    p_va = b.predict(Xva)
    idx = [i for i in range(len(p_va)) if p_va[i] >= thr]
    out = {"horizon": T.HORIZON, "top_q": T.TOP_Q, "short_threshold": round(thr, 6),
           "policy": {}}
    for fee in (0.0, 0.003, T.FEE):
        rets = [-float(fva[i]) - fee for i in idx]
        _, mean, sharpe, n = T._sharpe(rets)
        base = T._sharpe([-float(f) - fee for f in fva])
        out["policy"][f"fee_{fee}"] = {"sharpe": round(sharpe, 3), "n_trades": n,
                                       "pnl": round(sum(rets), 3),
                                       "always_short_sharpe": round(base[2], 3)}
    return out


@app.function(image=gpu_image, gpu="T4", timeout=1800)
def train_trader_gpu(series):
    """SOTA GPU training path: XGBoost with device="cuda" on the H=16 derivative
    trader. Returns AUC, the frozen policy metrics, wall time and the GPU name —
    so the GPU-vs-CPU comparison is honest at this data size."""
    import os
    import time
    import numpy as np
    import xgboost as xgb
    import train_deriv_trader as T

    rbs = [r for r in (T._rows(p) for p in series) if len(r) >= 5]

    def carve(rbs, frac):
        a, b = [], []
        for r in rbs:
            c = int(len(r) * (1 - frac))
            a += r[: max(0, c - T.HORIZON)]
            b += r[c:]
        return a, b

    trall, va = carve(rbs, 0.2)
    trreg = [r[: max(0, int(len(r) * 0.8) - T.HORIZON)] for r in rbs]
    tr, es = carve([r for r in trreg if len(r) >= 5], 0.2)

    def M(rs):
        return (np.array([x[0] for x in rs]), np.array([x[1] for x in rs], float),
                np.array([x[2] for x in rs], float))

    Xtr, ytr, _ = M(tr)
    Xes, yes, _ = M(es)
    Xva, yva, fva = M(va)

    params = dict(objective="binary:logistic", eval_metric="auc",
                  tree_method="hist", device="cuda", max_depth=6, eta=0.03,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                  scale_pos_weight=float((ytr == 0).sum() / max(1, (ytr == 1).sum())))
    dtr = xgb.DMatrix(Xtr, label=ytr)
    des = xgb.DMatrix(Xes, label=yes)
    t0 = time.time()
    bst = xgb.train(params, dtr, num_boost_round=2000, evals=[(des, "es")],
                    early_stopping_rounds=80, verbose_eval=False)
    train_s = time.time() - t0

    p_va = bst.predict(xgb.DMatrix(Xva))
    p_all = bst.predict(xgb.DMatrix(M(trall)[0]))
    thr = float(np.quantile(p_all, 1.0 - T.TOP_Q))
    idx = [i for i in range(len(p_va)) if p_va[i] >= thr]

    def auc(s, y):
        o = np.argsort(s, kind="mergesort")
        ss = s[o]; rr = np.empty(len(s)); i = 0
        while i < len(s):
            j = i
            while j < len(s) and ss[j] == ss[i]:
                j += 1
            rr[i:j] = (i + j - 1) / 2 + 1; i = j
        rk = np.empty(len(s)); rk[o] = rr
        P = y.sum(); N = len(y) - P
        return 0.5 if P == 0 or N == 0 else float((rk[y > 0].sum() - P * (P + 1) / 2) / (P * N))

    out = {"backend": "xgboost-cuda", "gpu": os.environ.get("NVIDIA_VISIBLE_DEVICES", "gpu"),
           "train_seconds": round(train_s, 2), "best_iteration": int(bst.best_iteration),
           "oot_auc": round(auc(p_va, yva), 4), "policy": {}}
    for fee in (0.0, 0.003, T.FEE):
        rets = [-float(fva[i]) - fee for i in idx]
        _, mean, sharpe, n = T._sharpe(rets)
        base = T._sharpe([-float(f) - fee for f in fva])
        out["policy"][f"fee_{fee}"] = {"sharpe": round(sharpe, 3), "n_trades": n,
                                       "pnl": round(sum(rets), 3),
                                       "always_short_sharpe": round(base[2], 3)}
    return out


@app.local_entrypoint()
def main(horizons: str = "4,8,16,32", gpu: bool = False):
    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    series = json.load(open(os.path.join(here, "data", "series.json"), encoding="utf-8"))
    hs = [int(h) for h in horizons.split(",") if h.strip()]

    print(f"fanning out {len(hs)} horizons across Modal containers...")
    results = sorted(run_horizon.map([(h, series) for h in hs]),
                     key=lambda r: r["horizon"])
    for r in results:
        print(f"H={r['horizon']:>2} AUC={r['auc']} n={r['n']} up={r['up_rate']} "
              f"mean|fwd|={r['mean_abs_fwd']}")
        for fee, s in r["strategies"].items():
            best = max((v["sharpe"] for k, v in s.items() if k != "always_short"),
                       default=0.0)
            print(f"   {fee}: always_short sh={s['always_short']['sharpe']:+.2f}  "
                  f"best_selective sh={best:+.2f}")

    trader = train_trader.remote(series)
    op = trader["policy"][f"fee_{0.005}"]
    print(f"\ntrader (CPU/LightGBM) H={trader['horizon']} thr={trader['short_threshold']}: "
          f"fee 0.005 Sharpe {op['sharpe']:+.2f} (pnl {op['pnl']:+.1f}, "
          f"{op['n_trades']} shorts) vs always-short {op['always_short_sharpe']:+.2f}")

    if gpu:
        g = train_trader_gpu.remote(series)
        gop = g["policy"][f"fee_{0.005}"]
        print(f"\ntrader (GPU/{g['backend']}) trained in {g['train_seconds']}s "
              f"({g['best_iteration']} rounds), AUC {g['oot_auc']}: "
              f"fee 0.005 Sharpe {gop['sharpe']:+.2f} (pnl {gop['pnl']:+.1f}, "
              f"{gop['n_trades']} shorts) vs always-short {gop['always_short_sharpe']:+.2f}")
