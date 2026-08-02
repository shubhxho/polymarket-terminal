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

image = (
    modal.Image.debian_slim()
    .pip_install("numpy>=1.26", "lightgbm>=4.0")
    .add_local_python_source(
        "features", "features_deriv", "backtest_horizons", "train_deriv_trader"
    )
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


@app.local_entrypoint()
def main(horizons: str = "4,8,16,32"):
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
    print(f"\ntrader H={trader['horizon']} thr={trader['short_threshold']}: "
          f"fee 0.005 Sharpe {op['sharpe']:+.2f} (pnl {op['pnl']:+.1f}, "
          f"{op['n_trades']} shorts) vs always-short {op['always_short_sharpe']:+.2f}")
