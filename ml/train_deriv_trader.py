"""Freeze a SOTA-methodology TRADING model from the derivative signal.

`backtest_horizons.py` located the regime where the signal actually pays: at a
~16-step horizon, a SELECTIVE SHORT — short only the windows whose up-probability
is in the top quantile — clears realistic fees and beats the always-short drift
baseline (Sharpe ~3.3–3.5 net of 0.5%). This trainer turns that finding into a
frozen, servable decision policy, the honest way:

  * label = sign of the h-step forward return; model = GBDT (the SOTA choice for
    tabular market data — the deep forecasters are baselines, see RESEARCH.md);
  * strictly out-of-time: inner early-stop slice for the round count, later-20%
    holdout (purged) touched only to measure the frozen policy;
  * the OPERATING POINT is frozen too — the top-quantile probability threshold
    (computed on the train region so it never sees the holdout) above which the
    policy shorts, plus the backtested net-of-fee Sharpe/P&L it earned.

Serving is `features_deriv.trade_signal(window)` — a pure-stdlib GBDT walk plus
the frozen threshold, emitting SHORT / FLAT with a conviction size. No numpy /
lightgbm at serve time.

Run:  ml/.venv/bin/python train_deriv_trader.py
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Sequence

import numpy as np
import lightgbm as lgb

from features import MIN_STD, WINDOW, _std
from features_deriv import DERIV_NAMES, deriv_features

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data", "series.json")
OUT = os.path.join(_HERE, "data", "deriv_trader.json")

HORIZON = 16          # the profitable regime found by backtest_horizons.py
TOP_Q = 0.2           # short the top 20% by up-probability (selective short)
FEE = 0.005           # the fee the operating point is required to survive

PARAMS = dict(
    objective="binary", metric="auc", feature_pre_filter=False, verbose=-1,
    is_unbalance=True, num_leaves=31, min_data_in_leaf=300, learning_rate=0.03,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
    seed=13, num_threads=2,
)


def _rows(prices: Sequence[float]):
    out = []
    n = len(prices)
    for i in range(WINDOW, n - HORIZON):
        w = prices[i - WINDOW : i]
        incs = [w[k] - w[k - 1] for k in range(1, len(w))]
        if _std(incs) < MIN_STD:
            continue
        f = deriv_features(w)
        fwd = prices[i + HORIZON] - prices[i]
        out.append(([f[k] for k in DERIV_NAMES], 1 if fwd > 0 else 0, fwd))
    return out


def _auc(scores, labels) -> float:
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
    return 0.5 if P == 0 or N == 0 else float((ranks[labels > 0].sum() - P * (P + 1) / 2.0) / (P * N))


def _sharpe(rets: List[float]):
    if not rets:
        return 0.0, 0.0, 0.0, 0
    n = len(rets)
    m = sum(rets) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / n)
    return sum(rets), m, (m / sd * math.sqrt(n) if sd > 0 else 0.0), n


def main() -> int:
    series = json.load(open(DATA, encoding="utf-8"))
    rbs = [r for r in (_rows(p) for p in series) if len(r) >= 5]

    def carve(rbs, frac):
        a, b = [], []
        for r in rbs:
            c = int(len(r) * (1 - frac))
            a += r[: max(0, c - HORIZON)]
            b += r[c:]
        return a, b

    trall, va = carve(rbs, 0.2)
    trreg = [r[: max(0, int(len(r) * 0.8) - HORIZON)] for r in rbs]
    tr, es = carve([r for r in trreg if len(r) >= 5], 0.2)

    def M(rs):
        return (np.array([x[0] for x in rs]), np.array([x[1] for x in rs], float),
                np.array([x[2] for x in rs], float))

    Xtr, ytr, _ = M(tr)
    Xes, yes, _ = M(es)
    Xva, yva, fva = M(va)
    booster = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=3000,
                        valid_sets=[lgb.Dataset(Xes, yes)],
                        callbacks=[lgb.early_stopping(80, verbose=False)])

    # Operating point: the short threshold = (1-TOP_Q) quantile of TRAIN-region
    # probabilities (never the holdout), frozen for serving.
    Xall, _, _ = M(trall)
    p_train = booster.predict(Xall)
    threshold = float(np.quantile(p_train, 1.0 - TOP_Q))

    # Evaluate the frozen policy on the untouched holdout, at several fees.
    p_va = booster.predict(Xva)
    short_idx = [i for i in range(len(p_va)) if p_va[i] >= threshold]
    policy = {}
    for fee in (0.0, 0.003, FEE):
        rets = [-float(fva[i]) - fee for i in short_idx]
        pnl, mean, sharpe, n = _sharpe(rets)
        base = _sharpe([-float(f) - fee for f in fva])
        policy[f"fee_{fee}"] = {"pnl": round(pnl, 3), "mean_ret": round(mean, 6),
                                "sharpe": round(sharpe, 3), "n_trades": n,
                                "always_short_sharpe": round(base[2], 3)}

    dump = booster.dump_model()
    model = {
        "kind": "lightgbm_gbdt_trader",
        "features": list(DERIV_NAMES),
        "horizon": HORIZON, "top_q": TOP_Q, "fee_assumed": FEE,
        "short_threshold": round(threshold, 6),
        "best_iteration": int(booster.best_iteration or booster.current_iteration()),
        "oot_auc": round(_auc(p_va, yva), 4),
        "policy": policy,
        "model": dump,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(model, fh)

    op = policy[f"fee_{FEE}"]
    print("SOTA derivative trading model frozen (out-of-time):")
    print(f"  horizon={HORIZON}  top_q={TOP_Q}  short_threshold={threshold:.4f}  "
          f"AUC={model['oot_auc']}")
    print(f"  policy @ fee {FEE}: shorts {op['n_trades']} windows, "
          f"pnl {op['pnl']:+.1f}, Sharpe {op['sharpe']:+.2f} "
          f"(always-short {op['always_short_sharpe']:+.2f})")
    verdict = ("BEATS drift baseline net of fee" if op["sharpe"] > op["always_short_sharpe"]
               and op["pnl"] > 0 else "does NOT beat baseline")
    print(f"  verdict: {verdict}. SHORT-BIASED, narrow — validate live spread.")
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
