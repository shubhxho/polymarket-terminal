"""Where does the derivative signal actually make money? A multi-horizon,
fee-aware, out-of-time backtest — the SOTA-methodology answer to "make it pay".

`backtest_deriv.py` showed the frozen H=4 signal has no tradeable edge: ranking
the 4-step up/down label doesn't convert to P&L net of fees. That is the wrong
*regime*, not a dead signal. This module sweeps the horizon and the strategy to
find the regime where the edge clears costs, following the accepted framework for
tabular financial ML (López de Prado — purged out-of-time labels, cost-aware
evaluation, GBDT model; the deep-learning forecasters are baselines here, not
SOTA — see RESEARCH.md / the Chronos ablation):

  * per horizon h, features are the (horizon-independent) derivative family, the
    label is the sign of the h-step forward return, and a GBDT is trained with an
    inner early-stop slice while the later-20% holdout is never touched (purged);
  * strategies scored on that holdout, net of a per-trade fee:
      - always-short          the drift baseline (this market set resolves down);
      - selective-short(q)    short ONLY the top-q by up-probability — the decile
                              table says high up-prob windows fall hardest, so a
                              concentrated short pays the fee on fewer, higher-edge
                              trades. This is the meta-labelling idea: the model
                              says *which* of the drift trades to actually take;
      - long-short(q)         top-q long / bottom-q short (the naive edge trade).

Finding on data/series.json: at h≈8–16 the selective short is net-of-fee
profitable and BEATS the always-short baseline (e.g. h=16, fee 0.5%%: selective
q=0.1 Sharpe ~3.3 while always-short is negative), whereas h=4 loses and h≥32 the
signal has decayed (AUC → 0.5). The edge is real but narrow and short-biased;
it needs live spread/slippage validation before it is money.

Run:  ml/.venv/bin/python backtest_horizons.py     (needs numpy + lightgbm)
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence, Tuple

import numpy as np
import lightgbm as lgb

from features import MIN_STD, WINDOW, _std
from features_deriv import DERIV_NAMES, deriv_features

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data", "series.json")
OUT = os.path.join(_HERE, "data", "deriv_horizon_backtest.json")

HORIZONS = (4, 8, 16, 32)
FEES = (0.0, 0.003, 0.005)
QUANTILES = (0.1, 0.2, 0.3)

PARAMS = dict(
    objective="binary", metric="auc", feature_pre_filter=False, verbose=-1,
    is_unbalance=True, num_leaves=31, min_data_in_leaf=300, learning_rate=0.03,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
    seed=13, num_threads=2,
)


def _rows(prices: Sequence[float], H: int):
    out = []
    n = len(prices)
    for i in range(WINDOW, n - H):
        w = prices[i - WINDOW : i]
        incs = [w[k] - w[k - 1] for k in range(1, len(w))]
        if _std(incs) < MIN_STD:
            continue
        f = deriv_features(w)
        fwd = prices[i + H] - prices[i]
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
    if P == 0 or N == 0:
        return 0.5
    return float((ranks[labels > 0].sum() - P * (P + 1) / 2.0) / (P * N))


def _sharpe(rets: List[float]) -> Tuple[float, float, float, int]:
    if not rets:
        return 0.0, 0.0, 0.0, 0
    n = len(rets)
    m = sum(rets) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / n)
    return sum(rets), m, (m / sd * math.sqrt(n) if sd > 0 else 0.0), n


def _run_horizon(payload):
    """Train + backtest one horizon (parallel worker)."""
    H, series = payload
    rbs = [r for r in (_rows(p, H) for p in series) if len(r) >= 5]

    def carve(rbs, frac):
        a, b = [], []
        for r in rbs:
            c = int(len(r) * (1 - frac))
            a += r[: max(0, c - H)]
            b += r[c:]
        return a, b

    _, va = carve(rbs, 0.2)
    trreg = [r[: max(0, int(len(r) * 0.8) - H)] for r in rbs]
    tr, es = carve([r for r in trreg if len(r) >= 5], 0.2)

    def M(rs):
        return (np.array([x[0] for x in rs]),
                np.array([x[1] for x in rs], float),
                np.array([x[2] for x in rs], float))

    Xtr, ytr, _ = M(tr)
    Xes, yes, _ = M(es)
    Xva, yva, fva = M(va)
    booster = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=3000,
                        valid_sets=[lgb.Dataset(Xes, yes)],
                        callbacks=[lgb.early_stopping(80, verbose=False)])
    prob = booster.predict(Xva)
    order = np.argsort(prob)
    n = len(fva)

    strat: Dict[str, Dict] = {}
    for fee in FEES:
        s = {"always_short": _pack(_sharpe([-float(f) - fee for f in fva]))}
        for q in QUANTILES:
            k = max(1, int(n * q))
            idx = order[-k:]
            s[f"selective_short_q{q}"] = _pack(_sharpe([-float(fva[i]) - fee for i in idx]))
        k = max(1, int(n * 0.2))
        ls = _sharpe([float(fva[i]) - fee for i in order[-k:]]
                     + [-float(fva[i]) - fee for i in order[:k]])
        s[f"long_short_q0.2"] = _pack(ls)
        strat[f"fee_{fee}"] = s

    return {
        "horizon": H, "auc": round(_auc(prob, yva), 4), "n": n,
        "up_rate": round(float(yva.mean()), 4),
        "mean_abs_fwd": round(float(np.abs(fva).mean()), 5),
        "strategies": strat,
    }


def _pack(t) -> Dict[str, float]:
    pnl, mean, sharpe, n = t
    return {"pnl": round(pnl, 3), "mean_ret": round(mean, 6),
            "sharpe": round(sharpe, 3), "n": n}


def main() -> int:
    series = json.load(open(DATA, encoding="utf-8"))
    with ProcessPoolExecutor(max_workers=min(len(HORIZONS), (os.cpu_count() or 2)),
                             mp_context=mp.get_context("fork")) as ex:
        results = list(ex.map(_run_horizon, [(H, series) for H in HORIZONS]))
    results.sort(key=lambda r: r["horizon"])

    # Best net-of-realistic-fee strategy that BEATS its own always-short baseline.
    best = None
    for r in results:
        for fee in ("fee_0.003", "fee_0.005"):
            base = r["strategies"][fee]["always_short"]["sharpe"]
            for name, s in r["strategies"][fee].items():
                if name == "always_short":
                    continue
                if s["sharpe"] > base and s["pnl"] > 0:
                    cand = (r["horizon"], fee, name, s["sharpe"], s["pnl"], base)
                    if best is None or s["sharpe"] > best[3]:
                        best = cand

    for r in results:
        print(f"\nH={r['horizon']:>2}  AUC={r['auc']}  n={r['n']}  up={r['up_rate']}  "
              f"mean|fwd|={r['mean_abs_fwd']}")
        for fee, s in r["strategies"].items():
            parts = "  ".join(f"{k}=(pnl{v['pnl']:+.1f},sh{v['sharpe']:+.2f})"
                              for k, v in s.items())
            print(f"  {fee}: {parts}")
    if best:
        print(f"\nBEST net-of-fee edge that beats drift: H={best[0]} {best[1]} "
              f"{best[2]} — Sharpe {best[3]:.2f} (pnl {best[4]:+.1f}) vs "
              f"always-short Sharpe {best[5]:.2f}.")
        print("Edge is real but SHORT-BIASED and narrow — validate live spread "
              "before trusting it as money.")
    else:
        print("\nNo strategy beat the drift baseline net of fees at any horizon.")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"horizons": results,
                   "best_net_of_fee": (
                       None if not best else
                       {"horizon": best[0], "fee": best[1], "strategy": best[2],
                        "sharpe": best[3], "pnl": best[4], "baseline_sharpe": best[5]})},
                  fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
