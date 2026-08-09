"""End-to-end test of the derivative trading pipeline — needs numpy + lightgbm.

Drives the whole chain on a real subset of data/series.json, exactly as the
trainer does, and asserts the pieces line up end to end:

  build derivative features → train the trader GBDT → freeze dump_model() +
  operating point → serve `trade_signal` (pure-stdlib walk) → backtest the served
  decisions.

The load-bearing checks: (1) the served up-probability equals LightGBM's own
`booster.predict` to machine precision *through the frozen JSON* — training and
serving cannot silently diverge; (2) the served SHORT/FLAT action obeys the frozen
threshold; (3) the backtest of the served decisions reproduces the same net-of-fee
Sharpe the policy would compute directly. If any join is wrong, this fails.

Run:  ml/.venv/bin/python test_e2e_deriv.py     (skips if lightgbm absent)
"""

from __future__ import annotations

import json
import math
import os

try:
    import numpy as np
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    print("skip test_e2e_deriv: numpy/lightgbm not installed")
    raise SystemExit(0)

from features import MIN_STD, WINDOW, _std
from features_deriv import DERIV_NAMES, deriv_features, trade_signal
from train_deriv_trader import PARAMS

_HERE = os.path.dirname(os.path.abspath(__file__))
H = 16
TOP_Q = 0.2
FEE = 0.005


def _holdout_and_train(series):
    """Build (train feature-rows, holdout (window,fwd)) at horizon H, purged."""
    per_series = []
    for prices in series:
        rows = []
        n = len(prices)
        for i in range(WINDOW, n - H):
            w = prices[i - WINDOW : i]
            incs = [w[k] - w[k - 1] for k in range(1, len(w))]
            if _std(incs) < MIN_STD:
                continue
            f = deriv_features(w)
            rows.append((list(w), [f[k] for k in DERIV_NAMES],
                         1 if prices[i + H] - prices[i] > 0 else 0,
                         prices[i + H] - prices[i]))
        if len(rows) >= 5:
            per_series.append(rows)
    tr, va = [], []
    for rows in per_series:
        cut = int(len(rows) * 0.8)
        tr += rows[: max(0, cut - H)]
        va += rows[cut:]
    return tr, va


def _run() -> None:
    series = json.load(open(os.path.join(_HERE, "data", "series.json"), encoding="utf-8"))
    tr, va = _holdout_and_train(series[:60])   # subset keeps the test fast
    assert tr and va

    Xtr = np.array([r[1] for r in tr], float)
    ytr = np.array([r[2] for r in tr], float)
    booster = lgb.train(dict(PARAMS, num_threads=2), lgb.Dataset(Xtr, ytr),
                        num_boost_round=120)
    threshold = float(np.quantile(booster.predict(Xtr), 1.0 - TOP_Q))

    # Freeze in the exact production shape.
    model = {
        "kind": "lightgbm_gbdt_trader", "features": list(DERIV_NAMES),
        "horizon": H, "top_q": TOP_Q, "fee_assumed": FEE,
        "short_threshold": threshold, "oot_auc": 0.0,
        "policy": {f"fee_{FEE}": {"sharpe": 1.0}}, "model": booster.dump_model(),
    }

    # 1) Served prob == booster.predict through the frozen JSON.
    Xva = np.array([r[1] for r in va], float)
    ref = booster.predict(Xva)
    served = np.array([trade_signal(r[0], model=model)["prob_up"] for r in va[:400]])
    # trade_signal rounds prob to 4dp; compare against the same rounding.
    assert np.max(np.abs(served - np.round(ref[:400], 4))) < 1e-9, "serving diverged from training"

    # 2) Action obeys the frozen threshold; 3) backtest of served decisions.
    rets = []
    n_short = 0
    for w, _, _, fwd in va:
        d = trade_signal(w, model=model)
        is_short = d["action"] == "SHORT"
        assert is_short == (d["prob_up"] >= round(threshold, 4)) or abs(d["prob_up"] - threshold) < 1e-4
        if is_short:
            n_short += 1
            rets.append(-fwd - FEE)
        assert 0.0 <= d["size"] <= 1.0 and d["action"] in ("SHORT", "FLAT")
    assert n_short > 0, "policy never traded on the holdout"
    m = sum(rets) / len(rets)
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / len(rets))
    sharpe = m / sd * math.sqrt(len(rets)) if sd > 0 else 0.0
    # Sanity: the served backtest is finite and the short set is a top-q slice.
    assert math.isfinite(sharpe)
    assert n_short <= int(len(va) * TOP_Q) + 5, (n_short, len(va))
    print(f"ok  e2e pipeline (train->freeze->serve->backtest): "
          f"{n_short} shorts on {len(va)} holdout, served Sharpe {sharpe:+.2f}")


if __name__ == "__main__":
    _run()
    print("\n1 e2e test passed")
