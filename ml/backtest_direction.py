"""Does the deployed direction model make money — after costs?

AUC 0.65 says the model *ranks* up-moves better than chance. It does NOT say the
edge survives the spread you cross to trade it. This is the honest economic test.

Strategy (strictly out-of-time val, calibrated probabilities):
  * go long the YES price when P(up) > 0.5 + band, short when < 0.5 - band, else flat;
  * realise the forward price move `fwd` over HORIZON (in probability points);
  * pay `cost` points per round-trip (the spread you cross in and out).
The conviction `band` is chosen on the TRAIN split by net Sharpe — never on val.

Reports per-trade expectancy, hit rate, Sharpe and total P&L at several cost
assumptions, the break-even cost where the edge dies, and a bootstrap 95% CI on
the val net expectancy (the "is it just noise?" test).

Run:  ml/.venv/bin/python ml/backtest_direction.py
"""

from __future__ import annotations

import json
import math
import os

import mlx.core as mx
from mlx.utils import tree_unflatten

from features import FEATURE_NAMES
from retrain_mlp14 import _split
from train_seq import FeatureMLP

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")

# A deterministic LCG for the bootstrap — no Math.random-style nondeterminism, so
# the reported confidence interval replays exactly.
_S = 0x2545F4914F6CDD1D


def _rand():
    global _S
    _S = (1103515245 * _S + 12345) & 0xFFFFFFFFFFFFFFFF
    return (_S >> 16) / 0xFFFFFFFFFFFF


def _probs(samples):
    """Calibrated P(up) for a list of Samples, batched."""
    norm = json.load(open(os.path.join(DATA, "seq_normalizer.json")))
    fmean, fstd = mx.array(norm["fmean"]), mx.array(norm["fstd"])
    temp = json.load(open(os.path.join(HERE, "..", "src", "lib", "mlModel.json"))).get("temperature", 1.0)
    model = FeatureMLP(len(FEATURE_NAMES))
    model.update(tree_unflatten(list(mx.load(os.path.join(DATA, "seq_model.safetensors")).items())))
    model.eval()
    feats = mx.array([s.feat for s in samples])
    logits = model(None, (feats - fmean) / fstd).reshape(-1) / temp
    return mx.sigmoid(logits).tolist()


def _trades(probs, fwd, band, cost):
    """Net per-trade returns (in points) for one band/cost, longs and shorts."""
    out = []
    for p, f in zip(probs, fwd):
        if p > 0.5 + band:
            out.append(f * 100 - cost)  # long
        elif p < 0.5 - band:
            out.append(-f * 100 - cost)  # short
    return out


def _stats(rets):
    n = len(rets)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sharpe": 0.0, "hit": 0.0, "pnl": 0.0}
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / n
    sd = math.sqrt(var) if var > 0 else 1e-9
    hit = sum(1 for r in rets if r > 0) / n
    return {"n": n, "mean": round(mean, 4), "sharpe": round(mean / sd, 4), "hit": round(hit, 4), "pnl": round(sum(rets), 2)}


def main():
    series = json.load(open(os.path.join(DATA, "series.json")))
    tr, va = _split(series, extra=False)
    tr_p, va_p = _probs(tr), _probs(va)
    tr_f, va_f = [s.fwd for s in tr], [s.fwd for s in va]

    REF_COST = 1.0  # 1¢ round-trip while choosing the band

    # Pick the conviction band on TRAIN by net Sharpe — no val lookahead.
    bands = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.2]
    best_band, best_sh = 0.0, -1e9
    for b in bands:
        sh = _stats(_trades(tr_p, tr_f, b, REF_COST))["sharpe"]
        if sh > best_sh:
            best_sh, best_band = sh, b

    print(f"chosen band (train, {REF_COST}c cost): {best_band}\n")
    print("VAL economics at that band, by round-trip cost:")
    rows = {}
    for cost in (0.0, 0.5, 1.0, 1.5, 2.0):
        rows[cost] = _stats(_trades(va_p, va_f, best_band, cost))
        r = rows[cost]
        print(f"  cost {cost:>3}c  n={r['n']:>6}  mean={r['mean']:+.3f}c/trade  hit={r['hit']:.3f}  sharpe={r['sharpe']:+.4f}  pnl={r['pnl']:+.1f}c")

    # Break-even cost: gross mean per trade at zero cost = the cost that zeroes it.
    breakeven = rows[0.0]["mean"]

    # Bootstrap 95% CI on net expectancy at 1c cost — the noise test.
    net = _trades(va_p, va_f, best_band, 1.0)
    boots = []
    for _ in range(2000):
        s = sum(net[int(_rand() * len(net))] for _ in range(len(net))) / len(net)
        boots.append(s)
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

    print(f"\nbreak-even cost: {breakeven:.3f}c/trade  (edge dies above this spread)")
    print(f"net expectancy @1c: {rows[1.0]['mean']:+.3f}c  95% CI [{lo:+.3f}, {hi:+.3f}]")
    makes_money = lo > 0 and rows[1.0]["mean"] > 0
    print("VERDICT:", "profitable after a 1c spread (CI excludes 0)" if makes_money
          else "edge does NOT survive a realistic spread — do not trade it naively")


if __name__ == "__main__":
    main()
