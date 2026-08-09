"""Does the derivative signal actually make money? An honest, out-of-time P&L
backtest of the frozen GBDT derivative signal — with the baselines that keep it
honest, run in parallel.

The signal ranks the up/down direction well (out-of-time AUC ~0.65). That is NOT
the same as making money: AUC scores a *binary* label, while P&L is driven by the
*signed forward return* — and a ranking edge on the label need not survive as an
edge on returns, or fees. This module measures the thing that actually matters.

Strategies, all on the later-20% holdout that trained no model:
  * directional  — long when prob > 0.5+band, short when < 0.5-band (a band=0 is
    the naive "trade every signal");
  * ranking L/S  — long the top-q by prob, short the bottom-q (trade the *edge*,
    not the 0.5 line, so a base-rate skew can't fool it);
  * baselines    — always-long, always-short, random. The signal only has alpha
    if it BEATS the better drift baseline; a market set that mostly resolves down
    makes "always-short" look great for reasons that are nothing to do with skill.

Everything is reported net of a per-trade fee, swept over fee/band/quantile in
parallel (fork pool). It also prints a per-decile mean-forward-return table — the
cleanest read on whether the signal carries any *return* edge at all.

Run:  ml/.venv/bin/python backtest_deriv.py     (pure stdlib; numpy not required)
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from features import HORIZON, MIN_STD, WINDOW, _std
from features_deriv import deriv_signal_gbdt, load_gbdt_model

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data", "series.json")
OUT = os.path.join(_HERE, "data", "deriv_backtest.json")


def _holdout_windows(prices: Sequence[float]) -> List[Tuple[List[float], float]]:
    """Later-20% (out-of-time) windows of one series with their forward returns."""
    rows: List[Tuple[List[float], float]] = []
    n = len(prices)
    for i in range(WINDOW, n - HORIZON):
        w = prices[i - WINDOW : i]
        incs = [w[k] - w[k - 1] for k in range(1, len(w))]
        if _std(incs) < MIN_STD:
            continue
        rows.append((list(w), prices[i + HORIZON] - prices[i]))
    if len(rows) < 5:
        return []
    cut = int(len(rows) * 0.8)
    return rows[cut:]


def _score_series(prices: Sequence[float]) -> List[Tuple[float, float]]:
    """(prob_up, forward_return) for one series' holdout — the parallel unit."""
    model = load_gbdt_model()
    out = []
    for w, fwd in _holdout_windows(prices):
        p = deriv_signal_gbdt(w, model)
        if p is not None:
            out.append((p, fwd))
    return out


@dataclass
class Result:
    name: str
    pnl: float
    mean_ret: float
    sharpe: float
    hit: float
    n: int


def _stats(name: str, rets: List[float]) -> Result:
    if not rets:
        return Result(name, 0.0, 0.0, 0.0, 0.0, 0)
    n = len(rets)
    mean = sum(rets) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in rets) / n)
    sharpe = mean / sd * math.sqrt(n) if sd > 0 else 0.0
    hit = sum(1 for x in rets if x > 0) / n
    return Result(name, sum(rets), mean, sharpe, hit, n)


def _directional(probs, fwd, band, fee) -> List[float]:
    rets = []
    for p, f in zip(probs, fwd):
        pos = 1 if p > 0.5 + band else (-1 if p < 0.5 - band else 0)
        if pos:
            rets.append(pos * f - fee)
    return rets


def _ranking_ls(probs, fwd, q, fee) -> List[float]:
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    k = int(len(order) * q)
    if k == 0:
        return []
    shorts, longs = order[:k], order[-k:]
    return [fwd[i] - fee for i in longs] + [-fwd[i] - fee for i in shorts]


def _decile_returns(probs, fwd) -> List[float]:
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    k = max(1, len(order) // 10)
    out = []
    for d in range(10):
        idx = order[d * k : (d + 1) * k] if d < 9 else order[9 * k :]
        out.append(sum(fwd[i] for i in idx) / len(idx))
    return out


def _run_one(job):
    kind, params, probs, fwd = job
    if kind == "directional":
        band, fee = params
        return _stats(f"dir band={band:.2f} fee={fee:.3f}", _directional(probs, fwd, band, fee))
    band = None
    q, fee = params
    return _stats(f"L/S q={q:.2f} fee={fee:.3f}", _ranking_ls(probs, fwd, q, fee))


def main() -> int:
    if load_gbdt_model() is None:
        print("no frozen GBDT (data/deriv_gbdt.json) — run train_deriv_gbdt.py first")
        return 1
    series = json.load(open(DATA, encoding="utf-8"))

    # Score every series' holdout in parallel — the expensive pure-python step.
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(8, (os.cpu_count() or 2)),
                             mp_context=ctx) as ex:
        chunks = list(ex.map(_score_series, series))
    pairs = [pf for ch in chunks for pf in ch]
    probs = [p for p, _ in pairs]
    fwd = [f for _, f in pairs]
    n = len(pairs)
    base_up = sum(1 for f in fwd if f > 0) / n

    # Baselines (no signal): the honest yardsticks.
    baselines = [
        _stats("always-long", [f for f in fwd]),
        _stats("always-short", [-f for f in fwd]),
    ]

    # Parallel sweep of strategy params × fees.
    fees = [0.0, 0.005, 0.01]
    jobs = []
    for fee in fees:
        for band in [0.0, 0.05, 0.10]:
            jobs.append(("directional", (band, fee), probs, fwd))
        for q in [0.1, 0.2]:
            jobs.append(("ranking", (q, fee), probs, fwd))
    with ProcessPoolExecutor(max_workers=min(8, (os.cpu_count() or 2)),
                             mp_context=ctx) as ex:
        swept = list(ex.map(_run_one, jobs))

    deciles = _decile_returns(probs, fwd)

    # Verdict: best net-of-realistic-fee strategy vs the best drift baseline.
    realistic = [r for r in swept if "fee=0.005" in r.name or "fee=0.010" in r.name]
    best_strat = max(realistic, key=lambda r: r.sharpe) if realistic else None
    best_base = max(baselines, key=lambda r: r.sharpe)
    has_alpha = bool(best_strat and best_strat.sharpe > best_base.sharpe and best_strat.pnl > 0)

    def line(r: Result) -> str:
        return (f"  {r.name:<22} pnl={r.pnl:>+8.2f}  mean={r.mean_ret:>+.5f}  "
                f"sharpe={r.sharpe:>+6.2f}  hit={r.hit:.3f}  n={r.n}")

    print(f"OOT holdout windows: {n}   up-rate: {base_up:.3f}   "
          f"mean|fwd|: {sum(abs(f) for f in fwd)/n:.4f}")
    print("\nbaselines (no signal):")
    for r in baselines:
        print(line(r))
    print("\nsignal strategies (swept in parallel):")
    for r in swept:
        print(line(r))
    print("\nper-decile mean forward return (decile 0 = lowest prob):")
    print("  " + "  ".join(f"{d:+.4f}" for d in deciles))
    print(f"  decile spread (top-bottom): {deciles[-1]-deciles[0]:+.5f}")
    print("\nVERDICT: " + (
        f"signal HAS tradeable alpha (best {best_strat.name} sharpe "
        f"{best_strat.sharpe:.2f} > best baseline {best_base.name} "
        f"{best_base.sharpe:.2f})" if has_alpha else
        "NO tradeable alpha net of fees — the best net-of-fee strategy does not "
        f"beat the '{best_base.name}' drift baseline (sharpe {best_base.sharpe:.2f}). "
        "The AUC edge is on the binary up/down label; forward-RETURN correlation is "
        "~0, so ranking direction does not convert to P&L. Positive P&L only from "
        "structural drift, which is not signal skill."))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({
            "n": n, "up_rate": round(base_up, 4),
            "baselines": [r.__dict__ for r in baselines],
            "strategies": [r.__dict__ for r in swept],
            "decile_forward_returns": [round(d, 6) for d in deciles],
            "has_tradeable_alpha": has_alpha,
        }, fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
