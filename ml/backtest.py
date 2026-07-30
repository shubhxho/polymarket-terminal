"""Signal-quality backtester — score a signal by REALIZED trading quality.

AUC tells you whether a signal *ranks* markets correctly. It does not tell you
whether trading the signal against the live market price actually makes money
after fees, nor whether the edge survives out-of-time. This harness answers the
question that matters for a prediction-market signal: if you had bought YES when
the model thought a market was underpriced and NO when overpriced, settling at
the realized 0/1 outcome, what would the P&L, hit-rate, Sharpe and drawdown have
been — and which edge threshold would you have picked *without looking ahead*?

Trading convention (per unit position = one share, price in probability units):
- A YES share costs `price`, pays 1 if the market resolves YES (outcome=1) else 0.
  Buy YES when the model's edge `model_prob - price` exceeds `+threshold`.
  Net profit  = (outcome - price) - cost.
- A NO share costs `1 - price`, pays 1 if it resolves NO (outcome=0) else 0.
  Buy NO when the edge is below `-threshold`.
  Net profit  = ((1 - outcome) - (1 - price)) - cost = (price - outcome) - cost.
- `cost` is a configurable per-trade fee + slippage, in the same probability units
  (e.g. 0.01 = one cent of slippage per share).

A record is a 4-tuple ``(predicted_prob, market_price, realized_outcome, timestamp)``
with the two probabilities in [0, 1], the outcome in {0, 1}, and any sortable
timestamp. Everything here is pure stdlib (``math``) — no numpy/pandas — so it
mirrors the style of ``features_ohlcv.py`` and the ``_auc`` / ``_decile_backtest``
helpers in ``train_seq.py`` and drops in without pulling mlx.

Run:  python ml/backtest.py     (selfcheck; exits non-zero on a failed assertion)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# A backtest record: (predicted_prob, market_price, realized_outcome, timestamp).
Record = Tuple[float, float, float, float]

# Default edge thresholds swept when picking the risk-adjusted-optimal cutoff.
DEFAULT_THRESHOLDS: Tuple[float, ...] = tuple(round(0.01 * i, 4) for i in range(0, 16))


# ── small stdlib stats helpers (mirrors features.py style) ────────────────────

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    """Population standard deviation (0.0 for <2 points)."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# ── result containers ─────────────────────────────────────────────────────────

@dataclass
class CalibrationBin:
    """One reliability bin: how the model's stated probability lined up with the
    realized YES rate for records whose predicted prob fell in [lo, hi)."""

    lo: float
    hi: float
    count: int
    mean_pred: float
    mean_outcome: float


@dataclass
class BacktestResult:
    """Realized trading quality of a signal at one edge threshold.

    All fields are plain floats/ints (JSON- and assert-friendly). `pnl` is the
    summed net profit across trades; `mean_return` / `sharpe` are per-trade.
    """

    n_records: int
    n_trades: int
    threshold: float
    fee: float
    pnl: float
    mean_return: float
    hit_rate: float
    avg_edge: float          # mean realized edge captured per trade, pre-fee
    avg_model_edge: float    # mean |model_prob - price| the signal bet on
    sharpe: float            # per-trade mean/std of net return
    max_drawdown: float      # largest peak-to-trough drop of the equity curve
    calibration: List[CalibrationBin] = field(default_factory=list)
    decile: dict = field(default_factory=dict)
    # Populated only when the threshold was chosen by a train-split sweep.
    chosen_on_train: bool = False
    train_threshold: Optional[float] = None
    sweep: Optional[dict] = None

    def as_dict(self) -> dict:
        d = {
            "n_records": self.n_records,
            "n_trades": self.n_trades,
            "threshold": round(self.threshold, 4),
            "fee": self.fee,
            "pnl": round(self.pnl, 4),
            "mean_return": round(self.mean_return, 5),
            "hit_rate": round(self.hit_rate, 4),
            "avg_edge": round(self.avg_edge, 5),
            "avg_model_edge": round(self.avg_model_edge, 5),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "calibration": [
                {
                    "lo": round(b.lo, 3),
                    "hi": round(b.hi, 3),
                    "count": b.count,
                    "mean_pred": round(b.mean_pred, 4),
                    "mean_outcome": round(b.mean_outcome, 4),
                }
                for b in self.calibration
            ],
            "decile": self.decile,
            "chosen_on_train": self.chosen_on_train,
        }
        if self.train_threshold is not None:
            d["train_threshold"] = round(self.train_threshold, 4)
        if self.sweep is not None:
            d["sweep"] = self.sweep
        return d


# ── core mechanics ────────────────────────────────────────────────────────────

def _validate(records: Sequence[Record]) -> None:
    if not records:
        raise ValueError("run_backtest: empty record list — nothing to backtest")
    for i, r in enumerate(records):
        if len(r) != 4:
            raise ValueError(f"record {i} must be (prob, price, outcome, timestamp), got {r!r}")
        p, price, y, _ = r
        for name, v in (("predicted_prob", p), ("market_price", price), ("outcome", y)):
            if not math.isfinite(v):
                raise ValueError(f"record {i} {name} is not finite: {v!r}")


def _sorted_by_time(records: Sequence[Record]) -> List[Record]:
    """Chronological order — the equity curve and drawdown are path-dependent, so
    trades must settle in time order, not input order."""
    return sorted(records, key=lambda r: r[3])


def _trades_at_threshold(
    records: Sequence[Record], threshold: float, fee: float
) -> Tuple[List[float], List[float], List[float], int]:
    """Walk records in time order, opening the model's edge trade on each.

    Returns (net_pnls, realized_edges, model_edges, hits) where net_pnls are the
    per-trade profits after fee in chronological order.
    """
    net_pnls: List[float] = []
    realized_edges: List[float] = []
    model_edges: List[float] = []
    hits = 0
    for p, price, y, _ in _sorted_by_time(records):
        edge = p - price
        if edge > threshold:                       # model says underpriced → buy YES
            gross = y - price
            won = y > 0.5
        elif edge < -threshold:                    # model says overpriced → buy NO
            gross = price - y
            won = y <= 0.5
        else:
            continue                               # edge too small to overcome cost band
        net_pnls.append(gross - fee)
        realized_edges.append(gross)
        model_edges.append(abs(edge))
        if won:
            hits += 1
    return net_pnls, realized_edges, model_edges, hits


def _max_drawdown(net_pnls: Sequence[float]) -> float:
    """Largest peak-to-trough drop of the cumulative-profit curve (>= 0)."""
    peak = 0.0
    equity = 0.0
    mdd = 0.0
    for r in net_pnls:
        equity += r
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)
    return mdd


def _calibration(records: Sequence[Record], nbins: int) -> List[CalibrationBin]:
    """Reliability bins over predicted_prob in [0, 1]. Empty bins are dropped so
    every returned bin has finite stats."""
    buckets: List[List[Tuple[float, float]]] = [[] for _ in range(nbins)]
    for p, _price, y, _ts in records:
        idx = min(nbins - 1, max(0, int(p * nbins)))
        buckets[idx].append((p, y))
    out: List[CalibrationBin] = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        preds = [p for p, _ in b]
        outs = [y for _, y in b]
        out.append(
            CalibrationBin(
                lo=i / nbins,
                hi=(i + 1) / nbins,
                count=len(b),
                mean_pred=_mean(preds),
                mean_outcome=_mean(outs),
            )
        )
    return out


def _decile_backtest(probs: List[float], fwd: List[float], q: float = 0.2) -> dict:
    """Signal quality by predicted-prob decile — same convention as
    ``train_seq._decile_backtest`` (reimplemented in stdlib so this module never
    imports mlx). Sort by score, compare the top and bottom `q` fractions by how
    they actually resolved; `up_rate` is the fraction that resolved YES (fwd > 0).
    Here `fwd` is the realized outcome (or outcome-minus-price excess) per record.
    A full 10-decile up-rate ladder is added for a monotonicity read.
    """
    if not probs:
        return {}
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    k = max(1, int(len(order) * q))
    top = [fwd[i] for i in order[-k:]]
    bottom = [fwd[i] for i in order[:k]]

    def _median(xs):
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    up = lambda xs: sum(1 for v in xs if v > 0) / len(xs)  # noqa: E731

    ladder = []
    n = len(order)
    for d in range(10):
        a = int(n * d / 10)
        c = int(n * (d + 1) / 10)
        if c <= a:
            continue
        seg = [fwd[i] for i in order[a:c]]
        ladder.append(round(up(seg), 3))

    return {
        "top_up_rate": round(up(top), 3),
        "bottom_up_rate": round(up(bottom), 3),
        "up_rate_spread": round(up(top) - up(bottom), 3),
        "top_median": round(_median(top), 4),
        "bottom_median": round(_median(bottom), 4),
        "decile_up_rates": ladder,
        "slice": k,
    }


def _result_at(
    records: Sequence[Record], threshold: float, fee: float, calib_bins: int
) -> BacktestResult:
    """All metrics for one (already-validated) record set at a fixed threshold."""
    net_pnls, realized_edges, model_edges, hits = _trades_at_threshold(records, threshold, fee)
    n_trades = len(net_pnls)
    std = _std(net_pnls)
    mean_ret = _mean(net_pnls)
    sharpe = mean_ret / std if std > 1e-12 else 0.0
    probs = [r[0] for r in records]
    outcomes = [r[2] for r in records]
    return BacktestResult(
        n_records=len(records),
        n_trades=n_trades,
        threshold=threshold,
        fee=fee,
        pnl=sum(net_pnls),
        mean_return=mean_ret,
        hit_rate=(hits / n_trades) if n_trades else 0.0,
        avg_edge=_mean(realized_edges),
        avg_model_edge=_mean(model_edges),
        sharpe=sharpe,
        max_drawdown=_max_drawdown(net_pnls),
        calibration=_calibration(records, calib_bins),
        decile=_decile_backtest(probs, outcomes),
    )


def _score(records: Sequence[Record], threshold: float, fee: float) -> Tuple[float, float, int]:
    """Risk-adjusted ranking key for the threshold sweep: Sharpe first, then mean
    return, then trade count — so a higher-Sharpe threshold wins, ties broken
    toward more return and then more (statistically firmer) trades.

    Computes only the trade-based stats it needs — not the calibration bins or
    decile backtest, which don't depend on the threshold — so sweeping N cutoffs
    stays cheap instead of rebuilding the full result N times.
    """
    net_pnls, _edges, _model_edges, _hits = _trades_at_threshold(records, threshold, fee)
    std = _std(net_pnls)
    mean_ret = _mean(net_pnls)
    sharpe = mean_ret / std if std > 1e-12 else 0.0
    return (sharpe, mean_ret, len(net_pnls))


# ── public entry point ────────────────────────────────────────────────────────

def run_backtest(
    records: Sequence[Record],
    threshold: float = 0.0,
    fee: float = 0.01,
    calib_bins: int = 10,
    train_frac: Optional[float] = None,
    thresholds: Optional[Sequence[float]] = None,
) -> BacktestResult:
    """Backtest a signal's realized trading quality.

    Two modes:

    * **Fixed threshold** (default) — score every record at `threshold` and return
      the full metric set.
    * **No-lookahead sweep** — pass `train_frac` (0<f<1). Records are ordered by
      timestamp and split into an earlier *train* slice and a later *test* slice.
      Every candidate in `thresholds` (defaults to 0..0.15 in 1c steps) is scored
      on train; the one with the best risk-adjusted return (Sharpe, then mean
      return, then trade count) is selected, and the returned result is the
      **test-split** performance at that threshold. `train_threshold`,
      `chosen_on_train` and a `sweep` breakdown record how it was picked — the
      test slice never influences the choice.

    Raises ``ValueError`` on empty or malformed input.
    """
    _validate(records)
    if train_frac is None:
        return _result_at(records, threshold, fee, calib_bins)

    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    cand = list(thresholds) if thresholds is not None else list(DEFAULT_THRESHOLDS)
    if not cand:
        raise ValueError("threshold sweep needs at least one candidate threshold")

    ordered = _sorted_by_time(records)
    cut = int(len(ordered) * train_frac)
    train, test = ordered[:cut], ordered[cut:]
    if not train or not test:
        raise ValueError(
            f"train_frac={train_frac} left an empty split ({len(train)}/{len(test)}) — "
            "need more records or a more central fraction"
        )

    sweep_rows = []
    best_t = cand[0]
    best_key = None
    for t in cand:
        key = _score(train, t, fee)              # sharpe, mean_return, n_trades
        sweep_rows.append(
            {"threshold": round(t, 4), "train_sharpe": round(key[0], 4),
             "train_mean_return": round(key[1], 5), "train_trades": key[2]}
        )
        if best_key is None or key > best_key:
            best_key, best_t = key, t

    # The reported result is the TEST slice at the train-chosen threshold.
    res = _result_at(test, best_t, fee, calib_bins)
    res.chosen_on_train = True
    res.train_threshold = best_t
    res.sweep = {
        "train_frac": train_frac,
        "n_train": len(train),
        "n_test": len(test),
        "candidates": sweep_rows,
        "chosen_threshold": round(best_t, 4),
    }
    return res


# ── self-check on synthetic signal-vs-market data ─────────────────────────────

def _synth(n: int, seed: int, informed: bool) -> List[Record]:
    """Synthetic records. Each market has a hidden true prob; the market price is
    a noisy quote around it and the outcome is Bernoulli(true). An *informed*
    model sits closer to the truth than the market (a real edge); a *random*
    model quotes noise unrelated to the truth (no edge)."""
    import random

    rng = random.Random(seed)
    out: List[Record] = []
    for t in range(n):
        true = rng.uniform(0.05, 0.95)
        price = min(0.99, max(0.01, true + rng.gauss(0.0, 0.10)))
        if informed:
            prob = min(0.99, max(0.01, true + rng.gauss(0.0, 0.03)))
        else:
            prob = rng.uniform(0.01, 0.99)
        outcome = 1.0 if rng.random() < true else 0.0
        out.append((prob, price, outcome, float(t)))
    return out


def _selfcheck() -> None:
    import json

    N, FEE = 6000, 0.01

    # 1) A genuinely predictive signal must make money and rank markets.
    good = run_backtest(_synth(N, seed=1, informed=True), threshold=0.03, fee=FEE)
    print("informed  :", json.dumps(good.as_dict(), indent=2)[:600], "...\n")
    assert good.n_trades > 0, "informed signal took no trades"
    assert good.pnl > 0, f"informed PnL not positive: {good.pnl}"
    assert good.sharpe > 0, f"informed Sharpe not positive: {good.sharpe}"
    assert good.decile["up_rate_spread"] > 0, "informed decile spread not positive"

    # 2) A random signal must be ~flat: no meaningful edge or Sharpe.
    rand = run_backtest(_synth(N, seed=2, informed=False), threshold=0.03, fee=FEE)
    print("random    : sharpe=%.4f  pnl=%.4f  mean_ret=%.5f  trades=%d"
          % (rand.sharpe, rand.pnl, rand.mean_return, rand.n_trades))
    assert abs(rand.sharpe) < 0.15, f"random Sharpe not ~0: {rand.sharpe}"
    assert good.sharpe > rand.sharpe, "informed must beat random on Sharpe"

    # 3) Calibration of the informed signal is monotone-ish: realized YES rate
    #    should rise with predicted probability across the bins.
    bins = good.calibration
    assert len(bins) >= 4, "too few calibration bins"
    ups = sum(1 for a, b in zip(bins, bins[1:]) if b.mean_outcome >= a.mean_outcome - 0.05)
    assert ups >= len(bins) - 2, f"calibration not monotone-ish: {[round(b.mean_outcome,2) for b in bins]}"
    assert bins[-1].mean_outcome > bins[0].mean_outcome, "top bin not above bottom bin"

    # 4) No-lookahead threshold sweep: chosen on train, reported on test, finite.
    swept = run_backtest(_synth(N, seed=3, informed=True), fee=FEE,
                         train_frac=0.6, thresholds=[0.0, 0.02, 0.05, 0.10])
    print("swept     : chosen_t=%.2f  test_sharpe=%.4f  test_pnl=%.4f"
          % (swept.train_threshold, swept.sharpe, swept.pnl))
    assert swept.chosen_on_train and swept.train_threshold is not None
    assert swept.sweep is not None and swept.sweep["chosen_threshold"] == swept.threshold
    for v in (swept.sharpe, swept.pnl, swept.mean_return, swept.max_drawdown):
        assert math.isfinite(v), "swept metric not finite"

    print("\nselfcheck OK — predictive signal profits, random ~0, calibration monotone-ish.")


if __name__ == "__main__":
    _selfcheck()
