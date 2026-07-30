"""Stdlib tests for the signal-quality backtester — no pytest, no GPU.

Run:  python ml/test_backtest.py   (exits non-zero on the first failed assertion)

Covers the trading mechanics (PnL sign for a rigged predictive vs random signal),
the no-lookahead threshold sweep (chosen on the train split only, reported on
test), metric finiteness, drawdown/calibration behaviour, and input guards. Fast
enough to run on every commit.
"""

from __future__ import annotations

import math
import random

from backtest import (
    BacktestResult,
    CalibrationBin,
    _decile_backtest,
    _max_drawdown,
    run_backtest,
)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _synth(n: int, seed: int, informed: bool):
    """A predictive signal sits near each market's hidden true prob; a random one
    quotes noise. Mirrors backtest._synth but kept local so tests are standalone."""
    rng = random.Random(seed)
    out = []
    for t in range(n):
        true = rng.uniform(0.05, 0.95)
        price = min(0.99, max(0.01, true + rng.gauss(0.0, 0.10)))
        prob = (min(0.99, max(0.01, true + rng.gauss(0.0, 0.03)))
                if informed else rng.uniform(0.01, 0.99))
        outcome = 1.0 if rng.random() < true else 0.0
        out.append((prob, price, outcome, float(t)))
    return out


# ── trading mechanics ─────────────────────────────────────────────────────────

def test_pnl_math_single_yes_trade():
    """One clean YES trade: edge over threshold, settle YES, profit = 1 - price - fee."""
    res = run_backtest([(0.9, 0.6, 1.0, 0.0)], threshold=0.1, fee=0.01)
    assert res.n_trades == 1
    assert approx(res.pnl, 1.0 - 0.6 - 0.01)
    assert res.hit_rate == 1.0
    assert approx(res.avg_edge, 1.0 - 0.6)          # realized edge is pre-fee


def test_pnl_math_single_no_trade():
    """Model far below price → buy NO; market resolves NO → profit = price - 0 - fee."""
    res = run_backtest([(0.2, 0.7, 0.0, 0.0)], threshold=0.1, fee=0.01)
    assert res.n_trades == 1
    assert approx(res.pnl, 0.7 - 0.0 - 0.01)
    assert res.hit_rate == 1.0


def test_no_trade_inside_threshold_band():
    """Edge below the threshold band → no position taken."""
    res = run_backtest([(0.52, 0.50, 1.0, 0.0)], threshold=0.1, fee=0.01)
    assert res.n_trades == 0
    assert res.pnl == 0.0 and res.sharpe == 0.0 and res.hit_rate == 0.0


def test_predictive_signal_profits_random_does_not():
    """The headline property: a genuinely predictive signal has positive PnL and
    Sharpe; a random signal is ~flat and strictly worse on Sharpe."""
    good = run_backtest(_synth(4000, seed=10, informed=True), threshold=0.03, fee=0.01)
    rand = run_backtest(_synth(4000, seed=11, informed=False), threshold=0.03, fee=0.01)
    assert good.pnl > 0 and good.sharpe > 0
    assert good.sharpe > rand.sharpe
    assert abs(rand.sharpe) < 0.2                    # random ≈ 0 risk-adjusted return


def test_fee_reduces_pnl_monotonically():
    recs = _synth(2000, seed=12, informed=True)
    lo = run_backtest(recs, threshold=0.02, fee=0.0)
    hi = run_backtest(recs, threshold=0.02, fee=0.05)
    assert lo.n_trades == hi.n_trades                # fee doesn't change which trades open
    assert lo.pnl > hi.pnl                           # higher cost → lower PnL
    assert approx(lo.pnl - hi.pnl, 0.05 * lo.n_trades, tol=1e-6)


# ── no-lookahead threshold sweep ──────────────────────────────────────────────

def test_sweep_chosen_on_train_only():
    """The sweep must pick the threshold on the earlier train slice and report the
    later test slice at that threshold — never peeking at test to choose."""
    recs = _synth(5000, seed=20, informed=True)
    res = run_backtest(recs, fee=0.01, train_frac=0.6,
                       thresholds=[0.0, 0.02, 0.05, 0.10])
    assert res.chosen_on_train
    assert res.train_threshold == res.threshold      # reported at the train-chosen cutoff
    assert res.sweep["chosen_threshold"] == res.threshold
    assert res.sweep["n_train"] + res.sweep["n_test"] == len(recs)
    # The chosen threshold maximises train Sharpe among candidates (train-only pick).
    rows = res.sweep["candidates"]
    best_row = max(rows, key=lambda r: (r["train_sharpe"], r["train_mean_return"], r["train_trades"]))
    assert best_row["threshold"] == res.threshold


def test_sweep_no_lookahead_invariance_to_test_labels():
    """Concretely no-lookahead: scrambling the TEST-slice outcomes must not change
    which threshold the sweep selects (the choice depends only on train)."""
    recs = _synth(3000, seed=21, informed=True)
    base = run_backtest(recs, fee=0.01, train_frac=0.6, thresholds=[0.0, 0.02, 0.05, 0.1])
    cut = int(len(recs) * 0.6)
    rng = random.Random(99)
    tampered = recs[:cut] + [(p, pr, float(rng.random() < 0.5), ts)
                             for (p, pr, _y, ts) in recs[cut:]]
    other = run_backtest(tampered, fee=0.01, train_frac=0.6, thresholds=[0.0, 0.02, 0.05, 0.1])
    assert base.train_threshold == other.train_threshold      # pick unchanged
    # ...but the reported test PnL did move, proving test labels were actually used
    # for scoring (just not for selection).
    assert base.pnl != other.pnl


def test_sweep_respects_time_order_not_input_order():
    """Records handed in shuffled order must split the same way as sorted-by-time."""
    recs = _synth(2000, seed=22, informed=True)
    shuffled = list(recs)
    random.Random(7).shuffle(shuffled)
    a = run_backtest(recs, fee=0.01, train_frac=0.5, thresholds=[0.0, 0.03, 0.06])
    b = run_backtest(shuffled, fee=0.01, train_frac=0.5, thresholds=[0.0, 0.03, 0.06])
    assert a.train_threshold == b.train_threshold
    assert approx(a.pnl, b.pnl, tol=1e-9)
    assert a.n_trades == b.n_trades


# ── component behaviour ───────────────────────────────────────────────────────

def test_max_drawdown_basic():
    # Equity path: +1, +1, -3, +1 → cum 1,2,-1,0; peak 2, trough -1 → dd 3.
    assert approx(_max_drawdown([1.0, 1.0, -3.0, 1.0]), 3.0)
    assert _max_drawdown([1.0, 2.0, 3.0]) == 0.0     # monotone up → no drawdown
    assert _max_drawdown([]) == 0.0


def test_calibration_bins_monotone_for_informed():
    res = run_backtest(_synth(5000, seed=30, informed=True), threshold=0.0, fee=0.01)
    bins = res.calibration
    assert all(isinstance(b, CalibrationBin) for b in bins)
    assert all(math.isfinite(b.mean_pred) and math.isfinite(b.mean_outcome) for b in bins)
    assert bins[-1].mean_outcome > bins[0].mean_outcome
    assert sum(b.count for b in bins) == res.n_records


def test_decile_convention_matches_train_seq():
    """Reuses train_seq._decile_backtest convention: up_rate = fraction with fwd>0,
    top-minus-bottom spread positive when the score ranks outcomes."""
    probs = [i / 100 for i in range(100)]
    fwd = [1.0 if i >= 50 else 0.0 for i in range(100)]   # high score → resolves YES
    d = _decile_backtest(probs, fwd, q=0.2)
    assert d["top_up_rate"] == 1.0 and d["bottom_up_rate"] == 0.0
    assert approx(d["up_rate_spread"], 1.0)
    assert len(d["decile_up_rates"]) == 10
    assert d["decile_up_rates"][0] == 0.0 and d["decile_up_rates"][-1] == 1.0


def test_all_metrics_finite():
    for informed in (True, False):
        res = run_backtest(_synth(1500, seed=40, informed=informed), threshold=0.02, fee=0.01)
        for v in (res.pnl, res.mean_return, res.hit_rate, res.avg_edge,
                  res.avg_model_edge, res.sharpe, res.max_drawdown):
            assert math.isfinite(v), (informed, v)


def test_as_dict_is_json_shaped():
    import json
    res = run_backtest(_synth(500, seed=41, informed=True), threshold=0.02, fee=0.01)
    s = json.dumps(res.as_dict())          # must not raise
    assert '"sharpe"' in s and '"decile"' in s and '"calibration"' in s


# ── input guards ──────────────────────────────────────────────────────────────

def test_empty_input_raises():
    raised = False
    try:
        run_backtest([])
    except ValueError:
        raised = True
    assert raised, "empty input must raise ValueError"


def test_malformed_record_raises():
    for bad in ([(0.5, 0.5, 1.0)], [(float("nan"), 0.5, 1.0, 0.0)]):
        raised = False
        try:
            run_backtest(bad)
        except ValueError:
            raised = True
        assert raised, f"malformed record must raise: {bad}"


def test_bad_train_frac_raises():
    recs = _synth(100, seed=50, informed=True)
    for frac in (0.0, 1.0, 1.5):
        raised = False
        try:
            run_backtest(recs, train_frac=frac)
        except ValueError:
            raised = True
        assert raised, f"train_frac={frac} must raise"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
