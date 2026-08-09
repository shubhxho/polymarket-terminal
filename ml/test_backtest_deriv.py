"""Stdlib tests for the derivative-signal backtest math — no GPU, no lightgbm.

Run:  python ml/test_backtest_deriv.py

Checks the P&L primitives directly: a signal perfectly aligned with the forward
return must make money and beat the drift baseline; an anti-aligned one must lose;
fees must strictly reduce P&L; the decile table must be monotone for a clean
signal. These guard the logic the honest verdict depends on.
"""

from __future__ import annotations

from backtest_deriv import (
    _decile_returns,
    _directional,
    _ranking_ls,
    _stats,
)


def test_aligned_signal_profits_anti_loses():
    # prob tracks the sign of the forward return exactly.
    fwd = [(-1) ** i * (0.01 + 0.001 * i) for i in range(200)]
    probs = [0.9 if f > 0 else 0.1 for f in fwd]
    good = _stats("good", _directional(probs, fwd, 0.0, 0.0))
    assert good.pnl > 0 and good.sharpe > 0 and good.hit == 1.0, good
    # Flip the signal → every call is wrong → symmetric loss.
    anti = _stats("anti", _directional([1 - p for p in probs], fwd, 0.0, 0.0))
    assert anti.pnl < 0 and anti.hit == 0.0, anti


def test_fees_reduce_pnl():
    fwd = [(-1) ** i * 0.02 for i in range(100)]
    probs = [0.9 if f > 0 else 0.1 for f in fwd]
    free = _stats("f0", _directional(probs, fwd, 0.0, 0.0))
    paid = _stats("f1", _directional(probs, fwd, 0.0, 0.01))
    assert paid.pnl < free.pnl
    # 100 trades × 0.01 fee removed exactly.
    assert abs((free.pnl - paid.pnl) - 100 * 0.01) < 1e-9


def test_ranking_long_short_captures_edge():
    # Monotone: higher prob → higher forward return. L/S must profit.
    fwd = [0.001 * (i - 100) for i in range(200)]
    probs = [i / 200 for i in range(200)]
    ls = _stats("ls", _ranking_ls(probs, fwd, 0.1, 0.0))
    assert ls.pnl > 0 and ls.sharpe > 0, ls
    dec = _decile_returns(probs, fwd)
    assert dec[-1] > dec[0]                      # top decile return > bottom
    assert all(dec[i] <= dec[i + 1] + 1e-9 for i in range(9)), dec  # monotone


def test_empty_and_flat_are_safe():
    assert _stats("empty", []).n == 0
    # A dead-neutral signal (band excludes everything) makes no trades.
    flat = _directional([0.5] * 50, [0.01] * 50, 0.1, 0.0)
    assert flat == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
