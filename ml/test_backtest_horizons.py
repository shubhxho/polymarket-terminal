"""Stdlib tests for the multi-horizon backtest primitives — no lightgbm needed.

The heavy per-horizon training lives behind lightgbm; these tests pin the parts
that decide the honest verdict: the Sharpe/P&L math, the result packer, and the
horizon-aware row builder (correct forward return, purge respected by construction).

Run:  python ml/test_backtest_horizons.py
"""

from __future__ import annotations

from backtest_horizons import _pack, _rows, _sharpe


def test_sharpe_sign_and_scale():
    # Positive-mean returns with real dispersion → positive Sharpe; flip → negative.
    rets = ([0.02, 0.01] * 50)               # mean 0.015, non-zero vol
    pnl, mean, sharpe, n = _sharpe(rets)
    assert n == 100 and abs(pnl - 1.5) < 1e-9 and mean > 0 and sharpe > 0
    pnl2, _, sharpe2, _ = _sharpe([-r for r in rets])
    assert abs(pnl2 + 1.5) < 1e-9 and sharpe2 < 0
    # Empty is safe.
    assert _sharpe([]) == (0.0, 0.0, 0.0, 0)


def test_zero_vol_gives_zero_sharpe():
    # Constant non-zero returns have zero dispersion → Sharpe defined as 0.
    _, _, sharpe, _ = _sharpe([0.02, 0.02, 0.02])
    assert sharpe == 0.0


def test_pack_rounds_and_keys():
    d = _pack((1.23456, 0.00012345, 2.34567, 42))
    assert set(d) == {"pnl", "mean_ret", "sharpe", "n"}
    assert d["n"] == 42 and d["pnl"] == 1.235 and d["sharpe"] == 2.346


def test_rows_forward_return_and_label():
    # A rising series with per-step variance (so windows aren't filtered as flat).
    prices = [0.2 + 0.01 * i + 0.001 * (i % 3) for i in range(80)]
    for H in (4, 8, 16):
        rows = _rows(prices, H)
        assert rows, H
        for feats, label, fwd in rows:
            assert len(feats) > 0
            assert (label == 1) == (fwd > 0)     # label is the sign of the fwd return
        # A predominantly rising series → mostly up labels and positive returns.
        assert sum(1 for _, lab, _ in rows if lab == 1) > 0.8 * len(rows)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
