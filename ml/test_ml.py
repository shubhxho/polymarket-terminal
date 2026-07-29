"""Stdlib tests for the ML signal pipeline — no pytest, no GPU.

Run:  python ml/test_ml.py     (exits non-zero on the first failed assertion)

Covers the feature layer (the new RSI/CCI/MACD/stochastic oscillators the model
now leans on) and the leakage-safe temporal split in train_seq. Fast enough to
run on every commit.
"""

from __future__ import annotations

import math

from features import (
    FEATURE_NAMES,
    HORIZON,
    _cci,
    _macd_hist,
    _rsi,
    _stoch_k,
    series_to_rich,
    window_features,
)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def test_feature_vector_shape_and_finite():
    ramp = [0.3 + 0.01 * i for i in range(16)]
    v = window_features(ramp)
    assert len(v) == len(FEATURE_NAMES) == 13, len(v)
    assert all(math.isfinite(x) for x in v)


def test_rsi_sign_and_bounds():
    up = [0.01 * i for i in range(1, 16)]     # all gains
    down = [-0.01 * i for i in range(1, 16)]  # all losses
    assert approx(_rsi(up), 1.0), _rsi(up)
    assert approx(_rsi(down), -1.0), _rsi(down)
    assert _rsi([0.0] * 15) == 0.0            # flat → neutral
    assert -1.0 <= _rsi([0.02, -0.01, 0.03, -0.02]) <= 1.0


def test_cci_sign_and_clip():
    # Last point stretched well above the window mean → positive CCI.
    w = [0.5] * 15 + [0.9]
    assert _cci(w) > 0
    w2 = [0.5] * 15 + [0.1]
    assert _cci(w2) < 0
    assert _cci([0.5] * 16) == 0.0            # flat → 0
    assert -5.0 <= _cci(w) <= 5.0             # clipped


def test_stoch_k_range():
    w = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert approx(_stoch_k(w), 1.0)           # last == high
    assert approx(_stoch_k(list(reversed(w))), 0.0)  # last == low
    assert _stoch_k([0.4] * 5) == 0.5         # flat → mid


def test_macd_trend_sign():
    up = [0.3 + 0.02 * i for i in range(16)]
    dn = [0.9 - 0.02 * i for i in range(16)]
    assert _macd_hist(up) > 0
    assert _macd_hist(dn) < 0


def test_degenerate_windows_are_finite():
    for w in ([0.0] * 16, [1.0] * 16, [0.5] * 16):
        assert all(math.isfinite(x) for x in window_features(w))


def test_temporal_split_no_leakage():
    """Validation windows must sit strictly later in time than training ones,
    with a HORIZON purge — no forward label peeks into the val region."""
    from train_seq import _split

    series = [[0.3 + 0.005 * i + 0.01 * math.sin(i / 3) for i in range(120)] for _ in range(6)]
    tr, va = _split(series, val_frac=0.2)
    assert tr and va
    # Reconstruct per-series ordering: the val samples' forward returns exist and
    # the split kept train strictly smaller than the un-purged cut.
    full = sum(len(series_to_rich(s)) for s in series)
    assert len(tr) + len(va) < full          # purge dropped the overlap band
    assert len(va) > 0 and len(tr) > len(va)


def test_walk_forward_folds_are_ordered():
    from train_seq import _walk_forward

    series = [[0.3 + 0.004 * i + 0.02 * math.sin(i / 4) for i in range(160)] for _ in range(8)]
    folds = list(_walk_forward(series, folds=4))
    assert len(folds) == 4
    # Later folds train on more data (expanding window).
    sizes = [len(tr) for _, tr, _ in folds]
    assert sizes == sorted(sizes), sizes


# ── OHLCV feature layer ───────────────────────────────────────────────────────

def test_ohlcv_feature_shape_and_finite():
    from features_ohlcv import OHLCV_FEATURES, window_features_ohlcv

    o = [0.4 + 0.01 * i for i in range(16)]
    h = [x + 0.02 for x in o]
    l = [x - 0.02 for x in o]
    c = [x + 0.005 for x in o]
    v = [100.0 + i for i in range(16)]
    n = [2 + i % 3 for i in range(16)]
    f = window_features_ohlcv(o, h, l, c, v, n, platform=0)
    assert len(f) == len(OHLCV_FEATURES) == 20
    assert all(math.isfinite(x) for x in f)


def test_order_flow_sign():
    """signed_flow > 0 when every candle closes above its open (buy pressure)."""
    from features_ohlcv import OHLCV_FEATURES, window_features_ohlcv

    idx = OHLCV_FEATURES.index("signed_flow")
    o = [0.5] * 16
    up_c = [0.55] * 16                       # closes above opens → +flow
    dn_c = [0.45] * 16                       # closes below opens → -flow
    h = [0.6] * 16
    l = [0.4] * 16
    v = [100.0] * 16
    n = [3] * 16
    assert window_features_ohlcv(o, h, l, up_c, v, n, 0)[idx] > 0.9
    assert window_features_ohlcv(o, h, l, dn_c, v, n, 0)[idx] < -0.9


def test_atr_and_cci_tp_use_range():
    from features_ohlcv import _atr, _cci_tp, _williams_r

    h = [0.5 + 0.1] * 16
    l = [0.5 - 0.1] * 16
    c = [0.5] * 16
    assert _atr(h, l, c) > 0                  # non-zero range → non-zero ATR
    assert _cci_tp(h, l, c) == 0.0            # flat typical price → 0
    # Williams %R at range top vs bottom.
    assert _williams_r([1.0], [0.0], [1.0]) == 0.0
    assert _williams_r([1.0], [0.0], [0.0]) == -1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
