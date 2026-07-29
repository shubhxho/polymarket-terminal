"""OHLCV feature extraction — the richer signal layer.

`fetch_hf.py` pulls real 1-hour **OHLCV** candles (open/high/low/close/volume/
trade_count). True highs and lows and real volume let us compute the indicators
*properly* instead of approximating them from a single price:

- **CCI** on the typical price (H+L+C)/3, as the textbook defines it — not on
  close alone.
- **Stochastic %K** and **Williams %R** over the true high-low range.
- **ATR** and **Parkinson** volatility, which need the intrabar range.
- **Order-flow imbalance** — signed volume, sign from each candle's close-vs-open
  — the microstructure signal that hand oscillators can't see. This is the piece
  the deep-research pass flags as adding real predictive value beyond RSI/CCI.

Emits `features.Sample` rows (return sequence + feature vector + label + forward
return), so it drops straight into the existing `train_seq.py` machinery.
"""

from __future__ import annotations

import math
from typing import List

from features import HORIZON, MIN_STD, WINDOW, Sample, _autocorr, _mean, _std

# Reuse the close-only oscillator helpers where the definition is identical.
from features import _macd_hist, _rsi  # noqa: E402

OHLCV_FEATURES = [
    # close path
    "last", "mean_ret", "vol", "drift", "band_z", "momentum", "autocorr",
    # proper oscillators (use true high/low / typical price)
    "rsi", "cci_tp", "stoch_k", "williams_r", "macd_hist",
    # volatility (need intrabar range)
    "atr", "parkinson_vol",
    # volume / order flow
    "vol_z", "trade_intensity", "signed_flow", "intrabar_pressure",
    # context
    "extremeness", "platform",
]


def _cci_tp(high, low, close) -> float:
    """Textbook CCI on typical price TP=(H+L+C)/3, scaled by /100 and clipped."""
    tp = [(high[i] + low[i] + close[i]) / 3 for i in range(len(close))]
    sma = _mean(tp)
    mad = _mean([abs(x - sma) for x in tp])
    if mad < 1e-9:
        return 0.0
    cci = (tp[-1] - sma) / (0.015 * mad)
    return max(-5.0, min(5.0, cci / 100.0))


def _stoch_k(high, low, close) -> float:
    hi, lo = max(high), min(low)
    if hi - lo < 1e-9:
        return 0.5
    return (close[-1] - lo) / (hi - lo)


def _williams_r(high, low, close) -> float:
    """Williams %R in -1..0 (0 = at range high, -1 = at range low)."""
    hi, lo = max(high), min(low)
    if hi - lo < 1e-9:
        return -0.5
    return -(hi - close[-1]) / (hi - lo)


def _atr(high, low, close) -> float:
    """Average true range over the window, normalised by last price."""
    trs = []
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        trs.append(tr)
    atr = _mean(trs) if trs else 0.0
    return atr / max(close[-1], 1e-3)


def _parkinson_vol(high, low) -> float:
    """Parkinson high-low volatility estimator (more efficient than close-close)."""
    vals = []
    for h, l in zip(high, low):
        if h > l and l > 0:
            vals.append((math.log(h / l)) ** 2)
    if not vals:
        return 0.0
    return math.sqrt(_mean(vals) / (4 * math.log(2)))


def window_features_ohlcv(o, h, l, c, v, n, platform: int) -> List[float]:
    """Feature vector for one OHLCV look-back window (lists length WINDOW)."""
    rets = [c[i] - c[i - 1] for i in range(1, len(c))]
    std = _std(rets)
    mean_c, std_c = _mean(c), _std(c)
    band_z = (c[-1] - mean_c) / std_c if std_c > 1e-9 else 0.0
    momentum = _mean(rets[-4:]) if len(rets) >= 4 else _mean(rets)

    # volume / order flow
    mean_v = _mean(v)
    vol_z = (v[-1] - mean_v) / (_std(v) + 1e-9) if len(v) >= 2 else 0.0
    mean_n = _mean(n)
    trade_intensity = (n[-1] - mean_n) / (_std(n) + 1e-9) if len(n) >= 2 else 0.0
    # signed volume: each candle's flow sign from close-vs-open, weighted by volume.
    signed = sum((1.0 if c[i] > o[i] else -1.0 if c[i] < o[i] else 0.0) * v[i] for i in range(len(c)))
    total_v = sum(v) + 1e-9
    signed_flow = signed / total_v                       # order-flow imbalance in -1..1
    # intrabar pressure: mean of (close-open)/(high-low) — where each bar closed in its range.
    press = []
    for i in range(len(c)):
        rng = h[i] - l[i]
        press.append((c[i] - o[i]) / rng if rng > 1e-9 else 0.0)
    intrabar_pressure = _mean(press)

    return [
        c[-1],
        _mean(rets),
        std,
        c[-1] - c[0],
        band_z,
        momentum,
        _autocorr(rets),
        _rsi(rets),
        _cci_tp(h, l, c),
        _stoch_k(h, l, c),
        _williams_r(h, l, c),
        _macd_hist(c),
        _atr(h, l, c),
        _parkinson_vol(h, l),
        vol_z,
        trade_intensity,
        signed_flow,
        intrabar_pressure,
        abs(c[-1] - 0.5) * 2,      # extremeness (favorite-longshot proxy)
        float(platform),
    ]


def market_to_rich(m: dict) -> List[Sample]:
    """Slide over one market's OHLCV series → Sample rows for train_seq."""
    o, h, l, c = m["open"], m["high"], m["low"], m["close"]
    v, n = m["volume"], m["trades"]
    plat = 1.0 if int(m.get("platform", 1)) != 1 else 0.0   # 0 = Polymarket, 1 = other
    out: List[Sample] = []
    N = len(c)
    for i in range(WINDOW, N - HORIZON):
        sl = slice(i - WINDOW, i)
        cw = c[sl]
        rets = [cw[k] - cw[k - 1] for k in range(1, len(cw))]
        if _std(rets) < MIN_STD:
            continue
        fwd = c[i + HORIZON] - c[i]
        feat = window_features_ohlcv(o[sl], h[sl], l[sl], cw, v[sl], n[sl], plat)
        out.append(Sample(rets, feat, 1 if fwd > 0 else 0, fwd))
    return out


def build(series: List[dict]) -> List[List[Sample]]:
    """Per-market Sample lists (kept grouped so the temporal split stays honest)."""
    return [market_to_rich(m) for m in series]


if __name__ == "__main__":
    import json
    import os

    path = os.path.join(os.path.dirname(__file__), "data", "ohlcv.json")
    series = json.load(open(path))
    groups = build(series)
    n = sum(len(g) for g in groups)
    ups = [s.label for g in groups for s in g]
    print(f"features/row: {len(OHLCV_FEATURES)}  markets: {len(series)}  windows: {n}  up-rate: {_mean([float(x) for x in ups]):.3f}")
    # finiteness check
    bad = sum(1 for g in groups for s in g if not all(math.isfinite(x) for x in s.feat))
    print(f"non-finite feature rows: {bad}")
    assert bad == 0
