"""Feature extraction for the direction classifier.

Turns a single market's price series (probabilities in 0..1, roughly uniform
cadence) into supervised (features, label) rows with a sliding window. The
feature set mirrors the terminal's own quant lib — drift, realised vol, a
Bollinger-style band z, short momentum, lag-1 autocorrelation of increments —
plus the classic technical oscillators traders actually watch: RSI, CCI, a
MACD histogram, a stochastic %K and a downside-vol risk proxy. The model learns
from the same microstructure the UI already surfaces.

Label: 1 if the price is higher HORIZON steps ahead, else 0. Pure stdlib +
math so it stays trivially runnable and testable without a GPU.
"""

from __future__ import annotations

import math
from typing import List, Tuple

WINDOW = 16          # look-back length for each sample
HORIZON = 4          # steps ahead the label looks
MIN_STD = 1e-4       # below this the window is settled/flat — skip it

RSI_EPS = 1e-9

FEATURE_NAMES = [
    "last",          # current probability
    "mean_ret",      # average increment over the window
    "vol",           # std of increments (realised vol proxy)
    "drift",         # net move across the window
    "band_z",        # (last - mean) / std  — how stretched
    "momentum",      # mean of the last 4 increments
    "autocorr",      # lag-1 autocorrelation of increments
    "activity",      # sum of |increments| — how much it moved at all
    "rsi",           # Wilder RSI, recentred to -1..1  (momentum oscillator)
    "cci",           # Commodity Channel Index / 100    (mean-reversion stretch)
    "macd_hist",     # MACD(3,8) minus its 4-EMA signal  (trend-change impulse)
    "stoch_k",       # stochastic %K in 0..1             (position in range)
    "downside_vol",  # std of negative increments only   (tail-risk proxy)
]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _autocorr(xs: List[float]) -> float:
    if len(xs) < 3:
        return 0.0
    m = _mean(xs)
    num = sum((xs[i] - m) * (xs[i - 1] - m) for i in range(1, len(xs)))
    den = sum((x - m) ** 2 for x in xs)
    return max(-1.0, min(1.0, num / den)) if den > 1e-12 else 0.0


def _rsi(rets: List[float]) -> float:
    """Wilder RSI over the window's increments, recentred to -1..1.

    Standard RSI is 100·avg_gain/(avg_gain+avg_loss); we map it to -1..1 so it
    sits on the same scale as the other signed features and z-scores cleanly.
    Flat window with no losses reads +1 (pinned overbought), no gains reads -1.
    """
    gains = sum(r for r in rets if r > 0)
    losses = -sum(r for r in rets if r < 0)
    denom = gains + losses
    if denom < RSI_EPS:
        return 0.0
    return (gains - losses) / denom  # == (RSI-50)/50


def _ema_series(xs: List[float], span: int) -> List[float]:
    a = 2.0 / (span + 1.0)
    out = [xs[0]]
    for x in xs[1:]:
        out.append(a * x + (1 - a) * out[-1])
    return out


def _cci(window: List[float]) -> float:
    """Commodity Channel Index / 100. Typical price is the probability itself
    (single series), so CCI = (last - sma) / (0.015·mean_abs_dev). Clipped to a
    sane band before scaling — raw CCI can spike to ±hundreds on a flat base."""
    sma = _mean(window)
    mad = _mean([abs(p - sma) for p in window])
    if mad < 1e-9:
        return 0.0
    cci = (window[-1] - sma) / (0.015 * mad)
    return max(-5.0, min(5.0, cci / 100.0))


def _macd_hist(window: List[float]) -> float:
    """MACD histogram: fast-EMA(3) minus slow-EMA(8), minus its 4-EMA signal.
    Short spans because the look-back window is only 16 points."""
    if len(window) < 4:
        return 0.0
    fast = _ema_series(window, 3)
    slow = _ema_series(window, 8)
    macd = [f - s for f, s in zip(fast, slow)]
    signal = _ema_series(macd, 4)
    return macd[-1] - signal[-1]


def _stoch_k(window: List[float]) -> float:
    """Stochastic %K: where the last price sits in the window's high-low range."""
    lo, hi = min(window), max(window)
    if hi - lo < 1e-9:
        return 0.5
    return (window[-1] - lo) / (hi - lo)


def window_features(window: List[float]) -> List[float]:
    """Feature vector for one look-back window of prices."""
    rets = [window[i] - window[i - 1] for i in range(1, len(window))]
    std = _std(rets)
    mean_w = _mean(window)
    std_w = _std(window)
    band_z = (window[-1] - mean_w) / std_w if std_w > 1e-9 else 0.0
    momentum = _mean(rets[-4:]) if len(rets) >= 4 else _mean(rets)
    neg = [r for r in rets if r < 0]
    return [
        window[-1],
        _mean(rets),
        std,
        window[-1] - window[0],
        band_z,
        momentum,
        _autocorr(rets),
        sum(abs(r) for r in rets),
        _rsi(rets),
        _cci(window),
        _macd_hist(window),
        _stoch_k(window),
        _std(neg) if len(neg) >= 2 else 0.0,
    ]


def series_to_samples(prices: List[float]) -> List[Tuple[List[float], int]]:
    """Slides over one price series, emitting (features, label) rows.

    Windows whose increments are essentially flat are dropped — a settled market
    has no direction to predict and would just teach the model the base rate.
    """
    out: List[Tuple[List[float], int]] = []
    n = len(prices)
    for i in range(WINDOW, n - HORIZON):
        window = prices[i - WINDOW : i]
        rets = [window[k] - window[k - 1] for k in range(1, len(window))]
        if _std(rets) < MIN_STD:
            continue
        label = 1 if prices[i + HORIZON] > prices[i] else 0
        out.append((window_features(window), label))
    return out


def build_dataset(series: List[List[float]]) -> Tuple[List[List[float]], List[int]]:
    """Flattens many markets' series into one (X, y) dataset."""
    X: List[List[float]] = []
    y: List[int] = []
    for prices in series:
        for feats, label in series_to_samples(prices):
            X.append(feats)
            y.append(label)
    return X, y


class Sample:
    """One training row for the sequence model: the return sequence, the hand
    features, the direction label, and the *signed forward return* (for the
    backtest, which asks whether the model's ranking separates winners)."""

    __slots__ = ("seq", "feat", "label", "fwd")

    def __init__(self, seq: List[float], feat: List[float], label: int, fwd: float):
        self.seq = seq
        self.feat = feat
        self.label = label
        self.fwd = fwd


def series_to_rich(prices: List[float]) -> List[Sample]:
    """Like `series_to_samples` but also carries the raw return sequence and the
    forward return, so one pass feeds both the feature-MLP and the GRU."""
    out: List[Sample] = []
    n = len(prices)
    for i in range(WINDOW, n - HORIZON):
        window = prices[i - WINDOW : i]
        rets = [window[k] - window[k - 1] for k in range(1, len(window))]
        if _std(rets) < MIN_STD:
            continue
        fwd = prices[i + HORIZON] - prices[i]
        out.append(Sample(rets, window_features(window), 1 if fwd > 0 else 0, fwd))
    return out


if __name__ == "__main__":
    # Self-check on a synthetic up-drifting ramp: later prices are higher, so
    # most labels should be 1 and features should be finite.
    ramp = [0.3 + 0.01 * i + 0.002 * math.sin(i) for i in range(60)]
    X, y = build_dataset([ramp])
    assert X and all(len(row) == len(FEATURE_NAMES) for row in X)
    assert all(all(math.isfinite(v) for v in row) for row in X)
    # On a rising ramp RSI should read positive (more gains than losses).
    rsi_i = FEATURE_NAMES.index("rsi")
    assert _mean([row[rsi_i] for row in X]) > 0
    print(f"features per row: {len(FEATURE_NAMES)}  samples: {len(X)}  up-rate: {_mean([float(v) for v in y]):.2f}")
