"""Feature extraction for the direction classifier.

Turns a single market's price series (probabilities in 0..1, roughly uniform
cadence) into supervised (features, label) rows with a sliding window. The
feature set mirrors the terminal's own quant lib — drift, realised vol, a
Bollinger-style band z, short momentum, lag-1 autocorrelation of increments —
so the model learns from the same microstructure the UI already surfaces.

Label: 1 if the price is higher HORIZON steps ahead, else 0. Pure stdlib +
math so it stays trivially runnable and testable without a GPU.
"""

from __future__ import annotations

import math
from typing import List, Tuple

WINDOW = 16          # look-back length for each sample
HORIZON = 4          # steps ahead the label looks
MIN_STD = 1e-4       # below this the window is settled/flat — skip it

FEATURE_NAMES = [
    "last",          # current probability
    "mean_ret",      # average increment over the window
    "vol",           # std of increments (realised vol proxy)
    "drift",         # net move across the window
    "band_z",        # (last - mean) / std  — how stretched
    "momentum",      # mean of the last 4 increments
    "autocorr",      # lag-1 autocorrelation of increments
    "activity",      # sum of |increments| — how much it moved at all
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


def window_features(window: List[float]) -> List[float]:
    """Feature vector for one look-back window of prices."""
    rets = [window[i] - window[i - 1] for i in range(1, len(window))]
    std = _std(rets)
    mean_w = _mean(window)
    std_w = _std(window)
    band_z = (window[-1] - mean_w) / std_w if std_w > 1e-9 else 0.0
    momentum = _mean(rets[-4:]) if len(rets) >= 4 else _mean(rets)
    return [
        window[-1],
        _mean(rets),
        std,
        window[-1] - window[0],
        band_z,
        momentum,
        _autocorr(rets),
        sum(abs(r) for r in rets),
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
    assert X and all(all(math.isfinite(v) for v in row) for row in X)
    print(f"features per row: {len(FEATURE_NAMES)}  samples: {len(X)}  up-rate: {_mean([float(v) for v in y]):.2f}")
