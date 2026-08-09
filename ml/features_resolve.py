"""Resolution-snapshot features — the calibrated resolution-probability layer.

Where `features.py` / `features_ohlcv.py` predict *short-horizon direction*, this
module predicts the **final YES/NO resolution** of a market from a single
*snapshot* of its state at some time `t_snapshot`. A snapshot describes "where
the market stands right now" — its price, how far into its life it is, recent
momentum / volatility, order-flow imbalance, how much of its total volume has
already traded — and the label is the market's *terminal* outcome. A model
trained on these rows learns a mapping from mid-life market state to the
probability that YES ultimately resolves true, which (once calibrated) is the
flagship resolution-probability signal.

Everything is pure stdlib (`math`, `json`, `dataclasses`) so it stays trivially
runnable and testable without numpy/pandas or a GPU. Every feature is bounded or
normalised so the vector is finite and well-scaled regardless of the raw inputs.

## Snapshot shape
A snapshot is a plain dict:

    {
        "price":        float,   # current YES (token1) price in 0..1
        "t_snapshot":   int,     # unix secs — time of this snapshot
        "t_created":    int,     # unix secs — market creation
        "t_end":        int,     # unix secs — market resolution/close
        "recent":       [float], # short recent-price window (oldest→newest)
        "volume":       float,   # cumulative volume traded so far
        "volume_total": float,   # market's final total volume
        "signed_flow":  float,   # recent signed aggressor flow (buys +, sells -)
        "flow_total":   float,   # recent total (absolute) flow
        "trade_count":  int,     # trades in the recent window
        # optional normalisation baselines for trade_count_z:
        "trade_count_mean": float,
        "trade_count_std":  float,
        # label source — either an explicit label or the raw outcome_prices:
        "label":          int,   # 1 = YES won, 0 = NO won
        "outcome_prices": str,   # e.g. '["0.99","0.01"]'  (see LABEL SEMANTICS)
    }

Missing keys fall back to safe defaults so `snapshot_features` never emits a
non-finite value.

## Label semantics
The training label comes from `markets.parquet`'s `outcome_prices`, a JSON-array
*string* of the two terminal outcome prices, ordered [token1/YES, token2/NO]:

    '["0.99","0.01"]'  → YES (answer1 / token1) won  → label = 1
    '["0.02","0.98"]'  → NO  (answer2 / token2) won  → label = 0

i.e. label = 1 iff the first (YES) outcome price finished higher than the second.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

# Clip price away from 0/1 before taking a logit so log() stays finite.
PRICE_EPS = 1e-4
# logit(1-PRICE_EPS) ≈ 9.2; divide by this so the scaled logit lands in ~[-1, 1].
LOGIT_SCALE = 9.2

RESOLVE_FEATURES = [
    "price",              # current YES price, 0..1
    "dist_from_half",     # |price - 0.5|, 0..0.5  (confidence away from a coin-flip)
    "logit_price",        # clipped log(p/(1-p)) scaled to ~-1..1
    "time_to_resolution", # fraction of market life still remaining, 0..1
    "age_fraction",       # fraction of market life already elapsed, 0..1
    "momentum",           # recent price change over the window, -1..1
    "realized_vol",       # recent realised vol (std of increments), 0..1
    "flow_imbalance",     # signed aggressor flow / total flow, -1..1
    "volume_maturity",    # cumulative volume / market total, 0..1
    "trade_count_z",      # z-scored recent trade count, clipped -5..5
    "calib_bucket",       # smooth encoding of the 0.1 price band, 0..1
    "price_x_time",       # interaction: dist_from_half * time_to_resolution
]


# ── bounded helpers ───────────────────────────────────────────────────────────

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _logit_scaled(p: float) -> float:
    """log(p/(1-p)), clipped away from the asymptotes and scaled to ~[-1, 1]."""
    p = _clip(p, PRICE_EPS, 1.0 - PRICE_EPS)
    return _clip(math.log(p / (1.0 - p)) / LOGIT_SCALE, -1.0, 1.0)


def _soft_bucket(p: float) -> float:
    """Smooth encoding of the 0.1 price decile band, in 0..1.

    A hard one-hot decile would be non-differentiable and stepped; instead we
    blend the band *centre* (which decile the price sits in) with the raw price,
    giving a value that is continuous in `p` yet still snaps toward the 0.1 grid
    a calibration curve is bucketed on. Flat within a band, monotone across it.
    """
    p = _clip(p, 0.0, 1.0)
    k = min(9, int(p * 10.0))          # decile index 0..9
    center = (k + 0.5) / 10.0          # band centre in 0..1
    return _clip(0.5 * center + 0.5 * p, 0.0, 1.0)


# ── label parsing ─────────────────────────────────────────────────────────────

def parse_outcome_prices(outcome_prices: str) -> int:
    """Resolution label from an `outcome_prices` JSON-array string.

    Returns 1 if YES (token1, the first outcome) finished strictly higher than
    NO (token2, the second), else 0. See the module LABEL SEMANTICS section.
    """
    prices = json.loads(outcome_prices)
    yes = float(prices[0])
    no = float(prices[1]) if len(prices) > 1 else 1.0 - yes
    return 1 if yes > no else 0


def label_from_snapshot(snapshot: dict) -> Optional[int]:
    """Extract the resolution label from a snapshot, or None if absent.

    Prefers an explicit integer `label`; otherwise parses `outcome_prices`.
    """
    if snapshot.get("label") is not None:
        return int(snapshot["label"])
    op = snapshot.get("outcome_prices")
    if op is not None:
        return parse_outcome_prices(op)
    return None


# ── features ──────────────────────────────────────────────────────────────────

def snapshot_features(snapshot: dict) -> List[float]:
    """Feature vector (one value per `RESOLVE_FEATURES` name) for one snapshot.

    Tolerant of missing keys — every field falls back to a neutral default so
    the returned vector is always finite and correctly ordered/sized.
    """
    price = _clip(float(snapshot.get("price", 0.5)), 0.0, 1.0)
    dist_from_half = abs(price - 0.5)
    logit_price = _logit_scaled(price)

    # Temporal position within the market's life.
    t_created = float(snapshot.get("t_created", 0.0))
    t_end = float(snapshot.get("t_end", t_created + 1.0))
    t_snapshot = float(snapshot.get("t_snapshot", t_created))
    life = t_end - t_created
    if life <= 0.0:
        time_to_resolution = 0.0
        age_fraction = 1.0
    else:
        time_to_resolution = _clip((t_end - t_snapshot) / life, 0.0, 1.0)
        age_fraction = _clip((t_snapshot - t_created) / life, 0.0, 1.0)

    # Recent-window momentum & realised vol (prices in 0..1 ⇒ diffs in -1..1).
    recent = [float(x) for x in snapshot.get("recent", [])] or [price]
    momentum = _clip(recent[-1] - recent[0], -1.0, 1.0)
    incs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    realized_vol = _clip(_std(incs), 0.0, 1.0)

    # Order-flow imbalance in -1..1.
    signed_flow = float(snapshot.get("signed_flow", 0.0))
    flow_total = float(snapshot.get("flow_total", abs(signed_flow)))
    flow_imbalance = _clip(signed_flow / flow_total, -1.0, 1.0) if flow_total > 1e-9 else 0.0

    # Volume maturity: how much of the market's lifetime volume has traded.
    vol_so_far = float(snapshot.get("volume", 0.0))
    vol_total = float(snapshot.get("volume_total", vol_so_far))
    volume_maturity = _clip(vol_so_far / vol_total, 0.0, 1.0) if vol_total > 1e-9 else 0.0

    # Recent trade-count z-score against a baseline (defaults keep it finite).
    trade_count = float(snapshot.get("trade_count", 0.0))
    tc_mean = float(snapshot.get("trade_count_mean", 50.0))
    tc_std = float(snapshot.get("trade_count_std", 50.0))
    trade_count_z = _clip((trade_count - tc_mean) / (tc_std + 1e-9), -5.0, 5.0)

    calib_bucket = _soft_bucket(price)
    price_x_time = dist_from_half * time_to_resolution

    return [
        price,
        dist_from_half,
        logit_price,
        time_to_resolution,
        age_fraction,
        momentum,
        realized_vol,
        flow_imbalance,
        volume_maturity,
        trade_count_z,
        calib_bucket,
        price_x_time,
    ]


@dataclass
class Sample:
    """One labeled resolution-prediction row: the feature vector and the market's
    terminal outcome label (1 = YES won, 0 = NO won)."""

    feat: List[float]
    label: int


def build(snapshots: Iterable[dict]) -> List[Sample]:
    """Turn labeled snapshots into `Sample` rows.

    Each input snapshot must carry a resolution label (an explicit `label` or an
    `outcome_prices` string). Snapshots without a label are skipped, so the same
    stream can mix resolved markets (usable) and still-open ones (dropped).
    """
    out: List[Sample] = []
    for snap in snapshots:
        label = label_from_snapshot(snap)
        if label is None:
            continue
        out.append(Sample(snapshot_features(snap), label))
    return out


# ── selfcheck ─────────────────────────────────────────────────────────────────

def _synthetic_snapshot(i: int, yes_won: bool) -> dict:
    """Deterministic synthetic snapshot for the selfcheck.

    A market that ends YES drifts its price up toward 1; one that ends NO drifts
    down toward 0. Snapshots are taken partway through the market's life, so the
    label is genuinely uncertain from state alone (as it is in real data).
    """
    base = 0.5 + (0.004 if yes_won else -0.004) * i + 0.01 * math.sin(i / 3.0)
    price = _clip(base, 0.02, 0.98)
    recent = [_clip(price - 0.02 + 0.005 * k + 0.003 * math.sin(k), 0.01, 0.99) for k in range(8)]
    t_created = 1_700_000_000
    t_end = t_created + 30 * 24 * 3600            # 30-day market
    t_snapshot = t_created + (i % 30) * 24 * 3600  # somewhere in its life
    return {
        "price": price,
        "t_snapshot": t_snapshot,
        "t_created": t_created,
        "t_end": t_end,
        "recent": recent,
        "volume": 1000.0 + 50.0 * i,
        "volume_total": 5000.0,
        "signed_flow": (30.0 if yes_won else -30.0) + 5.0 * math.sin(i),
        "flow_total": 120.0,
        "trade_count": 40 + (i % 25),
        "outcome_prices": '["0.99","0.01"]' if yes_won else '["0.02","0.98"]',
    }


if __name__ == "__main__":
    snaps = [_synthetic_snapshot(i, yes_won=(i % 2 == 0)) for i in range(60)]
    samples = build(snaps)

    assert samples, "build produced no samples"
    assert all(len(s.feat) == len(RESOLVE_FEATURES) for s in samples), "feature-vector length mismatch"
    assert all(all(math.isfinite(x) for x in s.feat) for s in samples), "non-finite feature"
    assert all(s.label in (0, 1) for s in samples), "label not in {0, 1}"

    # Bounded feature sanity on every row.
    pi = RESOLVE_FEATURES.index("price")
    di = RESOLVE_FEATURES.index("dist_from_half")
    fi = RESOLVE_FEATURES.index("flow_imbalance")
    vi = RESOLVE_FEATURES.index("volume_maturity")
    for s in samples:
        assert 0.0 <= s.feat[pi] <= 1.0
        assert 0.0 <= s.feat[di] <= 0.5
        assert -1.0 <= s.feat[fi] <= 1.0
        assert 0.0 <= s.feat[vi] <= 1.0

    up_rate = _mean([float(s.label) for s in samples])
    print(f"features/row: {len(RESOLVE_FEATURES)}  snapshots: {len(snaps)}  samples: {len(samples)}  yes-rate: {up_rate:.3f}")
    print("selfcheck ok")
