"""Order-flow feature extraction — the *true aggressor* signal layer.

The `SII-WANGZJ/Polymarket_data` dataset carries a real `taker_direction` per
trade: the side that crossed the spread (BUY = aggressive buyer, SELL =
aggressive seller). Prior repo models could only *proxy* order-flow from a
candle's close-vs-open sign (see `features_ohlcv.signed_flow`). With the true
aggressor side we can measure genuine microstructure toxicity — the pressure and
informed-trading signals that a close-based proxy blurs.

Buckets are pre-aggregated per time step and carry:
  open/high/low/close  — price (probabilities in 0..1)
  volume               — total USD traded in the bucket (buy + sell)
  trades               — trade count
  signed_flow          — net signed USD (BUY = +usd, SELL = -usd), so
                         |signed_flow| <= volume and buy_usd = (volume+flow)/2.

Emits `features.Sample` rows (return sequence + feature vector + label + forward
return) exactly like `features_ohlcv.py`, so it drops straight into the existing
`train_seq.py` machinery. Pure stdlib + `math` — trivially runnable, no GPU.
"""

from __future__ import annotations

import math
from typing import Dict, List

from features import HORIZON, MIN_STD, WINDOW, Sample, _autocorr, _mean, _std

FLOW_FEATURES = [
    # true aggressor order flow (taker_direction driven)
    "signed_imbalance",   # net signed flow / total flow, -1..1
    "vpin",               # volume-synced prob. of informed trading proxy, 0..1
    "toxicity",           # recency-weighted |per-bucket imbalance|, 0..1
    "trade_intensity",    # trades in the last bucket, z-scored over the window
    "flow_autocorr",      # lag-1 autocorrelation of the signed-flow series, -1..1
    "whale_ratio",        # share of window volume from the largest buckets, 0..1
    "buy_pressure",       # fraction of volume that was buy-aggressor, 0..1
    # price context so the vector stands alone
    "last",               # current probability
    "mean_ret",           # average close increment over the window
    "vol",                # std of close increments (realised vol proxy)
    "momentum",           # mean of the last 4 close increments
]

EPS = 1e-9


def _signed_imbalance(flows: List[float], vols: List[float]) -> float:
    """Net signed flow over total flow, clipped to -1..1.

    Since |signed_flow_i| <= volume_i this is already in [-1, 1]; the clip only
    guards floating-point overshoot."""
    total = sum(vols)
    if total < EPS:
        return 0.0
    return max(-1.0, min(1.0, sum(flows) / total))


def _vpin(flows: List[float], vols: List[float]) -> float:
    """VPIN proxy: mean per-bucket order-flow imbalance |buy-sell|/bucketVolume.

    Easley et al.'s volume-synchronised probability of informed trading. Each
    bucket is one volume clock tick; the fraction 0..1 rises when trade flow is
    persistently one-sided (informed) rather than balanced (uninformed)."""
    ratios = [abs(f) / v for f, v in zip(flows, vols) if v > EPS]
    if not ratios:
        return 0.0
    return max(0.0, min(1.0, _mean(ratios)))


def _toxicity(flows: List[float], vols: List[float]) -> float:
    """Recency-weighted mean of per-bucket |imbalance| (linear ramp weights).

    Like VPIN but leans on the most recent buckets — the toxicity a market maker
    feels *now* — so a fresh burst of one-sided flow shows even if the window
    was calm earlier. Bounded 0..1."""
    ratios = [abs(f) / v if v > EPS else 0.0 for f, v in zip(flows, vols)]
    if not ratios:
        return 0.0
    weights = [i + 1 for i in range(len(ratios))]  # newest bucket weighted most
    wsum = sum(weights)
    if wsum <= 0:
        return 0.0
    return max(0.0, min(1.0, sum(r * w for r, w in zip(ratios, weights)) / wsum))


def _whale_ratio(vols: List[float]) -> float:
    """Share of total window volume concentrated in the largest buckets.

    True per-trade sizes are aggregated away in the buckets, so we proxy whale
    activity by volume concentration: the fraction of volume in the top ~20% of
    buckets by size. ~0.2 when volume is uniform, → 1 when one bucket dominates.
    Bounded 0..1."""
    total = sum(vols)
    if total < EPS or not vols:
        return 0.0
    k = max(1, len(vols) // 5)
    top = sorted(vols, reverse=True)[:k]
    return max(0.0, min(1.0, sum(top) / total))


def _buy_pressure(flows: List[float], vols: List[float]) -> float:
    """Fraction of window volume that was buy-aggressor, 0..1.

    buy_usd = (volume + signed_flow) / 2, so aggregated buy pressure is
    (sum(vol) + sum(flow)) / (2·sum(vol)) = (1 + signed_imbalance) / 2. Balanced
    flow reads 0.5; all-buy reads 1.0, all-sell 0.0."""
    total = sum(vols)
    if total < EPS:
        return 0.5
    return max(0.0, min(1.0, (total + sum(flows)) / (2.0 * total)))


def window_features_flow(window: List[Dict]) -> List[float]:
    """Feature vector for one look-back window of bucket dicts (length WINDOW).

    Each bucket dict carries open/high/low/close/volume/trades/signed_flow.
    Returns one value per name in `FLOW_FEATURES`, in that order."""
    close = [b["close"] for b in window]
    vols = [max(0.0, float(b["volume"])) for b in window]
    flows = [float(b["signed_flow"]) for b in window]
    trades = [float(b["trades"]) for b in window]

    rets = [close[i] - close[i - 1] for i in range(1, len(close))]
    momentum = _mean(rets[-4:]) if len(rets) >= 4 else _mean(rets)

    # trade intensity: last bucket's trade count z-scored over the window.
    mean_n = _mean(trades)
    trade_intensity = (trades[-1] - mean_n) / (_std(trades) + EPS) if len(trades) >= 2 else 0.0

    return [
        _signed_imbalance(flows, vols),
        _vpin(flows, vols),
        _toxicity(flows, vols),
        trade_intensity,
        _autocorr(flows),
        _whale_ratio(vols),
        _buy_pressure(flows, vols),
        close[-1],
        _mean(rets),
        _std(rets),
        momentum,
    ]


def market_to_rich(m: dict) -> List[Sample]:
    """Slide over one market's bucketed series → Sample rows for train_seq."""
    buckets: List[Dict] = m["buckets"]
    close = [b["close"] for b in buckets]
    out: List[Sample] = []
    N = len(buckets)
    for i in range(WINDOW, N - HORIZON):
        win = buckets[i - WINDOW : i]
        cw = close[i - WINDOW : i]
        rets = [cw[k] - cw[k - 1] for k in range(1, len(cw))]
        if _std(rets) < MIN_STD:
            continue
        fwd = close[i + HORIZON] - close[i]
        feat = window_features_flow(win)
        out.append(Sample(rets, feat, 1 if fwd > 0 else 0, fwd))
    return out


def build(series: List[dict]) -> List[List[Sample]]:
    """Per-market Sample lists (kept grouped so the temporal split stays honest)."""
    return [market_to_rich(m) for m in series]


def _synth_market(n: int = 80, seed: int = 7) -> dict:
    """Deterministic synthetic bucketed market for the self-check.

    Up-drifting price with buy-leaning aggressor flow whose sign tracks the tick
    return, so signed_imbalance/buy_pressure land firmly on the buy side."""
    buckets: List[Dict] = []
    price = 0.35
    state = seed
    for i in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        wobble = ((state / 0x7FFFFFFF) - 0.5) * 0.01
        prev = price
        price = min(0.97, max(0.03, price + 0.006 + wobble))
        ret = price - prev
        o, c = prev, price
        h = max(o, c) + 0.004
        l = min(o, c) - 0.004
        vol = 800.0 + (state % 500) + (2500.0 if i % 17 == 0 else 0.0)  # occasional whale
        flow = (0.6 if ret >= 0 else -0.2) * vol                        # buy-leaning
        buckets.append({
            "open": o, "high": h, "low": l, "close": c,
            "volume": vol, "trades": 3 + (state % 7), "signed_flow": flow,
        })
    return {"buckets": buckets, "platform": 1}


if __name__ == "__main__":
    series = [_synth_market(80, 7), _synth_market(90, 19)]
    groups = build(series)
    n = sum(len(g) for g in groups)
    assert n > 0, "no windows built"
    ups = [s.label for g in groups for s in g]
    bad = sum(1 for g in groups for s in g if not all(math.isfinite(x) for x in s.feat))
    assert bad == 0, f"{bad} non-finite feature rows"
    assert all(len(s.feat) == len(FLOW_FEATURES) for g in groups for s in g)
    # A buy-leaning ramp must read positive net imbalance on average.
    si = FLOW_FEATURES.index("signed_imbalance")
    assert _mean([s.feat[si] for g in groups for s in g]) > 0
    print(
        f"features/row: {len(FLOW_FEATURES)}  markets: {len(series)}  "
        f"windows: {n}  up-rate: {_mean([float(x) for x in ups]):.3f}  non-finite: {bad}"
    )
