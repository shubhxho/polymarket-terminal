"""Microstructure feature extraction — the finest-grain signal layer.

The `SII-WANGZJ/Polymarket_data` dataset ships an `orderfilled.parquet` with
~689M raw **OrderFilled** events — every individual maker/taker fill, the finest
view of the tape and the one prior repo models never touched. There is *no order
book* in the dataset, so classic microstructure metrics (effective spread, Kyle's
lambda, VPIN-style toxicity) have to be derived *book-free*, straight from the
sequence of fills bucketed per outcome token per time bucket.

Each OrderFilled event pairs one USDC (collateral) leg with one outcome-token leg:

  timestamp, block_number, transaction_hash, log_index, contract, order_hash,
  maker, taker, maker_asset_id, taker_asset_id,
  maker_amount_filled, taker_amount_filled,          (uint256 wei, STORED AS STR)
  maker_fee, taker_fee, protocol_fee                 (uint256 wei, STORED AS STR)

Assumptions (documented; the self-check uses synthetic fills so exact on-chain
decoding is not required to test the maths):
  * The collateral (USDC) leg is the asset whose id is in ``COLLATERAL_IDS``
    (default ``{"0"}``). The other leg is the outcome token.
  * USDC and outcome tokens both carry **6 decimals** — parse the wei string to
    an int, divide by 1e6.
  * ``effective price = usdc_amount / token_amount`` (probability in 0..1).
  * The **taker** is the aggressor (crossed the spread). If the taker *receives*
    the token (gives USDC) the fill is a BUY (direction +1); if the taker *gives*
    the token (receives USDC) it is a SELL (direction -1).

Derived book-free microstructure features, per token per bucket window:
  effective_spread   — mean |trade price - trailing rolling mid| (relative)
  realized_spread    — signed price reversion a fill later (maker's edge)
  fill_size_mean     — mean USDC notional per fill (log1p-scaled)
  fill_size_cv       — coefficient of variation of USDC fill sizes
  large_fill_ratio   — share of volume in the top-decile fills, 0..1
  maker_concentration— Herfindahl of maker-wallet volume shares, 0..1
  taker_concentration— Herfindahl of taker-wallet volume shares, 0..1
  fill_rate          — fills in the last bucket, z-scored over the window
  price_impact       — Kyle's lambda: slope of dprice on signed notional
  fee_intensity      — (protocol+taker) fees per USDC volume
plus price context (last, mean_ret, vol, momentum) so the vector stands alone.

Emits `features.Sample` rows (return sequence + feature vector + label + forward
return) exactly like `features_ohlcv.py` / `features_flow.py`, so it drops
straight into the existing `train_seq.py` machinery. Pure stdlib + `math` — no
numpy/pandas, trivially runnable and testable without a GPU.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from features import HORIZON, MIN_STD, WINDOW, Sample, _mean, _std

MICRO_FEATURES = [
    # book-free spread / reversion
    "effective_spread",     # mean |price - rolling mid| / mid, >= 0
    "realized_spread",      # signed reversion one fill later (maker edge)
    # fill-size distribution
    "fill_size_mean",       # log1p(mean USDC notional per fill)
    "fill_size_cv",         # std/mean of USDC fill sizes, >= 0
    "large_fill_ratio",     # share of volume in top-decile fills, 0..1
    # participant concentration
    "maker_concentration",  # Herfindahl of maker volume shares, 0..1
    "taker_concentration",  # Herfindahl of taker volume shares, 0..1
    # activity / impact / cost
    "fill_rate",            # last bucket's fill count, z-scored over window
    "price_impact",         # Kyle's lambda proxy (dprice vs signed notional)
    "fee_intensity",        # (protocol+taker) fee per USDC volume, >= 0
    # price context so the vector stands alone
    "last",                 # current probability
    "mean_ret",             # average bucket close increment over the window
    "vol",                  # std of bucket close increments (realised vol proxy)
    "momentum",             # mean of the last 4 bucket close increments
]

EPS = 1e-9
USDC_DECIMALS = 6           # USDC and outcome tokens both carry 6 decimals
COLLATERAL_IDS = {"0"}      # asset ids treated as the USDC / collateral leg
BUCKET_SECONDS = 3600       # default time-bucket width (1 hour)
ROLL_K = 5                  # trailing fills used for the rolling-mid proxy
IMPACT_SCALE = 1e-3         # divides notional so Kyle's lambda is O(1), not tiny


class Fill:
    """One normalised OrderFilled event for a single outcome token.

    ``direction`` is +1 for a taker BUY (taker receives the token) and -1 for a
    taker SELL. ``signed`` is the taker-signed USDC notional (direction * usdc).
    """

    __slots__ = ("ts", "price", "usdc", "token", "direction", "signed",
                 "maker", "taker", "fee", "token_id")

    def __init__(self, ts, price, usdc, token, direction, maker, taker, fee, token_id):
        self.ts = ts
        self.price = price
        self.usdc = usdc
        self.token = token
        self.direction = direction
        self.signed = direction * usdc
        self.maker = maker
        self.taker = taker
        self.fee = fee
        self.token_id = token_id


class Bucket:
    """A time bucket of fills for one token, carrying a VWAP close price."""

    __slots__ = ("key", "price", "fills")

    def __init__(self, key, price, fills):
        self.key = key
        self.price = price
        self.fills = fills


def _wei_to_units(s, decimals: int = USDC_DECIMALS) -> float:
    """Parse a uint256 wei value stored as a decimal string into token units.

    Robust to ints, whitespace, empty/None (→ 0.0). Division by ``10**decimals``
    is exact for the 6-decimal USDC/outcome-token scale used here."""
    if s is None:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s) / (10 ** decimals)
    s = str(s).strip()
    if not s:
        return 0.0
    return int(s) / (10 ** decimals)


def normalize_fill(ev: dict, collateral_ids=COLLATERAL_IDS):
    """Turn one raw OrderFilled event dict into a `Fill`, or None if unusable.

    Determines which leg is USDC from the asset ids, derives the effective price
    and the taker-signed direction, and parses every wei string to token units.
    """
    maker_id = str(ev.get("maker_asset_id", ""))
    taker_id = str(ev.get("taker_asset_id", ""))
    maker_amt = _wei_to_units(ev.get("maker_amount_filled"))
    taker_amt = _wei_to_units(ev.get("taker_amount_filled"))

    if maker_id in collateral_ids and taker_id not in collateral_ids:
        # maker gives USDC, taker gives token → taker SELLS the token.
        usdc, token, token_id, direction = maker_amt, taker_amt, taker_id, -1
    elif taker_id in collateral_ids and maker_id not in collateral_ids:
        # taker gives USDC, receives token → taker BUYS the token.
        usdc, token, token_id, direction = taker_amt, maker_amt, maker_id, +1
    else:
        # both or neither leg is collateral — not a clean USDC/token pair.
        return None

    if token <= EPS or usdc <= EPS:
        return None
    price = usdc / token
    if not math.isfinite(price) or price <= 0.0:
        return None

    fee = _wei_to_units(ev.get("protocol_fee")) + _wei_to_units(ev.get("taker_fee"))
    try:
        ts = int(ev.get("timestamp", 0))
    except (TypeError, ValueError):
        ts = 0
    return Fill(ts, price, usdc, token, direction,
                str(ev.get("maker", "")), str(ev.get("taker", "")), fee, str(token_id))


def _hhi(weights: List[float]) -> float:
    """Herfindahl-Hirschman index of a set of non-negative weights, in 0..1.

    Sum of squared shares. One participant → 1.0; N equal participants → 1/N."""
    total = sum(weights)
    if total <= EPS:
        return 0.0
    return max(0.0, min(1.0, sum((w / total) ** 2 for w in weights)))


def _cv(xs: List[float]) -> float:
    """Coefficient of variation std/mean, clamped to >= 0. 0 if mean ~ 0."""
    m = _mean(xs)
    if abs(m) < EPS:
        return 0.0
    return max(0.0, _std(xs) / m)


def _linreg_slope(xs: List[float], ys: List[float]) -> float:
    """OLS slope of y on x. 0 when x has no variance or fewer than 2 points."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var < EPS:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = cov / var
    return slope if math.isfinite(slope) else 0.0


def _effective_spread(fills: List[Fill]) -> float:
    """Mean relative gap between each fill price and a trailing rolling mid.

    Book-free half-spread proxy: with no quotes, the "mid" is the trailing mean
    of the last ``ROLL_K`` fill prices. A wide, choppy tape (fills landing far
    from the running mid) reads high; a calm one-price tape reads ~0."""
    if len(fills) < 2:
        return 0.0
    prices = [f.price for f in fills]
    gaps = []
    for i in range(1, len(prices)):
        lo = max(0, i - ROLL_K)
        mid = _mean(prices[lo:i])
        if mid > EPS:
            gaps.append(abs(prices[i] - mid) / mid)
    return _mean(gaps) if gaps else 0.0


def _realized_spread(fills: List[Fill]) -> float:
    """Signed price reversion one fill later — the market maker's realised edge.

    For each fill: direction * (price - price_next) / price. After an aggressive
    BUY that pushed price up transiently, price tends to fall back → the maker
    who sold profits → positive. Sign convention matches classic realized spread
    (aggressor loses, maker gains) up to the constant 2x quoting factor we drop."""
    if len(fills) < 2:
        return 0.0
    vals = []
    for i in range(len(fills) - 1):
        p, pn = fills[i].price, fills[i + 1].price
        if p > EPS:
            vals.append(fills[i].direction * (p - pn) / p)
    return _mean(vals) if vals else 0.0


def _large_fill_ratio(sizes: List[float]) -> float:
    """Share of total USDC volume concentrated in the top-decile fills, 0..1.

    ~0.1 when sizes are uniform, → 1 when one whale fill dominates."""
    total = sum(sizes)
    if total <= EPS or not sizes:
        return 0.0
    k = max(1, len(sizes) // 10)
    top = sorted(sizes, reverse=True)[:k]
    return max(0.0, min(1.0, sum(top) / total))


def window_features_micro(buckets: List[Bucket]) -> List[float]:
    """Feature vector for one look-back window of `Bucket`s (length WINDOW).

    Bucket-level price context comes from the VWAP close series; the
    microstructure features are computed over every fill in the window, in
    time order. Returns one value per name in `MICRO_FEATURES`, in that order."""
    close = [b.price for b in buckets]
    rets = [close[i] - close[i - 1] for i in range(1, len(close))]
    momentum = _mean(rets[-4:]) if len(rets) >= 4 else _mean(rets)

    fills = [f for b in buckets for f in b.fills]
    sizes = [f.usdc for f in fills]

    # participant concentration: Herfindahl of per-wallet USDC volume.
    maker_vol: Dict[str, float] = defaultdict(float)
    taker_vol: Dict[str, float] = defaultdict(float)
    for f in fills:
        maker_vol[f.maker] += f.usdc
        taker_vol[f.taker] += f.usdc

    # fill rate: last bucket's fill count z-scored over the window's counts.
    counts = [float(len(b.fills)) for b in buckets]
    mean_c = _mean(counts)
    fill_rate = (counts[-1] - mean_c) / (_std(counts) + EPS) if len(counts) >= 2 else 0.0

    # Kyle's lambda: regress fill-to-fill price change on signed notional.
    dprice, signed = [], []
    for i in range(1, len(fills)):
        dprice.append(fills[i].price - fills[i - 1].price)
        signed.append(fills[i].signed / IMPACT_SCALE)
    price_impact = _linreg_slope(signed, dprice)

    total_vol = sum(sizes)
    fee_intensity = (sum(f.fee for f in fills) / total_vol) if total_vol > EPS else 0.0

    return [
        _effective_spread(fills),
        _realized_spread(fills),
        math.log1p(_mean(sizes)),
        _cv(sizes),
        _large_fill_ratio(sizes),
        _hhi(list(maker_vol.values())),
        _hhi(list(taker_vol.values())),
        fill_rate,
        price_impact,
        fee_intensity,
        close[-1],
        _mean(rets),
        _std(rets),
        momentum,
    ]


def _bucketize(fills: List[Fill], bucket_seconds: int = BUCKET_SECONDS) -> List[Bucket]:
    """Group time-sorted fills into VWAP-priced time buckets, ordered by time."""
    by_key: Dict[int, List[Fill]] = defaultdict(list)
    for f in fills:
        by_key[f.ts // bucket_seconds].append(f)
    out: List[Bucket] = []
    for key in sorted(by_key):
        group = by_key[key]
        vol = sum(f.usdc for f in group)
        if vol > EPS:
            price = sum(f.price * f.usdc for f in group) / vol   # VWAP close
        else:
            price = group[-1].price
        out.append(Bucket(key, price, group))
    return out


def _buckets_to_rich(buckets: List[Bucket]) -> List[Sample]:
    """Slide a WINDOW over one token's bucket series → Sample rows."""
    close = [b.price for b in buckets]
    out: List[Sample] = []
    N = len(buckets)
    for i in range(WINDOW, N - HORIZON):
        win = buckets[i - WINDOW:i]
        cw = close[i - WINDOW:i]
        rets = [cw[k] - cw[k - 1] for k in range(1, len(cw))]
        if _std(rets) < MIN_STD:
            continue
        fwd = close[i + HORIZON] - close[i]
        feat = window_features_micro(win)
        out.append(Sample(rets, feat, 1 if fwd > 0 else 0, fwd))
    return out


def _normalize_market(m: dict) -> Dict[str, List[Fill]]:
    """Parse a market dict's raw fills and group the valid ones by token id."""
    collateral_ids = set(m.get("collateral_ids", COLLATERAL_IDS))
    by_tok: Dict[str, List[Fill]] = defaultdict(list)
    for ev in m.get("fills", []):
        f = normalize_fill(ev, collateral_ids)
        if f is not None:
            by_tok[f.token_id].append(f)
    for tok in by_tok:
        by_tok[tok].sort(key=lambda f: (f.ts,))
    return by_tok


def market_to_rich(m: dict, bucket_seconds: int = BUCKET_SECONDS) -> List[Sample]:
    """All Sample rows for one market dict (concatenated across its tokens)."""
    out: List[Sample] = []
    for fills in _normalize_market(m).values():
        out.extend(_buckets_to_rich(_bucketize(fills, bucket_seconds)))
    return out


def build(series: List[dict], bucket_seconds: int = BUCKET_SECONDS) -> List[List[Sample]]:
    """Per-token Sample lists, one group per outcome token.

    Each ``series`` element is a dict with a ``fills`` list of raw OrderFilled
    events (optionally ``collateral_ids``). Fills are bucketed *per token* so the
    temporal split downstream stays honest (no cross-token windows)."""
    groups: List[List[Sample]] = []
    for m in series:
        for fills in _normalize_market(m).values():
            samples = _buckets_to_rich(_bucketize(fills, bucket_seconds))
            if samples:
                groups.append(samples)
    return groups


def _synth_market(n_buckets: int = 80, seed: int = 7, token_id: str = "TKN") -> dict:
    """Deterministic synthetic market of raw OrderFilled events for the check.

    Buy-leaning aggressor flow with a genuine price-impact mechanism: each fill
    nudges a running price up by a lambda proportional to its signed notional, so
    the impact regression recovers a positive Kyle's lambda and the price drifts
    up (mostly up-labels). Emitted as raw wei-string events to exercise parsing.
    """
    makers = [f"0xmaker{k}" for k in range(6)]
    takers = [f"0xtaker{k}" for k in range(9)]
    events = []
    price = 0.35
    state = seed
    ts = 1_700_000_000

    def rng():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    for b in range(n_buckets):
        m_fills = 3 + int(rng() * 6)                       # 3..8 fills per bucket
        for _ in range(m_fills):
            buy = rng() < 0.7                              # buy-leaning tape
            direction = 1 if buy else -1
            base = 50.0 + rng() * 450.0
            usdc = base + (3000.0 if rng() < 0.04 else 0.0)   # occasional whale
            signed = direction * usdc
            price = min(0.97, max(0.03, price + 1e-5 * signed + (rng() - 0.5) * 5e-4))
            token = usdc / price
            fee = 0.001 * usdc
            maker = makers[int(rng() * len(makers))]
            taker = takers[int(rng() * len(takers))]
            usdc_wei = str(int(round(usdc * 10 ** USDC_DECIMALS)))
            token_wei = str(int(round(token * 10 ** USDC_DECIMALS)))
            fee_wei = str(int(round(fee * 10 ** USDC_DECIMALS)))
            if buy:   # taker gives USDC, receives token
                ev = {"maker_asset_id": token_id, "taker_asset_id": "0",
                      "maker_amount_filled": token_wei, "taker_amount_filled": usdc_wei}
            else:     # maker gives USDC, taker gives token
                ev = {"maker_asset_id": "0", "taker_asset_id": token_id,
                      "maker_amount_filled": usdc_wei, "taker_amount_filled": token_wei}
            ev.update({"timestamp": ts, "maker": maker, "taker": taker,
                       "protocol_fee": fee_wei, "taker_fee": "0", "maker_fee": "0"})
            events.append(ev)
            ts += 7
        ts = (b + 1) * BUCKET_SECONDS + 1_700_000_000       # next bucket
    return {"fills": events, "collateral_ids": ["0"]}


if __name__ == "__main__":
    series = [_synth_market(80, 7), _synth_market(90, 19)]
    groups = build(series)
    n = sum(len(g) for g in groups)
    assert n > 0, "no windows built"
    ups = [s.label for g in groups for s in g]
    bad = sum(1 for g in groups for s in g if not all(math.isfinite(x) for x in s.feat))
    assert bad == 0, f"{bad} non-finite feature rows"
    assert all(len(s.feat) == len(MICRO_FEATURES) for g in groups for s in g)
    # A buy-leaning, positive-impact tape must read positive Kyle's lambda.
    pi = MICRO_FEATURES.index("price_impact")
    assert _mean([s.feat[pi] for g in groups for s in g]) > 0
    print(
        f"features/row: {len(MICRO_FEATURES)}  tokens: {len(groups)}  "
        f"windows: {n}  up-rate: {_mean([float(x) for x in ups]):.3f}  non-finite: {bad}"
    )
