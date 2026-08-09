"""Smart-money / behavioral features from the user-level tape.

The HF dataset `SII-WANGZJ/Polymarket_data` ships a `users.parquet` — the raw
*who-traded-what* tape the terminal has never modelled before. Every row is one
fill:

    timestamp     uint64 seconds
    market_id     market identifier
    condition_id  on-chain condition (resolution key)
    event_id      parent event
    user          wallet address (the thing that makes this "smart money")
    role          "maker" | "taker"
    price         YES-normalised fill price in 0..1
    usd_amount    notional of the fill (magnitude, USD)
    token_amount  SIGNED share count: +buy / -sell

That signed token flow plus a per-market **final** (the resolution, or the last
observed price as a fallback) is all we need to (a) score every wallet by a
mark-to-final PnL and crown a "smart" cohort, and (b) turn each market's tape
into behavioral features — who is on which side, how concentrated the flow is,
whether the smart cohort is leaning, and how fast new wallets are piling in.

PnL proxy (assumptions, documented):
  A wallet's mark-to-final PnL over a market is
      Σ  token_amount · (final − price)
  summed over its fills. Buying (+tokens) at price p and marking to final F earns
  (F − p) per token; selling (−tokens) earns the negative of that. This is the
  net-position value minus net cost — realised + marked-unrealised in one line,
  no per-fill inventory matching needed. Wallet scores sum across all markets.

Everything is pure stdlib (`math`, `collections`) and emits `features.Sample`
rows (return sequence + feature vector + label + forward return), so it drops
straight into the same `train_seq.py` / temporal-split machinery as the OHLCV
and close-only feature layers.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Set

from features import HORIZON, MIN_STD, WINDOW, Sample, _mean, _std

EPS = 1e-9
SMART_TOP_FRAC = 0.2  # top 20% of wallets by PnL are the "smart" cohort

SMART_FEATURES = [
    # smart-cohort leaning
    "smart_net_flow",      # net signed USD from smart wallets / gross   ∈ [-1, 1]
    "smart_share",         # smart gross volume / total gross volume      ∈ [0, 1]
    "smart_minus_dumb",    # (smart_net − rest_net) / gross               ∈ [-2, 2]
    # crowd behaviour
    "herding",             # |net flow| / gross  (one-sidedness)          ∈ [0, 1]
    "flow_imbalance",      # net signed USD / gross (signed herding)      ∈ [-1, 1]
    "crowding",            # first-time-wallet entry rate in the window   ∈ [0, 1]
    # microstructure / concentration
    "maker_taker_ratio",   # maker volume / (maker + taker volume)        ∈ [0, 1]
    "whale_concentration", # largest single-wallet share of gross volume  ∈ [0, 1]
    # price context (favorite-longshot)
    "last",                # current YES price                            ∈ [0, 1]
    "extremeness",         # |last − 0.5| · 2                             ∈ [0, 1]
]


# ── wallet scoring / cohort tagging ──────────────────────────────────────────

def _signed_usd(usd: float, token: float) -> float:
    """USD notional signed by the direction of `token_amount` (+buy / −sell)."""
    if token > 0:
        return abs(usd)
    if token < 0:
        return -abs(usd)
    return 0.0


def _market_final(m: dict) -> float:
    """Resolution / mark price for a market: explicit `final`, else last price."""
    if m.get("final") is not None:
        return float(m["final"])
    prices = m["price"]
    return float(prices[-1]) if prices else 0.5


def wallet_pnl(m: dict) -> Dict[str, float]:
    """Mark-to-final PnL per wallet for one market: Σ token·(final − price)."""
    final = _market_final(m)
    pnl: Dict[str, float] = defaultdict(float)
    for user, price, token in zip(m["user"], m["price"], m["token_amount"]):
        pnl[user] += token * (final - price)
    return dict(pnl)


def build_leaderboard(series: Sequence[dict]) -> Dict[str, float]:
    """Aggregate mark-to-final PnL per wallet across every market in `series`."""
    board: Dict[str, float] = defaultdict(float)
    for m in series:
        for user, p in wallet_pnl(m).items():
            board[user] += p
    return dict(board)


def tag_smart(leaderboard: Dict[str, float], top_frac: float = SMART_TOP_FRAC) -> Set[str]:
    """Top-cohort wallets by PnL. Deterministic: sort by (−pnl, wallet) and take
    the top ceil(top_frac · N). Wallet address is the tiebreak so ties never make
    the cohort order depend on dict insertion."""
    if not leaderboard:
        return set()
    top_frac = max(0.0, min(1.0, top_frac))
    ranked = sorted(leaderboard.items(), key=lambda kv: (-kv[1], kv[0]))
    k = max(1, math.ceil(top_frac * len(ranked))) if top_frac > 0 else 0
    return {w for w, _ in ranked[:k]}


# ── per-window behavioral features ───────────────────────────────────────────

def window_features_smart(
    users: Sequence[str],
    roles: Sequence[str],
    prices: Sequence[float],
    usds: Sequence[float],
    tokens: Sequence[float],
    smart_set: Set[str],
    seen_before: Set[str],
) -> List[float]:
    """Feature vector for one window of the user tape (all lists length WINDOW).

    `smart_set` is the globally-tagged smart cohort; `seen_before` is the set of
    wallets that traded this market strictly *before* the window (for crowding).
    """
    net = smart_net = smart_gross = dumb_net = 0.0
    maker_vol = taker_vol = 0.0
    per_wallet_gross: Dict[str, float] = defaultdict(float)
    window_wallets: Set[str] = set()

    for user, role, usd, token in zip(users, roles, usds, tokens):
        g = abs(usd)
        s = _signed_usd(usd, token)
        net += s
        per_wallet_gross[user] += g
        window_wallets.add(user)
        if user in smart_set:
            smart_net += s
            smart_gross += g
        else:
            dumb_net += s
        if role == "maker":
            maker_vol += g
        else:  # taker (default for any non-maker label)
            taker_vol += g

    gross = sum(per_wallet_gross.values()) + EPS

    new_wallets = sum(1 for w in window_wallets if w not in seen_before)
    crowding = new_wallets / len(window_wallets) if window_wallets else 0.0
    whale = (max(per_wallet_gross.values()) / gross) if per_wallet_gross else 0.0

    last = prices[-1]
    return [
        smart_net / gross,                       # smart_net_flow
        smart_gross / gross,                     # smart_share
        (smart_net - dumb_net) / gross,          # smart_minus_dumb
        abs(net) / gross,                        # herding
        net / gross,                             # flow_imbalance
        crowding,                                # crowding
        maker_vol / (maker_vol + taker_vol + EPS),  # maker_taker_ratio
        whale,                                   # whale_concentration
        last,                                    # last
        abs(last - 0.5) * 2.0,                   # extremeness
    ]


def market_to_smart(m: dict, smart_set: Set[str]) -> List[Sample]:
    """Slide over one market's (time-ordered) tape → Sample rows for train_seq."""
    idx = sorted(range(len(m["timestamp"])), key=lambda i: (m["timestamp"][i], i))
    users = [m["user"][i] for i in idx]
    roles = [m["role"][i] for i in idx]
    prices = [float(m["price"][i]) for i in idx]
    usds = [float(m["usd_amount"][i]) for i in idx]
    tokens = [float(m["token_amount"][i]) for i in idx]

    out: List[Sample] = []
    N = len(prices)
    for i in range(WINDOW, N - HORIZON):
        sl = slice(i - WINDOW, i)
        pw = prices[sl]
        rets = [pw[k] - pw[k - 1] for k in range(1, len(pw))]
        if _std(rets) < MIN_STD:
            continue  # settled/flat price window — nothing directional to learn
        seen_before = set(users[: i - WINDOW])
        feat = window_features_smart(
            users[sl], roles[sl], pw, usds[sl], tokens[sl], smart_set, seen_before
        )
        fwd = prices[i + HORIZON] - prices[i]
        out.append(Sample(rets, feat, 1 if fwd > 0 else 0, fwd))
    return out


def build(series: List[dict], top_frac: float = SMART_TOP_FRAC) -> List[List[Sample]]:
    """Per-market Sample lists. The smart cohort is tagged *once* from the global
    leaderboard, then applied to every market (kept grouped so the temporal split
    in train_seq stays honest).

    Leakage caveat: the leaderboard here scores wallets using each market's own
    `final`, so a cohort tag can carry information from a market's resolution into
    that same market's feature rows. For an honest backtest, tag the cohort from a
    *past* slice of markets and pass those wallets (or a pre-tagged `smart` set)
    forward — see how `tag_smart` is decoupled from `market_to_smart` for exactly
    this. The default path is fine for the in-sample selfcheck and cohort study."""
    smart = tag_smart(build_leaderboard(series), top_frac)
    return [market_to_smart(m, smart) for m in series]


# ── synthetic tape (shared by __main__ selfcheck and test_smart.py) ──────────

def synthesize(n_wallets: int = 40, n_markets: int = 6, trades_per_market: int = 90,
               n_skilled: int = 8, seed: int = 7) -> List[dict]:
    """Build a deterministic synthetic user tape with a skilled sub-population.

    Skilled wallets (`w0..w{n_skilled-1}`) trade *with* each market's eventual
    resolution — they buy YES when it resolves 1, sell when it resolves 0 — so a
    mark-to-final leaderboard should surface them as the smart cohort. The rest
    trade noise. Each market's price drifts toward its resolution.
    """
    import random

    rng = random.Random(seed)
    skilled = {f"w{i}" for i in range(n_skilled)}
    series: List[dict] = []
    t0 = 1_700_000_000

    for mkt in range(n_markets):
        final = mkt % 2  # alternate resolutions 0/1
        ts, user, role, price, usd, token = [], [], [], [], [], []
        p = 0.5
        for k in range(trades_per_market):
            # price drifts toward the resolution with noise, clamped off the rails
            p += (final - p) * 0.03 + rng.uniform(-0.02, 0.02)
            p = max(0.02, min(0.98, p))
            w = f"w{rng.randrange(n_wallets)}"
            if w in skilled:
                # informed: lean toward the resolution
                sign = 1.0 if final == 1 else -1.0
            else:
                sign = 1.0 if rng.random() < 0.5 else -1.0
            amt = rng.uniform(50, 500)
            ts.append(t0 + mkt * 1_000_000 + k * 60)
            user.append(w)
            role.append("maker" if rng.random() < 0.4 else "taker")
            price.append(round(p, 4))
            usd.append(round(amt, 2))
            token.append(round(sign * amt / max(p, 0.02), 2))
        series.append({
            "market_id": f"m{mkt}",
            "condition_id": f"c{mkt}",
            "event_id": f"e{mkt}",
            "final": float(final),
            "timestamp": ts, "user": user, "role": role,
            "price": price, "usd_amount": usd, "token_amount": token,
        })
    return series


if __name__ == "__main__":
    series = synthesize()

    board = build_leaderboard(series)
    smart = tag_smart(board)
    skilled = {f"w{i}" for i in range(8)}
    hit = len(smart & skilled)
    print(f"wallets: {len(board)}  smart cohort: {len(smart)}  "
          f"skilled captured: {hit}/{len(skilled)}")

    groups = build(series)
    n = sum(len(g) for g in groups)
    labels = [float(s.label) for g in groups for s in g]
    print(f"features/row: {len(SMART_FEATURES)}  markets: {len(series)}  "
          f"windows: {n}  up-rate: {_mean(labels):.3f}")

    # finiteness + bound checks across every emitted row
    bad = 0
    for g in groups:
        for s in g:
            v = dict(zip(SMART_FEATURES, s.feat))
            if not all(math.isfinite(x) for x in s.feat):
                bad += 1
            elif not (0.0 <= v["smart_share"] <= 1.0 and 0.0 <= v["herding"] <= 1.0):
                bad += 1
    print(f"non-finite / out-of-bound rows: {bad}")
    assert bad == 0
    assert n > 0
