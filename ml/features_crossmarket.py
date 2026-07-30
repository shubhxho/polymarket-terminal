"""Cross-market / neg-risk basket features — the structure single-market models miss.

Every other feature module in this suite (`features.py`, `features_ohlcv.py`,
`features_resolve.py`) looks at **one** market in isolation. But Polymarket's
`markets.parquet` links markets that belong to the same real-world question: they
share an `event_id`, and events carry a `neg_risk` flag. In a *neg-risk* event
the outcome markets are mutually exclusive and collectively exhaustive — exactly
one resolves YES — so their YES prices should sum to ~1. When the sum drifts away
from 1 the basket is mispriced, and that deviation is a tradeable arb signal that
a per-market model literally cannot see, because it only ever looks at one leg.

This module computes, for each market, features derived from its **sibling**
markets in the same event:

- **basket_sum** — sum of every leg's YES price in the event (the basket total).
- **basket_residual** — `basket_sum - 1`, the raw arb signal (0 ⇒ fairly priced).
- **share_of_basket** — this leg's price / basket_sum, its normalised weight in 0..1.
- **sibling_count** — how many *other* markets share the event.
- **rank_in_basket** — this leg's price rank among the legs, 0 (favourite) .. 1
  (longshot), normalised.
- **favorite_gap** — this leg's price minus the top *sibling* price. Positive only
  for the favourite (its lead over second place); negative for everyone else.
- **dispersion** — stdev of the legs' prices (how lopsided the field is).
- **is_neg_risk** — the event's neg-risk flag (1.0 / 0.0).

Only a genuine neg-risk basket (`neg_risk` set **and** ≥2 legs) gets the full
cross-market treatment. For non-neg-risk events (where "prices sum to 1" is not a
law) and singleton events (no siblings) the basket features **degrade
gracefully**: `basket_residual = 0`, `share_of_basket = 1`, and the market is
treated as its own whole basket. `sibling_count` and `is_neg_risk` stay factual
so the model can still tell the regimes apart.

Everything is pure stdlib (`math`, `ast`, `collections`) — no numpy / pandas — so
it stays trivially runnable and testable without a GPU, and every feature is
bounded or finite regardless of the raw inputs.

## Label semantics
The label comes from `markets.parquet`'s `outcome_prices`. **Note the format:** in
this dataset it is a *Python-list repr string* like `"['1', '0']"` (single
quotes), not JSON — so it must be parsed with `ast.literal_eval`, **not**
`json.loads` (which rejects single-quoted strings). `ast.literal_eval` also
happily parses the double-quoted variant `'["0.99","0.01"]'`, so it covers both.
The first element is token1 / YES, the second token2 / NO:

    "['1', '0']"        → YES won  → label = 1
    '["0.99","0.01"]'   → YES won  → label = 1
    "['0.02','0.98']"   → NO won   → label = 0

i.e. label = 1 iff the first (YES) outcome price finished at least as high as the
second.
"""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional

CROSSMARKET_FEATURES = [
    "basket_sum",       # Σ sibling YES prices in the event   (≈1 for a fair neg-risk basket)
    "basket_residual",  # basket_sum − 1                      (the arb signal; 0 ⇒ fair)
    "share_of_basket",  # this leg's price / basket_sum       (normalised weight, 0..1)
    "sibling_count",    # number of *other* markets in the event
    "rank_in_basket",   # price rank among legs, 0 favourite .. 1 longshot (normalised)
    "favorite_gap",     # price − top sibling price           (+ only for the favourite)
    "dispersion",       # stdev of the legs' prices           (how lopsided the field is)
    "is_neg_risk",      # event neg-risk flag                 (1.0 / 0.0)
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


def _price(market: dict) -> float:
    """Current YES (token1) price of a market, clipped to 0..1. Missing → 0.5."""
    try:
        p = float(market.get("price", 0.5))
    except (TypeError, ValueError):
        p = 0.5
    return _clip(p, 0.0, 1.0)


def _is_neg_risk(market: dict) -> bool:
    """Coerce the event's `neg_risk` flag (bool / int / 'true' / '1') to a bool."""
    v = market.get("neg_risk", False)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y")
    return bool(v)


def event_key(market: dict):
    """The market's event grouping key. Falls back to a per-market id (so a market
    with no `event_id` becomes its own singleton event) rather than colliding on
    ``None``."""
    eid = market.get("event_id")
    if eid is not None:
        return ("event", eid)
    mid = market.get("market_id", market.get("id"))
    return ("solo", id(market) if mid is None else mid)


# ── label parsing ─────────────────────────────────────────────────────────────

def parse_outcome_prices(outcome_prices) -> Optional[int]:
    """Resolution label from an `outcome_prices` list-repr string (or list).

    Uses ``ast.literal_eval`` — the dataset stores this as a *Python* list repr
    (`"['1', '0']"`, single quotes) which ``json.loads`` cannot parse. Returns 1
    if YES (token1, first outcome) finished ≥ NO (token2, second), else 0; or
    ``None`` if the value can't be parsed into ≥1 numeric outcome.
    """
    if outcome_prices is None:
        return None
    if isinstance(outcome_prices, (list, tuple)):
        seq = outcome_prices
    else:
        try:
            seq = ast.literal_eval(str(outcome_prices).strip())
        except (ValueError, SyntaxError):
            return None
    if not isinstance(seq, (list, tuple)) or not seq:
        return None
    try:
        yes = float(seq[0])
        no = float(seq[1]) if len(seq) > 1 else 1.0 - yes
    except (ValueError, TypeError):
        return None
    return 1 if yes >= no else 0


def label_from_market(market: dict) -> Optional[int]:
    """Resolution label for a market, or ``None`` if absent.

    Prefers an explicit integer ``label``; otherwise parses ``outcome_prices``.
    """
    if market.get("label") is not None:
        try:
            return int(market["label"])
        except (ValueError, TypeError):
            return None
    return parse_outcome_prices(market.get("outcome_prices"))


# ── features ──────────────────────────────────────────────────────────────────

def market_features(market: dict, event_markets: Iterable[dict]) -> List[float]:
    """Feature vector (one value per `CROSSMARKET_FEATURES`) for one market given
    every market in its event (``event_markets`` includes ``market`` itself).

    A genuine neg-risk basket (`neg_risk` set **and** ≥2 legs) gets the full
    cross-market features; anything else degrades gracefully (see module docstring).
    The vector is always finite and correctly sized.
    """
    legs = list(event_markets)
    price = _price(market)
    prices = [_price(m) for m in legs]
    sibling_prices = [_price(m) for m in legs if m is not market]
    sibling_count = float(len(legs) - 1)
    neg_risk = _is_neg_risk(market)
    has_basket = neg_risk and len(legs) >= 2

    if has_basket:
        basket_sum = sum(prices)
        basket_residual = basket_sum - 1.0
        share_of_basket = _clip(price / basket_sum, 0.0, 1.0) if basket_sum > 1e-12 else 1.0
        # rank: 0 for the favourite (nothing priced higher) .. 1 for the longshot.
        n_higher = sum(1 for p in prices if p > price + 1e-12)
        rank_in_basket = n_higher / (len(legs) - 1)
        favorite_gap = price - max(sibling_prices) if sibling_prices else 0.0
        dispersion = _std(prices)
    else:
        # Non-neg-risk or singleton: no "sum to 1" law → no basket arb structure.
        basket_sum = price
        basket_residual = 0.0
        share_of_basket = 1.0
        rank_in_basket = 0.0
        favorite_gap = 0.0
        dispersion = 0.0

    return [
        basket_sum,
        basket_residual,
        share_of_basket,
        sibling_count,
        rank_in_basket,
        favorite_gap,
        dispersion,
        1.0 if neg_risk else 0.0,
    ]


@dataclass
class Sample:
    """One labeled cross-market row: the feature vector and the market's terminal
    outcome label (1 = YES / token1 won, 0 = NO / token2 won)."""

    feat: List[float]
    label: int


def build(markets: Iterable[dict]) -> List[Sample]:
    """Group markets by `event_id`, compute each market's cross-market features
    from its event siblings, and emit labeled `Sample` rows.

    Markets without a resolution label (explicit ``label`` or a parseable
    ``outcome_prices``) are skipped, so a mixed stream of resolved and still-open
    markets yields only the usable, labeled rows.
    """
    events: dict = defaultdict(list)
    for m in markets:
        events[event_key(m)].append(m)

    out: List[Sample] = []
    for legs in events.values():
        for m in legs:
            label = label_from_market(m)
            if label is None:
                continue
            out.append(Sample(market_features(m, legs), label))
    return out


# ── selfcheck ─────────────────────────────────────────────────────────────────

def _synthetic_event(event_id: int, n_legs: int, neg_risk: bool, imbalance: float = 0.0):
    """A synthetic multi-market event for the selfcheck.

    Builds ``n_legs`` legs whose YES prices sum to ``1 + imbalance`` (so a
    neg-risk basket with ``imbalance == 0`` is fair). The favourite (leg 0)
    resolves YES; the rest resolve NO — a plausible mutually-exclusive outcome.
    """
    # Descending weights → the first leg is the favourite.
    raw = [1.0 / (k + 1.5) for k in range(n_legs)]
    total = sum(raw)
    prices = [(1.0 + imbalance) * r / total for r in raw]
    legs = []
    for k, p in enumerate(prices):
        won = k == 0                       # exactly one leg wins the basket
        legs.append({
            "market_id": f"{event_id}-{k}",
            "event_id": event_id,
            "neg_risk": neg_risk,
            "price": max(0.001, min(0.999, p)),
            "outcome_prices": "['1', '0']" if won else "['0', '1']",
        })
    return legs


if __name__ == "__main__":
    markets: List[dict] = []
    # A mix: balanced neg-risk baskets, imbalanced ones, non-neg-risk events, and
    # singletons — the full range of regimes the feature layer has to handle.
    for e in range(12):
        markets += _synthetic_event(e, n_legs=3 + (e % 3), neg_risk=True,
                                    imbalance=0.0 if e % 2 == 0 else 0.15)
    for e in range(12, 16):
        markets += _synthetic_event(e, n_legs=2, neg_risk=False)     # non-neg-risk pair
    for e in range(16, 20):
        markets += _synthetic_event(e, n_legs=1, neg_risk=True)      # singleton

    samples = build(markets)
    assert samples, "build produced no samples"
    assert all(len(s.feat) == len(CROSSMARKET_FEATURES) for s in samples), "feature-vector length mismatch"
    assert all(all(math.isfinite(x) for x in s.feat) for s in samples), "non-finite feature"
    assert all(s.label in (0, 1) for s in samples), "label not in {0, 1}"

    idx = {name: i for i, name in enumerate(CROSSMARKET_FEATURES)}
    for s in samples:
        assert 0.0 <= s.feat[idx["share_of_basket"]] <= 1.0
        assert 0.0 <= s.feat[idx["rank_in_basket"]] <= 1.0

    # A balanced neg-risk basket residual ≈ 0; an imbalanced one is clearly off.
    balanced = market_features(_synthetic_event(0, 3, neg_risk=True, imbalance=0.0)[0],
                               _synthetic_event(0, 3, neg_risk=True, imbalance=0.0))
    imbalanced = market_features(_synthetic_event(0, 3, neg_risk=True, imbalance=0.25)[0],
                                 _synthetic_event(0, 3, neg_risk=True, imbalance=0.25))
    assert abs(balanced[idx["basket_residual"]]) < 1e-9, balanced[idx["basket_residual"]]
    assert abs(imbalanced[idx["basket_residual"]] - 0.25) < 1e-9, imbalanced[idx["basket_residual"]]

    yes_rate = _mean([float(s.label) for s in samples])
    print(f"features/row: {len(CROSSMARKET_FEATURES)}  markets: {len(markets)}  samples: {len(samples)}  yes-rate: {yes_rate:.3f}")
    print("selfcheck ok")
