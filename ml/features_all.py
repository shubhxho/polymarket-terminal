"""Unified feature builder — one vector that spans all six merged feature families.

Each feature module in this suite looks at the dataset through one lens:

  * ``features_flow``        — true-aggressor order flow (bucketed taker direction)
  * ``features_resolve``     — resolution-snapshot / calibrated-probability state
  * ``features_smart``       — smart-money / behavioral user-tape features
  * ``features_crossmarket`` — neg-risk basket / cross-market arb structure
  * ``features_event``       — event-level co-movement / relative value
  * ``features_micro``       — book-free microstructure from raw fills

Every trainer so far had to pick *one* family. This module composes all six into a
single, namespaced feature space so a model can consume the full picture per
market/window. It does **not** reimplement any feature — it imports each merged
module and calls its existing public feature function, then concatenates the
results in a fixed order aligned to :data:`ALL_FEATURES`.

## Context shape
``build_all`` takes one ``context`` dict. Each family reads its own slice of it and
every slice is optional — when a family's inputs are absent the family's block is
**zero-filled** (neutral), so a caller that only has, say, a price/flow series still
gets a full-width vector with the other families zeroed. The keys:

    {
      # features_flow — one look-back window of bucket dicts
      #   (open/high/low/close/volume/trades/signed_flow), length WINDOW
      "flow_window":  [ {...}, ... ],

      # features_resolve — a single resolution snapshot dict
      "snapshot":     { "price": .., "t_created": .., "t_end": .., ... },

      # features_smart — one window of the user tape (all lists aligned)
      "smart": {
          "users": [..], "roles": [..], "prices": [..],
          "usds": [..], "tokens": [..],
          "smart_set": set(), "seen_before": set(),
      },

      # features_crossmarket — this market leg plus its event siblings
      #   (siblings EXCLUDES the leg itself; may be empty → singleton basket)
      "market":   { "price": .., "event_id": .., "neg_risk": .., ... },
      "siblings": [ { ...market dicts... }, ... ],

      # features_event — this market's price window + its event peers' windows
      "event": {
          "self_close": [..],            # length WINDOW
          "peer_closes": [ [..], ... ],  # aligned peer windows (may be empty)
          "self_flow": [..] | None,      # optional signed-flow window
          "peer_flows": [ [..], ... ] | None,
      },

      # features_micro — one look-back window of micro Buckets
      "micro_window": [ Bucket, ... ],
    }

A family whose top-level key is missing (or whose essential inner input is empty)
is zero-filled. A family that has its input but only *partial* context — a market
with no siblings, an event with no peers — uses that module's own graceful neutral
path (e.g. a singleton basket), exactly as documented in those modules.

Pure stdlib (`math`) plus the six merged feature modules. No numpy/pandas, no GPU.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

from features_crossmarket import CROSSMARKET_FEATURES, market_features
from features_event import EVENT_FEATURES, window_features_event
from features_flow import FLOW_FEATURES, window_features_flow
from features_micro import MICRO_FEATURES, window_features_micro
from features_resolve import RESOLVE_FEATURES, snapshot_features
from features_smart import SMART_FEATURES, window_features_smart

# A family adaptor maps a context dict to that family's raw feature vector, or
# returns None when the family's required inputs are absent (→ zero-fill).
Adaptor = Callable[[dict], Optional[List[float]]]


# ── per-family adaptors ───────────────────────────────────────────────────────
# Each pulls its family's inputs out of the shared context and calls the merged
# module's own public feature fn. None ⇒ inputs absent ⇒ build_all zero-fills.

def _flow_vec(ctx: dict) -> Optional[List[float]]:
    window = ctx.get("flow_window")
    if not window:
        return None
    return window_features_flow(window)


def _resolve_vec(ctx: dict) -> Optional[List[float]]:
    snapshot = ctx.get("snapshot")
    if snapshot is None:
        return None
    # snapshot_features is tolerant of missing keys and always finite.
    return snapshot_features(snapshot)


def _smart_vec(ctx: dict) -> Optional[List[float]]:
    s = ctx.get("smart")
    # window_features_smart indexes prices[-1]; an empty tape has no window.
    if not s or not s.get("prices"):
        return None
    return window_features_smart(
        s["users"],
        s["roles"],
        s["prices"],
        s["usds"],
        s["tokens"],
        s.get("smart_set", set()),
        s.get("seen_before", set()),
    )


def _crossmarket_vec(ctx: dict) -> Optional[List[float]]:
    market = ctx.get("market")
    if market is None:
        return None
    # market_features expects event_markets to INCLUDE the leg itself. With no
    # siblings this is a singleton event and the module degrades gracefully.
    siblings = ctx.get("siblings") or []
    return market_features(market, [market, *siblings])


def _event_vec(ctx: dict) -> Optional[List[float]]:
    e = ctx.get("event")
    # window_features_event indexes self_close[-1]; no self window ⇒ nothing.
    if not e or not e.get("self_close"):
        return None
    return window_features_event(
        e["self_close"],
        e.get("peer_closes", []),
        e.get("self_flow"),
        e.get("peer_flows"),
    )


def _micro_vec(ctx: dict) -> Optional[List[float]]:
    window = ctx.get("micro_window")
    if not window:
        return None
    return window_features_micro(window)


# Family registry — the ONE source of order for both names and vectors. Adding a
# family here extends ALL_FEATURES and build_all together, keeping them aligned.
_FAMILIES: List[Tuple[str, List[str], Adaptor]] = [
    ("flow", FLOW_FEATURES, _flow_vec),
    ("resolve", RESOLVE_FEATURES, _resolve_vec),
    ("smart", SMART_FEATURES, _smart_vec),
    ("crossmarket", CROSSMARKET_FEATURES, _crossmarket_vec),
    ("event", EVENT_FEATURES, _event_vec),
    ("micro", MICRO_FEATURES, _micro_vec),
]


def _build_names() -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """Concatenated, namespaced, de-duplicated feature names + per-family spans.

    Each family's names are prefixed with ``"<ns>."`` (e.g. ``flow.vpin``), which
    makes collisions across families impossible even when two families share a
    bare name (``flow.last`` vs ``micro.last``). The de-dup guard is a safety net
    for exact namespaced repeats; with distinct prefixes it never drops anything,
    so ``len(ALL_FEATURES)`` stays equal to the sum of the family widths and the
    concatenated build_all vector aligns one-to-one with ALL_FEATURES.
    """
    names: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    seen: set = set()
    for ns, feats, _ in _FAMILIES:
        start = len(names)
        for name in feats:
            key = f"{ns}.{name}"
            if key in seen:
                continue
            seen.add(key)
            names.append(key)
        spans[ns] = (start, len(names))
    return names, spans


ALL_FEATURES, FAMILY_SPANS = _build_names()

# Invariant: namespacing guarantees uniqueness, so no de-dup drop occurs and the
# unified width equals the sum of the family widths. build_all relies on this to
# stay aligned to ALL_FEATURES.
assert len(ALL_FEATURES) == sum(len(feats) for _, feats, _ in _FAMILIES), (
    "namespaced feature names collided — build_all/ALL_FEATURES misaligned"
)


def _finite(x: float) -> float:
    """Coerce to a finite float; map NaN/±inf to 0.0 (defensive, families are
    already finite by contract)."""
    x = float(x)
    return x if math.isfinite(x) else 0.0


def build_all(context: Optional[dict] = None) -> List[float]:
    """Assemble the full unified feature vector for one market/window context.

    Calls each family's public feature fn on its slice of ``context`` and
    concatenates the results in :data:`ALL_FEATURES` order. A family whose inputs
    are absent is zero-filled (neutral) so the returned vector is always exactly
    ``len(ALL_FEATURES)`` long and every value is finite.
    """
    ctx = context or {}
    vector: List[float] = []
    for ns, feats, fn in _FAMILIES:
        vec = fn(ctx)
        if vec is None:
            vec = [0.0] * len(feats)
        elif len(vec) != len(feats):
            raise ValueError(
                f"family '{ns}' returned {len(vec)} features, expected {len(feats)}"
            )
        vector.extend(_finite(x) for x in vec)
    return vector


def family_slice(vector: List[float], ns: str) -> List[float]:
    """The sub-vector for one family out of a full :func:`build_all` vector."""
    start, end = FAMILY_SPANS[ns]
    return vector[start:end]


# ── synthetic fully-populated context (selfcheck + tests share this) ──────────

def _synth_context(window: int = 16, seed: int = 7) -> dict:
    """A deterministic, fully-populated context exercising every family.

    Pure stdlib LCG so the vector is reproducible with no external inputs.
    """
    # Import here so the classes are local to context construction only.
    from features_micro import Bucket, Fill

    state = seed

    def rng() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    # --- flow: an up-drifting window of bucket dicts with buy-leaning flow ------
    flow_window = []
    price = 0.4
    for i in range(window):
        prev = price
        price = min(0.97, max(0.03, price + 0.006 + (rng() - 0.5) * 0.01))
        vol = 800.0 + rng() * 500.0 + (2500.0 if i % 7 == 0 else 0.0)
        flow_window.append({
            "open": prev, "high": max(prev, price) + 0.004,
            "low": min(prev, price) - 0.004, "close": price,
            "volume": vol, "trades": 3 + int(rng() * 7),
            "signed_flow": 0.6 * vol,
        })

    # --- resolve: a mid-life resolution snapshot ------------------------------
    t_created = 1_700_000_000
    snapshot = {
        "price": 0.62, "t_snapshot": t_created + 10 * 24 * 3600,
        "t_created": t_created, "t_end": t_created + 30 * 24 * 3600,
        "recent": [0.55 + 0.01 * k for k in range(8)],
        "volume": 3000.0, "volume_total": 5000.0,
        "signed_flow": 40.0, "flow_total": 120.0, "trade_count": 55,
        "outcome_prices": '["0.99","0.01"]',
    }

    # --- smart: one window of the user tape -----------------------------------
    users, roles, prices, usds, tokens = [], [], [], [], []
    p = 0.5
    for k in range(window):
        p = min(0.98, max(0.02, p + (rng() - 0.4) * 0.02))
        w = f"w{int(rng() * 10)}"
        amt = 50.0 + rng() * 450.0
        users.append(w)
        roles.append("maker" if rng() < 0.4 else "taker")
        prices.append(round(p, 4))
        usds.append(round(amt, 2))
        tokens.append(round((1.0 if rng() < 0.6 else -1.0) * amt / max(p, 0.02), 2))
    smart = {
        "users": users, "roles": roles, "prices": prices,
        "usds": usds, "tokens": tokens,
        "smart_set": {"w0", "w1", "w2"},
        "seen_before": {"w0", "w3"},
    }

    # --- crossmarket: a neg-risk basket leg + siblings ------------------------
    market = {
        "market_id": "leg0", "event_id": "E1", "neg_risk": True,
        "price": 0.5, "outcome_prices": "['1', '0']",
    }
    siblings = [
        {"market_id": "leg1", "event_id": "E1", "neg_risk": True,
         "price": 0.3, "outcome_prices": "['0', '1']"},
        {"market_id": "leg2", "event_id": "E1", "neg_risk": True,
         "price": 0.2, "outcome_prices": "['0', '1']"},
    ]

    # --- event: a self price window + two peer windows ------------------------
    self_close = [flow_window[i]["close"] for i in range(window)]
    peer_a = [min(0.97, max(0.03, c - 0.03 + (rng() - 0.5) * 0.004)) for c in self_close]
    peer_b = [min(0.97, max(0.03, c + 0.02 + (rng() - 0.5) * 0.004)) for c in self_close]
    event = {
        "self_close": self_close,
        "peer_closes": [peer_a, peer_b],
        "self_flow": [(1.0 if rng() < 0.6 else -1.0) * (400.0 + rng() * 300.0)
                      for _ in range(window)],
        "peer_flows": [[(1.0 if rng() < 0.5 else -1.0) * 300.0 for _ in range(window)],
                       [(1.0 if rng() < 0.5 else -1.0) * 300.0 for _ in range(window)]],
    }

    # --- micro: a window of Buckets, each with several Fills ------------------
    micro_window: List[Bucket] = []
    mp = 0.4
    makers = [f"0xmaker{k}" for k in range(4)]
    takers = [f"0xtaker{k}" for k in range(6)]
    for b in range(window):
        fills: List[Fill] = []
        for _ in range(3 + int(rng() * 5)):
            direction = 1 if rng() < 0.65 else -1
            usdc = 50.0 + rng() * 450.0 + (2500.0 if rng() < 0.05 else 0.0)
            mp = min(0.97, max(0.03, mp + 1e-5 * direction * usdc + (rng() - 0.5) * 5e-4))
            token = usdc / mp
            fills.append(Fill(
                ts=1_700_000_000 + b * 3600, price=mp, usdc=usdc, token=token,
                direction=direction, maker=makers[int(rng() * len(makers))],
                taker=takers[int(rng() * len(takers))], fee=0.001 * usdc,
                token_id="TKN",
            ))
        vwap = sum(f.price * f.usdc for f in fills) / sum(f.usdc for f in fills)
        micro_window.append(Bucket(key=b, price=vwap, fills=fills))

    return {
        "flow_window": flow_window,
        "snapshot": snapshot,
        "smart": smart,
        "market": market,
        "siblings": siblings,
        "event": event,
        "micro_window": micro_window,
    }


# ── selfcheck ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Names: namespaced, unique, and the summed family width.
    assert len(ALL_FEATURES) == len(set(ALL_FEATURES)), "duplicate feature names"
    assert all("." in n for n in ALL_FEATURES), "un-namespaced feature name"
    total = sum(len(feats) for _, feats, _ in _FAMILIES)
    assert len(ALL_FEATURES) == total, (len(ALL_FEATURES), total)

    # Full context ⇒ full-width, all-finite vector that fired every family.
    ctx = _synth_context()
    vec = build_all(ctx)
    assert len(vec) == len(ALL_FEATURES), (len(vec), len(ALL_FEATURES))
    assert all(math.isfinite(x) for x in vec), "non-finite feature in unified vector"
    fired = [ns for ns, _, _ in _FAMILIES if any(family_slice(vec, ns))]
    assert len(fired) == len(_FAMILIES), f"some families all-zero: fired={fired}"

    # Empty context ⇒ all zeros, still full width.
    empty = build_all({})
    assert len(empty) == len(ALL_FEATURES)
    assert all(x == 0.0 for x in empty), "empty context should zero-fill"

    print(
        f"families: {len(_FAMILIES)}  total features: {len(ALL_FEATURES)}  "
        f"fired: {len(fired)}/{len(_FAMILIES)}  finite: True"
    )
    print("  spans: " + "  ".join(f"{ns}[{a}:{b}]" for ns, (a, b) in FAMILY_SPANS.items()))
    print("selfcheck ok")
