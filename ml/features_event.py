"""Event-level relative-value / co-movement feature extraction.

Polymarket markets don't move in isolation. Every market hangs off an *event*
(shared `event_id` / `event_title` — e.g. the several "who wins the primary"
markets under one race), and those sibling markets move together: a shock to the
event drags the whole basket, while a market's move *relative to its peers* is
the part that actually carries information about that market specifically.

This module turns a panel of per-market price/flow time series, grouped by
event, into per-market co-movement features. Distinct from a static basket-arb
check (does the yes/no book sum to 1 right now) — this is about the *time-series*
relationship between a market and the event it belongs to:

- **beta_to_event** — regression slope of this market's returns on the
  event-average returns (how strongly it co-moves with the basket).
- **idio_return** — this market's realised drift minus beta·event drift: the
  idiosyncratic move the event doesn't explain. ~0 for a market that perfectly
  tracks its event.
- **event_momentum** — the event-average recent drift (basket-wide trend).
- **rel_strength** — this market's momentum minus the event's (relative strength
  vs its peers).
- **lead_lag** — cross-correlation sign: does this market *lead* the event
  (its moves are followed by the basket, +) or *lag* it (-)?
- **event_vol** — event-average realised vol (how choppy the basket is).
- **flow_divergence** — this market's signed order-flow direction vs the
  event-aggregate flow, in -1..1 (is it being bought while the basket is sold?).
- **peer_count** — how many sibling markets share the event.

Emits `features.Sample` rows (return sequence + feature vector + label + forward
return) exactly like `features_ohlcv.py` / `features_flow.py`, so it drops
straight into the existing `train_seq.py` machinery. Pure stdlib + `math` /
`collections` — trivially runnable and testable, no numpy/pandas, no GPU.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Optional

from features import HORIZON, MIN_STD, WINDOW, Sample, _mean, _std

EVENT_FEATURES = [
    # relative-value / co-movement (this market vs its event peers)
    "beta_to_event",    # slope of self returns on event-average returns
    "idio_return",      # self drift - beta*event drift (idiosyncratic move)
    "event_momentum",   # event-average recent drift
    "rel_strength",     # self momentum - event momentum
    "lead_lag",         # cross-corr sign: +lead / -lag the event, -1..1
    "event_vol",        # event-average realised vol
    "flow_divergence",  # self flow direction vs event-aggregate flow, -1..1
    "peer_count",       # number of sibling markets in the event
    # standalone price context so the vector stands on its own
    "last",             # current probability
    "self_momentum",    # mean of the last 4 self increments
    "self_vol",         # std of self increments (realised vol proxy)
]

EPS = 1e-9
BETA_CLIP = 10.0  # guard: regression slope stays finite/sane on a near-flat basket


def _diffs(xs: List[float]) -> List[float]:
    """First differences (per-step returns) of a price series."""
    return [xs[i] - xs[i - 1] for i in range(1, len(xs))]


def _ols_beta(y: List[float], x: List[float]) -> float:
    """Slope of the OLS fit y = a + beta*x, i.e. cov(x,y)/var(x).

    Returns 0 on a near-flat regressor (var(x)~0 → undefined slope) and clips to
    a sane band so a barely-moving event basket can't blow the feature up."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    mx, my = _mean(x[:n]), _mean(y[:n])
    var = sum((x[i] - mx) ** 2 for i in range(n))
    if var < EPS:
        return 0.0
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    return max(-BETA_CLIP, min(BETA_CLIP, cov / var))


def _corr(a: List[float], b: List[float]) -> float:
    """Pearson correlation of two aligned series, clipped to -1..1.

    0 when either series is (near-)flat — no variance to correlate."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    ma, mb = _mean(a[:n]), _mean(b[:n])
    va = sum((a[i] - ma) ** 2 for i in range(n))
    vb = sum((b[i] - mb) ** 2 for i in range(n))
    if va < EPS or vb < EPS:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return max(-1.0, min(1.0, cov / math.sqrt(va * vb)))


def _lead_lag(self_ret: List[float], event_ret: List[float]) -> float:
    """Signed lead/lag score in -1..1 from one-step cross-correlations.

    A *leader*'s move at t is followed by the event basket at t+1, so
    corr(self[:-1], event[1:]) is high. A *lagger* follows the basket, so
    corr(event[:-1], self[1:]) is high. The difference is +1 for a pure leader,
    -1 for a pure lagger, ~0 when the two are contemporaneous."""
    if len(self_ret) < 3 or len(event_ret) < 3:
        return 0.0
    lead = _corr(self_ret[:-1], event_ret[1:])  # self's past predicts event's future
    lag = _corr(event_ret[:-1], self_ret[1:])   # event's past predicts self's future
    return max(-1.0, min(1.0, lead - lag))


def _net_imbalance(flows: List[float]) -> float:
    """Net signed flow over total absolute flow, in -1..1.

    +1 = all buy-aggressor, -1 = all sell, 0 = balanced or no flow."""
    total = sum(abs(f) for f in flows)
    if total < EPS:
        return 0.0
    return max(-1.0, min(1.0, sum(flows) / total))


def _event_return_series(peer_rets: List[List[float]]) -> List[float]:
    """Element-wise mean across peer return series (the event index returns).

    Truncates to the shortest peer so ragged windows stay aligned."""
    if not peer_rets:
        return []
    L = min(len(r) for r in peer_rets)
    return [_mean([pr[k] for pr in peer_rets]) for k in range(L)]


def window_features_event(
    self_close: List[float],
    peer_closes: List[List[float]],
    self_flow: Optional[List[float]] = None,
    peer_flows: Optional[List[List[float]]] = None,
) -> List[float]:
    """Feature vector for one look-back window of a market and its event peers.

    `self_close` is this market's price window (length WINDOW); `peer_closes` is
    a list of the sibling markets' aligned price windows (excluding self, may be
    empty for a singleton event). `self_flow` / `peer_flows` are optional signed
    order-flow series (per bar); omitted → flow features read neutral.

    Returns one value per name in `EVENT_FEATURES`, in that order. Every value is
    finite for any input, including the singleton-event (no-peer) path."""
    self_ret = _diffs(self_close)
    peer_rets = [_diffs(pc) for pc in peer_closes]
    peer_count = len(peer_closes)

    event_ret = _event_return_series(peer_rets)
    # Align self returns to the event index length for the joint stats.
    L = min(len(self_ret), len(event_ret)) if event_ret else 0
    self_ret_a = self_ret[:L]
    event_ret_a = event_ret[:L]

    beta = _ols_beta(self_ret_a, event_ret_a) if event_ret_a else 0.0
    # Full-window self move minus the part the event explains. With no peers the
    # event term is 0, so idio collapses to the market's own drift.
    self_drift = sum(self_ret)
    event_drift = sum(event_ret_a)
    idio_return = self_drift - beta * event_drift

    event_momentum = _mean(event_ret[-4:]) if event_ret else 0.0
    self_momentum = _mean(self_ret[-4:]) if self_ret else 0.0
    rel_strength = self_momentum - event_momentum

    lead_lag = _lead_lag(self_ret_a, event_ret_a) if event_ret_a else 0.0
    event_vol = _mean([_std(pr) for pr in peer_rets]) if peer_rets else 0.0
    self_vol = _std(self_ret)

    self_imb = _net_imbalance(self_flow) if self_flow else 0.0
    if peer_flows:
        agg = [f for pf in peer_flows for f in pf]  # event-aggregate signed flow
        event_imb = _net_imbalance(agg)
    else:
        event_imb = 0.0
    # difference of two -1..1 imbalances, halved back into -1..1.
    flow_divergence = max(-1.0, min(1.0, (self_imb - event_imb) / 2.0))

    return [
        beta,
        idio_return,
        event_momentum,
        rel_strength,
        lead_lag,
        event_vol,
        flow_divergence,
        float(peer_count),
        self_close[-1],
        self_momentum,
        self_vol,
    ]


def _event_key(m: dict) -> str:
    """Group key for a market: prefer `event_id`, fall back to `event_title`.

    Markets with no event marker each become their own singleton event (keyed by
    market id / object id) so they still get a graceful no-peer feature row."""
    key = m.get("event_id") or m.get("event_title")
    if key is None:
        key = m.get("market_id", id(m))
    return str(key)


def _slice_flow(m: dict, start: int, stop: int) -> Optional[List[float]]:
    """Signed-flow window for a market, or None if the series is absent."""
    flow = m.get("signed_flow")
    if flow is None:
        return None
    return [float(x) for x in flow[start:stop]]


def event_to_rich(self_m: dict, peers: List[dict]) -> List[Sample]:
    """Slide over one market's series → Sample rows using its event peers.

    Peers are the sibling markets under the same event. Every window looks at the
    same time index across the aligned panel; windows where the market itself is
    essentially flat are skipped (nothing to predict), matching the sibling
    feature modules."""
    close = [float(x) for x in self_m["close"]]
    # Align to the shortest series in the (self + peers) panel so every index is
    # populated for every market.
    N = len(close)
    for p in peers:
        N = min(N, len(p["close"]))
    out: List[Sample] = []
    for i in range(WINDOW, N - HORIZON):
        cw = close[i - WINDOW : i]
        rets = _diffs(cw)
        if _std(rets) < MIN_STD:
            continue
        peer_closes = [[float(x) for x in p["close"][i - WINDOW : i]] for p in peers]
        self_flow = _slice_flow(self_m, i - WINDOW, i)
        peer_flows = None
        pf = [_slice_flow(p, i - WINDOW, i) for p in peers]
        if any(f is not None for f in pf):
            peer_flows = [f for f in pf if f is not None]
        fwd = close[i + HORIZON] - close[i]
        feat = window_features_event(cw, peer_closes, self_flow, peer_flows)
        out.append(Sample(rets, feat, 1 if fwd > 0 else 0, fwd))
    return out


def build(series: List[dict]) -> List[List[Sample]]:
    """Per-market Sample lists, grouping the input panel by event.

    `series` is a flat list of market dicts, each with a `close` series and an
    `event_id`/`event_title` (plus optional `signed_flow`). Markets are grouped
    by event; each market's samples are computed against its event peers and kept
    as their own group so the downstream temporal split stays honest. Group order
    follows first-appearance of the market within its event."""
    events: "defaultdict[str, List[dict]]" = defaultdict(list)
    for m in series:
        events[_event_key(m)].append(m)

    groups: List[List[Sample]] = []
    for members in events.values():
        for idx, self_m in enumerate(members):
            peers = [p for j, p in enumerate(members) if j != idx]
            groups.append(event_to_rich(self_m, peers))
    return groups


# ── self-check ────────────────────────────────────────────────────────────────

def _synth_event(n: int = 80, n_markets: int = 4, seed: int = 3) -> List[dict]:
    """A synthetic event: one *planted leader* plus followers that lag it.

    The leader is a random walk in probability space. Each follower relaxes
    toward the leader's *previous* price (a one-step lag) plus small idiosyncratic
    noise, so the followers co-move with the leader but trail it. From the
    leader's viewpoint its peers (the followers) lag → the leader's `lead_lag`
    reads clearly positive, which the self-check recovers."""

    def rng(state: int) -> int:
        return (1103515245 * state + 12345) & 0x7FFFFFFF

    # Leader path.
    lead: List[float] = []
    lead_flow: List[float] = []
    price, state = 0.5, seed
    for _ in range(n):
        state = rng(state)
        step = ((state / 0x7FFFFFFF) - 0.5) * 0.03
        prev = price
        price = min(0.97, max(0.03, price + step))
        lead.append(price)
        lead_flow.append((1.0 if price >= prev else -1.0) * (500.0 + state % 400))

    markets: List[dict] = [{
        "market_id": "leader", "event_id": "E",
        "close": lead[:], "signed_flow": lead_flow[:],
    }]

    # Followers: track the leader lagged by one step.
    for j in range(1, n_markets):
        fclose: List[float] = []
        fflow: List[float] = []
        fp, state = 0.5, seed * 31 + j * 7
        for i in range(n):
            state = rng(state)
            noise = ((state / 0x7FFFFFFF) - 0.5) * 0.004
            target = lead[i - 1] if i > 0 else lead[0]
            fprev = fp
            fp = min(0.97, max(0.03, 0.6 * target + 0.4 * fp + noise))
            fclose.append(fp)
            fflow.append((1.0 if fp >= fprev else -1.0) * (400.0 + state % 300))
        markets.append({
            "market_id": f"follower{j}", "event_id": "E",
            "close": fclose, "signed_flow": fflow,
        })
    return markets


if __name__ == "__main__":
    markets = _synth_event(80, 4, seed=3)
    # Add a lone singleton-event market to exercise the graceful no-peer path.
    singleton = {
        "market_id": "solo", "event_id": "SOLO",
        "close": [0.4 + 0.006 * i + 0.01 * math.sin(i / 3) for i in range(80)],
    }
    series = markets + [singleton]

    groups = build(series)
    n_windows = sum(len(g) for g in groups)
    assert n_windows > 0, "no windows built"

    bad = sum(1 for g in groups for s in g if not all(math.isfinite(x) for x in s.feat))
    assert bad == 0, f"{bad} non-finite feature rows"
    assert all(len(s.feat) == len(EVENT_FEATURES) for g in groups for s in g)

    # Recover the planted leader: it is the first market of event E (group 0),
    # and its mean lead_lag must be the highest and positive.
    ll = EVENT_FEATURES.index("lead_lag")
    lead_lags = [
        (_mean([s.feat[ll] for s in g]) if g else float("-inf"))
        for g in groups[:4]  # the 4 markets of event E, in input order
    ]
    leader_idx = max(range(len(lead_lags)), key=lambda k: lead_lags[k])
    assert leader_idx == 0, f"leader not recovered: lead_lag by market = {lead_lags}"
    assert lead_lags[0] > 0, f"leader lead_lag not positive: {lead_lags[0]:.3f}"

    ups = [s.label for g in groups for s in g]
    print(
        f"features/row: {len(EVENT_FEATURES)}  markets: {len(series)}  "
        f"windows: {n_windows}  up-rate: {_mean([float(x) for x in ups]):.3f}  "
        f"non-finite: {bad}  leader lead_lag: {lead_lags[0]:+.3f}"
    )
