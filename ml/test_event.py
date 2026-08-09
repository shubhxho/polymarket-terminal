"""Stdlib tests for the event-level co-movement feature layer — no pytest, no GPU.

Run:  python ml/test_event.py     (exits non-zero on the first failed assertion)

Covers the relative-value / co-movement features in `features_event.py`: vector
shape and finiteness, a finite beta, the idiosyncratic return vanishing for a
market that perfectly tracks its event, the bounded flow divergence, the
graceful singleton-event (no-peer) path, lead/lag recovery of a planted leader,
and binary labels out of `build`.
"""

from __future__ import annotations

import math

from features_event import (
    EVENT_FEATURES,
    _lead_lag,
    _net_imbalance,
    _synth_event,
    build,
    window_features_event,
)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _ramp(start: float, step: float, n: int = 16):
    return [start + step * i for i in range(n)]


def test_feature_vector_shape_and_finite():
    self_close = _ramp(0.4, 0.01)
    peers = [_ramp(0.4, 0.01), _ramp(0.5, -0.005), _ramp(0.3, 0.008)]
    v = window_features_event(self_close, peers)
    assert len(v) == len(EVENT_FEATURES) == 11, len(v)
    assert all(math.isfinite(x) for x in v)


def test_beta_to_event_is_finite():
    self_close = [0.4 + 0.01 * i + 0.02 * math.sin(i / 2) for i in range(16)]
    # Peers share the same rising, oscillating path plus a small idiosyncratic
    # wobble, so they genuinely co-move with self (positive beta).
    peers = [[0.4 + 0.01 * i + 0.02 * math.sin(i / 2) + 0.003 * math.sin(i + p)
              for i in range(16)] for p in range(3)]
    idx = EVENT_FEATURES.index("beta_to_event")
    beta = window_features_event(self_close, peers)[idx]
    assert math.isfinite(beta)
    # A market that co-moves with a rising basket has a positive beta.
    assert beta > 0, beta


def test_idio_return_zero_for_perfect_tracker():
    """A market whose returns equal the event-average returns has ~0 idio move.

    With identical peers the event index equals the market's own path, so
    beta≈1 and idio_return = self_drift - beta*event_drift ≈ 0."""
    path = [0.3 + 0.01 * i + 0.02 * math.sin(i / 2) for i in range(16)]
    peers = [list(path) for _ in range(3)]  # peers identical to self → perfect tracker
    idx = EVENT_FEATURES.index("idio_return")
    idio = window_features_event(path, peers)[idx]
    assert approx(idio, 0.0, tol=1e-9), idio
    # And beta should be ~1 in that case.
    beta = window_features_event(path, peers)[EVENT_FEATURES.index("beta_to_event")]
    assert approx(beta, 1.0, tol=1e-6), beta


def test_flow_divergence_bounded():
    self_close = _ramp(0.4, 0.01)
    peers = [_ramp(0.4, 0.01), _ramp(0.4, -0.01)]
    idx = EVENT_FEATURES.index("flow_divergence")
    # Self all-buy, peers all-sell → maximally divergent, but stays in [-1, 1].
    self_flow = [100.0] * 16
    peer_flows = [[-100.0] * 16, [-100.0] * 16]
    fd = window_features_event(self_close, peers, self_flow, peer_flows)[idx]
    assert -1.0 <= fd <= 1.0, fd
    assert fd > 0, fd  # self buying while basket sells → positive divergence
    # Reverse the sign.
    fd2 = window_features_event(self_close, peers, [-100.0] * 16, [[100.0] * 16, [100.0] * 16])[idx]
    assert -1.0 <= fd2 <= 1.0, fd2
    assert fd2 < 0, fd2
    # No flow supplied → neutral zero.
    assert window_features_event(self_close, peers)[idx] == 0.0


def test_net_imbalance_bounds():
    assert approx(_net_imbalance([5.0, 5.0, 5.0]), 1.0)
    assert approx(_net_imbalance([-2.0, -2.0]), -1.0)
    assert _net_imbalance([3.0, -3.0]) == 0.0
    assert _net_imbalance([]) == 0.0
    assert _net_imbalance([0.0, 0.0]) == 0.0


def test_singleton_event_graceful():
    """A market with no event peers still yields a finite, well-formed vector."""
    self_close = [0.5 + 0.01 * i for i in range(16)]
    v = window_features_event(self_close, peer_closes=[])
    assert len(v) == len(EVENT_FEATURES)
    assert all(math.isfinite(x) for x in v)
    # No peers → no co-movement signal: beta, event momentum/vol, lead_lag all 0.
    for name in ("beta_to_event", "event_momentum", "event_vol", "lead_lag", "flow_divergence"):
        assert v[EVENT_FEATURES.index(name)] == 0.0, name
    # peer_count reads 0, and idio collapses to the market's own drift.
    assert v[EVENT_FEATURES.index("peer_count")] == 0.0
    idio = v[EVENT_FEATURES.index("idio_return")]
    assert approx(idio, self_close[-1] - self_close[0]), idio


def test_lead_lag_sign():
    """A series whose moves are echoed one step later by the basket leads (+)."""
    base = [0.5 + 0.02 * math.sin(i / 2) for i in range(20)]
    self_ret = [base[i] - base[i - 1] for i in range(1, len(base))]
    # Event lags self by one step: event[t] ~ self[t-1].
    event_ret = [0.0] + self_ret[:-1]
    assert _lead_lag(self_ret, event_ret) > 0
    # And the reverse case lags.
    assert _lead_lag(event_ret, self_ret) < 0
    # Degenerate short inputs are graceful.
    assert _lead_lag([0.1, 0.2], [0.1, 0.2]) == 0.0


def test_build_produces_binary_labels_and_leader():
    series = _synth_event(80, 4, seed=5)
    groups = build(series)
    assert groups, "no groups built"
    n = sum(len(g) for g in groups)
    assert n > 0, "no windows built"
    # Labels are strictly binary.
    for g in groups:
        for s in g:
            assert s.label in (0, 1), s.label
            assert len(s.feat) == len(EVENT_FEATURES)
            assert all(math.isfinite(x) for x in s.feat)
    # The planted leader (group 0) leads its peers on average.
    ll = EVENT_FEATURES.index("lead_lag")
    from features_event import _mean
    lead_lags = [(_mean([s.feat[ll] for s in g]) if g else float("-inf")) for g in groups[:4]]
    assert max(range(len(lead_lags)), key=lambda k: lead_lags[k]) == 0, lead_lags
    assert lead_lags[0] > 0, lead_lags[0]


def test_build_handles_multiple_events_and_singletons():
    """Two real events plus a lone market all coexist in one panel."""
    ev_a = _synth_event(60, 3, seed=1)
    ev_b = _synth_event(60, 3, seed=2)
    for m in ev_b:
        m["event_id"] = "B"
    solo = {"market_id": "solo", "event_id": "SOLO",
            "close": [0.4 + 0.007 * i for i in range(60)]}
    groups = build(ev_a + ev_b + [solo])
    # 3 + 3 + 1 markets → 7 per-market groups.
    assert len(groups) == 7, len(groups)
    assert all(all(math.isfinite(x) for x in s.feat) for g in groups for s in g)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
