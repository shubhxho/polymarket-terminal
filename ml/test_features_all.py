"""Stdlib tests for the unified feature builder — no pytest, no GPU.

Run:  python ml/test_features_all.py   (exits non-zero on the first failed assert)

Covers the assembly contract: full namespaced width, unique names, per-family
zero-fill when a family's inputs are absent, alignment of each family's slice to
its own module output, and finiteness throughout.
"""

from __future__ import annotations

import math

from features_all import (
    ALL_FEATURES,
    FAMILY_SPANS,
    _FAMILIES,
    _synth_context,
    build_all,
    family_slice,
)
from features_crossmarket import CROSSMARKET_FEATURES, market_features
from features_event import EVENT_FEATURES, window_features_event
from features_flow import FLOW_FEATURES, window_features_flow
from features_micro import MICRO_FEATURES, window_features_micro
from features_resolve import RESOLVE_FEATURES, snapshot_features
from features_smart import SMART_FEATURES, window_features_smart

# top-level context key that feeds each family (drop it → that family zero-fills)
_FAMILY_KEY = {
    "flow": "flow_window",
    "resolve": "snapshot",
    "smart": "smart",
    "crossmarket": "market",
    "event": "event",
    "micro": "micro_window",
}

_EXPECTED_WIDTH = (
    len(FLOW_FEATURES) + len(RESOLVE_FEATURES) + len(SMART_FEATURES)
    + len(CROSSMARKET_FEATURES) + len(EVENT_FEATURES) + len(MICRO_FEATURES)
)


def test_all_features_width_and_unique():
    # Concatenated width equals the sum of the six family widths.
    assert len(ALL_FEATURES) == _EXPECTED_WIDTH == 66, (len(ALL_FEATURES), _EXPECTED_WIDTH)
    # Namespacing makes every name unique.
    assert len(ALL_FEATURES) == len(set(ALL_FEATURES)), "duplicate feature names"


def test_all_features_namespaced():
    prefixes = {ns for ns, _, _ in _FAMILIES}
    for name in ALL_FEATURES:
        assert "." in name, name
        ns = name.split(".", 1)[0]
        assert ns in prefixes, name


def test_family_spans_partition():
    # Spans are contiguous, in registry order, and cover [0, width) exactly.
    cursor = 0
    for ns, feats, _ in _FAMILIES:
        start, end = FAMILY_SPANS[ns]
        assert start == cursor, (ns, start, cursor)
        assert end - start == len(feats), (ns, end - start, len(feats))
        # namespaced names in this span carry this family's prefix
        for name in ALL_FEATURES[start:end]:
            assert name.startswith(ns + "."), name
        cursor = end
    assert cursor == len(ALL_FEATURES)


def test_full_context_length_and_finite():
    vec = build_all(_synth_context())
    assert len(vec) == len(ALL_FEATURES)
    assert all(math.isfinite(x) for x in vec), "non-finite feature"


def test_empty_context_zero_fills_all():
    for ctx in (None, {}):
        vec = build_all(ctx)
        assert len(vec) == len(ALL_FEATURES)
        assert all(x == 0.0 for x in vec), "empty context must be all zeros"


def test_full_context_every_family_fires():
    vec = build_all(_synth_context())
    for ns, _, _ in _FAMILIES:
        assert any(v != 0.0 for v in family_slice(vec, ns)), f"family {ns} all-zero"


def test_missing_family_zero_fills_only_that_family():
    full_ctx = _synth_context()
    full_vec = build_all(full_ctx)
    for drop_ns, _, _ in _FAMILIES:
        ctx = dict(full_ctx)
        del ctx[_FAMILY_KEY[drop_ns]]
        vec = build_all(ctx)
        assert len(vec) == len(ALL_FEATURES)
        assert all(math.isfinite(x) for x in vec)
        # dropped family is entirely zero...
        start, end = FAMILY_SPANS[drop_ns]
        assert all(x == 0.0 for x in vec[start:end]), f"{drop_ns} not zero-filled"
        # ...and every other family is byte-for-byte unchanged.
        for ns, _, _ in _FAMILIES:
            if ns == drop_ns:
                continue
            assert family_slice(vec, ns) == family_slice(full_vec, ns), (
                f"dropping {drop_ns} disturbed {ns}"
            )


def test_slices_match_direct_module_calls():
    """Each family's slice equals calling that module's own feature fn directly —
    proves the vector is aligned and un-permuted."""
    ctx = _synth_context()
    vec = build_all(ctx)

    flow = window_features_flow(ctx["flow_window"])
    assert family_slice(vec, "flow") == [float(x) for x in flow]

    resolve = snapshot_features(ctx["snapshot"])
    assert family_slice(vec, "resolve") == [float(x) for x in resolve]

    s = ctx["smart"]
    smart = window_features_smart(
        s["users"], s["roles"], s["prices"], s["usds"], s["tokens"],
        s["smart_set"], s["seen_before"],
    )
    assert family_slice(vec, "smart") == [float(x) for x in smart]

    cross = market_features(ctx["market"], [ctx["market"], *ctx["siblings"]])
    assert family_slice(vec, "crossmarket") == [float(x) for x in cross]

    e = ctx["event"]
    event = window_features_event(
        e["self_close"], e["peer_closes"], e["self_flow"], e["peer_flows"],
    )
    assert family_slice(vec, "event") == [float(x) for x in event]

    micro = window_features_micro(ctx["micro_window"])
    assert family_slice(vec, "micro") == [float(x) for x in micro]


def test_partial_crossmarket_no_siblings_still_fires():
    """A market with no siblings is a graceful singleton basket, NOT a zero-fill —
    zero-fill only triggers when the whole family input is absent."""
    ctx = _synth_context()
    ctx["siblings"] = []
    vec = build_all(ctx)
    assert len(vec) == len(ALL_FEATURES)
    assert all(math.isfinite(x) for x in vec)
    # singleton path: share_of_basket == 1.0 (not the zero-filled 0.0)
    idx = CROSSMARKET_FEATURES.index("share_of_basket")
    start, _ = FAMILY_SPANS["crossmarket"]
    assert vec[start + idx] == 1.0


def test_partial_event_no_peers_still_finite():
    """An event window with no peers uses the module's singleton path (finite)."""
    ctx = _synth_context()
    ctx["event"] = {"self_close": ctx["event"]["self_close"], "peer_closes": []}
    vec = build_all(ctx)
    assert all(math.isfinite(x) for x in vec)
    # 'last' (self price) is populated even with no peers → family fired.
    idx = EVENT_FEATURES.index("last")
    start, _ = FAMILY_SPANS["event"]
    assert vec[start + idx] != 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
