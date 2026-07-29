"""Stdlib tests for the SII dataset loader — no pytest, no network.

Run:  python ml/test_fetch_sii.py     (exits non-zero on the first failed assertion)

Exercises the pure builders on the synthetic fixture and validates the two
artifacts on disk (writing them via the fixture path if they aren't there yet),
so it runs fully offline on every commit.
"""

from __future__ import annotations

import json
import math
import os

import fetch_sii as fs

DATA_DIR = fs.DATA_DIR


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ── builder unit tests (synthetic fixture, no I/O) ────────────────────────────

def test_synthetic_fixture_shape():
    trades, markets = fs._synthetic()
    assert trades and markets
    fs._assert_schema(trades, fs.TRADE_COLS, "trades")
    fs._assert_schema(list(markets.values()), fs.MARKET_COLS, "markets")


def test_label_from_outcome():
    assert fs._label_from_outcome('["0.99","0.01"]') == 1
    assert fs._label_from_outcome('["0.02","0.98"]') == 0
    assert fs._label_from_outcome(["0.99", "0.01"]) == 1
    assert fs._label_from_outcome('["0.5","0.5"]') is None   # unresolved
    assert fs._label_from_outcome("not json") is None
    assert fs._label_from_outcome(None) is None


def test_signed_flow_sign():
    """A bucket of pure BUY aggressor flow is positive; pure SELL is negative."""
    base = 1_600_000_000
    buy = [{
        "timestamp": base + i, "asset_id": 7, "market_id": 1,
        "price": 0.5, "usd_amount": 100.0, "taker_direction": "BUY",
        "nonusdc_side": "token1",
    } for i in range(50)]
    series = fs.build_series(buy, min_buckets=1)
    assert len(series) == 1
    assert all(f > 0 for f in series[0]["signed_flow"])
    sell = [dict(t, taker_direction="SELL") for t in buy]
    s2 = fs.build_series(sell, min_buckets=1)
    assert all(f < 0 for f in s2[0]["signed_flow"])


def test_ohlc_bounds_and_order():
    """open/close sit within [low, high] and buckets are time-ordered."""
    trades, _ = fs._synthetic()
    series = fs.build_series(trades, min_buckets=fs.MIN_BUCKETS)
    assert series, "expected tokens with >= MIN_BUCKETS buckets"
    for s in series:
        ts = s["timestamp"]
        assert ts == sorted(ts) and len(set(ts)) == len(ts)   # strictly ordered, unique
        for o, h, l, c in zip(s["open"], s["high"], s["low"], s["close"]):
            assert l <= o <= h and l <= c <= h
            assert all(math.isfinite(x) for x in (o, h, l, c))


def test_resolve_labels_and_features():
    trades, markets = fs._synthetic()
    rows = fs.build_resolve(trades, markets)
    assert rows
    for r in rows:
        assert r["label"] in (0, 1)
        assert 0.0 <= r["price"] <= 1.0
        assert r["cum_volume"] >= 0.0
        assert r["time_to_resolution"] >= 0.0
        assert all(math.isfinite(r[k]) for k in ("price", "cum_volume", "recent_flow", "time_to_resolution"))
    labels = {r["label"] for r in rows}
    assert labels <= {0, 1}


# ── artifact integration tests (write then validate on disk) ──────────────────

def _ensure_artifacts():
    if not (os.path.exists(fs.OUT_SERIES) and os.path.exists(fs.OUT_RESOLVE)):
        trades, markets = fs._synthetic()
        os.makedirs(DATA_DIR, exist_ok=True)
        json.dump(fs.build_series(trades, fs.MIN_BUCKETS), open(fs.OUT_SERIES, "w"))
        json.dump(fs.build_resolve(trades, markets), open(fs.OUT_RESOLVE, "w"))


def test_artifacts_exist_and_nonempty():
    _ensure_artifacts()
    series = json.load(open(fs.OUT_SERIES))
    resolve = json.load(open(fs.OUT_RESOLVE))
    assert isinstance(series, list) and series, "sii_series.json empty"
    assert isinstance(resolve, list) and resolve, "sii_resolve.json empty"

    for s in series:
        for key in ("open", "high", "low", "close", "volume", "trades", "signed_flow", "timestamp"):
            assert key in s and isinstance(s[key], list) and s[key]
        ts = s["timestamp"]
        assert ts == sorted(ts)                       # buckets ordered in time
        assert all(math.isfinite(x) for x in s["close"])
        assert all(math.isfinite(x) for x in s["signed_flow"])

    for r in resolve:
        assert r["label"] in (0, 1)                   # labels are strictly binary
        assert all(math.isfinite(r[k]) for k in ("price", "cum_volume", "recent_flow", "time_to_resolution"))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
