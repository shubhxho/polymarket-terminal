"""Build signal artifacts from the SII-WANGZJ/Polymarket_data HF dataset.

`SII-WANGZJ/Polymarket_data` (MIT, 163 GB, 1.9 B records) is the deepest public
Polymarket dump: the full CLOB trade history (2022-11-21 → 2026-03-04) plus a
market table with the *final resolution* of 538 k markets. That resolution is
the piece the public OHLCV feeds (`fetch_hf.py`) can't give us — it's the honest
label for a flagship "will this market resolve YES?" model.

This module streams two tables and writes two artifacts into `ml/data/`:

- `sii_series.json` — per-token time-bucketed OHLCV + order flow. Trades are
  bucketed by `asset_id` into fixed time windows; each bucket records
  open/high/low/close price, USD volume, trade count, and *signed aggressor
  flow* = Σ usd_amount · (+1 if taker_direction==BUY else −1). Same shape/spirit
  as `data/ohlcv.json` (adds `timestamp` and `signed_flow`). Consumers:
  `features_ohlcv.py`.

- `sii_resolve.json` — mid-market snapshots labelled by final resolution. Each
  market is joined to `markets.parquet.outcome_prices` (a JSON-array string such
  as `["0.99","0.01"]` → answer1/token1 won → label 1). Snapshots of market
  state (price, cumulative volume, recent flow, time-to-resolution) are taken at
  several points before `end_date`, each carrying the binary resolution label.
  This is the label source for the flagship resolution model.

    python ml/fetch_sii.py --sample     # ~2000 rows (default for tests)
    python ml/fetch_sii.py              # full stream (418 M trades — slow)

The sample path streams row groups straight off the Hub with
`huggingface_hub` + `pyarrow` (no `datasets` lib required). If the network / Hub
is unavailable it falls back to a deterministic synthetic fixture and prints
which path was used, so the pipeline and its tests run fully offline.

Needs: huggingface_hub, pyarrow (heavy imports are kept inside functions).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

REPO = "SII-WANGZJ/Polymarket_data"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_SERIES = os.path.join(DATA_DIR, "sii_series.json")
OUT_RESOLVE = os.path.join(DATA_DIR, "sii_resolve.json")

# Trade columns we pull (subset of the 18-col schema — the ones we aggregate on).
TRADE_COLS = [
    "timestamp", "market_id", "condition_id", "price", "usd_amount",
    "taker_direction", "nonusdc_side", "asset_id",
]
MARKET_COLS = [
    "id", "question", "condition_id", "token1", "token2",
    "closed", "outcome_prices", "volume", "end_date",
]

BUCKET_SECONDS = 3600          # 1-hour OHLCV buckets, like the 1h upstream
MIN_BUCKETS = 40               # enough buckets for windows + a horizon
MIN_MARKET_TRADES = 6          # need a few trades to snapshot a resolution
SNAPSHOT_FRACS = (0.2, 0.4, 0.6, 0.8)   # where along the trade timeline to snapshot
RECENT_TRADES = 10             # trailing trades for the "recent flow" feature
SAMPLE_ROWS = 2000


# ── streaming ────────────────────────────────────────────────────────────────

def _stream_trades(sample: bool, max_rows: int):
    """Yield trade-row dicts from `trades.parquet`.

    Prefers streaming row groups off the Hub via `HfFileSystem` (no full 28 GB
    download); falls back to a cached `hf_hub_download`. Raises on any failure so
    the caller can drop to the synthetic fixture.
    """
    import pyarrow.parquet as pq

    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        handle = fs.open(f"datasets/{REPO}/trades.parquet", "rb")
        pf = pq.ParquetFile(handle)
    except Exception:
        # Last resort: materialise the file locally (only sane for the full run).
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(REPO, "trades.parquet", repo_type="dataset")
        pf = pq.ParquetFile(local)

    batch_size = 512 if sample else 65536
    n = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=TRADE_COLS):
        d = batch.to_pydict()
        rows_in_batch = len(d["timestamp"])
        for i in range(rows_in_batch):
            yield {c: d[c][i] for c in TRADE_COLS}
            n += 1
            if sample and n >= max_rows:
                return


def _load_markets(market_ids: set):
    """Download `markets.parquet` (85 MB) and return {id: market-dict} filtered
    to the markets we actually saw trades for."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(REPO, "markets.parquet", repo_type="dataset")
    table = pq.read_table(local, columns=MARKET_COLS)
    d = table.to_pydict()
    out: dict = {}
    for i in range(len(d["id"])):
        mid = d["id"][i]
        if market_ids and mid not in market_ids:
            continue
        out[mid] = {c: d[c][i] for c in MARKET_COLS}
    return out


# ── synthetic fixture (offline fallback) ──────────────────────────────────────

def _synthetic(n_markets: int = 24, trades_per_token: int = 60):
    """Deterministic trades + markets so the pipeline runs with no network.

    Each market gets a YES (token1) and NO (token2) asset with an hourly random
    walk; the market's `outcome_prices` reflect the final YES price, so labels
    are self-consistent with the price path the snapshots see.
    """
    import random

    rng = random.Random(1234)
    base_ts = 1_600_000_000
    hour = BUCKET_SECONDS

    trades = []
    markets: dict = {}
    for m in range(n_markets):
        mid = 100_000 + m
        cond = f"0xcond{m:04d}"
        tok_yes = 200_000 + m * 2
        tok_no = 200_000 + m * 2 + 1
        price = rng.uniform(0.25, 0.75)
        start = base_ts + m * 7 * 24 * hour
        mkt_volume = 0.0
        for k in range(trades_per_token):
            # gentle random walk, clamped inside the CLOB's (0,1) price band
            price = min(0.98, max(0.02, price + rng.uniform(-0.04, 0.04)))
            ts = start + k * hour + rng.randint(0, hour - 1)
            usd = round(rng.uniform(20, 5000), 2)
            mkt_volume += usd
            taker = "BUY" if rng.random() < 0.5 else "SELL"
            trades.append({
                "timestamp": ts, "market_id": mid, "condition_id": cond,
                "price": price, "usd_amount": usd,
                "taker_direction": taker, "nonusdc_side": "token1",
                "asset_id": tok_yes,
            })
            # sparser NO-side flow (complementary price)
            if k % 2 == 0:
                no_usd = round(rng.uniform(20, 2000), 2)
                mkt_volume += no_usd
                trades.append({
                    "timestamp": ts + 1, "market_id": mid, "condition_id": cond,
                    "price": round(1.0 - price, 4),
                    "usd_amount": no_usd,
                    "taker_direction": "SELL" if taker == "BUY" else "BUY",
                    "nonusdc_side": "token2", "asset_id": tok_no,
                })
        won = price > 0.5
        markets[mid] = {
            "id": mid, "question": f"Synthetic market {m}?",
            "condition_id": cond, "token1": tok_yes, "token2": tok_no,
            "closed": 1,
            "outcome_prices": json.dumps(["0.99", "0.01"] if won else ["0.02", "0.98"]),
            "volume": mkt_volume,
            "end_date": _iso(start + (trades_per_token + 2) * hour),
        }
    return trades, markets


def _iso(epoch: int) -> str:
    import datetime

    return datetime.datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── parsing helpers ────────────────────────────────────────────────────────────

def _parse_end(val):
    """`end_date` → epoch seconds. Accepts datetime objects, ISO strings, epochs."""
    if val is None:
        return None
    import datetime

    if isinstance(val, datetime.datetime):
        return val.timestamp()
    if isinstance(val, datetime.date):
        return datetime.datetime(val.year, val.month, val.day).timestamp()
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)

    s = str(val).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.datetime.strptime(str(val)[:19], fmt).timestamp()
            except ValueError:
                continue
    return None


def _parse_outcome_array(outcome_prices):
    """`outcome_prices` → list[float], or None. The dataset stores it as either a
    JSON array string (`["0.99","0.01"]`) or a Python-list repr (`"['1', '0']"`),
    so try JSON first then `ast.literal_eval`."""
    if outcome_prices is None:
        return None
    if isinstance(outcome_prices, (list, tuple)):
        seq = outcome_prices
    else:
        s = str(outcome_prices).strip()
        seq = None
        for parse in (json.loads, __import__("ast").literal_eval):
            try:
                seq = parse(s)
                break
            except (ValueError, SyntaxError, TypeError):
                continue
        if seq is None:
            return None
    try:
        return [float(x) for x in seq]
    except (ValueError, TypeError):
        return None


def _label_from_outcome(outcome_prices):
    """`["0.99","0.01"]` → 1 (answer1/token1 won), `["0.02","0.98"]` → 0.

    Returns None when the market isn't resolved to a clear winner."""
    vals = _parse_outcome_array(outcome_prices)
    if vals is None or len(vals) < 2:
        return None
    if max(vals) < 0.9:        # not settled to ~1 on either side
        return None
    return 1 if vals[0] >= vals[1] else 0


def _sign(taker_direction) -> float:
    return 1.0 if str(taker_direction).upper() == "BUY" else -1.0


# ── artifact builders ──────────────────────────────────────────────────────────

def build_series(trades, min_buckets: int):
    """Bucket trades by asset_id into fixed windows → per-token OHLCV series.

    Mirrors `data/ohlcv.json` (open/high/low/close/volume/trades) and adds
    `timestamp` (bucket-start epoch) and `signed_flow` (aggressor USD flow)."""
    # asset_id -> bucket_ts -> aggregation dict
    buckets: dict = defaultdict(dict)
    meta: dict = {}
    for t in trades:
        aid = t["asset_id"]
        ts = int(t["timestamp"])
        price = float(t["price"])
        usd = float(t["usd_amount"])
        if not (math.isfinite(price) and math.isfinite(usd)):
            continue
        flow = usd * _sign(t["taker_direction"])
        bkt = (ts // BUCKET_SECONDS) * BUCKET_SECONDS
        meta.setdefault(aid, (t["market_id"], t.get("nonusdc_side")))
        b = buckets[aid].get(bkt)
        if b is None:
            buckets[aid][bkt] = {
                "open_ts": ts, "open": price, "close_ts": ts, "close": price,
                "high": price, "low": price, "vol": usd, "n": 1, "flow": flow,
            }
        else:
            if ts < b["open_ts"]:
                b["open_ts"], b["open"] = ts, price
            if ts >= b["close_ts"]:
                b["close_ts"], b["close"] = ts, price
            b["high"] = max(b["high"], price)
            b["low"] = min(b["low"], price)
            b["vol"] += usd
            b["n"] += 1
            b["flow"] += flow

    series = []
    for aid, bmap in buckets.items():
        if len(bmap) < min_buckets:
            continue
        ordered = sorted(bmap.items())          # chronological by bucket start
        mid, side = meta.get(aid, (None, None))
        series.append({
            "asset_id": aid,
            "market_id": mid,
            "token_side": side,
            "timestamp": [int(ts) for ts, _ in ordered],
            "open":  [float(b["open"]) for _, b in ordered],
            "high":  [float(b["high"]) for _, b in ordered],
            "low":   [float(b["low"]) for _, b in ordered],
            "close": [float(b["close"]) for _, b in ordered],
            "volume": [float(b["vol"]) for _, b in ordered],
            "trades": [int(b["n"]) for _, b in ordered],
            "signed_flow": [float(b["flow"]) for _, b in ordered],
        })
    return series


def build_resolve(trades, markets):
    """Mid-market snapshots labelled by final resolution.

    Groups trades per market, joins to `markets` for the label + `end_date`, and
    emits a flat list of snapshot rows (each carrying the binary label)."""
    by_market: dict = defaultdict(list)
    for t in trades:
        by_market[t["market_id"]].append(t)

    rows = []
    for mid, mts in by_market.items():
        market = markets.get(mid)
        if market is None:
            continue
        label = _label_from_outcome(market.get("outcome_prices"))
        if label is None:
            continue
        if len(mts) < MIN_MARKET_TRADES:
            continue
        mts.sort(key=lambda r: int(r["timestamp"]))
        token1 = market.get("token1")
        end_ts = _parse_end(market.get("end_date"))
        if end_ts is None:
            end_ts = float(int(mts[-1]["timestamp"]))

        # running aggregates indexed by trade position
        cum_vol = 0.0
        cum_vols, last_yes_price = [], 0.5
        yes_prices = []
        for t in mts:
            cum_vol += float(t["usd_amount"])
            cum_vols.append(cum_vol)
            if _is_yes(t, token1):
                last_yes_price = float(t["price"])
            yes_prices.append(last_yes_price)

        n = len(mts)
        seen_ts = set()
        for frac in SNAPSHOT_FRACS:
            idx = min(n - 1, max(0, int(frac * n)))
            snap_ts = int(mts[idx]["timestamp"])
            if snap_ts in seen_ts:               # avoid duplicate snapshots on tiny markets
                continue
            seen_ts.add(snap_ts)
            lo = max(0, idx - RECENT_TRADES + 1)
            # YES-oriented aggressor flow: a BUY of the NO (token2) asset is
            # bearish for YES, so flip its sign — keeps recent_flow's sign
            # consistent with the YES resolution label.
            recent_flow = sum(
                float(mts[j]["usd_amount"]) * _sign(mts[j]["taker_direction"])
                * (1.0 if _is_yes(mts[j], token1) else -1.0)
                for j in range(lo, idx + 1)
            )
            price = yes_prices[idx]
            if not math.isfinite(price):
                continue
            rows.append({
                "market_id": mid,
                "condition_id": market.get("condition_id"),
                "ts": snap_ts,
                "price": float(price),
                "cum_volume": float(cum_vols[idx]),
                "recent_flow": float(recent_flow),
                "time_to_resolution": float(max(0.0, end_ts - snap_ts)),
                "label": int(label),
            })
    return rows


def _is_yes(trade, token1) -> bool:
    """A trade prices the YES token when its asset is token1 (or the side says so)."""
    if token1 is not None and trade.get("asset_id") == token1:
        return True
    return str(trade.get("nonusdc_side")) == "token1"


# ── schema guard ───────────────────────────────────────────────────────────────

def _assert_schema(rows, required, name: str) -> None:
    assert rows, f"{name}: no rows to validate"
    keys = set(rows[0].keys())
    missing = [c for c in required if c not in keys]
    assert not missing, f"{name}: missing columns {missing}"


# ── driver ─────────────────────────────────────────────────────────────────────

def gather(sample: bool, max_rows: int):
    """Return (trades, markets, source-label). Tries HF, falls back to synthetic."""
    try:
        trades = list(_stream_trades(sample, max_rows))
        _assert_schema(trades, TRADE_COLS, "trades.parquet")
        market_ids = {t["market_id"] for t in trades}
        markets = _load_markets(market_ids)
        _assert_schema(list(markets.values()), MARKET_COLS, "markets.parquet")
        return trades, markets, "hf-stream"
    except Exception as e:  # noqa: BLE001 - any HF/network/parse failure → synthetic
        print(f"  HF path unavailable ({type(e).__name__}: {e}); using synthetic fixture", flush=True)
        trades, markets = _synthetic()
        return trades, markets, "synthetic"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true", help="stream ~2000 rows (default for tests)")
    ap.add_argument("--rows", type=int, default=SAMPLE_ROWS, help="row cap in --sample mode")
    ap.add_argument("--min-buckets", type=int, default=None,
                    help="min buckets per token (default 40, or 6 in --sample mode)")
    args = ap.parse_args()

    min_buckets = args.min_buckets
    if min_buckets is None:
        min_buckets = 6 if args.sample else MIN_BUCKETS

    print(f"gathering trades (sample={args.sample}, rows={args.rows})…", flush=True)
    trades, markets, source = gather(args.sample, args.rows)
    print(f"  data source: {source}  ({len(trades)} trades, {len(markets)} markets)", flush=True)

    series = build_series(trades, min_buckets)
    resolve = build_resolve(trades, markets)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_SERIES, "w") as f:
        json.dump(series, f)
    with open(OUT_RESOLVE, "w") as f:
        json.dump(resolve, f)

    buckets_total = sum(len(s["close"]) for s in series)
    pos = sum(1 for r in resolve if r["label"] == 1)
    print(f"wrote {len(series)} token series ({buckets_total} buckets) → {OUT_SERIES}")
    print(f"wrote {len(resolve)} resolution snapshots "
          f"({pos} pos / {len(resolve) - pos} neg) → {OUT_RESOLVE}")


if __name__ == "__main__":
    main()
