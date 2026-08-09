"""Fetch the best public Polymarket dataset: real 1-hour OHLCV candles.

`ImpliedData/prediction-markets` is the cross-platform prediction-market data
layer (Polymarket + Manifold), built on Polymarket's 404M-fill on-chain tick
history. The free sample is 272k rows of **OHLCV** — open/high/low/close plus
volume and trade_count — which is a big upgrade over the single-price hourly
series `fetch_data.py` pulls from the CLOB: true highs and lows give a *proper*
CCI (typical price), stochastic and ATR, and volume/trade_count give real
liquidity and order-flow-proxy features.

Downloads the auto-converted parquet (17 MB), groups by market, and writes
`data/ohlcv.json` — a list of per-market OHLCV series that `features_ohlcv.py`
turns into training windows.

    python ml/fetch_hf.py                 # all platforms
    python ml/fetch_hf.py --platform 3    # one platform only

Needs: huggingface_hub, pyarrow (see requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "ImpliedData/prediction-markets"
# Named OHLCV files on the main branch (the auto-converted parquet mixes tables).
FILES = ["polymarket_ohlcv_1h_sample.parquet", "manifold_ohlcv_1h_sample.parquet"]
OUT = os.path.join(os.path.dirname(__file__), "data", "ohlcv.json")

MIN_CANDLES = 40   # need enough for windows + a horizon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", type=int, default=None, help="keep only this platform_id")
    ap.add_argument("--polymarket-only", action="store_true", help="skip the Manifold file")
    ap.add_argument("--min-candles", type=int, default=MIN_CANDLES)
    args = ap.parse_args()

    files = FILES[:1] if args.polymarket_only else FILES
    # (platform_id, market_id) -> list of (ts, o, h, l, c, v, n)
    rows: dict[tuple, list] = defaultdict(list)
    for fn in files:
        path = hf_hub_download(REPO, fn, repo_type="dataset")
        t = pq.read_table(path, columns=[
            "timestamp", "platform_id", "market_id", "open", "high", "low", "close", "volume", "trade_count",
        ])
        d = t.to_pydict()
        n = len(d["timestamp"])
        print(f"  {fn}: {n} rows", flush=True)
        for i in range(n):
            pid = d["platform_id"][i]
            if args.platform is not None and pid != args.platform:
                continue
            key = (pid, d["market_id"][i])
            rows[key].append((
                d["timestamp"][i], d["open"][i], d["high"][i], d["low"][i],
                d["close"][i], d["volume"][i], d["trade_count"][i],
            ))

    series = []
    platforms: dict = defaultdict(int)
    for (pid, mid), pts in rows.items():
        if len(pts) < args.min_candles:
            continue
        pts.sort(key=lambda r: r[0])   # chronological
        series.append({
            "platform": int(pid),
            "market_id": int(mid),
            "open":  [float(r[1]) for r in pts],
            "high":  [float(r[2]) for r in pts],
            "low":   [float(r[3]) for r in pts],
            "close": [float(r[4]) for r in pts],
            "volume": [float(r[5]) for r in pts],
            "trades": [int(r[6]) for r in pts],
        })
        platforms[int(pid)] += 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(series, f)
    total = sum(len(s["close"]) for s in series)
    print(f"\nwrote {len(series)} markets ({total} candles) → {OUT}")
    print("per-platform market counts:", dict(platforms))


if __name__ == "__main__":
    main()
