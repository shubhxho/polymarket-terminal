"""Pull a real training set from the public Polymarket APIs.

Same two upstreams the terminal uses: Gamma for the market list (and its
`clobTokenIds`), CLOB for each token's price history. No key required. Writes
`ml/data/series.json` — a list of price series, one per liquid market — which
`train.py` turns into (features, label) rows via `features.py`.

Stdlib only (urllib + json), so it runs in any Python without extra installs.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, List

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
OUT = os.path.join(os.path.dirname(__file__), "data", "series.json")

# How much to pull. History is one request per token, so this is the main knob
# on both dataset size and wall-clock.
N_MARKETS = int(os.environ.get("N_MARKETS", "150"))
INTERVAL = os.environ.get("INTERVAL", "1w")   # 1h|6h|1d|1w|1m|max
FIDELITY = os.environ.get("FIDELITY", "60")   # minutes per bucket


# Gamma 403s the default python-urllib agent; a normal browser UA is fine.
_HEADERS = {
    "accept": "application/json",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
}


def _get(url: str, tries: int = 3) -> Any:
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001 - retry any transient upstream error
            last = e
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url} :: {last}")


def market_tokens(limit: int) -> List[str]:
    """First-outcome CLOB token id for each active, liquid market."""
    q = urllib.parse.urlencode(
        {
            "limit": limit,
            "order": "volume24hr",
            "ascending": "false",
            "active": "true",
            "closed": "false",
        }
    )
    rows = _get(f"{GAMMA}/markets?{q}")
    tokens: List[str] = []
    for m in rows:
        raw = m.get("clobTokenIds")
        ids = json.loads(raw) if isinstance(raw, str) else raw
        if ids:
            tokens.append(str(ids[0]))
    return tokens


def price_series(token: str) -> List[float]:
    q = urllib.parse.urlencode({"market": token, "interval": INTERVAL, "fidelity": FIDELITY})
    data = _get(f"{CLOB}/prices-history?{q}")
    pts = data.get("history", []) if isinstance(data, dict) else []
    return [float(p["p"]) for p in pts if "p" in p]


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tokens = market_tokens(N_MARKETS)
    print(f"fetched {len(tokens)} tokens; pulling history…")

    series: List[List[float]] = []
    for i, tok in enumerate(tokens):
        try:
            s = price_series(tok)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] skip {tok[:10]}… ({e})")
            continue
        # Need enough points for at least a few windows.
        if len(s) >= 24:
            series.append(s)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(tokens)}  kept {len(series)}")
        time.sleep(0.05)  # be polite to the public API

    with open(OUT, "w") as f:
        json.dump(series, f)
    total = sum(len(s) for s in series)
    print(f"wrote {len(series)} series ({total} points) → {OUT}")


if __name__ == "__main__":
    main()
