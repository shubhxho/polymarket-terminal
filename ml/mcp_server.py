"""Polymarket signal MCP server.

Serves the trained signal model over the Model Context Protocol, so any MCP
client (Claude Code, the terminal, an agent) can ask for live trading signals.
It loads the local MLX sequence model (train_seq.py output), fetches live price
history from Polymarket's public CLOB, and returns a calibrated up-probability
plus the microstructure features behind it.

Runs locally on Apple silicon, where MLX is native.

Register with Claude Code:
    claude mcp add pmt-signals -- \
      /path/to/ml/.venv/bin/python /path/to/ml/mcp_server.py

Tools:
    market_signal(token_id)   signal for one market's CLOB token
    scan_signals(limit)       top markets ranked by model conviction
    model_info()              what model is loaded and how it scored
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

import mlx.core as mx
from mlx.utils import tree_unflatten

from features import FEATURE_NAMES, WINDOW, window_features
from train_seq import FeatureMLP, SeqGRU

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
_UA = {
    "accept": "application/json",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
}

# ── Load the trained model once ──────────────────────────────────────────────
with open(os.path.join(DATA, "seq_normalizer.json")) as f:
    NORM = json.load(f)
_Model = SeqGRU if NORM["winner"] == "seq_gru" else FeatureMLP
_model = _Model(len(FEATURE_NAMES))
_model.update(tree_unflatten(list(mx.load(os.path.join(DATA, "seq_model.safetensors")).items())))
_model.eval()
_FMEAN = mx.array(NORM["fmean"], dtype=mx.float32)
_FSTD = mx.array(NORM["fstd"], dtype=mx.float32)
_RSTD = float(NORM["rstd"])


def _get(url: str, tries: int = 2) -> Any:
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"GET failed: {url} :: {last}")


def _history(token_id: str, interval: str = "1w") -> list[float]:
    q = urllib.parse.urlencode({"market": token_id, "interval": interval, "fidelity": "60"})
    d = _get(f"{CLOB}/prices-history?{q}")
    pts = d.get("history", []) if isinstance(d, dict) else []
    return [float(p["p"]) for p in pts if "p" in p]


def _oscillators(feat_map: dict) -> dict:
    """Human read of the classic oscillators, so a caller sees *why* — RSI and
    CCI especially, the two the terminal leans on for over-extension."""
    rsi = feat_map.get("rsi", 0.0)          # -1..1  (== (RSI-50)/50)
    cci = feat_map.get("cci", 0.0) * 100    # undo the /100 scaling
    stoch = feat_map.get("stoch_k", 0.5)
    rsi_100 = round(rsi * 50 + 50, 1)
    return {
        "rsi": rsi_100,
        "rsi_state": "overbought" if rsi_100 >= 70 else "oversold" if rsi_100 <= 30 else "neutral",
        "cci": round(cci, 1),
        "cci_state": "overbought" if cci >= 100 else "oversold" if cci <= -100 else "neutral",
        "stoch_k": round(stoch, 3),
        "macd_hist": round(feat_map.get("macd_hist", 0.0), 6),
    }


def _score(prices: list[float]) -> dict | None:
    """Model up-probability + features for the latest window of a price series."""
    if len(prices) < WINDOW + 1:
        return None
    w = prices[-WINDOW:]
    rets = [w[i] - w[i - 1] for i in range(1, len(w))]
    feat = window_features(w)
    seq = mx.array([[[r] for r in rets]], dtype=mx.float32) / _RSTD
    f = (mx.array([feat], dtype=mx.float32) - _FMEAN) / _FSTD
    p = float(mx.sigmoid(_model(seq, f)).item())
    feat_map = {k: round(v, 5) for k, v in zip(FEATURE_NAMES, feat)}
    return {
        "up_probability": round(p, 4),
        "direction": "bullish" if p > 0.5 else "bearish",
        "conviction": round(abs(p - 0.5) * 2, 4),
        "last_price": round(w[-1], 4),
        "oscillators": _oscillators(feat_map),
        "features": feat_map,
    }


# ── MCP surface ──────────────────────────────────────────────────────────────
# The decorator server was FastMCP in the 1.x SDK and was renamed MCPServer in
# 2.x; the .tool()/.run() surface is identical, so accept either.
try:
    from mcp.server.fastmcp import FastMCP  # SDK 1.x  # noqa: E402
except ImportError:  # SDK 2.x
    from mcp.server.mcpserver import MCPServer as FastMCP  # noqa: E402

mcp = FastMCP("pmt-signals")


@mcp.tool()
def model_info() -> dict:
    """Which signal model is loaded and how it scored in validation."""
    metrics = {}
    mp = os.path.join(DATA, "seq_metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            metrics = json.load(f)
    return {
        "model": NORM["winner"],
        "features": FEATURE_NAMES,
        "window": WINDOW,
        "validation": metrics.get("models", {}),
        "walk_forward": metrics.get("walk_forward", {}),
        "note": "up_probability is the model's estimate that price rises over the next few hours. "
        "walk_forward AUC across time folds shows whether the signal persists (>0.5 = it does).",
    }


@mcp.tool()
def market_signal(token_id: str, interval: str = "1w") -> dict:
    """Signal for one Polymarket outcome, by its CLOB token id.

    Returns the model's up-probability, direction, conviction and the
    microstructure features behind the call.
    """
    prices = _history(token_id, interval)
    sig = _score(prices)
    if sig is None:
        return {"error": f"not enough history for {token_id} ({len(prices)} points)"}
    return {"token_id": token_id, "points": len(prices), **sig}


@mcp.tool()
def scan_signals(limit: int = 12) -> dict:
    """Scan the highest-volume markets and rank them by model conviction.

    Returns the strongest directional signals right now, each with the market
    question, current price, direction and up-probability.
    """
    limit = max(1, min(limit, 30))
    q = urllib.parse.urlencode(
        {"limit": limit, "order": "volume24hr", "ascending": "false", "active": "true", "closed": "false"}
    )
    markets = _get(f"{GAMMA}/markets?{q}")
    out = []
    for m in markets:
        raw = m.get("clobTokenIds")
        ids = json.loads(raw) if isinstance(raw, str) else raw
        if not ids:
            continue
        try:
            sig = _score(_history(str(ids[0])))
        except Exception:  # noqa: BLE001
            sig = None
        if not sig:
            continue
        out.append(
            {
                "question": str(m.get("question", ""))[:90],
                "token_id": str(ids[0]),
                "direction": sig["direction"],
                "up_probability": sig["up_probability"],
                "conviction": sig["conviction"],
                "last_price": sig["last_price"],
            }
        )
    out.sort(key=lambda r: r["conviction"], reverse=True)
    return {"scanned": len(markets), "signals": out}


if __name__ == "__main__":
    mcp.run()
