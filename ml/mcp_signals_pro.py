"""Polymarket resolution-probability MCP server — the flagship.

`pmt-signals-pro` serves the *calibrated resolution model*: given a market's
live price path it estimates P(the YES outcome ultimately resolves true) and
compares that to the current market price to surface mispricing (edge). This is
a separate server from `mcp_server.py` (which serves the short-horizon
direction model) — different name, different model artifacts, different tools.

It loads a local MLX snapshot classifier (resolve_model.safetensors) plus its
normalizer (resolve_normalizer.json), fetches live price history from
Polymarket's public Gamma + CLOB APIs, builds snapshot features inline, and
returns a *calibrated* probability with an edge vs the market. Runs locally on
Apple silicon where MLX is native.

Artifacts live in ml/data/ (produced by the resolution-model training unit):
    resolve_model.safetensors   the trained snapshot classifier
    resolve_normalizer.json     feature mean/std + optional calibration params
    resolve_metrics.json        validation Brier / log-loss / AUC / calibration

Every artifact load is guarded: if a file is missing the server still starts on
a random-init fallback (flagged in model_info) so the process never crashes.

Register with Claude Code:
    claude mcp add pmt-signals-pro -- \
      /path/to/ml/.venv/bin/python /path/to/ml/mcp_signals_pro.py

Self-test (no network, no server):
    /path/to/ml/.venv/bin/python ml/mcp_signals_pro.py --selftest

Tools:
    model_info()                       which resolution model is loaded + metrics
    resolution_signal(token_id, ...)   calibrated P(YES resolves) + edge for one token
    scan_resolution(limit)             active markets ranked by |model prob − price|
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
_UA = {
    "accept": "application/json",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
}

MODEL_PATH = os.path.join(DATA, "resolve_model.safetensors")
NORM_PATH = os.path.join(DATA, "resolve_normalizer.json")
METRICS_PATH = os.path.join(DATA, "resolve_metrics.json")

SNAP_WINDOW = 16          # look-back length for the snapshot
MIN_POINTS = 8            # fewest price points we'll score
_HIDDEN_DEFAULT = 64

# Snapshot feature set — a point-in-time read of the market's path, tuned for
# *resolution* (does YES ultimately win?) rather than next-hour direction. Kept
# inline on purpose: this server must not import sibling training modules, which
# may not have merged. If the loaded normalizer declares its own feature list we
# surface that in model_info, and the vector is length-matched to the model.
SNAPSHOT_FEATURES = [
    "last",          # current price ≈ market-implied P(YES)
    "mean_w",        # mean price over the window
    "drift",         # net move across the window (last − first)
    "vol",           # std of increments (realised vol proxy)
    "momentum",      # mean of the last 4 increments
    "band_z",        # (last − mean) / std — how stretched
    "rsi",           # Wilder RSI recentred to −1..1
    "autocorr",      # lag-1 autocorrelation of increments (trend persistence)
    "stoch_k",       # position of last price in the window range (0..1)
    "hi_lo_range",   # window high − low (how much it has swung)
    "extremeness",   # |last − 0.5| · 2 — how decided the market already is
    "activity",      # sum |increments| — total path length
]


# ── inline snapshot feature extraction ───────────────────────────────────────
def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _autocorr(xs: List[float]) -> float:
    if len(xs) < 3:
        return 0.0
    m = _mean(xs)
    num = sum((xs[i] - m) * (xs[i - 1] - m) for i in range(1, len(xs)))
    den = sum((x - m) ** 2 for x in xs)
    return max(-1.0, min(1.0, num / den)) if den > 1e-12 else 0.0


def _rsi(rets: List[float]) -> float:
    """Wilder RSI over increments, recentred to −1..1 (== (RSI−50)/50)."""
    gains = sum(r for r in rets if r > 0)
    losses = -sum(r for r in rets if r < 0)
    denom = gains + losses
    if denom < 1e-9:
        return 0.0
    return (gains - losses) / denom


def _stoch_k(window: List[float]) -> float:
    lo, hi = min(window), max(window)
    if hi - lo < 1e-9:
        return 0.5
    return (window[-1] - lo) / (hi - lo)


def snapshot_features(prices: List[float]) -> List[float]:
    """Point-in-time feature vector for a price series (probabilities in 0..1).

    Uses the most recent SNAP_WINDOW points. Pure stdlib + math so it stays
    trivially testable and needs no GPU.
    """
    window = prices[-SNAP_WINDOW:]
    rets = [window[i] - window[i - 1] for i in range(1, len(window))]
    mean_w = _mean(window)
    std_w = _std(window)
    last = window[-1]
    band_z = (last - mean_w) / std_w if std_w > 1e-9 else 0.0
    momentum = _mean(rets[-4:]) if len(rets) >= 4 else _mean(rets)
    return [
        last,
        mean_w,
        last - window[0],
        _std(rets),
        momentum,
        band_z,
        _rsi(rets),
        _autocorr(rets),
        _stoch_k(window),
        max(window) - min(window),
        abs(last - 0.5) * 2.0,
        sum(abs(r) for r in rets),
    ]


# ── inline resolution model ──────────────────────────────────────────────────
class ResolveMLP(nn.Module):
    """Snapshot classifier → logit for P(YES resolves).

    Two hidden layers matching the repo's house `FeatureMLP` style (l1/l2/out),
    so weights saved as a flat `dict(tree_flatten(model.parameters()))` load
    back cleanly. Dropout is a no-op in eval mode; kept so parameter names line
    up with a model trained with regularisation.
    """

    def __init__(self, n_in: int, hidden: int = _HIDDEN_DEFAULT, p: float = 0.0):
        super().__init__()
        self.l1 = nn.Linear(n_in, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(p)

    def __call__(self, feat):
        x = self.drop(nn.relu(self.l1(feat)))
        x = self.drop(nn.relu(self.l2(x)))
        return self.out(x)


def _synthetic_norm(n: int) -> dict:
    """Identity normalizer over SNAPSHOT_FEATURES — for the dummy fallback."""
    return {
        "fmean": [0.0] * n,
        "fstd": [1.0] * n,
        "features": list(SNAPSHOT_FEATURES),
        "hidden": _HIDDEN_DEFAULT,
        "calibration": {"method": "identity"},
    }


def _valid_stats(norm: Optional[dict], n: int) -> bool:
    """True iff `norm` carries usable fmean/fstd vectors of length `n`."""
    if not isinstance(norm, dict):
        return False
    fmean, fstd = norm.get("fmean"), norm.get("fstd")
    return (
        isinstance(fmean, list) and isinstance(fstd, list)
        and len(fmean) == n and len(fstd) == n and n > 0
    )


def _load() -> tuple:
    """Load (model, norm, meta), guarding every missing/broken artifact.

    Returns a random-init fallback (meta['dummy']=True) when the trained model
    is absent or fails to load, so the server can still start and --selftest can
    still run. When real weights are present, layer dims are inferred from the
    weight shapes themselves, so this loads correctly regardless of the exact
    hidden size the training unit chose.

    The returned normalizer is *always* consistent with the model's input width:
    if the on-disk normalizer is missing, malformed, or a different length than
    the model expects, an identity normalizer sized to the model is substituted
    (any calibration/feature metadata that *is* present is preserved). This
    keeps the module-level fmean/fstd tensors and the matmul in `_score` in
    lockstep, so neither import nor inference can crash on a bad artifact.
    """
    meta: dict[str, Any] = {"dummy": True, "model_path": MODEL_PATH, "reason": None}

    disk_norm: Optional[dict] = None
    if os.path.exists(NORM_PATH):
        try:
            with open(NORM_PATH) as f:
                disk_norm = json.load(f)
        except Exception as e:  # noqa: BLE001
            meta["reason"] = f"normalizer unreadable: {e}"
    elif meta["reason"] is None:
        meta["reason"] = "no resolve_normalizer.json"

    model = None
    n_in: Optional[int] = None
    hidden = int(disk_norm.get("hidden", _HIDDEN_DEFAULT)) if isinstance(disk_norm, dict) else _HIDDEN_DEFAULT
    if os.path.exists(MODEL_PATH):
        try:
            weights = dict(mx.load(MODEL_PATH).items())
            w1 = weights.get("l1.weight")
            if w1 is None:
                raise ValueError("resolve_model.safetensors missing 'l1.weight' — "
                                 "unexpected architecture")
            hidden, n_in = int(w1.shape[0]), int(w1.shape[1])
            model = ResolveMLP(n_in, hidden)
            model.update(tree_unflatten(list(weights.items())))
            model.eval()
            meta.update(dummy=False, reason=None)
        except Exception as e:  # noqa: BLE001
            model, n_in = None, None
            meta["reason"] = f"model load failed ({e}); using random-init fallback"
    elif meta["reason"] is None or meta["reason"] == "no resolve_normalizer.json":
        meta["reason"] = "no resolve_model.safetensors"

    # Settle the input width: trust the model when we have it, else the on-disk
    # normalizer's fmean length, else our own snapshot vector.
    if n_in is None:
        disk_fmean = disk_norm.get("fmean") if isinstance(disk_norm, dict) else None
        n_in = len(disk_fmean) if isinstance(disk_fmean, list) and disk_fmean else len(SNAPSHOT_FEATURES)

    # Build a normalizer guaranteed to match the input width.
    if _valid_stats(disk_norm, n_in):
        norm = disk_norm
    else:
        norm = _synthetic_norm(n_in)
        if isinstance(disk_norm, dict):          # preserve usable metadata
            if isinstance(disk_norm.get("calibration"), dict):
                norm["calibration"] = disk_norm["calibration"]
            if isinstance(disk_norm.get("features"), list):
                norm["features"] = disk_norm["features"]
            if not meta["dummy"] and meta["reason"] is None:
                meta["reason"] = "normalizer stats missing/mismatched — using identity scaling"

    if model is None:
        model = ResolveMLP(n_in, hidden)
        mx.eval(model.parameters())
        model.eval()

    meta.update(n_in=n_in, hidden=hidden)
    return model, norm, meta


_MODEL, _NORM, _META = _load()
_FMEAN = mx.array(_NORM["fmean"], dtype=mx.float32)
_FSTD = mx.array([s if s else 1.0 for s in _NORM["fstd"]], dtype=mx.float32)
_N_IN = int(_META["n_in"])


def _calibrate(logit: float) -> float:
    """Map a raw logit to a calibrated probability.

    Honours calibration params baked into the normalizer:
      {"method": "temperature", "temperature": T}   → sigmoid(logit / T)
      {"method": "platt", "a": a, "b": b}            → sigmoid(a·logit + b)
    Anything else (or absent) falls back to the plain sigmoid.
    """
    cal = _NORM.get("calibration") or {}
    method = cal.get("method")
    if method == "temperature":
        t = float(cal.get("temperature", 1.0)) or 1.0
        z = logit / t
    elif method == "platt":
        z = float(cal.get("a", 1.0)) * logit + float(cal.get("b", 0.0))
    else:
        z = logit
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def _fit_len(vec: List[float], n: int) -> List[float]:
    """Pad/truncate a feature vector to the model's expected input width, so a
    minor feature-set drift between this server and the training unit degrades
    gracefully instead of crashing."""
    if len(vec) == n:
        return vec
    if len(vec) > n:
        return vec[:n]
    return vec + [0.0] * (n - len(vec))


def _score(prices: List[float]) -> Optional[dict]:
    """Calibrated resolution signal for the latest snapshot of a price series."""
    if len(prices) < MIN_POINTS:
        return None
    feats = snapshot_features(prices)
    x = _fit_len(feats, _N_IN)
    f = (mx.array([x], dtype=mx.float32) - _FMEAN) / _FSTD
    logit = float(_MODEL(f).reshape(-1)[0].item())
    prob = _calibrate(logit)                     # calibrated P(YES resolves)
    market = round(float(prices[-1]), 4)
    edge = prob - market                          # + = model thinks YES underpriced
    # Confidence blends how far the model sits from the market with how far it
    # sits from a coin-flip (conviction). Both matter: a big gap on a 50/50 read
    # is weaker than a big gap the model is sure about.
    conviction = abs(prob - 0.5) * 2.0
    confidence = round(min(1.0, (abs(edge) / 0.15) * (0.5 + 0.5 * conviction)), 4)
    if abs(edge) < 0.02:
        rec = "fair"
    elif edge > 0:
        rec = "buy YES"
    else:
        rec = "buy NO"
    feat_map = {k: round(v, 5) for k, v in zip(SNAPSHOT_FEATURES, feats)}
    return {
        "resolution_probability": round(prob, 4),   # calibrated P(YES resolves)
        "market_price": market,
        "edge": round(edge, 4),
        "edge_pct_pts": round(edge * 100, 2),
        "resolves": "YES" if prob >= 0.5 else "NO",
        "conviction": round(conviction, 4),
        "confidence": confidence,
        "recommendation": rec,
        "calibrated": _NORM.get("calibration", {}).get("method", "sigmoid"),
        "snapshot": feat_map,
    }


# ── live data (Gamma + CLOB), same approach as mcp_server.py ──────────────────
def _get(url: str, tries: int = 2) -> Any:
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"GET failed: {url} :: {last}")


def _history(token_id: str, interval: str = "1w") -> List[float]:
    q = urllib.parse.urlencode({"market": token_id, "interval": interval, "fidelity": "60"})
    d = _get(f"{CLOB}/prices-history?{q}")
    pts = d.get("history", []) if isinstance(d, dict) else []
    return [float(p["p"]) for p in pts if "p" in p]


# ── MCP surface ──────────────────────────────────────────────────────────────
# FastMCP in the 1.x SDK, renamed MCPServer in 2.x; the .tool()/.run() surface
# is identical, so accept either.
try:
    from mcp.server.fastmcp import FastMCP  # SDK 1.x  # noqa: E402
except ImportError:  # SDK 2.x
    from mcp.server.mcpserver import MCPServer as FastMCP  # noqa: E402

mcp = FastMCP("pmt-signals-pro")


@mcp.tool()
def model_info() -> dict:
    """Which resolution model is loaded and how it scored in validation.

    Reports the served artifact's status plus its Brier / log-loss / AUC and
    calibration numbers from resolve_metrics.json when present.
    """
    metrics = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
        except Exception as e:  # noqa: BLE001
            metrics = {"error": f"metrics unreadable: {e}"}
    return {
        "server": "pmt-signals-pro",
        "flagship": "calibrated resolution-probability model",
        "artifacts_present": not _META["dummy"],
        "status": "loaded" if not _META["dummy"] else f"fallback ({_META['reason']})",
        "model_path": MODEL_PATH,
        "input_features": _N_IN,
        "hidden": _META["hidden"],
        "features": _NORM.get("features", SNAPSHOT_FEATURES),
        "calibration": _NORM.get("calibration", {"method": "sigmoid"}),
        "validation": {
            k: metrics.get(k)
            for k in ("brier", "log_loss", "logloss", "auc", "val_auc", "calibration", "ece")
            if k in metrics
        },
        "metrics": metrics,
        "note": "resolution_probability is the calibrated P(the YES outcome resolves true). "
        "edge = model prob − current market price; positive means YES looks underpriced.",
    }


@mcp.tool()
def resolution_signal(token_id: str, interval: str = "1w") -> dict:
    """Calibrated resolution signal for one Polymarket outcome, by CLOB token id.

    Fetches live price history, builds a snapshot, and returns the model's
    calibrated P(YES resolves), the current market price, the edge between them,
    and a confidence read.
    """
    prices = _history(token_id, interval)
    sig = _score(prices)
    if sig is None:
        return {"error": f"not enough history for {token_id} ({len(prices)} points)"}
    return {"token_id": token_id, "points": len(prices), **sig}


@mcp.tool()
def scan_resolution(limit: int = 12) -> dict:
    """Scan active markets and rank them by mispricing (|model prob − price|).

    Returns the markets where the calibrated resolution model most disagrees
    with the current price — the biggest edges — each with the question,
    current price, model probability and recommended side.
    """
    limit = max(1, min(limit, 30))
    q = urllib.parse.urlencode(
        {"limit": limit, "order": "volume24hr", "ascending": "false",
         "active": "true", "closed": "false"}
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
                "resolution_probability": sig["resolution_probability"],
                "market_price": sig["market_price"],
                "edge": sig["edge"],
                "recommendation": sig["recommendation"],
                "confidence": sig["confidence"],
            }
        )
    out.sort(key=lambda r: abs(r["edge"]), reverse=True)
    return {"scanned": len(markets), "mispriced": out}


def _selftest() -> int:
    """Offline smoke test: score a synthetic snapshot, print the signal, exit 0.

    Loads the real artifacts if present, otherwise the random-init fallback.
    Never starts the server and never touches the network.
    """
    mx.random.seed(7)
    # Synthetic YES-token path drifting up toward resolution.
    prices = [0.42 + 0.02 * i + 0.01 * math.sin(i / 2.0) for i in range(SNAP_WINDOW + 4)]
    sig = _score(prices)
    report = {
        "selftest": "ok",
        "artifacts_present": not _META["dummy"],
        "loader_status": _META["reason"] or "loaded",
        "input_features": _N_IN,
        "signal": sig,
    }
    print(json.dumps(report, indent=2))
    ok = sig is not None and 0.0 <= sig["resolution_probability"] <= 1.0
    if not ok:
        print("SELFTEST FAILED: invalid signal", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    mcp.run()
