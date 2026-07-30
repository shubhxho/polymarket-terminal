"""Polymarket blended-signal MCP server — the fused best-signal.

`pmt-ensemble` serves ONE calibrated signal fused from the suite's independent
base models. Where `mcp_signals_pro.py` serves the resolution model alone (and
`mcp_server.py` the short-horizon direction model), this server scores every
base model that is available, folds their probabilities into a single estimate
via the merged `ensemble.blend` combiner, and reports the blend's calibrated
probability, its edge versus the live market price, and the agreement /
confidence that make the call trustworthy.

The base models this server knows about (order mirrors `ensemble.MODELS`):

    resolve   calibrated P(the YES outcome ultimately resolves true)
    flow      short-horizon direction from the aggressor order-flow tape
    smart     the smart-money cohort's lean

Each base model is a snapshot MLP with its own weights + normalizer. This server
loads every one it can find (`resolve_*`, `flow_*`, `smart_*` in ml/data/),
guarding missing/broken artifacts: an absent model simply does not contribute,
and the ensemble renormalises over whichever models are present (see
`ensemble.blend`). Only the merged `ensemble` module is imported — never a
sibling wave-3 training/MCP unit, which may not have merged.

Feature extraction is inline (pure stdlib + math): a single point-in-time
snapshot of the price path, length-matched per model to whatever input width its
weights declare. This mirrors `mcp_signals_pro.py` exactly and keeps the server
independent of the training modules. (The flow / smart models were trained on
order-flow and wallet tapes that are not available live here; when their
artifacts are present this server scores them off the price snapshot as a proxy.
Absent — the normal case today — they never enter the blend at all.)

Calibration is honoured per base model. A normalizer may carry either an
isotonic table (`calibration` = a list of `[pred, actual]` pairs, as the shipped
resolve model does) or a parametric map (`{"method": "temperature"|"platt", ...}`);
both are applied, falling back to a plain sigmoid when neither is present.

Artifacts live in ml/data/ (produced by the base-model training units):
    resolve_model.safetensors / resolve_normalizer.json   (+ *_metrics.json)
    flow_model.safetensors    / flow_normalizer.json       (optional)
    smart_model.safetensors   / smart_normalizer.json      (optional)

Register with Claude Code:
    claude mcp add pmt-ensemble -- \
      /path/to/ml/.venv/bin/python /path/to/ml/mcp_ensemble.py

Self-test (no network, no server):
    /path/to/ml/.venv/bin/python ml/mcp_ensemble.py --selftest

Tools:
    model_info()                    which base models are loaded + ensemble config
    best_signal(token_id, ...)      blended calibrated P(YES) + edge + confidence
    scan_best(limit)                active markets ranked by risk-adjusted edge
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

# The merged stdlib combiner — the ONLY sibling module we import. Never a
# wave-3 training/MCP unit (they may not have merged). cwd must be ml/ so this
# resolves; the registration command above runs the file by absolute path from
# ml/, and --selftest is run the same way.
import ensemble

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
_UA = {
    "accept": "application/json",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
}

# Base models this server fuses, keyed by artifact prefix in ml/data/. Order and
# names mirror `ensemble.MODELS` so a blend contribution maps straight back to a
# model. Kept in sync defensively: if the merged ensemble ever grows a model we
# don't have artifacts for, it just never contributes.
BASE_MODELS: List[str] = list(ensemble.MODELS)

SNAP_WINDOW = 16          # look-back length for the snapshot
MIN_POINTS = 8            # fewest price points we'll score
_HIDDEN_DEFAULT = 64
_EDGE_FAIR = 0.02         # |edge| under this reads as "fair" (no side)

# Snapshot feature set — a point-in-time read of the market's path. Inlined on
# purpose (this server must not import sibling training modules) and length-
# matched per base model in `_score_model`, so it degrades gracefully whatever
# input width a given model's weights declare.
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


# ── inline base-model architecture ────────────────────────────────────────────
class SnapshotMLP(nn.Module):
    """Snapshot classifier → logit for P(YES). Matches the repo's house
    `FeatureMLP` style (l1/l2/out) so weights saved as a flat
    `dict(tree_flatten(model.parameters()))` load back cleanly regardless of the
    hidden width the training unit chose (dims are read from the weight shapes).
    Dropout is a no-op in eval mode; kept so parameter names line up with a model
    trained with regularisation."""

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


def _paths(prefix: str) -> Tuple[str, str, str]:
    return (
        os.path.join(DATA, f"{prefix}_model.safetensors"),
        os.path.join(DATA, f"{prefix}_normalizer.json"),
        os.path.join(DATA, f"{prefix}_metrics.json"),
    )


def _synthetic_norm(n: int) -> dict:
    """Identity normalizer of width `n` — for the dummy fallback."""
    return {
        "fmean": [0.0] * n,
        "fstd": [1.0] * n,
        "features": list(SNAPSHOT_FEATURES[:n]) if n <= len(SNAPSHOT_FEATURES) else None,
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


def _load_base(prefix: str) -> dict:
    """Load one base model into a state dict, guarding every missing/broken
    artifact so import can never crash.

    Returns {model, fmean, fstd, norm, n_in, hidden, dummy, reason, model_path}.
    An absent/broken model yields a random-init fallback flagged `dummy=True`
    (it still scores, but callers exclude it from the real blend). Layer dims are
    read from the weight shapes, and the normalizer is always length-matched to
    the model's input width (identity fallback, preserving any calibration /
    feature metadata that is present) so the matmul in `_score_model` can never
    crash on a mismatched artifact.
    """
    model_path, norm_path, _ = _paths(prefix)
    reason: Optional[str] = None

    disk_norm: Optional[dict] = None
    if os.path.exists(norm_path):
        try:
            with open(norm_path) as f:
                disk_norm = json.load(f)
        except Exception as e:  # noqa: BLE001
            reason = f"normalizer unreadable: {e}"

    model = None
    dummy = True
    n_in: Optional[int] = None
    hidden = int(disk_norm.get("hidden", _HIDDEN_DEFAULT)) if isinstance(disk_norm, dict) else _HIDDEN_DEFAULT
    if os.path.exists(model_path):
        try:
            weights = dict(mx.load(model_path).items())
            w1 = weights.get("l1.weight")
            if w1 is None:
                raise ValueError(f"{prefix}_model.safetensors missing 'l1.weight' — "
                                 "unexpected architecture")
            hidden, n_in = int(w1.shape[0]), int(w1.shape[1])
            model = SnapshotMLP(n_in, hidden)
            model.update(tree_unflatten(list(weights.items())))
            model.eval()
            dummy, reason = False, None
        except Exception as e:  # noqa: BLE001
            model, n_in = None, None
            reason = f"model load failed ({e}); random-init fallback"
    elif reason is None:
        reason = f"no {prefix}_model.safetensors"

    # Settle the input width: trust the model, else the on-disk normalizer's
    # fmean length, else our own snapshot vector.
    if n_in is None:
        disk_fmean = disk_norm.get("fmean") if isinstance(disk_norm, dict) else None
        n_in = len(disk_fmean) if isinstance(disk_fmean, list) and disk_fmean else len(SNAPSHOT_FEATURES)

    if _valid_stats(disk_norm, n_in):
        norm = disk_norm
    else:
        norm = _synthetic_norm(n_in)
        if isinstance(disk_norm, dict):          # preserve usable metadata
            if disk_norm.get("calibration") is not None:
                norm["calibration"] = disk_norm["calibration"]
            if isinstance(disk_norm.get("features"), list):
                norm["features"] = disk_norm["features"]
            if not dummy and reason is None:
                reason = "normalizer stats missing/mismatched — identity scaling"

    if model is None:
        model = SnapshotMLP(n_in, hidden)
        mx.eval(model.parameters())
        model.eval()

    return {
        "model": model,
        "fmean": mx.array(norm["fmean"], dtype=mx.float32),
        "fstd": mx.array([s if s else 1.0 for s in norm["fstd"]], dtype=mx.float32),
        "norm": norm,
        "n_in": int(n_in),
        "hidden": int(hidden),
        "dummy": dummy,
        "reason": reason,
        "model_path": model_path,
    }


# Load every base model once at import (guarded — never raises).
_STATE: Dict[str, dict] = {name: _load_base(name) for name in BASE_MODELS}
# Equal-weight prior — the sane default the ensemble module itself documents. No
# fitted stacker artifact is shipped whose semantics match this stdlib combiner
# (modal_ensemble's ensemble_normalizer.json z-scores base probs, a different
# pipeline), so we deliberately do not load one.
_STACK = ensemble.StackWeights.default()


def _isotonic(x: float, table: List) -> float:
    """Piecewise-linear interpolation of `x` through an isotonic calibration
    table of `[pred, actual]` pairs (sorted by pred). Clamped at the ends."""
    pts = [
        (float(p[0]), float(p[1]))
        for p in table
        if isinstance(p, (list, tuple)) and len(p) >= 2
        and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float))
    ]
    if not pts:
        return x
    pts.sort(key=lambda t: t[0])
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        if x <= x1:
            span = x1 - x0
            return y0 if span < 1e-12 else y0 + (y1 - y0) * (x - x0) / span
    return pts[-1][1]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def _calibrate(norm: dict, logit: float) -> float:
    """Map a raw logit to a calibrated probability, honouring the normalizer's
    calibration spec. Supports three forms:
      list of [pred, actual]         → isotonic map of sigmoid(logit)
      {"method":"temperature",T}     → sigmoid(logit / T)
      {"method":"platt","a","b"}     → sigmoid(a·logit + b)
    Anything else (or absent) → plain sigmoid(logit)."""
    cal = norm.get("calibration")
    if isinstance(cal, list) and cal:
        return max(0.0, min(1.0, _isotonic(_sigmoid(logit), cal)))
    if isinstance(cal, dict):
        method = cal.get("method")
        if method == "temperature":
            t = float(cal.get("temperature", 1.0)) or 1.0
            return _sigmoid(logit / t)
        if method == "platt":
            return _sigmoid(float(cal.get("a", 1.0)) * logit + float(cal.get("b", 0.0)))
    return _sigmoid(logit)


def _calibration_kind(norm: dict) -> str:
    cal = norm.get("calibration")
    if isinstance(cal, list) and cal:
        return "isotonic"
    if isinstance(cal, dict):
        return str(cal.get("method", "sigmoid"))
    return "sigmoid"


def _fit_len(vec: List[float], n: int) -> List[float]:
    """Pad/truncate a feature vector to a model's expected input width so minor
    feature-set drift degrades gracefully instead of crashing."""
    if len(vec) == n:
        return vec
    if len(vec) > n:
        return vec[:n]
    return vec + [0.0] * (n - len(vec))


def _score_model(name: str, feats: List[float]) -> Optional[float]:
    """Calibrated P(YES) from one base model for a precomputed snapshot vector."""
    st = _STATE.get(name)
    if st is None:
        return None
    x = _fit_len(feats, st["n_in"])
    f = (mx.array([x], dtype=mx.float32) - st["fmean"]) / st["fstd"]
    logit = float(st["model"](f).reshape(-1)[0].item())
    return _calibrate(st["norm"], logit)


# ── the blended signal ────────────────────────────────────────────────────────
def _blend_signal(prices: List[float], include_dummy: bool = False) -> Optional[dict]:
    """Fuse every available base model into one calibrated signal for a price
    path. `include_dummy` lets the selftest exercise multi-model fusion even when
    only the random-init fallbacks are loaded; production callers keep it False so
    a random dummy never pollutes a real blend.
    """
    if len(prices) < MIN_POINTS:
        return None
    feats = snapshot_features(prices)

    preds: Dict[str, float] = {}
    per_model: List[dict] = []
    for name in BASE_MODELS:
        st = _STATE[name]
        if st["dummy"] and not include_dummy:
            continue
        p = _score_model(name, feats)
        if p is None:
            continue
        preds[name] = p
        per_model.append({
            "model": name,
            "label": ensemble.MODEL_LABELS.get(name, name),
            "prob": round(p, 4),
            "dummy": st["dummy"],
        })

    sig = ensemble.blend(preds, weights=_STACK)
    prob = sig.prob
    market = round(float(prices[-1]), 4)
    edge = prob - market                              # + = model thinks YES underpriced
    if not sig.contributing:
        rec = "no-signal"
    elif abs(edge) < _EDGE_FAIR:
        rec = "fair"
    elif edge > 0:
        rec = "buy YES"
    else:
        rec = "buy NO"
    # Risk-adjusted edge discounts the raw edge by how much we trust the blend —
    # a big edge from a shaky, disagreeing set of models is worth less than a
    # smaller edge everyone concurs on. This is the scan_best ranking key.
    risk_adj_edge = edge * sig.confidence
    return {
        "blended_probability": round(prob, 4),        # calibrated fused P(YES)
        "market_price": market,
        "edge": round(edge, 4),
        "edge_pct_pts": round(edge * 100, 2),
        "risk_adjusted_edge": round(risk_adj_edge, 4),
        "resolves": None if not sig.contributing else ("YES" if prob >= 0.5 else "NO"),
        "direction": rec,
        "confidence": round(sig.confidence, 4),
        "agreement": round(sig.agreement, 4),
        "contributing": list(sig.contributing),
        "n_models": len(sig.contributing),
        "per_model": per_model,
        "snapshot": {k: round(v, 5) for k, v in zip(SNAPSHOT_FEATURES, feats)},
    }


# ── live data (Gamma + CLOB), same approach as mcp_signals_pro.py ─────────────
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

mcp = FastMCP("pmt-ensemble")


def _base_status(name: str) -> dict:
    st = _STATE[name]
    return {
        "model": name,
        "label": ensemble.MODEL_LABELS.get(name, name),
        "loaded": not st["dummy"],
        "status": "loaded" if not st["dummy"] else f"fallback ({st['reason']})",
        "model_path": st["model_path"],
        "input_features": st["n_in"],
        "hidden": st["hidden"],
        "features": st["norm"].get("features"),
        "calibration": _calibration_kind(st["norm"]),
    }


@mcp.tool()
def model_info() -> dict:
    """Which base models are loaded and how the ensemble fuses them.

    Reports each base model's load status (real artifacts vs random-init
    fallback), its input width / calibration, and the ensemble configuration —
    the models in play, the stacker (equal-weight prior unless a fitted one is
    supplied), and how the blended signal is defined.
    """
    bases = [_base_status(name) for name in BASE_MODELS]
    loaded = [b["model"] for b in bases if b["loaded"]]
    return {
        "server": "pmt-ensemble",
        "role": "blended best-signal — fuses the base models via ensemble.blend",
        "base_models": bases,
        "loaded_models": loaded,
        "any_loaded": bool(loaded),
        "ensemble": {
            "models": list(ensemble.MODELS),
            "labels": ensemble.MODEL_LABELS,
            "stacker": "fitted" if _STACK.fitted else "default (equal weights)",
            "weights": {k: round(v, 6) for k, v in _STACK.weights.items()},
            "bias": round(_STACK.bias, 6),
        },
        "note": "blended_probability is the calibrated fused P(YES). edge = blend "
        "prob − market price (positive → YES underpriced). confidence is agreement-"
        "gated: it falls when the base models disagree, are wishy-washy, or are "
        "missing. risk_adjusted_edge = edge × confidence (the scan_best ranking).",
    }


@mcp.tool()
def best_signal(token_id: str, interval: str = "1w") -> dict:
    """Blended best-signal for one Polymarket outcome, by CLOB token id.

    Fetches live price history, builds a snapshot, scores every available base
    model, and fuses them via `ensemble.blend`. Returns the calibrated fused
    P(YES), the market price, the edge between them, the recommended side, and
    the agreement / confidence behind the call.
    """
    prices = _history(token_id, interval)
    sig = _blend_signal(prices)
    if sig is None:
        return {"error": f"not enough history for {token_id} ({len(prices)} points)"}
    return {"token_id": token_id, "points": len(prices), **sig}


@mcp.tool()
def scan_best(limit: int = 12) -> dict:
    """Scan active markets and rank them by risk-adjusted edge.

    Scores each market's YES token through the blend and ranks by
    |edge × confidence| — the biggest edges the base models most agree on — so
    the top of the list is the best risk-adjusted opportunities, not just the
    loudest disagreements with the market.
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
            sig = _blend_signal(_history(str(ids[0])))
        except Exception:  # noqa: BLE001
            sig = None
        if not sig or not sig["contributing"]:
            continue
        out.append(
            {
                "question": str(m.get("question", ""))[:90],
                "token_id": str(ids[0]),
                "blended_probability": sig["blended_probability"],
                "market_price": sig["market_price"],
                "edge": sig["edge"],
                "risk_adjusted_edge": sig["risk_adjusted_edge"],
                "direction": sig["direction"],
                "confidence": sig["confidence"],
                "agreement": sig["agreement"],
                "n_models": sig["n_models"],
            }
        )
    out.sort(key=lambda r: abs(r["risk_adjusted_edge"]), reverse=True)
    return {"scanned": len(markets), "opportunities": out}


def _selftest() -> int:
    """Offline smoke test: score a synthetic snapshot through every base model,
    fuse via `ensemble.blend`, print the blended signal, exit 0.

    Loads real artifacts where present and random-init fallbacks otherwise, and
    (uniquely) folds the fallbacks in too, so the fusion path is exercised end to
    end even before flow/smart ship. Never starts the server, never touches the
    network.
    """
    mx.random.seed(7)
    # Synthetic YES-token path drifting up toward resolution.
    prices = [0.42 + 0.02 * i + 0.01 * math.sin(i / 2.0) for i in range(SNAP_WINDOW + 4)]
    sig = _blend_signal(prices, include_dummy=True)
    report = {
        "selftest": "ok",
        "loaded_base_models": [n for n in BASE_MODELS if not _STATE[n]["dummy"]],
        "fallback_base_models": [n for n in BASE_MODELS if _STATE[n]["dummy"]],
        "stacker": "fitted" if _STACK.fitted else "default (equal weights)",
        "signal": sig,
    }
    print(json.dumps(report, indent=2))
    ok = (
        sig is not None
        and 0.0 <= sig["blended_probability"] <= 1.0
        and 0.0 <= sig["confidence"] <= 1.0
        and 0.0 <= sig["agreement"] <= 1.0
        and sig["contributing"]  # the blend actually fused something
    )
    if not ok:
        print("SELFTEST FAILED: invalid blended signal", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    mcp.run()
