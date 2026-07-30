"""End-to-end local signal-pipeline orchestrator — one command, no cloud.

Everything in `ml/` produces a *piece* of the picture:

  * `fetch_sii`      — samples Polymarket trades → `data/sii_series.json` +
                       `data/sii_resolve.json` (synthetic fallback offline).
  * `features_all`   — folds the six feature families into one unified vector.
  * `train_resolve`  — the local MLX resolution model → `resolve_model.safetensors`
                       + `resolve_metrics.json`.
  * `evaluate_all`   — ranks models (calling `backtest` for realized PnL/Sharpe)
                       → `data/leaderboard.json`.
  * `signal_engine`  — turns model predictions into tradeable BUY/HOLD signals.

This module is the single entry point that runs those pieces **in order** so a
user can reproduce the whole local pipeline with one command. It does not
reimplement anything: heavy steps are shelled out to the module's own CLI, and
the pure-Python steps import the merged module and call it directly. Every stage
is guarded and skippable — when a dependency, artifact, or the network is
missing the stage degrades gracefully and says so, rather than aborting the run.

Run:
    python ml/run_all.py --dry-run     # print the 5-stage plan, touch nothing
    python ml/run_all.py --sample      # run the whole pipeline at smoke scale
    python ml/run_all.py --sample --skip train   # skip a stage by name

Pure stdlib (+ subprocess / import of the merged modules). No numpy/pandas.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# Status vocabulary for a stage's outcome. RAN = did real work; SKIPPED = a
# precondition was absent and we honestly stepped over it; FAILED = it tried and
# errored (still non-fatal to the pipeline); PLANNED = --dry-run, nothing ran.
RAN = "ran"
SKIPPED = "skipped"
FAILED = "failed"
PLANNED = "planned"


@dataclass
class StageResult:
    name: str
    status: str
    detail: str = ""
    artifacts: List[str] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class Ctx:
    """Shared run context. Redirectable `here`/`python` make stages unit-testable."""

    here: str = HERE
    python: str = sys.executable
    dry_run: bool = False
    sample: bool = True
    epochs: int = 6
    top: int = 5
    results: Dict[str, StageResult] = field(default_factory=dict)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.here, "data")

    def data_path(self, name: str) -> str:
        return os.path.join(self.data_dir, name)


# ── small helpers ─────────────────────────────────────────────────────────────

def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _banner(n: int, total: int, name: str, headline: str) -> None:
    _log()
    _log(f"[{n}/{total}] {name.upper()} — {headline}")
    _log("-" * 72)


def _module_importable(python: str, module: str) -> bool:
    """True if `module` imports under `python`. Used to gate the MLX train step
    without importing a heavy dependency into this orchestrator's own process."""
    try:
        r = subprocess.run(
            [python, "-c", f"import {module}"],
            cwd=HERE, capture_output=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


def _run_script(ctx: Ctx, script: str, args: List[str], timeout: int = 900) -> subprocess.CompletedProcess:
    """Run a merged module's own CLI as a subprocess (streams via inherited stdio)."""
    cmd = [ctx.python, os.path.join(ctx.here, script), *args]
    _log(f"$ {' '.join([os.path.basename(ctx.python), script, *args])}")
    return subprocess.run(cmd, cwd=ctx.here, timeout=timeout)


def _existing(paths: List[str]) -> List[str]:
    return [p for p in paths if os.path.exists(p)]


# ── stage 1: data ─────────────────────────────────────────────────────────────

def run_data(ctx: Ctx) -> StageResult:
    outs = [ctx.data_path("sii_series.json"), ctx.data_path("sii_resolve.json")]
    script = os.path.join(ctx.here, "fetch_sii.py")
    if not os.path.exists(script):
        return StageResult("data", SKIPPED, "fetch_sii.py not found", [])
    t0 = time.time()
    try:
        # --sample streams a small row cap; fetch_sii falls back to a synthetic
        # fixture on any HF/network failure, so this stage works fully offline.
        proc = _run_script(ctx, "fetch_sii.py", ["--sample"])
    except Exception as e:  # noqa: BLE001
        return StageResult("data", FAILED, f"{type(e).__name__}: {e}", [], time.time() - t0)
    dt = time.time() - t0
    made = _existing(outs)
    if proc.returncode != 0 or not made:
        return StageResult("data", FAILED, f"exit {proc.returncode}", made, dt)
    return StageResult("data", RAN, f"sampled trades → {len(made)} json artifact(s)", made, dt)


# ── stage 2: features ─────────────────────────────────────────────────────────

def _feature_context_from_sample(ctx: Ctx) -> Dict:
    """Build a features_all context. Prefer a real sampled resolve snapshot so the
    resolve family reflects live sampled data; fall back to the module's own
    synthetic fully-populated context (shared with its selfcheck) otherwise."""
    resolve_path = ctx.data_path("sii_resolve.json")
    if os.path.exists(resolve_path):
        try:
            with open(resolve_path) as f:
                rows = json.load(f)
            if rows:
                r = rows[0]
                # Map sampled snapshot fields onto snapshot_features' tolerant keys.
                # (Deliberately no `label` — this is an inference-side feature
                # vector, so the resolution target never enters the context.)
                snap = {
                    "price": r.get("price", 0.5),
                    "signed_flow": r.get("recent_flow", 0.0),
                    "volume": r.get("cum_volume", 0.0),
                }
                return {"snapshot": snap, "_source": "sampled sii_resolve.json"}
        except Exception:  # noqa: BLE001
            pass
    try:
        import features_all  # noqa: E402
        ctx_synth = features_all._synth_context()  # type: ignore[attr-defined]
        ctx_synth["_source"] = "synthetic fully-populated context"
        return ctx_synth
    except Exception:  # noqa: BLE001
        return {"_source": "empty (neutral zero-fill)"}


def run_features(ctx: Ctx) -> StageResult:
    if ctx.here not in sys.path:
        sys.path.insert(0, ctx.here)
    t0 = time.time()
    try:
        import features_all  # noqa: E402
    except Exception as e:  # noqa: BLE001
        return StageResult("features", SKIPPED, f"features_all import failed ({type(e).__name__})", [])
    fctx = _feature_context_from_sample(ctx)
    source = fctx.pop("_source", "unknown")
    try:
        vec = features_all.build_all(fctx)
    except Exception as e:  # noqa: BLE001
        return StageResult("features", FAILED, f"{type(e).__name__}: {e}", [], time.time() - t0)
    dt = time.time() - t0
    names = getattr(features_all, "ALL_FEATURES", None)
    nlen = len(names) if names else len(vec)
    nonzero = sum(1 for x in vec if x != 0.0)
    _log(f"built unified vector: {len(vec)}/{nlen} features "
         f"({nonzero} non-zero) from {source}")
    return StageResult("features", RAN, f"{len(vec)}-dim vector, {nonzero} non-zero ({source})", [], dt)


# ── stage 3: train ────────────────────────────────────────────────────────────

def _resolve_is_pricepath(path: str) -> bool:
    """True if sii_resolve.json is in the price-path *market* shape train_resolve
    consumes (has a 'prices' array), rather than fetch_sii's flat snapshot rows."""
    try:
        with open(path) as f:
            data = json.load(f)
        rows = data.get("markets", data) if isinstance(data, dict) else data
        return bool(rows) and isinstance(rows[0], dict) and "prices" in rows[0]
    except Exception:  # noqa: BLE001
        return False


def run_train(ctx: Ctx) -> StageResult:
    outs = [
        ctx.data_path("resolve_model.safetensors"),
        ctx.data_path("resolve_normalizer.json"),
        ctx.data_path("resolve_metrics.json"),
    ]
    script = os.path.join(ctx.here, "train_resolve.py")
    if not os.path.exists(script):
        return StageResult("train", SKIPPED, "train_resolve.py not found", [])
    if not _module_importable(ctx.python, "mlx.core"):
        return StageResult("train", SKIPPED, "mlx not installed (local MLX train unavailable)", [])

    # The sampled sii_resolve.json holds flat snapshot rows, not the price-path
    # markets train_resolve slides over — so for a reliable smoke run we train on
    # the bundled synthetic fixture unless a price-path resolve set is present.
    resolve_path = ctx.data_path("sii_resolve.json")
    use_fixture = not _resolve_is_pricepath(resolve_path)
    args = ["--epochs", str(ctx.epochs)]
    why = ""
    if use_fixture:
        args = ["--fixture", *args]
        why = " (bundled fixture — sampled resolve rows are snapshots, not price paths)"

    t0 = time.time()
    try:
        proc = _run_script(ctx, "train_resolve.py", args)
    except Exception as e:  # noqa: BLE001
        return StageResult("train", FAILED, f"{type(e).__name__}: {e}", [], time.time() - t0)
    dt = time.time() - t0
    made = _existing(outs)
    if proc.returncode != 0 or not made:
        return StageResult("train", FAILED, f"exit {proc.returncode}", made, dt)
    return StageResult("train", RAN, f"MLX resolve model trained{why}", made, dt)


# ── stage 4: evaluate ─────────────────────────────────────────────────────────

def run_evaluate(ctx: Ctx) -> StageResult:
    out = ctx.data_path("leaderboard.json")
    script = os.path.join(ctx.here, "evaluate_all.py")
    if not os.path.exists(script):
        return StageResult("evaluate", SKIPPED, "evaluate_all.py not found", [])
    t0 = time.time()
    try:
        # evaluate_all --report ranks the model suite (calling backtest.run_backtest
        # per model for realized PnL/Sharpe) and writes the leaderboard.
        proc = _run_script(ctx, "evaluate_all.py", ["--report"])
    except Exception as e:  # noqa: BLE001
        return StageResult("evaluate", FAILED, f"{type(e).__name__}: {e}", [], time.time() - t0)
    dt = time.time() - t0
    made = _existing([out])
    if proc.returncode != 0 or not made:
        return StageResult("evaluate", FAILED, f"exit {proc.returncode}", made, dt)

    # Tie backtest in explicitly on a holdout, printing one realized-trading line.
    _demo_backtest(ctx)
    return StageResult("evaluate", RAN, "ranked model suite → leaderboard", made, dt)


def _demo_backtest(ctx: Ctx) -> None:
    """Run backtest on a small informed holdout and print a one-line summary,
    demonstrating the realized-trading step evaluate_all folds in per model."""
    if ctx.here not in sys.path:
        sys.path.insert(0, ctx.here)
    try:
        import backtest  # noqa: E402
        try:
            import evaluate_all  # noqa: E402
            holdout = evaluate_all._synth_holdout(400, seed=1, noise=0.05)  # type: ignore[attr-defined]
            src = "evaluate_all synthetic holdout"
        except Exception:  # noqa: BLE001
            holdout = backtest._synth(400, seed=1, informed=True)  # type: ignore[attr-defined]
            src = "backtest synthetic holdout"
        res = backtest.run_backtest(holdout, train_frac=0.6)
        _log(f"backtest ({src}): sharpe={res.sharpe:.3f} pnl={res.pnl:.1f} "
             f"trades={res.n_trades} max_dd={res.max_drawdown:.2f}")
    except Exception as e:  # noqa: BLE001
        _log(f"backtest demo unavailable ({type(e).__name__}: {e})")


# ── stage 5: signal ───────────────────────────────────────────────────────────

def run_signal(ctx: Ctx) -> StageResult:
    if ctx.here not in sys.path:
        sys.path.insert(0, ctx.here)
    t0 = time.time()
    try:
        import signal_engine  # noqa: E402
    except Exception as e:  # noqa: BLE001
        return StageResult("signal", SKIPPED, f"signal_engine import failed ({type(e).__name__})", [])

    # A few demo markets: base-model predictions vs the live market price. One
    # carries a genuinely-informed price/outcome history so its confidence is
    # fused with a real backtested reliability, exactly as the terminal uses it.
    try:
        history = signal_engine._informed_history(60, seed=3)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        history = None
    markets = [
        {"id": "mkt-strong-yes", "market_price": 0.42,
         "model_preds": {"resolve": 0.71, "flow": 0.68, "smart": 0.66}, "history": history},
        {"id": "mkt-lean-no", "market_price": 0.63,
         "model_preds": {"resolve": 0.44, "flow": 0.47, "smart": 0.40}},
        {"id": "mkt-agree-hold", "market_price": 0.50,
         "model_preds": {"resolve": 0.51, "flow": 0.49, "smart": 0.50}},
        {"id": "mkt-split", "market_price": 0.55,
         "model_preds": {"resolve": 0.72, "flow": 0.33, "smart": 0.58}},
    ]
    try:
        ranked = signal_engine.rank_signals(markets)
    except Exception as e:  # noqa: BLE001
        return StageResult("signal", FAILED, f"{type(e).__name__}: {e}", [], time.time() - t0)
    dt = time.time() - t0

    top = ranked[: ctx.top]
    _log(f"top {len(top)} signal(s) by risk-adjusted edge:")
    _log(f"  {'market':<18} {'call':<8} {'prob':>6} {'price':>6} {'edge':>7} {'conf':>6} {'score':>7}")
    for s in top:
        _log(f"  {str(s.market_id):<18} {s.direction:<8} {s.prob:>6.3f} "
             f"{s.market_price:>6.3f} {s.edge:>+7.3f} {s.confidence:>6.3f} {s.score:>7.4f}")

    artifacts: List[str] = []
    try:
        out = ctx.data_path("top_signals.json")
        os.makedirs(ctx.data_dir, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                [{"market_id": s.market_id, "direction": s.direction, "prob": round(s.prob, 6),
                  "market_price": s.market_price, "edge": round(s.edge, 6),
                  "confidence": round(s.confidence, 6), "score": round(s.score, 6)} for s in ranked],
                f, indent=2,
            )
        artifacts = [out]
    except Exception:  # noqa: BLE001
        pass
    live = sum(1 for s in ranked if s.direction != signal_engine.HOLD)
    return StageResult("signal", RAN, f"{live}/{len(ranked)} live calls, {len(top)} shown", artifacts, dt)


# ── pipeline registry ─────────────────────────────────────────────────────────

# (name, headline, runner, expected-artifact basenames) — the single ordered
# source of truth shared by the plan (dry-run) and the real run.
PIPELINE: List = [
    ("data", "sample trades → sii_series / sii_resolve", run_data,
     ["sii_series.json", "sii_resolve.json"]),
    ("features", "fold six families → unified feature vector", run_features, []),
    ("train", "local MLX resolve model → safetensors + metrics", run_train,
     ["resolve_model.safetensors", "resolve_normalizer.json", "resolve_metrics.json"]),
    ("evaluate", "rank models + backtest → leaderboard", run_evaluate,
     ["leaderboard.json"]),
    ("signal", "predictions → tradeable BUY/HOLD signals", run_signal,
     ["top_signals.json"]),
]

STAGES: List[str] = [name for name, *_ in PIPELINE]


def print_plan(ctx: Ctx, skip: Optional[set] = None) -> None:
    skip = skip or set()
    _log("PLAN — local signal pipeline (no cloud), in order:")
    _log(f"  python : {ctx.python}")
    _log(f"  cwd    : {ctx.here}")
    _log(f"  mode   : {'sample' if ctx.sample else 'full'}  (dry-run — nothing will run)")
    _log()
    for i, (name, headline, _runner, arts) in enumerate(PIPELINE, 1):
        tag = "  (skipped by --skip)" if name in skip else ""
        _log(f"  {i}. {name:<9} {headline}{tag}")
        if arts:
            _log(f"       → artifacts: {', '.join(arts)}")
    _log()
    _log("no side effects performed (--dry-run).")


def _summary(ctx: Ctx) -> None:
    _log()
    _log("=" * 72)
    _log("PIPELINE SUMMARY")
    _log("=" * 72)
    for name, _headline, _runner, _arts in PIPELINE:
        r = ctx.results.get(name)
        if r is None:
            _log(f"  {name:<9} -        (not run)")
            continue
        _log(f"  {name:<9} {r.status:<8} {r.detail}  [{r.seconds:.2f}s]")

    _log()
    _log("Artifacts produced:")
    seen = set()
    any_art = False
    for name, _h, _r, _a in PIPELINE:
        r = ctx.results.get(name)
        for p in (r.artifacts if r else []):
            if p in seen:
                continue
            seen.add(p)
            if os.path.exists(p):
                size = os.path.getsize(p)
                _log(f"  {os.path.relpath(p, ctx.here):<40} {size:>8} bytes")
                any_art = True
    if not any_art:
        _log("  (none — every artifact-producing stage was skipped)")


def run_pipeline(ctx: Ctx, skip: Optional[set] = None) -> int:
    skip = skip or set()
    total = len(PIPELINE)
    _log("=" * 72)
    _log("polymarket-terminal — end-to-end local signal pipeline")
    _log("=" * 72)
    for i, (name, headline, runner, _arts) in enumerate(PIPELINE, 1):
        _banner(i, total, name, headline)
        if name in skip:
            ctx.results[name] = StageResult(name, SKIPPED, "skipped by --skip")
            _log("skipped (--skip)")
            continue
        try:
            res = runner(ctx)
        except Exception as e:  # noqa: BLE001 — a stage must never abort the pipeline
            res = StageResult(name, FAILED, f"unexpected {type(e).__name__}: {e}")
        ctx.results[name] = res
        _log(f"→ {res.status.upper()}: {res.detail}")

    _summary(ctx)
    # Exit 0 as long as the orchestration completed; skips/failures are honestly
    # reported, not fatal — the pipeline is meant to degrade gracefully.
    return 0


def build_ctx(args: argparse.Namespace) -> Ctx:
    return Ctx(
        here=HERE,
        python=args.python or sys.executable,
        dry_run=bool(args.dry_run),
        sample=True,  # smoke scale is the only implemented mode
        epochs=args.epochs,
        top=args.top,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true",
                    help="run the pipeline at fast smoke scale (default behaviour)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the 5-stage plan and exit without side effects")
    ap.add_argument("--skip", action="append", default=[], metavar="STAGE",
                    help=f"skip a stage by name (one of: {', '.join(STAGES)}); repeatable")
    ap.add_argument("--epochs", type=int, default=6,
                    help="MLX train epochs for the smoke train (default 6)")
    ap.add_argument("--top", type=int, default=5, help="how many top signals to print")
    ap.add_argument("--python", default=None,
                    help="interpreter for the subprocess steps (default: this one)")
    args = ap.parse_args(argv)

    bad = [s for s in args.skip if s not in STAGES]
    if bad:
        ap.error(f"unknown --skip stage(s): {', '.join(bad)} (valid: {', '.join(STAGES)})")
    skip = set(args.skip)

    ctx = build_ctx(args)
    if args.dry_run:
        print_plan(ctx, skip)
        return 0
    return run_pipeline(ctx, skip)


if __name__ == "__main__":
    raise SystemExit(main())
