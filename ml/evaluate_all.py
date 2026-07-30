"""Model leaderboard — evaluate every signal model on a common out-of-time holdout.

The suite has many models (resolution, order-flow, smart-money, ensemble, the
big-data "mega" model, the sequence transformer, …), each shipping its own
``data/*_metrics.json`` with the validation numbers it was trained to report.
Those numbers are not comparable across models: each was measured on a different
split, sometimes after a different calibration, and "reported" is not the same as
"reproduced". This harness answers *which model gives the best signal* with
evidence rather than vibes — it runs every model through the **same** merged
``backtest.run_backtest`` (realized P&L, Sharpe, drawdown, calibration) on the
**same** holdout, adds the classification metrics a trading backtest doesn't
cover (AUC, Brier, log-loss, ECE calibration error), surfaces each model's own
reported numbers next to the freshly reproduced ones, flags where the two
disagree, and emits a ``Leaderboard`` ranked by risk-adjusted realized return.

Ranking key: **risk-adjusted realized return** (the backtest Sharpe — mean net
return per trade divided by its volatility), tie-broken by **Brier** (lower is
better). A model that merely ranks markets well (high AUC) but whose edge does
not survive fees will sit below one that actually makes risk-adjusted money.

A per-model holdout is a list of ``backtest`` records — 4-tuples
``(predicted_prob, market_price, realized_outcome, timestamp)``. The realized
trading metrics come straight from ``backtest.run_backtest`` (imported, never
reimplemented); only the classification metrics — which a P&L backtest does not
produce — are computed here, in pure stdlib.

Run:
  python ml/evaluate_all.py            # selfcheck: strong > mediocre > random
  python ml/evaluate_all.py --report   # + writes data/leaderboard.json, prints table
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Import the merged signal-quality harness — its realized-trading metrics
# (PnL / Sharpe / drawdown / calibration / decile) are the source of truth and
# are never reimplemented here.
from backtest import BacktestResult, Record, run_backtest

# Default place the models drop their reported metrics.
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Canonical model-name → reported-metrics filename. Names not listed here fall
# back to ``{model}_metrics.json``; a missing file is handled gracefully (the
# reproduced holdout eval still runs, reported columns are just blank).
DEFAULT_METRICS_FILES: Dict[str, str] = {
    "resolution": "resolve_metrics.json",
    "resolve": "resolve_metrics.json",
    "mega": "bigdata_metrics.json",
    "bigdata": "bigdata_metrics.json",
    "transformer": "seq_metrics.json",
    "seq": "seq_metrics.json",
    "ensemble": "ohlcv_metrics.json",
    "ohlcv": "ohlcv_metrics.json",
    "flow": "modal_flow_metrics.json",
    "smart": "multilingual_metrics.json",
    "multilingual": "multilingual_metrics.json",
    "distill": "distill_metrics.json",
    "chronos": "chronos_metrics.json",
    "modal": "modal_metrics.json",
    "base": "metrics.json",
}

# Tolerances above which reported-vs-reproduced is flagged as a disagreement.
DEFAULT_AUC_TOL = 0.05
DEFAULT_BRIER_TOL = 0.05


# ── classification metrics (stdlib; these are NOT backtest's metrics) ─────────

def _auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Area under the ROC curve via the rank-sum (Mann–Whitney) identity, with
    average ranks so tied scores are handled correctly. Returns 0.5 when one
    class is absent (AUC is undefined — treat as no discrimination)."""
    data = sorted(zip(scores, labels), key=lambda t: t[0])
    n = len(data)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and data[j][0] == data[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based average rank of the tie group
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    n_pos = sum(1 for _, y in data if y > 0.5)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, data) if y > 0.5)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _brier(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Mean squared error between predicted probability and 0/1 outcome."""
    if not scores:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(scores, labels)) / len(scores)


def _log_loss(scores: Sequence[float], labels: Sequence[float], eps: float = 1e-12) -> float:
    """Binary cross-entropy; predictions clamped into [eps, 1-eps] so a confident
    miss is a large-but-finite penalty rather than infinity."""
    if not scores:
        return float("nan")
    total = 0.0
    for p, y in zip(scores, labels):
        p = min(1.0 - eps, max(eps, p))
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(scores)


def _ece(scores: Sequence[float], labels: Sequence[float], nbins: int = 10) -> float:
    """Expected Calibration Error: bin predictions into ``nbins`` equal-width
    probability buckets and take the sample-weighted mean gap between each
    bucket's mean predicted probability and its realized YES rate."""
    if not scores:
        return float("nan")
    sums_pred = [0.0] * nbins
    sums_out = [0.0] * nbins
    counts = [0] * nbins
    for p, y in zip(scores, labels):
        idx = min(nbins - 1, max(0, int(p * nbins)))
        sums_pred[idx] += p
        sums_out[idx] += y
        counts[idx] += 1
    n = len(scores)
    ece = 0.0
    for b in range(nbins):
        if counts[b] == 0:
            continue
        mean_pred = sums_pred[b] / counts[b]
        frac_pos = sums_out[b] / counts[b]
        ece += (counts[b] / n) * abs(mean_pred - frac_pos)
    return ece


# ── reported-metrics extraction from a model's *_metrics.json ─────────────────

_AUC_KEYS = ("val_auc", "auc", "fusion_auc", "oot_auc")
_BRIER_KEYS = ("brier", "val_brier", "raw_brier")
_LOGLOSS_KEYS = ("logloss", "log_loss", "val_bce", "bce", "raw_logloss")


def _finite(v: object) -> Optional[float]:
    """Coerce to a finite float or return None (so missing/NaN reads are blank)."""
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _first_metric(scopes: Sequence[Mapping], keys: Sequence[str]) -> Optional[float]:
    """First finite value found across candidate scopes (in priority order) for
    any of ``keys``."""
    for scope in scopes:
        if not isinstance(scope, Mapping):
            continue
        for k in keys:
            if k in scope:
                v = _finite(scope[k])
                if v is not None:
                    return v
    return None


def _reported_from_obj(obj: Mapping) -> Dict[str, Optional[float]]:
    """Best-effort {auc, brier, logloss} from a parsed metrics JSON, following
    winner/overall_best hints into the ``models`` block and common nested
    sub-model keys before falling back to the top level."""
    scopes: List[Mapping] = []
    models = obj.get("models") if isinstance(obj, Mapping) else None
    winner = None
    for hint in ("overall_best", "winner", "served_nn_winner"):
        if isinstance(obj.get(hint), str):
            winner = obj[hint]
            break
    if isinstance(models, Mapping) and isinstance(models.get(winner), Mapping):
        scopes.append(models[winner])
    for key in ("ensemble", "fine_tuned", "student_distilled", "gbdt", "neural"):
        sub = obj.get(key)
        if isinstance(sub, Mapping):
            scopes.append(sub)
    scopes.append(obj)  # top level, lowest priority
    return {
        "auc": _first_metric(scopes, _AUC_KEYS),
        "brier": _first_metric(scopes, _BRIER_KEYS),
        "logloss": _first_metric(scopes, _LOGLOSS_KEYS),
    }


def _load_reported(
    model: str, data_dir: str, metrics_files: Mapping[str, str]
) -> Tuple[Optional[str], Dict[str, Optional[float]]]:
    """Locate and parse a model's reported metrics. Returns (path_or_None, dict).
    Missing file, unreadable file, or malformed JSON all degrade gracefully to
    (None-ish path, all-None metrics) — never an exception."""
    fname = metrics_files.get(model, f"{model}_metrics.json")
    path = os.path.join(data_dir, fname)
    empty = {"auc": None, "brier": None, "logloss": None}
    if not os.path.isfile(path):
        return None, empty
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError):
        return path, empty
    if not isinstance(obj, Mapping):
        return path, empty
    return path, _reported_from_obj(obj)


# ── result containers ─────────────────────────────────────────────────────────

@dataclass
class ModelEval:
    """One model's fresh holdout evaluation plus its own reported numbers."""

    model: str
    n_records: int
    # reproduced classification metrics on the common holdout
    auc: float
    brier: float
    log_loss: float
    ece: float
    # reproduced realized-trading metrics (straight from backtest.run_backtest)
    sharpe: float
    pnl: float
    mean_return: float
    max_drawdown: float
    n_trades: int
    hit_rate: float
    # the model's own reported validation numbers (None when unavailable)
    metrics_file: Optional[str] = None
    reported_auc: Optional[float] = None
    reported_brier: Optional[float] = None
    reported_logloss: Optional[float] = None
    # reported-vs-reproduced disagreement
    auc_delta: Optional[float] = None
    brier_delta: Optional[float] = None
    disagrees: bool = False
    rank: int = 0
    backtest: Optional[BacktestResult] = field(default=None, repr=False)

    def as_dict(self) -> dict:
        d = {
            "rank": self.rank,
            "model": self.model,
            "n_records": self.n_records,
            "sharpe": _round(self.sharpe, 4),
            "pnl": _round(self.pnl, 4),
            "mean_return": _round(self.mean_return, 5),
            "max_drawdown": _round(self.max_drawdown, 4),
            "n_trades": self.n_trades,
            "hit_rate": _round(self.hit_rate, 4),
            "auc": _round(self.auc, 4),
            "brier": _round(self.brier, 4),
            "log_loss": _round(self.log_loss, 4),
            "ece": _round(self.ece, 4),
            "metrics_file": (
                os.path.basename(self.metrics_file) if self.metrics_file else None
            ),
            "reported_auc": _round(self.reported_auc, 4),
            "reported_brier": _round(self.reported_brier, 4),
            "reported_logloss": _round(self.reported_logloss, 4),
            "auc_delta": _round(self.auc_delta, 4),
            "brier_delta": _round(self.brier_delta, 4),
            "disagrees": self.disagrees,
        }
        return d


@dataclass
class Leaderboard:
    """Ranked models plus the parameters the evaluation was run under."""

    ranked: List[ModelEval]
    fee: float
    threshold: float
    train_frac: Optional[float]
    ece_bins: int
    auc_tol: float
    brier_tol: float

    @property
    def best(self) -> Optional[ModelEval]:
        return self.ranked[0] if self.ranked else None

    def as_dict(self) -> dict:
        return {
            "params": {
                "fee": self.fee,
                "threshold": self.threshold,
                "train_frac": self.train_frac,
                "ece_bins": self.ece_bins,
                "auc_tol": self.auc_tol,
                "brier_tol": self.brier_tol,
            },
            "ranking_key": "risk-adjusted realized return (sharpe), tie-break brier",
            "n_models": len(self.ranked),
            "best_model": self.best.model if self.best else None,
            "leaderboard": [m.as_dict() for m in self.ranked],
        }


def _round(v: Optional[float], n: int) -> Optional[float]:
    # Non-finite (NaN/inf) collapses to None so the JSON we emit stays valid for
    # strict parsers (json.dump would otherwise write the non-standard `NaN`).
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return round(v, n)


# ── public entry point ────────────────────────────────────────────────────────

def evaluate_all(
    holdout_records_by_model: Mapping[str, Sequence[Record]],
    *,
    data_dir: Optional[str] = None,
    metrics_files: Optional[Mapping[str, str]] = None,
    threshold: float = 0.0,
    fee: float = 0.01,
    train_frac: Optional[float] = None,
    ece_bins: int = 10,
    auc_tol: float = DEFAULT_AUC_TOL,
    brier_tol: float = DEFAULT_BRIER_TOL,
) -> Leaderboard:
    """Evaluate every model on its holdout and return a ranked ``Leaderboard``.

    ``holdout_records_by_model`` maps a model name to that model's holdout —
    a list of ``backtest`` records ``(predicted_prob, market_price, outcome, ts)``.
    For a like-for-like comparison the caller should pass the SAME underlying
    markets/outcomes to each model (same holdout), differing only in the model's
    predicted probability.

    For each model:
      * run the merged ``backtest.run_backtest`` for realized trading quality
        (PnL, Sharpe, drawdown …) — Sharpe is the risk-adjusted-return rank key;
      * compute classification metrics a P&L backtest does not (AUC, Brier,
        log-loss, ECE);
      * read the model's ``data/*_metrics.json`` if present and surface its
        reported AUC/Brier/log-loss, flagging disagreement with the reproduced
        numbers beyond ``auc_tol`` / ``brier_tol``.

    The ``Leaderboard`` is sorted by risk-adjusted realized return (Sharpe,
    descending), ties broken by Brier (ascending), then model name for a stable
    order. Raises ``ValueError`` on an empty model set or an empty holdout.
    """
    if not holdout_records_by_model:
        raise ValueError("evaluate_all: no models supplied")
    ddir = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    mfiles = dict(DEFAULT_METRICS_FILES)
    if metrics_files:
        mfiles.update(metrics_files)

    evals: List[ModelEval] = []
    for model, records in holdout_records_by_model.items():
        if not records:
            raise ValueError(f"evaluate_all: model {model!r} has an empty holdout")

        # Realized trading quality — delegated wholesale to the merged harness.
        bt = run_backtest(records, threshold=threshold, fee=fee, train_frac=train_frac)

        # Classification metrics on the same holdout records.
        scores = [r[0] for r in records]
        labels = [r[2] for r in records]
        auc = _auc(scores, labels)
        brier = _brier(scores, labels)
        ll = _log_loss(scores, labels)
        ece = _ece(scores, labels, nbins=ece_bins)

        # Reported numbers + disagreement flag.
        mfile, rep = _load_reported(model, ddir, mfiles)
        auc_delta = (auc - rep["auc"]) if rep["auc"] is not None else None
        brier_delta = (brier - rep["brier"]) if rep["brier"] is not None else None
        disagrees = bool(
            (auc_delta is not None and abs(auc_delta) > auc_tol)
            or (brier_delta is not None and abs(brier_delta) > brier_tol)
        )

        evals.append(
            ModelEval(
                model=model,
                n_records=len(records),
                auc=auc,
                brier=brier,
                log_loss=ll,
                ece=ece,
                sharpe=bt.sharpe,
                pnl=bt.pnl,
                mean_return=bt.mean_return,
                max_drawdown=bt.max_drawdown,
                n_trades=bt.n_trades,
                hit_rate=bt.hit_rate,
                metrics_file=mfile,
                reported_auc=rep["auc"],
                reported_brier=rep["brier"],
                reported_logloss=rep["logloss"],
                auc_delta=auc_delta,
                brier_delta=brier_delta,
                disagrees=disagrees,
                backtest=bt,
            )
        )

    # Risk-adjusted realized return first (higher Sharpe wins); Brier breaks ties
    # (lower is better); model name keeps the order deterministic.
    evals.sort(key=lambda m: (-m.sharpe, m.brier, m.model))
    for i, m in enumerate(evals, start=1):
        m.rank = i

    return Leaderboard(
        ranked=evals,
        fee=fee,
        threshold=threshold,
        train_frac=train_frac,
        ece_bins=ece_bins,
        auc_tol=auc_tol,
        brier_tol=brier_tol,
    )


# ── human-readable rendering ──────────────────────────────────────────────────

def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and not math.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def render(leaderboard: Leaderboard) -> str:
    """Markdown table of the leaderboard, ranked best-first. A ``!`` in the flag
    column marks a model whose reported metrics disagree with the reproduced
    holdout eval beyond tolerance."""
    headers = [
        "#", "model", "sharpe", "pnl", "mean_ret", "max_dd",
        "auc", "brier", "logloss", "ece", "rep_auc", "Δauc", "flag",
    ]
    rows: List[List[str]] = []
    for m in leaderboard.ranked:
        rows.append([
            str(m.rank),
            m.model,
            _fmt(m.sharpe),
            _fmt(m.pnl),
            _fmt(m.mean_return, 5),
            _fmt(m.max_drawdown),
            _fmt(m.auc),
            _fmt(m.brier),
            _fmt(m.log_loss),
            _fmt(m.ece),
            _fmt(m.reported_auc),
            _fmt(m.auc_delta),
            "!" if m.disagrees else "",
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    best = leaderboard.best
    title = (
        f"Model leaderboard — {len(leaderboard.ranked)} models, "
        f"ranked by risk-adjusted realized return (Sharpe; tie-break Brier). "
        f"fee={leaderboard.fee}"
    )
    if best is not None:
        title += f"\nBest signal: **{best.model}** (Sharpe {_fmt(best.sharpe)}, AUC {_fmt(best.auc)})"
    return "\n".join([title, "", line(headers), sep, *(line(r) for r in rows)])


# ── synthetic per-model holdouts for the selfcheck / --report demo ────────────

def _synth_holdout(n: int, seed: int, noise: Optional[float]) -> List[Record]:
    """A synthetic per-model holdout on a shared market. Each market has a hidden
    true probability; the market price is a noisy quote around it and the outcome
    is Bernoulli(true). ``noise`` sets how close the model's predicted prob sits
    to the truth — small noise = a strong signal, large noise = mediocre, and
    ``None`` = a random model quoting noise unrelated to the truth (no edge)."""
    import random

    rng = random.Random(seed)
    out: List[Record] = []
    for t in range(n):
        true = rng.uniform(0.05, 0.95)
        price = min(0.99, max(0.01, true + rng.gauss(0.0, 0.10)))
        if noise is None:
            prob = rng.uniform(0.01, 0.99)
        else:
            prob = min(0.99, max(0.01, true + rng.gauss(0.0, noise)))
        outcome = 1.0 if rng.random() < true else 0.0
        out.append((prob, price, outcome, float(t)))
    return out


def _synthetic_suite(n: int = 6000) -> Dict[str, List[Record]]:
    """A strong, a mediocre and a random model on the same synthetic holdout."""
    return {
        "strong": _synth_holdout(n, seed=1, noise=0.03),
        "mediocre": _synth_holdout(n, seed=1, noise=0.14),
        "random": _synth_holdout(n, seed=1, noise=None),
    }


def _selfcheck() -> Leaderboard:
    """Rank a strong / mediocre / random model and assert the ordering holds."""
    board = evaluate_all(_synthetic_suite(), threshold=0.02, fee=0.01)
    order = [m.model for m in board.ranked]
    assert order[0] == "strong", f"strong model must rank #1, got {order}"
    assert order[-1] == "random", f"random model must rank last, got {order}"
    strong = board.ranked[0]
    random_m = board.ranked[-1]
    assert strong.sharpe > random_m.sharpe, "strong must beat random on Sharpe"
    assert strong.auc > random_m.auc, "strong must out-rank random on AUC"
    for m in board.ranked:
        for v in (m.sharpe, m.pnl, m.auc, m.brier, m.log_loss, m.ece):
            assert math.isfinite(v), f"{m.model} produced a non-finite metric"
    assert strong.auc > 0.6, f"strong AUC unexpectedly low: {strong.auc}"
    assert abs(random_m.auc - 0.5) < 0.05, f"random AUC not ~0.5: {random_m.auc}"
    return board


def _report(path: Optional[str] = None) -> Leaderboard:
    """Run the synthetic suite, print the table, and write ``leaderboard.json``."""
    board = _selfcheck()
    out_path = path or os.path.join(DEFAULT_DATA_DIR, "leaderboard.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(board.as_dict(), fh, indent=2)
    print(render(board))
    print(f"\nwrote {out_path}")
    return board


if __name__ == "__main__":
    import sys

    if "--report" in sys.argv[1:]:
        _report()
    else:
        board = _selfcheck()
        print(render(board))
        print("\nselfcheck OK — strong ranks #1, random ranks last, all metrics finite.")
