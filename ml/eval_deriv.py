"""Rank the price-derivative family out-of-time to find the single best signal.

`features_deriv.py` builds a whole family of derivative candidates (raw finite
differences, denoised Savitzky–Golay velocity/acceleration/jerk, and the
volatility-normalised variants). This harness answers *which derivative is the
best signal* with the same honesty bar the rest of `ml/` uses: strictly
**out-of-time** validation (train on each market's earlier windows, score its
later ones, with a HORIZON purge — mirrors `train_seq._split`), plus a 4-fold
**walk-forward** so we see stability across epochs rather than a one-off fit.

Each candidate is a single scalar per window, so "how good a signal is it" is its
univariate **AUC** against the up-label (price higher HORIZON steps ahead). AUC is
direction-agnostic: a value that predicts *down* lands below 0.5, so signal
*strength* is |AUC−0.5| and the sign tells you whether the derivative is a
momentum tell (AUC>0.5) or a mean-reversion tell (AUC<0.5). We also report Pearson
correlation with the signed forward return and the up-rate spread between the top
and bottom deciles of the candidate (the tradeable separation).

Ranking key: **out-of-time |AUC−0.5|**, required to hold up in walk-forward
(reported as the min fold AUC-strength) so a lucky single slice can't win.

Run:
  ml/.venv/bin/python eval_deriv.py                 # ranks on data/series.json
  ml/.venv/bin/python eval_deriv.py --data X.json   # any [[float,…], …] series file
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from features import WINDOW, HORIZON, MIN_STD, _std  # same windowing contract
from features_deriv import (
    DERIV_NAMES,
    DERIV_SIGNAL_FEATURES,
    deriv_features,
    load_signal_model,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(_HERE, "data", "series.json")


# ── sample building (mirrors features.series_to_rich, but keeps the raw window) ─

@dataclass
class Row:
    feats: Dict[str, float]
    label: int
    fwd: float


def _series_rows(prices: Sequence[float]) -> List[Row]:
    """Sliding (derivative-features, up-label, forward-return) rows for one series.
    Flat windows are dropped exactly as `series_to_samples` does."""
    out: List[Row] = []
    n = len(prices)
    for i in range(WINDOW, n - HORIZON):
        window = prices[i - WINDOW : i]
        incs = [window[k] - window[k - 1] for k in range(1, len(window))]
        if _std(incs) < MIN_STD:
            continue
        fwd = prices[i + HORIZON] - prices[i]
        out.append(Row(deriv_features(window), 1 if fwd > 0 else 0, fwd))
    return out


# ── metrics (stdlib) ───────────────────────────────────────────────────────────

def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC via the Mann–Whitney rank-sum with average ranks for ties."""
    data = sorted(zip(scores, labels), key=lambda t: t[0])
    n = len(data)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and data[j][0] == data[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    n_pos = sum(1 for _, y in data if y > 0)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, data) if y > 0)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    d = math.sqrt(sxx * syy)
    return sxy / d if d > 1e-12 else 0.0


def _decile_spread(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Up-rate in the top decile of the candidate minus the up-rate in the bottom
    decile — the separation a trader could actually act on. Sign follows the
    candidate's own direction (top decile = most-positive values)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    k = max(1, len(order) // 10)
    bot = order[:k]
    top = order[-k:]
    up = lambda idx: sum(labels[i] for i in idx) / len(idx)
    return up(top) - up(bot)


# ── splits (mirror train_seq) ──────────────────────────────────────────────────

def _oot_split(rows_by_series: List[List[Row]], val_frac: float = 0.2
               ) -> Tuple[List[Row], List[Row]]:
    tr, va = [], []
    for rows in rows_by_series:
        if len(rows) < 5:
            tr.extend(rows)
            continue
        cut = int(len(rows) * (1 - val_frac))
        tr.extend(rows[: max(0, cut - HORIZON)])
        va.extend(rows[cut:])
    return tr, va


def _walk_forward(rows_by_series: List[List[Row]], folds: int = 4):
    blocks = folds + 1
    for k in range(1, blocks):
        va: List[Row] = []
        for rows in rows_by_series:
            if len(rows) < blocks:
                continue
            step = len(rows) / blocks
            va.extend(rows[int(step * k): int(step * (k + 1))])
        if va:
            yield k, va


# ── candidate evaluation ───────────────────────────────────────────────────────

@dataclass
class DerivEval:
    name: str
    auc: float            # out-of-time AUC (>0.5 momentum, <0.5 mean-reversion)
    strength: float       # |auc-0.5|
    corr: float           # Pearson(candidate, forward return)
    spread: float         # top-decile minus bottom-decile up-rate
    wf_min_strength: float  # worst-fold |auc-0.5| across walk-forward
    wf_mean_auc: float
    direction: str        # "momentum" / "mean-revert" / "flat"


def evaluate(series: List[List[float]]) -> Tuple[List[DerivEval], int, int, float]:
    rows_by_series = [_series_rows(p) for p in series]
    rows_by_series = [r for r in rows_by_series if r]
    tr, va = _oot_split(rows_by_series)
    if not va:
        raise ValueError("no out-of-time validation rows — series too short")
    up_rate = sum(r.label for r in va) / len(va)

    wf = list(_walk_forward(rows_by_series))

    evals: List[DerivEval] = []
    for name in DERIV_NAMES:
        v_scores = [r.feats[name] for r in va]
        v_labels = [r.label for r in va]
        v_fwd = [r.fwd for r in va]
        auc = _auc(v_scores, v_labels)
        wf_strengths = []
        wf_aucs = []
        for _, fold in wf:
            a = _auc([r.feats[name] for r in fold], [r.label for r in fold])
            wf_strengths.append(abs(a - 0.5))
            wf_aucs.append(a)
        strength = abs(auc - 0.5)
        direction = ("momentum" if auc > 0.5 + 1e-6
                     else "mean-revert" if auc < 0.5 - 1e-6 else "flat")
        evals.append(DerivEval(
            name=name,
            auc=auc,
            strength=strength,
            corr=_pearson(v_scores, v_fwd),
            spread=_decile_spread(v_scores, v_labels),
            wf_min_strength=min(wf_strengths) if wf_strengths else 0.0,
            wf_mean_auc=sum(wf_aucs) / len(wf_aucs) if wf_aucs else 0.5,
            direction=direction,
        ))
    # Best = strongest out-of-time signal that also survives every walk-forward fold.
    evals.sort(key=lambda e: (min(e.strength, e.wf_min_strength), e.strength), reverse=True)

    # The trained COMBINER (all derivatives folded into one logistic) — scored on
    # the same holdout so it sits in the same table, pinned first when present.
    model = load_signal_model()
    if model is not None:
        def _combi(feats: Dict[str, float]) -> float:
            mean, std, w = model["mean"], model["std"], model["weights"]
            z = float(model["bias"])
            for i, nm in enumerate(model["features"]):
                s = std[i] if std[i] > 1e-12 else 1.0
                z += w[i] * ((feats[nm] - mean[i]) / s)
            return z  # monotone in prob → AUC-equivalent, no sigmoid needed
        c_scores = [_combi(r.feats) for r in va]
        auc = _auc(c_scores, [r.label for r in va])
        wf_str, wf_aucs = [], []
        for _, fold in wf:
            a = _auc([_combi(r.feats) for r in fold], [r.label for r in fold])
            wf_str.append(abs(a - 0.5))
            wf_aucs.append(a)
        combiner = DerivEval(
            name="COMBINED (deriv_signal)",
            auc=auc, strength=abs(auc - 0.5),
            corr=_pearson(c_scores, [r.fwd for r in va]),
            spread=_decile_spread(c_scores, [r.label for r in va]),
            wf_min_strength=min(wf_str) if wf_str else 0.0,
            wf_mean_auc=sum(wf_aucs) / len(wf_aucs) if wf_aucs else 0.5,
            direction="combined",
        )
        evals.insert(0, combiner)
    return evals, len(tr), len(va), up_rate


def _render(evals: List[DerivEval], n_tr: int, n_va: int, up_rate: float) -> str:
    lines = [
        f"Derivative-signal leaderboard — out-of-time AUC on {n_va} val rows "
        f"({n_tr} train), base up-rate {up_rate:.3f}.",
        "Ranked by min(OOT strength, worst walk-forward fold). "
        "AUC>0.5 momentum, <0.5 mean-reversion.",
        "",
    ]
    hdr = ["#", "candidate", "auc", "strength", "wf_min", "wf_mean", "corr_fwd", "decile_spread", "reads"]
    rows = [[
        str(i), e.name, f"{e.auc:.4f}", f"{e.strength:.4f}", f"{e.wf_min_strength:.4f}",
        f"{e.wf_mean_auc:.4f}", f"{e.corr:+.4f}", f"{e.spread:+.4f}", e.direction,
    ] for i, e in enumerate(evals, 1)]
    w = [len(h) for h in hdr]
    for r in rows:
        for j, c in enumerate(r):
            w[j] = max(w[j], len(c))
    fmt = lambda cells: "| " + " | ".join(c.ljust(w[j]) for j, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * w[j] for j in range(len(hdr))) + " |"
    best = evals[0]
    lines += [fmt(hdr), sep, *(fmt(r) for r in rows)]
    lines += ["", f"Best derivative signal: **{best.name}** "
              f"(OOT AUC {best.auc:.4f}, {best.direction}, worst-fold strength "
              f"{best.wf_min_strength:.4f}, decile spread {best.spread:+.4f})."]
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    data_path = DEFAULT_DATA
    if "--data" in argv:
        data_path = argv[argv.index("--data") + 1]
    with open(data_path, "r", encoding="utf-8") as fh:
        series = json.load(fh)
    evals, n_tr, n_va, up_rate = evaluate(series)
    print(_render(evals, n_tr, n_va, up_rate))
    out = os.path.join(_HERE, "data", "deriv_leaderboard.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "base_up_rate": round(up_rate, 6),
            "n_train": n_tr, "n_val": n_va,
            "ranked": [e.__dict__ for e in evals],
            "best": evals[0].name,
        }, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
