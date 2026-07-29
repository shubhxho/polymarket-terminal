"""Local MLX trainer for the resolution-probability model.

This is the runnable, laptop-sized counterpart to the flagship Modal resolution
job. It answers a different question than the direction model in `train_seq.py`:
given a mid-market snapshot of a *binary* market — its price, how far the market
still has to run, its recent momentum and flow — what is the calibrated
probability the market ultimately **resolves YES**?

Two things make it honest rather than a demo:

  * **Out-of-time split by market end time.** A market's snapshots all share one
    resolution label, so the only leakage-safe partition is by *whole market*:
    train on markets that settle earlier, validate on markets that settle later.
    Splitting snapshots within a market would leak the outcome.

  * **PAV isotonic calibration.** A raw classifier's scores rank well but are not
    probabilities. Pool-adjacent-violators (implemented in-repo, no sklearn)
    fits a monotone score→probability map on the training predictions; we then
    report Brier / log-loss / a reliability table on the *validation* markets so
    the calibration quality is measured out-of-sample.

Data:
  * Reads `ml/data/sii_resolve.json` (labeled snapshots from the SII loader) if
    present. Schema: a list of markets, each
        {"end_time": <sortable>, "resolution": 0|1,
         "prices": [float, ...], "volume": [float, ...]?}
  * `--fixture` (or absent data) trains on a small bundled synthetic fixture so
    the pipeline always runs end to end.

Reuses `train_seq.py`'s eval kit — `FeatureMLP`, `_auc`, `_normalizers`,
`_train_one`, `_probs`, and the `_split`/`_walk_forward` walk-forward harness
(for an auxiliary price-path stability diagnostic) — rather than reimplementing.

Emits (into `ml/data/`):
  * `resolve_model.safetensors`   — trained FeatureMLP weights
  * `resolve_normalizer.json`     — z-score normaliser (same schema as
                                    seq_normalizer.json) + the calibration curve
  * `resolve_metrics.json`        — auc, brier, logloss, calibration bins

Run:  python ml/train_resolve.py --fixture
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import List, Optional, Sequence, Tuple

# Reuse train_seq's model + eval kit (do not reimplement). Insert this file's
# directory first so the sibling modules resolve regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_seq  # noqa: E402
from train_seq import FeatureMLP, _auc, _normalizers, _split, _walk_forward  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
SII = os.path.join(DATA_DIR, "sii_resolve.json")
WEIGHTS = os.path.join(DATA_DIR, "resolve_model.safetensors")
NORMALIZER = os.path.join(DATA_DIR, "resolve_normalizer.json")
METRICS = os.path.join(DATA_DIR, "resolve_metrics.json")

# Snapshot feature set (order is the contract for the served normaliser).
FEATURE_NAMES = [
    "price",               # current mid-market probability
    "dist_from_half",      # |price - 0.5| — how decided the market already looks
    "time_to_resolution",  # fraction of the market's life still remaining
    "momentum",            # mean of the last few price increments
    "realized_vol",        # std of recent increments
    "flow_imbalance",      # (#up - #down) / n over recent increments
    "volume_maturity",     # cumulative volume so far / total volume
]

WINDOW = 8            # look-back for momentum / vol / flow
MIN_I = 4             # need this much history before emitting a snapshot
VAL_FRAC = 0.2
SEED = 17
HIDDEN = train_seq.HIDDEN


class Snap:
    """One labeled snapshot. Carries `.feat`/`.seq`/`.label` so it plugs straight
    into `train_seq._normalizers` (which reads `.feat` and `.seq`) and its
    `.end_time` drives the out-of-time split."""

    __slots__ = ("seq", "feat", "label", "end_time")

    def __init__(self, seq: List[float], feat: List[float], label: int, end_time: float):
        self.seq = seq
        self.feat = feat
        self.label = label
        self.end_time = end_time


# ── feature extraction ────────────────────────────────────────────────────────

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _flow_imbalance(rets: Sequence[float]) -> float:
    if not rets:
        return 0.0
    up = sum(1 for r in rets if r > 0)
    dn = sum(1 for r in rets if r < 0)
    return (up - dn) / len(rets)


def market_to_snapshots(mkt: dict) -> List[Snap]:
    """Slide over one market's price path, emitting a labeled snapshot per step.

    Every snapshot of a market carries the same resolution label — the split is
    by market, so this is not leakage."""
    prices = [float(p) for p in mkt.get("prices", [])]
    n = len(prices)
    if n <= MIN_I:
        return []
    # Resolution label is required; tolerate a couple of loader key spellings but
    # fail loudly (not with a bare KeyError) if none is present.
    raw_label = next((mkt[k] for k in ("resolution", "resolved", "outcome") if k in mkt), None)
    if raw_label is None:
        raise SystemExit(
            "sii_resolve.json market is missing a resolution label "
            "(expected key 'resolution', 'resolved', or 'outcome')"
        )
    label = int(raw_label)
    end_time = mkt.get("end_time", 0)
    # Only use volume if it lines up with the price path — a mismatched-length
    # array would silently mis-scale volume_maturity, so fall back to the index.
    vol = mkt.get("volume")
    if not vol or len(vol) != n:
        vol = None
    total_vol = float(sum(vol)) if vol else float(n)
    if total_vol <= 0:
        total_vol = float(n)

    out: List[Snap] = []
    for i in range(MIN_I, n):
        w = prices[max(0, i - WINDOW): i + 1]
        rets = [w[k] - w[k - 1] for k in range(1, len(w))]
        price = prices[i]
        if vol:
            vol_mat = sum(vol[: i + 1]) / total_vol
        else:
            vol_mat = (i + 1) / n
        feat = [
            price,
            abs(price - 0.5),
            (n - 1 - i) / n,                        # time_to_resolution
            _mean(rets[-4:]) if rets else 0.0,      # momentum
            _std(rets),                             # realized_vol
            _flow_imbalance(rets),                  # flow_imbalance
            vol_mat,                                # volume_maturity
        ]
        # `.seq` feeds _normalizers' return-std; guard against an empty window.
        out.append(Snap(rets if rets else [0.0], feat, label, end_time))
    return out


# ── data loading / fixture ────────────────────────────────────────────────────

def make_fixture(n_markets: int = 160, seed: int = SEED) -> List[dict]:
    """Synthetic binary markets: each has a latent YES/NO outcome; its price
    path starts near 0.5 and mean-reverts toward the outcome with noise. Higher
    late prices therefore genuinely predict a YES resolution, so AUC lands
    comfortably above 0.5 and every metric is finite — a real, if easy, signal.
    `end_time` increases with the market index so the out-of-time split has a
    meaningful chronological ordering."""
    rng = random.Random(seed)
    markets: List[dict] = []
    for m in range(n_markets):
        outcome = 1 if rng.random() < 0.5 else 0
        n = rng.randint(24, 56)
        target = 0.85 if outcome else 0.15
        p = 0.5 + rng.uniform(-0.08, 0.08)
        prices, volume = [], []
        for _ in range(n):
            p += 0.05 * (target - p) + rng.gauss(0.0, 0.05)
            p = min(0.99, max(0.01, p))
            prices.append(p)
            volume.append(abs(rng.gauss(1.0, 0.3)) + 0.1)
        markets.append(
            {"end_time": m, "resolution": outcome, "prices": prices, "volume": volume}
        )
    return markets


def load_markets(use_fixture: bool) -> Tuple[List[dict], str]:
    if not use_fixture and os.path.exists(SII):
        with open(SII) as f:
            markets = json.load(f)
        if isinstance(markets, dict) and "markets" in markets:
            markets = markets["markets"]
        if markets:
            return markets, SII
        print(f"note: {SII} was empty — falling back to the bundled fixture")
    return make_fixture(), "fixture"


def split_by_end_time(
    markets: List[dict], val_frac: float = VAL_FRAC
) -> Tuple[List[Snap], List[Snap]]:
    """Honest out-of-time split: order markets by end time, the earliest
    (1-val_frac) settle into TRAIN, the latest into VAL. The cut lands on a
    *distinct end-time boundary*, so markets that share an end time are never
    split across train/val — a market's resolution never straddles the split."""
    ordered = sorted(markets, key=lambda mk: mk.get("end_time", 0))
    idx = max(1, int(len(ordered) * (1 - val_frac)))
    idx = min(idx, len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    # Walk the cut forward past any run of markets tied on the boundary end time
    # so all of them land in train (never straddling into val).
    if 0 < idx < len(ordered):
        boundary = ordered[idx - 1].get("end_time", 0)
        while idx < len(ordered) and ordered[idx].get("end_time", 0) == boundary:
            idx += 1
    tr_m, va_m = ordered[:idx], ordered[idx:]
    tr = [s for mk in tr_m for s in market_to_snapshots(mk)]
    va = [s for mk in va_m for s in market_to_snapshots(mk)]
    return tr, va


# ── PAV isotonic calibration (pool-adjacent-violators, no sklearn) ─────────────

def pav_calibration(scores: Sequence[float], labels: Sequence[float]) -> List[Tuple[float, float]]:
    """Fit a monotone (non-decreasing) score→probability map by pool-adjacent-
    violators. Returns breakpoints [(x, p), ...] sorted by x, where each block's
    x is its highest raw score and p its pooled label mean — a step/interpolation
    table consumed by `apply_calibration`."""
    pairs = sorted(zip((float(s) for s in scores), (float(y) for y in labels)), key=lambda t: t[0])
    if not pairs:
        return [(0.0, 0.5), (1.0, 0.5)]
    # Each block: [sum_labels, count, x_hi] where value = sum_labels / count.
    blocks: List[List[float]] = []
    for x, y in pairs:
        blocks.append([y, 1.0, x])
        # Merge left while the sequence of block means would decrease (violation).
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) >= (blocks[-1][0] / blocks[-1][1]):
            s_y = blocks[-2][0] + blocks[-1][0]
            n = blocks[-2][1] + blocks[-1][1]
            x_hi = blocks[-1][2]  # keep the block's upper score edge
            blocks[-2:] = [[s_y, n, x_hi]]
    curve = [(x_hi, s_y / n) for s_y, n, x_hi in blocks]
    # Guarantee coverage of the full score range for interpolation lookups.
    if curve[0][0] > pairs[0][0]:
        curve.insert(0, (pairs[0][0], curve[0][1]))
    if curve[-1][0] < pairs[-1][0]:
        curve.append((pairs[-1][0], curve[-1][1]))
    return curve


def apply_calibration(curve: Sequence[Tuple[float, float]], score: float) -> float:
    """Map one raw score to a calibrated probability by linear interpolation
    across the isotonic breakpoints, clamped at the ends."""
    if not curve:
        return score
    if score <= curve[0][0]:
        return curve[0][1]
    if score >= curve[-1][0]:
        return curve[-1][1]
    lo = 0
    hi = len(curve) - 1
    # Binary search for the bracketing breakpoints.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if curve[mid][0] <= score:
            lo = mid
        else:
            hi = mid
    x0, p0 = curve[lo]
    x1, p1 = curve[hi]
    if x1 == x0:
        return p1
    t = (score - x0) / (x1 - x0)
    return p0 + t * (p1 - p0)


# ── metrics ───────────────────────────────────────────────────────────────────

def _clip(p: float, eps: float = 1e-7) -> float:
    return min(1.0 - eps, max(eps, p))


def brier(probs: Sequence[float], labels: Sequence[float]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def logloss(probs: Sequence[float], labels: Sequence[float]) -> float:
    tot = 0.0
    for p, y in zip(probs, labels):
        c = _clip(p)
        tot += -(y * math.log(c) + (1 - y) * math.log(1 - c))
    return tot / len(probs)


def calibration_bins(probs: Sequence[float], labels: Sequence[float], n_bins: int = 10) -> List[dict]:
    """Reliability table: bucket by predicted probability, report mean predicted
    vs observed frequency per bucket (the diagonal is perfect calibration)."""
    bins: List[dict] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if (p >= lo and (p < hi or (b == n_bins - 1 and p <= hi)))]
        if not idx:
            bins.append({"bin": b, "lo": round(lo, 3), "hi": round(hi, 3),
                         "count": 0, "mean_pred": None, "frac_pos": None})
            continue
        mp = sum(probs[i] for i in idx) / len(idx)
        fp = sum(labels[i] for i in idx) / len(idx)
        bins.append({"bin": b, "lo": round(lo, 3), "hi": round(hi, 3),
                     "count": len(idx), "mean_pred": round(mp, 4), "frac_pos": round(fp, 4)})
    return bins


# ── tensors / training ────────────────────────────────────────────────────────

def _tensors(snaps: Sequence[Snap], fmean, fstd):
    """(dummy_seq, z-scored feat, y). FeatureMLP ignores the seq argument, so a
    minimal placeholder keeps `train_seq._train_one`'s signature satisfied."""
    import mlx.core as mx

    feat = (mx.array([s.feat for s in snaps], dtype=mx.float32) - fmean) / fstd
    seq = mx.zeros((len(snaps), 1, 1), dtype=mx.float32)
    y = mx.array([float(s.label) for s in snaps], dtype=mx.float32)
    return seq, feat, y


def path_signal_diagnostic(markets: List[dict]) -> Optional[dict]:
    """Auxiliary reuse of `train_seq._split` / `_walk_forward`: does the price
    *path's* own momentum predict its short-horizon direction, and is that stable
    over time? Scores each direction-window by its momentum feature (no training)
    and reports AUC via `_auc`. Pure diagnostic — never affects the resolution
    model — and wrapped so a data shape that these helpers reject can't break the
    run."""
    try:
        from features import FEATURE_NAMES as DIR_FEATURES

        mom = DIR_FEATURES.index("momentum")
        series = [[float(p) for p in mk.get("prices", [])] for mk in markets]

        _, va = _split(series, val_frac=VAL_FRAC)
        if not va:
            return None
        oot_auc = _auc([s.feat[mom] for s in va], [float(s.label) for s in va])

        folds = []
        for k, _tr_k, va_k in _walk_forward(series, folds=3):
            if not va_k:
                continue
            a = _auc([s.feat[mom] for s in va_k], [float(s.label) for s in va_k])
            folds.append({"fold": k, "auc": round(a, 4), "n": len(va_k)})
        return {
            "note": "price-path momentum → direction, out-of-time; sanity check that the "
                    "path carries signal (independent of the resolution head).",
            "oot_auc": round(oot_auc, 4),
            "walk_forward": folds,
        }
    except Exception as exc:  # noqa: BLE001 — diagnostic must never fail the run
        return {"note": f"diagnostic skipped: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", action="store_true",
                    help="ignore ml/data/sii_resolve.json and train on the bundled synthetic fixture")
    ap.add_argument("--epochs", type=int, default=None, help="override training epochs")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx.utils import tree_flatten

    mx.random.seed(SEED)
    random.seed(SEED)
    if args.epochs is not None:
        train_seq.EPOCHS = args.epochs

    markets, source = load_markets(args.fixture)
    tr_s, va_s = split_by_end_time(markets)
    if not tr_s or not va_s:
        raise SystemExit("not enough markets to form an out-of-time split")
    print(f"source: {source} — {len(markets)} markets → "
          f"train {len(tr_s)} / val {len(va_s)} snapshots")

    # Normalisers from TRAIN snapshots only (reused from train_seq — no leakage).
    fmean, fstd, _rstd = _normalizers(tr_s)

    tr = _tensors(tr_s, fmean, fstd)
    va = _tensors(va_s, fmean, fstd)

    model = FeatureMLP(len(FEATURE_NAMES))
    print("training resolution FeatureMLP …")
    val_auc = train_seq._train_one("resolve", model, tr, va, len(FEATURE_NAMES))

    # Raw (uncalibrated) probabilities.
    raw_tr = train_seq._probs(model, tr[0], tr[1])
    raw_va = train_seq._probs(model, va[0], va[1])
    y_tr = tr[2].tolist()
    y_va = va[2].tolist()

    # Fit PAV isotonic calibration on TRAIN predictions; measure on VAL.
    curve = pav_calibration(raw_tr, y_tr)
    cal_va = [apply_calibration(curve, s) for s in raw_va]

    metrics = {
        "source": source,
        "markets": len(markets),
        "train_snapshots": len(tr_s),
        "val_snapshots": len(va_s),
        "auc": round(_auc(cal_va, y_va), 4),
        "raw_auc": round(val_auc, 4),
        "brier": round(brier(cal_va, y_va), 5),
        "raw_brier": round(brier(raw_va, y_va), 5),
        "logloss": round(logloss(cal_va, y_va), 5),
        "raw_logloss": round(logloss(raw_va, y_va), 5),
        "base_rate": round(sum(y_va) / len(y_va), 4),
        "calibration_bins": calibration_bins(cal_va, y_va),
        "features": FEATURE_NAMES,
        "note": "auc/brier/logloss are on out-of-time validation markets, after "
                "PAV isotonic calibration fit on the training predictions.",
    }
    diag = path_signal_diagnostic(markets)
    if diag is not None:
        metrics["path_signal"] = diag

    os.makedirs(DATA_DIR, exist_ok=True)
    mx.save_safetensors(WEIGHTS, dict(tree_flatten(model.parameters())))
    # Same schema as seq_normalizer.json (feature order + z-score mean/std +
    # winner/meta), plus the calibration curve so inference can reproduce
    # calibrated probabilities without refitting.
    with open(NORMALIZER, "w") as f:
        json.dump(
            {
                "winner": "resolve_mlp",
                "task": "resolve",
                "fmean": fmean.tolist(),
                "fstd": fstd.tolist(),
                "rstd": _rstd,
                "features": FEATURE_NAMES,
                "seq_len": 0,
                "hidden": HIDDEN,
                "calibration": [[round(x, 6), round(p, 6)] for x, p in curve],
            },
            f,
            indent=2,
        )
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n== resolve metrics ==")
    print(json.dumps(metrics, indent=2))
    print(f"\nmodel      → {WEIGHTS}")
    print(f"normalizer → {NORMALIZER}")
    print(f"metrics    → {METRICS}")


if __name__ == "__main__":
    main()
