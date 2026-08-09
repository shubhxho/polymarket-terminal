"""META-ENSEMBLE — a calibrated stacker that blends the three base signal models.

The suite trains three specialists on `SII-WANGZJ/Polymarket_data`:

  * **resolution** (`shubhxho/polymarket-resolution-model`) — P(market resolves YES)
    from its mid-market state,
  * **flow** (`shubhxho/polymarket-flow-model`) — short-horizon direction from true
    order-flow imbalance,
  * **smart-money** (`shubhxho/polymarket-smartmoney-model`) — direction from the
    behaviour of historically-profitable cohorts.

Each is good at a different thing. This job trains a **stacker** — a small model
whose *inputs are the three base models' predictions* — into one blended signal,
and shows (honestly) whether the blend actually beats the best single base model
out-of-time. Pipeline, all inside the Modal function:

1. **Assemble out-of-fold (OOF) base predictions.** For each base repo, try to
   download a published OOF table (`oof/oof_preds.parquet`, columns
   `market_id,timestamp,pred,label,end_date`). Any base without one is
   **regenerated** from a small streamed slice of `SII-WANGZJ/Polymarket_data`:
   build shared mid-market snapshots and score them with that base's *already
   trained* LightGBM booster + normaliser. Align all three by
   `(market_id, timestamp)` (inner join) so every row has all three base preds.
2. **Train the stacker** (logistic over the base preds) on an **out-of-time split
   by market `end_date`** — earliest markets train, latest are the held-out test,
   so the stacker never sees a base pred from the test fold. **Isotonic (PAV)
   calibration** — implemented here, no sklearn — on a middle held-out fold.
3. **Report** Brier / log-loss / AUC of the blend vs *each base model alone* on
   the same test rows (so "did stacking help?" is answerable), plus a reliability
   table. Push artifacts + normaliser JSON + metrics + a model card to
   `shubhxho/polymarket-ensemble-model`.

    modal run ml/modal_ensemble.py --max-markets 6000 --push

Local proof (zero heavy deps — pure-stdlib synthetic stacker + this PAV):

    python ml/modal_ensemble.py --smoke

Heavy imports (modal/torch/lightgbm/datasets/pyarrow/numpy/huggingface_hub) live
INSIDE functions so the module imports cleanly under a bare interpreter with none
of them installed.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

# Modal is optional at import time — guarded so a bare `python` (or the smoke
# path) can import this module with modal uninstalled. The App is only built when
# modal is present (i.e. under `modal run`).
try:
    import modal
    _HAS_MODAL = True
except Exception:  # pragma: no cover - depends on install env
    modal = None
    _HAS_MODAL = False

REPO = "SII-WANGZJ/Polymarket_data"
OUT_REPO = "shubhxho/polymarket-ensemble-model"

# Base models, in a fixed order — this order is the stacker's feature order and is
# persisted in the normaliser so inference feeds the base preds in the same slots.
BASE_ORDER = ["resolution", "flow", "smartmoney"]
BASE_REPOS = {
    "resolution": "shubhxho/polymarket-resolution-model",
    "flow": "shubhxho/polymarket-flow-model",
    "smartmoney": "shubhxho/polymarket-smartmoney-model",
}
# Where each base repo keeps its LightGBM booster + z-score normaliser, and the
# OOF-prediction tables we try first. Matches what modal_resolve/flow/smart push.
BASE_ARTIFACTS = {
    "resolution": {"gbdt": "resolve_gbdt.txt", "norm": "resolve_normalizer.json"},
    "flow": {"gbdt": "flow/flow_gbdt.txt", "norm": "flow/flow_normalizer.json"},
    "smartmoney": {"gbdt": "smartmoney_gbdt.txt", "norm": "smartmoney_normalizer.json"},
}
OOF_CANDIDATES = ("oof/oof_preds.parquet", "oof_preds.parquet")

# Mid-market snapshot config (shared by the regeneration path; mirrors the
# resolution model so regenerated base preds sit on a comparable grid).
SNAP_FRACS = (0.15, 0.35, 0.55, 0.75, 0.90)
RECENT = 24
MIN_TRADES = 12
DECISIVE = 0.90
EPS = 1e-9


# ─────────────────────────── pure-stdlib helpers ────────────────────────────
# Shared by BOTH the smoke path and the Modal path. No numpy, no heavy deps.

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _parse_outcome_prices(outcome_prices: object) -> Optional[list]:
    """Parse `outcome_prices` into a Python list, tolerant of the real dataset's
    formats. On HF this field is NOT valid JSON — it's a single-quoted Python-list
    repr like `"['1', '0']"` (json.loads → zero labels). Try JSON first, then fall
    back to `ast.literal_eval` for the Python-repr case.
    """
    if isinstance(outcome_prices, (list, tuple)):
        return list(outcome_prices)
    if not isinstance(outcome_prices, str):
        return None
    s = outcome_prices.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        val = ast.literal_eval(s)
        return list(val) if isinstance(val, (list, tuple)) else None
    except Exception:
        return None


def resolution_label(outcome_prices: object) -> Optional[int]:
    """Binary YES/NO label from a market's `outcome_prices` (ground-truth target
    for the stacker). `["0.99","0.01"]` → 1, `"['0.02','0.98']"` → 0. Returns None
    for unparseable or non-decisive (e.g. 0.5/0.5, void) markets so they never
    pollute training/alignment.
    """
    arr = _parse_outcome_prices(outcome_prices)
    if arr is None or len(arr) < 2:
        return None
    try:
        p1, p2 = float(arr[0]), float(arr[1])
    except Exception:
        return None
    if not (math.isfinite(p1) and math.isfinite(p2)):
        return None
    if max(p1, p2) < DECISIVE or min(p1, p2) > 1.0 - DECISIVE:
        return None
    return 1 if p1 > p2 else 0


# ── metrics (pure) ───────────────────────────────────────────────────────────

def auc(scores: List[float], labels: List[float]) -> float:
    """Tie-aware ROC AUC (Mann–Whitney U). Matches _auc in train_seq / modal_*."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(1 for _, y in pairs if y > 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i, r = 0, 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (r + r + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] > 0.5:
                rank_sum += avg
        r += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier(probs: List[float], labels: List[float]) -> float:
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def log_loss(probs: List[float], labels: List[float]) -> float:
    if not probs:
        return 0.0
    s = 0.0
    for p, y in zip(probs, labels):
        p = min(max(p, 1e-6), 1 - 1e-6)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(probs)


def calibration_table(probs: List[float], labels: List[float], bins: int = 10) -> List[dict]:
    """Reliability bins: mean prediction vs empirical positive rate per band."""
    table = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        last = b == bins - 1
        idx = [i for i, p in enumerate(probs) if p >= lo and (p < hi or (last and p <= hi))]
        if not idx:
            table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "mean_pred": None, "frac_pos": None})
            continue
        table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": len(idx),
                      "mean_pred": round(_mean([probs[i] for i in idx]), 4),
                      "frac_pos": round(_mean([float(labels[i]) for i in idx]), 4)})
    return table


def _decile_lift(probs: List[float], labels: List[float], bins: int = 10) -> List[dict]:
    """Up-rate per score-decile — the tradeable structure (higher score → higher
    hit-rate, monotonically?). Mirrors _lift_table / _decile_backtest conventions."""
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    n = len(order)
    out = []
    for b in range(bins):
        idx = order[b * n // bins:(b + 1) * n // bins]
        if idx:
            out.append({"decile": b + 1,
                        "up_rate": round(_mean([float(labels[i]) for i in idx]), 3),
                        "mean_score": round(_mean([probs[i] for i in idx]), 3),
                        "n": len(idx)})
    return out


# ── isotonic calibration via PAV (pool-adjacent-violators) ───────────────────

def isotonic_fit(scores: List[float], labels: List[float]) -> List[Tuple[float, float]]:
    """Fit a non-decreasing calibration map score→prob with the PAV algorithm.
    Returns compact breakpoints [(threshold, calibrated_prob), …] ascending in
    threshold; `isotonic_predict` looks a raw score up against them."""
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    xs = [scores[i] for i in order]
    ys = [float(labels[i]) for i in order]

    blocks: List[List[float]] = []
    for y in ys:
        blocks.append([y, 1.0])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) >= (blocks[-1][0] / blocks[-1][1]):
            s2, c2 = blocks.pop()
            s1, c1 = blocks.pop()
            blocks.append([s1 + s2, c1 + c2])
    cal: List[float] = []
    for s, c in blocks:
        cal.extend([s / c] * int(c))

    bp: List[Tuple[float, float]] = []
    for x, v in zip(xs, cal):
        if not bp or abs(bp[-1][1] - v) > 1e-12:
            bp.append((x, v))
    return bp


def isotonic_predict(bp: List[Tuple[float, float]], s: float) -> float:
    if not bp:
        return s
    i = bisect.bisect_right(bp, (s, math.inf)) - 1
    if i < 0:
        return bp[0][1]
    return bp[i][1]


# ── z-score normaliser + pure logistic stacker (smoke + Modal) ───────────────

def zstats(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    cols = list(zip(*X))
    fmean = [_mean(list(c)) for c in cols]
    fstd = [(_std(list(c)) or 1.0) for c in cols]
    return fmean, fstd


def znorm(X: List[List[float]], fmean: List[float], fstd: List[float]) -> List[List[float]]:
    return [[(row[j] - fmean[j]) / (fstd[j] + 1e-9) for j in range(len(row))] for row in X]


def logreg_fit(X: List[List[float]], y: List[float], epochs: int = 500,
               lr: float = 0.3, l2: float = 1e-3) -> Tuple[List[float], float]:
    """Full-batch logistic regression (the stacker). Tiny — a handful of inputs
    (the base preds) — so plain gradient descent converges fast."""
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(w[j] * xi[j] for j in range(d))
            e = _sigmoid(z) - yi
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * (gb / n)
    return w, b


def logreg_predict(w: List[float], b: float, X: List[List[float]]) -> List[float]:
    return [_sigmoid(b + sum(w[j] * row[j] for j in range(len(row)))) for row in X]


def stacker_inputs(base_preds: List[float]) -> List[float]:
    """Feature vector the stacker sees for one row: each base model's probability
    in logit space (a linear stacker over logits ≈ a weighted opinion pool, which
    is the principled way to combine probabilities) plus the raw mean as a robust
    anchor. Order matches BASE_ORDER."""
    logits = [_logit(p) for p in base_preds]
    return logits + [_mean(base_preds)]


# ── blend-vs-base reporting (shared) ─────────────────────────────────────────

def _base_vs_blend(base_cols: Dict[str, List[float]], blend: List[float],
                   y: List[float]) -> dict:
    """AUC/Brier/log-loss for each base model alone and for the blend, on the same
    rows, plus whether the blend beat the best single base (the honesty check)."""
    per_base = {}
    for name in BASE_ORDER:
        col = base_cols[name]
        per_base[name] = {"auc": round(auc(col, y), 4), "brier": round(brier(col, y), 4),
                          "log_loss": round(log_loss(col, y), 4)}
    blend_m = {"auc": round(auc(blend, y), 4), "brier": round(brier(blend, y), 4),
               "log_loss": round(log_loss(blend, y), 4)}
    best_single = max(BASE_ORDER, key=lambda n: per_base[n]["auc"])
    return {"per_base": per_base, "blend": blend_m, "best_single_base": best_single,
            "blend_beats_best_single_auc": bool(blend_m["auc"] >= per_base[best_single]["auc"]),
            "auc_gain_vs_best_single": round(blend_m["auc"] - per_base[best_single]["auc"], 4)}


# ─────────────────────────── smoke pipeline (stdlib) ────────────────────────

def _synth_oof(n_markets: int = 900, seed: int = 11):
    """Synthetic OOF table: each market has a latent YES/NO label; the three base
    models emit correlated-but-independently-noisy probabilities of it (different
    skill + different miscalibration per model). Later markets get later
    `end_date` so the out-of-time split is meaningful. A well-behaved stacker
    should denoise the three into a blend that beats any single one.

    Returns rows of (market_id, timestamp, {name: pred}, label, end_ts).
    """
    rng = random.Random(seed)
    base_ts = 1_600_000_000
    # per-model skill (logit slope on the latent signal) and calibration skew
    skill = {"resolution": 2.2, "flow": 1.6, "smartmoney": 1.4}
    noise = {"resolution": 1.1, "flow": 1.4, "smartmoney": 1.5}
    skew = {"resolution": 1.0, "flow": 0.7, "smartmoney": 1.3}  # !=1 → miscalibrated
    rows = []
    for m in range(n_markets):
        label = 1 if rng.random() < 0.5 else 0
        signal = (label - 0.5) * 2.0            # ±1 latent direction
        end = base_ts + m * 3600                # strictly increasing end_date
        n_snaps = rng.randint(2, 5)
        for s in range(n_snaps):
            preds = {}
            for name in BASE_ORDER:
                z = skill[name] * signal + rng.gauss(0.0, noise[name])
                preds[name] = _sigmoid(skew[name] * z)   # miscalibration via skew
            ts = end - (n_snaps - s) * 600
            rows.append((f"m{m}", ts, preds, float(label), float(end)))
    return rows


def run_smoke() -> int:
    print("== META-ENSEMBLE stacker — SMOKE (pure-stdlib, synthetic OOF) ==")
    rows = _synth_oof()

    # Out-of-time split by end_date: earliest→train, middle→calibrate, latest→test.
    rows.sort(key=lambda r: r[4])
    n = len(rows)
    tr_end, cal_end = int(n * 0.55), int(n * 0.70)
    tr, ca, te = rows[:tr_end], rows[tr_end:cal_end], rows[cal_end:]
    if not (tr and ca and te):
        print("!! not enough rows for a 3-way temporal split")
        return 1

    def cols(chunk):
        base = {name: [r[2][name] for r in chunk] for name in BASE_ORDER}
        y = [r[3] for r in chunk]
        return base, y

    tr_base, ytr = cols(tr)
    ca_base, yca = cols(ca)
    te_base, yte = cols(te)
    print(f"rows — train {len(tr)}  calib {len(ca)}  test {len(te)}  "
          f"(train YES-rate {_mean(ytr):.3f})")

    # Stacker: logistic over the base preds (in logit space + mean anchor).
    def feats(base, i):
        return stacker_inputs([base[name][i] for name in BASE_ORDER])

    Xtr = [feats(tr_base, i) for i in range(len(ytr))]
    Xca = [feats(ca_base, i) for i in range(len(yca))]
    Xte = [feats(te_base, i) for i in range(len(yte))]
    fmean, fstd = zstats(Xtr)
    w, b = logreg_fit(znorm(Xtr, fmean, fstd), ytr)

    blend_raw = logreg_predict(w, b, znorm(Xte, fmean, fstd))
    # Calibrate on the held-out calibration fold, apply to test.
    curve = isotonic_fit(logreg_predict(w, b, znorm(Xca, fmean, fstd)), yca)
    blend_cal = [isotonic_predict(curve, p) for p in blend_raw]

    rep = _base_vs_blend(te_base, blend_cal, yte)
    print("\n-- out-of-time test: blend vs each base model --")
    print("  model          AUC     Brier    LogLoss")
    for name in BASE_ORDER:
        mm = rep["per_base"][name]
        print(f"  {name:<12}  {mm['auc']:.4f}  {mm['brier']:.4f}   {mm['log_loss']:.4f}")
    braw = {"auc": auc(blend_raw, yte), "brier": brier(blend_raw, yte)}
    print(f"  {'blend(raw)':<12}  {braw['auc']:.4f}  {braw['brier']:.4f}")
    bm = rep["blend"]
    print(f"  {'blend(cal)':<12}  {bm['auc']:.4f}  {bm['brier']:.4f}   {bm['log_loss']:.4f}")
    print(f"\n  best single base: {rep['best_single_base']}  |  "
          f"blend beats it (AUC): {rep['blend_beats_best_single_auc']}  "
          f"(gain {rep['auc_gain_vs_best_single']:+.4f})")

    print("\n-- reliability table (calibrated blend) --")
    print("  bin          n     mean_pred  frac_pos")
    for row in calibration_table(blend_cal, yte):
        mp = "  -  " if row["mean_pred"] is None else f"{row['mean_pred']:.3f}"
        fp = "  -  " if row["frac_pos"] is None else f"{row['frac_pos']:.3f}"
        print(f"  {row['bin']:<11} {row['n']:>4}    {mp:>6}     {fp:>6}")

    ok = (bm["auc"] > 0.5 and 0.0 <= bm["brier"] <= 1.0
          and rep["blend_beats_best_single_auc"])
    print("\nSMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ─────────────────────────── Modal H100 pipeline ────────────────────────────
# Everything below imports heavy deps INSIDE the function bodies. Registered on
# the Modal App only when modal is installed (see the guard at the bottom).

def _snapshot_named_features(ts, yp, sg, uv, cum_uv, k, created_ts, end_ts) -> Dict[str, float]:
    """Named mid-market snapshot features for trades[0:k] of one market, under the
    names the base models use, so a base booster can be re-scored on this grid.

    Full coverage of the resolution model's 12 features (canonical names); the
    flow/smart models' overlapping notions are exposed under their own names too
    (aliases). Names a given base doesn't recognise are simply not selected; names
    it wants that we don't produce default to 0 at selection time.
    """
    price = yp[k - 1]
    now = ts[k - 1]
    signed_dist = price - 0.5
    dist = abs(signed_dist)

    span = max(float(end_ts - created_ts), 1.0)
    elapsed = min(max((now - created_ts) / span, 0.0), 1.0)
    time_rem = 1.0 - elapsed
    log_hours = math.log1p(max(float(end_ts - now), 0.0) / 3600.0)

    r0 = max(0, k - RECENT)
    recent_yp = yp[r0:k]
    rets = [recent_yp[i] - recent_yp[i - 1] for i in range(1, len(recent_yp))]
    momentum = (recent_yp[-1] - recent_yp[0]) if len(recent_yp) >= 2 else 0.0
    vol = _std(rets)

    recent_uv = sum(uv[r0:k]) + EPS
    flow_recent = sum(sg[r0:k]) / recent_uv
    seen_uv = cum_uv[k]
    total_uv = cum_uv[-1]
    maturity = min(max(seen_uv / (total_uv + EPS), 0.0), 1.0)
    flow_cum = sum(sg[:k]) / (seen_uv + EPS)

    gains = sum(x for x in rets if x > 0)
    losses = -sum(x for x in rets if x < 0)
    rsi = (gains - losses) / (gains + losses) if (gains + losses) > EPS else 0.0
    band = _std(recent_yp)
    band_z = (price - _mean(recent_yp)) / band if band > EPS else 0.0

    return {
        # resolution model (canonical, full coverage)
        "yes_price": price, "signed_dist": signed_dist, "dist_half": dist,
        "time_remaining_frac": time_rem, "log_hours_to_res": log_hours,
        "momentum_recent": momentum, "vol_recent": vol,
        "flow_imb_recent": flow_recent, "flow_imb_cum": flow_cum,
        "vol_maturity": maturity, "log_trades": math.log1p(float(k)),
        "price_x_maturity": signed_dist * maturity,
        # flow / smart aliases for the same underlying notions
        "last": price, "mean_ret": _mean(rets), "vol": vol, "drift": momentum,
        "band_z": band_z, "momentum": momentum, "rsi": rsi,
        "signed_imbalance": flow_recent, "ofi_last": flow_recent, "ofi_trend": flow_cum,
        "buy_pressure": (flow_recent + 1.0) / 2.0, "trade_intensity": math.log1p(float(k)),
        "extremeness": dist * 2.0, "log_volume": math.log1p(seen_uv),
    }


def _yes_view(price: float, side: str, direction: str, usd: float) -> Tuple[float, float]:
    """One raw trade → (YES price, signed YES USD flow). token2 (NO) inverts."""
    yp = price if side == "token1" else 1.0 - price
    buy = direction == "BUY"
    yes_up = (buy and side == "token1") or ((not buy) and side == "token2")
    return yp, (usd if yes_up else -usd)


def _load_base_scorer(name: str, hf_token: Optional[str]):
    """Load a base model's LightGBM booster + its feature order from the HF repo.
    Returns (booster, features) or None if unavailable.

    NOTE: the base repos train their GBDT on RAW features — `fmean`/`fstd` in the
    normaliser are for those models' *neural* branch only. A LightGBM booster must
    be fed features in the exact space it was trained on (split thresholds are in
    raw feature units), so we score on raw values and deliberately do NOT z-score."""
    import lightgbm as lgb
    from huggingface_hub import hf_hub_download
    art = BASE_ARTIFACTS[name]
    repo = BASE_REPOS[name]
    try:
        gp = hf_hub_download(repo, art["gbdt"], repo_type="model", token=hf_token)
        npth = hf_hub_download(repo, art["norm"], repo_type="model", token=hf_token)
    except Exception as e:
        print(f"  [{name}] booster/normaliser unavailable: {type(e).__name__}", flush=True)
        return None
    with open(npth) as f:
        norm = json.load(f)
    return lgb.Booster(model_file=gp), list(norm.get("features", []))


def _try_download_oof(name: str, hf_token: Optional[str]):
    """Try to pull a published OOF table for one base model. Returns a dict
    {(market_id, ts): (pred, label, end_ts)} or None."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    for cand in OOF_CANDIDATES:
        try:
            p = hf_hub_download(BASE_REPOS[name], cand, repo_type="model", token=hf_token)
        except Exception:
            continue
        t = pq.read_table(p).to_pydict()
        cols = set(t.keys())
        if not {"market_id", "timestamp", "pred"} <= cols:
            continue
        out = {}
        for i in range(len(t["market_id"])):
            key = (str(t["market_id"][i]), int(t["timestamp"][i]))
            lab = float(t["label"][i]) if "label" in cols else None
            end = float(t["end_date"][i]) if "end_date" in cols else float(t["timestamp"][i])
            out[key] = (float(t["pred"][i]), lab, end)
        print(f"  [{name}] downloaded OOF table: {len(out)} rows", flush=True)
        return out
    return None


def _regen_snapshots(max_markets: int, slice_row_groups: int, hf_token: Optional[str]):
    """Build shared mid-market snapshots from a small streamed slice of the dataset,
    keyed by (market_id, snapshot_ts), carrying the resolution label + end_date and
    the named feature dict. This is the substrate the base boosters re-score."""
    import datetime as _dt
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def _epoch(x):
        if x is None:
            return None
        if isinstance(x, _dt.datetime):
            return x.timestamp()
        if isinstance(x, _dt.date):
            return _dt.datetime(x.year, x.month, x.day, tzinfo=_dt.timezone.utc).timestamp()
        if isinstance(x, (int, float)):
            return float(x)
        try:
            return float(x.timestamp())
        except Exception:
            return None

    mpath = hf_hub_download(REPO, "markets.parquet", repo_type="dataset", token=hf_token)
    mt = pq.read_table(mpath, columns=["id", "outcome_prices", "created_at", "end_date"]).to_pydict()
    label_of, created_of, end_of = {}, {}, {}
    for i in range(len(mt["id"])):
        lab = resolution_label(mt["outcome_prices"][i])
        if lab is None:
            continue
        mid = str(mt["id"][i])
        c, e = _epoch(mt["created_at"][i]), _epoch(mt["end_date"][i])
        if c is None or e is None or e <= c:
            continue
        label_of[mid] = lab
        created_of[mid] = c
        end_of[mid] = e
        if len(label_of) >= max_markets:
            break
    print(f"  regen: {len(label_of)} labeled markets from markets.parquet", flush=True)

    # Stream a slice of trades.parquet, collecting per-market YES-view trades.
    tpath = hf_hub_download(REPO, "trades.parquet", repo_type="dataset", token=hf_token)
    pf = pq.ParquetFile(tpath)
    cols = ["market_id", "timestamp", "price", "usd_amount", "nonusdc_side", "taker_direction"]
    by_market: Dict[str, list] = {}
    n_groups = pf.num_row_groups if slice_row_groups <= 0 else min(slice_row_groups, pf.num_row_groups)
    for g in range(n_groups):
        d = pf.read_row_group(g, columns=cols).to_pydict()
        for i in range(len(d["market_id"])):
            mid = str(d["market_id"][i])
            if mid not in label_of:
                continue
            price = float(d["price"][i])
            if not (0.0 < price < 1.0):
                continue
            yp, sg = _yes_view(price, d["nonusdc_side"][i], d["taker_direction"][i], float(d["usd_amount"][i]))
            by_market.setdefault(mid, []).append((int(d["timestamp"][i]), yp, sg, abs(float(d["usd_amount"][i]))))

    snaps = {}   # (market_id, ts) -> {"feats": {...}, "label": int, "end": float}
    for mid, trades in by_market.items():
        if len(trades) < MIN_TRADES:
            continue
        trades.sort(key=lambda t: t[0])
        ts = [t[0] for t in trades]
        yp = [t[1] for t in trades]
        sg = [t[2] for t in trades]
        uv = [t[3] for t in trades]
        cum = [0.0] * (len(trades) + 1)
        for i in range(len(trades)):
            cum[i + 1] = cum[i] + uv[i]
        seen = set()
        for fr in SNAP_FRACS:
            k = int(len(trades) * fr)
            if k < 2 or k in seen:
                continue
            seen.add(k)
            feats = _snapshot_named_features(ts, yp, sg, uv, cum, k, created_of[mid], end_of[mid])
            if all(math.isfinite(v) for v in feats.values()):
                snaps[(mid, int(ts[k - 1]))] = {"feats": feats, "label": label_of[mid], "end": end_of[mid]}
    print(f"  regen: {len(snaps)} snapshots over {len(by_market)} markets", flush=True)
    return snaps


def _score_base_on_snaps(name: str, snaps: dict, hf_token: Optional[str]):
    """Regenerate one base model's OOF preds by scoring the shared snapshots with
    its trained booster. Returns {(market_id, ts): (pred, label, end)} or None."""
    import numpy as np
    scorer = _load_base_scorer(name, hf_token)
    if scorer is None:
        return None
    booster, feats = scorer
    if not feats:
        return None
    keys = list(snaps.keys())
    rows = []
    for key in keys:
        fd = snaps[key]["feats"]
        rows.append([fd.get(fn, 0.0) for fn in feats])
    # Raw features — the booster was trained on raw values (see _load_base_scorer).
    preds = booster.predict(np.asarray(rows, dtype=np.float64))
    out = {}
    for key, p in zip(keys, preds):
        out[key] = (float(p), float(snaps[key]["label"]), float(snaps[key]["end"]))
    print(f"  [{name}] regenerated {len(out)} OOF preds from booster", flush=True)
    return out


def _train_ensemble(max_markets: int, slice_row_groups: int, push: bool, hf_token: str) -> dict:
    """The real job: assemble aligned OOF base preds → out-of-time stacker →
    PAV calibration → blend-vs-base eval → artifacts. Runs on the H100 worker."""
    tok = hf_token or None
    if tok:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # 1) Assemble OOF preds per base — download if published, else regenerate.
    oof: Dict[str, dict] = {}
    source: Dict[str, str] = {}
    snaps = None
    for name in BASE_ORDER:
        tbl = _try_download_oof(name, tok)
        if tbl is not None:
            oof[name], source[name] = tbl, "downloaded"
            continue
        if snaps is None:
            snaps = _regen_snapshots(max_markets, slice_row_groups, tok)
        tbl = _score_base_on_snaps(name, snaps, tok)
        if tbl is None:
            raise SystemExit(f"could not obtain OOF preds for base '{name}' "
                             "(no published table and booster unavailable)")
        oof[name], source[name] = tbl, "regenerated"

    # Align by (market_id, timestamp): inner join across all three bases. Label /
    # end_date taken from whichever table carries them (regen always does).
    common = set(oof[BASE_ORDER[0]])
    for name in BASE_ORDER[1:]:
        common &= set(oof[name])
    aligned = []
    for key in common:
        preds = {name: oof[name][key][0] for name in BASE_ORDER}
        lab = next((oof[n][key][1] for n in BASE_ORDER if oof[n][key][1] is not None), None)
        end = next((oof[n][key][2] for n in BASE_ORDER if oof[n][key][2] is not None), None)
        if lab is None or end is None:
            continue
        aligned.append((key[0], key[1], preds, float(lab), float(end)))
    if len(aligned) < 300:
        raise SystemExit(f"only {len(aligned)} aligned OOF rows — need more overlap")
    print(f"aligned OOF rows: {len(aligned)} (sources: {source})", flush=True)

    # 2) Out-of-time split by end_date.
    aligned.sort(key=lambda r: r[4])
    n = len(aligned)
    tr_end, cal_end = int(n * 0.60), int(n * 0.75)
    tr, ca, te = aligned[:tr_end], aligned[tr_end:cal_end], aligned[cal_end:]

    def cols(chunk):
        base = {name: [r[2][name] for r in chunk] for name in BASE_ORDER}
        y = [r[3] for r in chunk]
        return base, y

    tr_base, ytr = cols(tr)
    ca_base, yca = cols(ca)
    te_base, yte = cols(te)

    def feats(base, i):
        return stacker_inputs([base[name][i] for name in BASE_ORDER])

    Xtr = [feats(tr_base, i) for i in range(len(ytr))]
    Xca = [feats(ca_base, i) for i in range(len(yca))]
    Xte = [feats(te_base, i) for i in range(len(yte))]
    fmean, fstd = zstats(Xtr)
    w, b = logreg_fit(znorm(Xtr, fmean, fstd), ytr)

    blend_raw = logreg_predict(w, b, znorm(Xte, fmean, fstd))
    curve = isotonic_fit(logreg_predict(w, b, znorm(Xca, fmean, fstd)), yca)
    blend_cal = [isotonic_predict(curve, p) for p in blend_raw]

    # 3) Report blend vs each base alone (on the same held-out rows).
    rep_raw = _base_vs_blend(te_base, blend_raw, yte)
    rep_cal = _base_vs_blend(te_base, blend_cal, yte)

    result = {
        "runtime": "modal H100 / meta-ensemble stacker (logistic over base OOF preds)",
        "dataset": REPO, "base_models": {n: BASE_REPOS[n] for n in BASE_ORDER},
        "oof_source": source,
        "aligned_rows": n, "train_rows": len(tr), "calib_rows": len(ca), "test_rows": len(te),
        "test_yes_rate": round(_mean(yte), 4),
        "stacker": {"type": "logistic", "inputs": [f"logit_{n}" for n in BASE_ORDER] + ["mean_prob"],
                    "weights": [round(float(x), 5) for x in w], "bias": round(float(b), 5)},
        "base_vs_blend_raw": rep_raw,
        "base_vs_blend_calibrated": rep_cal,
        "reliability_calibrated": calibration_table(blend_cal, yte),
        "decile_lift_calibrated": _decile_lift(blend_cal, yte),
        "baseline_brier": round(brier([_mean(ytr)] * len(yte), yte), 4),
    }
    print(f"[blend raw]  {rep_raw['blend']} | beats best single: {rep_raw['blend_beats_best_single_auc']}", flush=True)
    print(f"[blend cal]  {rep_cal['blend']} | beats best single: {rep_cal['blend_beats_best_single_auc']}", flush=True)

    # ── Artifacts: normaliser JSON (stacker weights + isotonic curve) + metrics ──
    import base64
    norm = {
        "winner": "logistic_stacker_calibrated",
        "base_order": BASE_ORDER, "base_repos": {n: BASE_REPOS[n] for n in BASE_ORDER},
        "stacker_input_order": [f"logit_{n}" for n in BASE_ORDER] + ["mean_prob"],
        "fmean": [round(float(x), 6) for x in fmean], "fstd": [round(float(x), 6) for x in fstd],
        "weights": [round(float(x), 6) for x in w], "bias": round(float(b), 6),
        "isotonic_breakpoints": [[round(float(x), 6), round(float(v), 6)] for x, v in curve],
        "label": "1 = market resolves YES (token1); meta-target = ground-truth resolution",
        "note": "inputs are each base model's probability in logit space + their mean; "
                "z-score with fmean/fstd, apply weights+bias (sigmoid), then map through "
                "isotonic_breakpoints.",
    }
    artifacts = {"ensemble_normalizer.json": json.dumps(norm, indent=2).encode()}
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    artifacts["ensemble_metrics.json"] = json.dumps(clean, indent=2).encode()
    result["_artifacts_b64"] = {k: base64.b64encode(v).decode() for k, v in artifacts.items()}

    if push and hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=OUT_REPO, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=artifacts["ensemble_normalizer.json"],
                            path_in_repo="ensemble_normalizer.json", repo_id=OUT_REPO, repo_type="model")
            api.upload_file(path_or_fileobj=artifacts["ensemble_metrics.json"],
                            path_in_repo="metrics/ensemble_metrics.json", repo_id=OUT_REPO, repo_type="model")
            api.upload_file(path_or_fileobj=model_card(clean).encode(),
                            path_in_repo="README.md", repo_id=OUT_REPO, repo_type="model")
            result["hf_repo"] = f"https://huggingface.co/{OUT_REPO}"
            print(f"pushed → {result['hf_repo']}", flush=True)
        except Exception as e:
            result["push_error"] = f"{type(e).__name__}: {e}"[:300]
            print("push failed:", result["push_error"], flush=True)

    return result


def model_card(m: dict) -> str:
    cal = m.get("base_vs_blend_calibrated", {})
    per = cal.get("per_base", {})
    blend = cal.get("blend", {})
    rows = "\n".join(
        f"| {n} (base) | {per.get(n, {}).get('auc')} | {per.get(n, {}).get('brier')} | "
        f"{per.get(n, {}).get('log_loss')} |" for n in BASE_ORDER)
    return f"""---
license: apache-2.0
tags:
- tabular-classification
- ensemble
- stacking
- calibration
- polymarket
- prediction-markets
language:
- en
---

# Polymarket meta-ensemble stacker (calibrated)

Blends the three base signal models — **resolution**, **flow**, **smart-money** —
into one calibrated probability by *stacking*: a small logistic model whose inputs
are the base models' own predictions (in logit space, plus their mean). Trained on
out-of-fold base predictions with a strict **out-of-time split by market
`end_date`**, so the stacker never sees a base pred from the test fold. Calibrated
with **isotonic regression via PAV** (implemented in-repo, no sklearn).

Base models: {', '.join(f'`{BASE_REPOS[n]}`' for n in BASE_ORDER)}.
OOF sources: {m.get('oof_source')}.

## Out-of-time test metrics (calibrated blend vs each base alone)
| model | AUC | Brier | log-loss |
|---|---|---|---|
{rows}
| **blend (calibrated)** | **{blend.get('auc')}** | **{blend.get('brier')}** | **{blend.get('log_loss')}** |

Best single base: {cal.get('best_single_base')}. Blend beats it on AUC:
**{cal.get('blend_beats_best_single_auc')}** (gain {cal.get('auc_gain_vs_best_single')}).
Baseline (predict train YES-rate): Brier {m.get('baseline_brier')}.
Aligned OOF rows: {m.get('aligned_rows')} (train {m.get('train_rows')} / calib
{m.get('calib_rows')} / test {m.get('test_rows')}).

## Honesty
Stacking only helps when the base models make *complementary* errors. The table
above is on held-out, later-resolving markets — if the blend does **not** beat the
best single base, that is reported as-is (`blend_beats_best_single_auc`), not hidden.
Not financial advice.

## Inference
Feed each base model's probability into `ensemble_normalizer.json`'s
`stacker_input_order` (logit of each base pred, in `base_order`, then their mean),
z-score with `fmean`/`fstd`, apply `weights`+`bias` through a sigmoid, then map the
result through `isotonic_breakpoints`.
"""


def _build_modal_app():
    """Construct the Modal App + remote function. Called only when modal is
    installed, so a bare import never touches modal."""
    app = modal.App("pmt-ensemble")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install("pyarrow", "numpy", "lightgbm", "datasets",
                     "huggingface_hub", "safetensors")
    )

    @app.function(image=image, gpu="H100", cpu=16.0, memory=131072, timeout=10800)
    def run(max_markets: int = 6000, slice_row_groups: int = 8,
            push: bool = False, hf_token: str = "") -> dict:
        return _train_ensemble(max_markets, slice_row_groups, push, hf_token)

    @app.local_entrypoint()
    def main(max_markets: int = 6000, slice_row_groups: int = 8, push: bool = False):
        import base64
        token = os.environ.get("HF_TOKEN", "")
        report = run.remote(max_markets=max_markets, slice_row_groups=slice_row_groups,
                            push=push, hf_token=token)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        for name, b64 in report.pop("_artifacts_b64", {}).items():
            with open(os.path.join(data_dir, name), "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"saved data/{name}")
        with open(os.path.join(data_dir, "ensemble_metrics.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print(f"\nwrote {data_dir}/ensemble_metrics.json")

    return app, run, main


if _HAS_MODAL:
    # Module-level `app` so `modal run ml/modal_ensemble.py` discovers it.
    app, run, main = _build_modal_app()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Meta-ensemble stacker trainer (Modal H100)")
    ap.add_argument("--smoke", action="store_true",
                    help="run the stacker on synthetic OOF preds with pure stdlib (no GPU/Modal)")
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(run_smoke())
    print("Nothing to do. Use --smoke for the local pipeline, or:\n"
          "  modal run ml/modal_ensemble.py --max-markets 6000 --push")
