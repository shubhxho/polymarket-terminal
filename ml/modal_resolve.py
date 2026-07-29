"""FLAGSHIP — calibrated market-resolution-probability trainer, on Modal H100.

The rest of the suite predicts short-horizon *direction*. This model answers the
higher-value question a trader actually cares about: **given a market's current
mid-market state, what is the probability it finally resolves YES?** The label is
ground truth — each market's recorded `outcome_prices` — so this is a genuine
resolution model, the first in the repo.

Pipeline (all inside the Modal function, streaming from HF):
1. Read `markets.parquet`; derive the binary YES/NO resolution label from the
   `outcome_prices` JSON string (`["0.99","0.01"]` → token1/YES won → 1).
2. Stream `trades.parquet` row-group by row-group, join to the resolved markets,
   fold every trade into a YES-perspective price + signed-aggressor USD flow.
3. Build labeled **mid-market snapshots** per market (price level, dist-from-0.5,
   time-to-resolution, recent momentum/vol, signed flow imbalance, volume
   maturity) — the same market resolves to one label, sampled at several points
   along its life.
4. Train a LightGBM GBDT + a small torch MLP; blend the probabilities.
5. **Isotonic calibration via PAV** (pool-adjacent-violators, implemented here —
   no sklearn) fit on a held-out fold.
6. Honest eval: **out-of-time / walk-forward split by market `end_date`** (train
   on earlier-resolving markets, test on later). Report Brier, log-loss, AUC and
   a reliability/calibration table. Expect modest AUC — resolution is hard.
7. Push safetensors + gbdt + normaliser + metrics + a generated card to
   `shubhxho/polymarket-resolution-model`.

    modal run ml/modal_resolve.py --max-markets 6000 --push

Local proof (zero heavy deps — pure stdlib logistic baseline + this PAV):

    python ml/modal_resolve.py --smoke

Heavy imports (modal/torch/lightgbm/pyarrow/numpy) live INSIDE functions so the
module imports cleanly under a bare interpreter with none of them installed.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import json
import math
import os
import random
from typing import List, Optional, Tuple

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
HF_MODEL_REPO = "shubhxho/polymarket-resolution-model"

# Snapshot fractions along each market's trade timeline (mid-market states).
SNAP_FRACS = (0.15, 0.35, 0.55, 0.75, 0.90)
RECENT = 24          # trailing trades for momentum / vol / recent flow
MIN_TRADES = 12      # markets with fewer trades give no usable snapshots
DECISIVE = 0.90      # a resolved market's winning outcome_price must clear this
EPS = 1e-9

FEATURE_NAMES = [
    "yes_price",           # current YES mid (level)
    "signed_dist",         # price - 0.5  (which side, how far)
    "dist_half",           # |price - 0.5|
    "time_remaining_frac", # 1 - fraction of market lifetime elapsed
    "log_hours_to_res",    # log1p(hours until end_date)
    "momentum_recent",     # YES price change over the recent window
    "vol_recent",          # std of recent YES price increments
    "flow_imb_recent",     # signed aggressor USD / USD over recent window
    "flow_imb_cum",        # signed aggressor USD / USD, cumulative to snapshot
    "vol_maturity",        # cumulative USD so far / market's total USD
    "log_trades",          # log1p(trades seen so far)
    "price_x_maturity",    # (price-0.5) * maturity  (mature + extreme = strong)
]


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
    """Binary YES/NO label from a market's `outcome_prices`.

    `["0.99","0.01"]` / `"['0.99','0.01']"` → token1/answer1 (YES) won → 1;
    `"['0.02','0.98']"` → 0. Returns None when unparseable or not decisively
    resolved (e.g. 0.5/0.5), so ambiguous/void markets never pollute training.
    """
    arr = _parse_outcome_prices(outcome_prices)
    if arr is None or len(arr) < 2:
        return None
    try:
        p1 = float(arr[0])
        p2 = float(arr[1])
    except Exception:
        return None
    if not (math.isfinite(p1) and math.isfinite(p2)):
        return None
    # Decisive resolution: the winner cleared DECISIVE and the loser collapsed
    # (min <= 1 - DECISIVE). This keeps clean resolutions like [0.92, 0.08] while
    # rejecting ties (0.5/0.5) and malformed rows that don't sum to ~1.
    if max(p1, p2) < DECISIVE or min(p1, p2) > 1.0 - DECISIVE:
        return None
    return 1 if p1 > p2 else 0


def yes_view(price: float, side: str, direction: str, usd: float) -> Tuple[float, float]:
    """Map one raw trade to (YES price, signed YES USD flow).

    `nonusdc_side` says which token traded; `token2` (NO) prices invert to the YES
    frame. Signed flow is +USD when the aggressor pushes YES up (buying YES, or
    selling NO), −USD otherwise.
    """
    yp = price if side == "token1" else 1.0 - price
    buy = direction == "BUY"
    yes_up = (buy and side == "token1") or ((not buy) and side == "token2")
    return yp, (usd if yes_up else -usd)


def snapshot_features(ts, yp, sg, uv, cum_uv, k, created_ts, end_ts) -> List[float]:
    """12 features for the mid-market state using trades[0:k] of one market.

    `cum_uv[j]` is the cumulative USD through trade j (prefix sum, len == n+1).
    """
    price = yp[k - 1]
    now = ts[k - 1]
    signed_dist = price - 0.5
    dist = abs(signed_dist)

    span = max(float(end_ts - created_ts), 1.0)
    elapsed = min(max((now - created_ts) / span, 0.0), 1.0)
    time_rem = 1.0 - elapsed
    hours_to_res = max(float(end_ts - now), 0.0) / 3600.0
    log_hours = math.log1p(hours_to_res)

    r0 = max(0, k - RECENT)
    recent_yp = yp[r0:k]
    rets = [recent_yp[i] - recent_yp[i - 1] for i in range(1, len(recent_yp))]
    momentum = (recent_yp[-1] - recent_yp[0]) if len(recent_yp) >= 2 else 0.0
    vol = _std(rets)

    recent_sg = sum(sg[r0:k])
    recent_uv = sum(uv[r0:k])
    flow_recent = recent_sg / (recent_uv + EPS)

    seen_uv = cum_uv[k]
    total_uv = cum_uv[-1]
    maturity = min(max(seen_uv / (total_uv + EPS), 0.0), 1.0)
    flow_cum = sum(sg[:k]) / (seen_uv + EPS)

    return [
        price,
        signed_dist,
        dist,
        time_rem,
        log_hours,
        momentum,
        vol,
        flow_recent,
        flow_cum,
        maturity,
        math.log1p(float(k)),
        signed_dist * maturity,
    ]


def market_snapshots(trades: List[Tuple[float, float, float, float]],
                     created_ts: float, end_ts: float, label: int
                     ) -> List[Tuple[List[float], int, float]]:
    """Build labeled snapshots for one market.

    `trades` are (ts, yes_price, signed_usd, usd) in ascending time. Returns rows
    of (features, label, end_ts) — end_ts carried for the out-of-time split.
    """
    n = len(trades)
    if n < MIN_TRADES:
        return []
    trades = sorted(trades, key=lambda t: t[0])
    ts = [t[0] for t in trades]
    yp = [t[1] for t in trades]
    sg = [t[2] for t in trades]
    uv = [max(t[3], 0.0) for t in trades]
    cum_uv = [0.0] * (n + 1)
    for i in range(n):
        cum_uv[i + 1] = cum_uv[i] + uv[i]

    rows = []
    seen = set()
    for fr in SNAP_FRACS:
        k = int(n * fr)
        if k < 2 or k in seen:
            continue
        seen.add(k)
        feats = snapshot_features(ts, yp, sg, uv, cum_uv, k, created_ts, end_ts)
        if all(math.isfinite(v) for v in feats):
            rows.append((feats, label, float(end_ts)))
    return rows


# ── metrics (pure) ───────────────────────────────────────────────────────────

def auc(scores: List[float], labels: List[float]) -> float:
    """Tie-aware ROC AUC (Mann–Whitney U)."""
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
    """Reliability bins: for each [b/bins, (b+1)/bins) predicted band, the mean
    prediction vs the empirical positive rate. A calibrated model has them equal.
    """
    table = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        last = b == bins - 1
        idx = [i for i, p in enumerate(probs) if p >= lo and (p < hi or (last and p <= hi))]
        if not idx:
            table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "mean_pred": None, "frac_pos": None})
            continue
        mp = _mean([probs[i] for i in idx])
        fp = _mean([float(labels[i]) for i in idx])
        table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": len(idx),
                      "mean_pred": round(mp, 4), "frac_pos": round(fp, 4)})
    return table


# ── isotonic calibration via PAV (pool-adjacent-violators) ───────────────────

def isotonic_fit(scores: List[float], labels: List[float]) -> List[Tuple[float, float]]:
    """Fit a non-decreasing calibration map score→prob with the PAV algorithm.

    Returns compact breakpoints [(threshold, calibrated_prob), …] ascending in
    threshold; `isotonic_predict` looks a raw score up against them.
    """
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    xs = [scores[i] for i in order]
    ys = [float(labels[i]) for i in order]

    # blocks: [sum_of_y, count]; merge while the running means violate monotonicity
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

    # collapse to breakpoints where the calibrated value changes
    bp: List[Tuple[float, float]] = []
    for x, v in zip(xs, cal):
        if not bp or abs(bp[-1][1] - v) > 1e-12:
            bp.append((x, v))
    return bp


def isotonic_predict(bp: List[Tuple[float, float]], s: float) -> float:
    if not bp:
        return s
    # bp is ascending in threshold; bisect the first tuple element directly via a
    # key so we don't rebuild an xs list on every call (matters when mapping a
    # whole test fold through a large calibration curve).
    i = bisect.bisect_right(bp, (s, math.inf)) - 1
    if i < 0:
        return bp[0][1]
    return bp[i][1]


# ── z-score normaliser + pure logistic-regression baseline (smoke only) ──────

def zstats(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    cols = list(zip(*X))
    fmean = [_mean(list(c)) for c in cols]
    fstd = [(_std(list(c)) or 1.0) for c in cols]
    return fmean, fstd


def znorm(X: List[List[float]], fmean: List[float], fstd: List[float]) -> List[List[float]]:
    return [[(row[j] - fmean[j]) / (fstd[j] + 1e-9) for j in range(len(row))] for row in X]


def logreg_fit(X: List[List[float]], y: List[float], epochs: int = 400,
               lr: float = 0.3, l2: float = 1e-3) -> Tuple[List[float], float]:
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


# ─────────────────────────── smoke pipeline (stdlib) ────────────────────────

def _synth_markets(n_markets: int = 240, seed: int = 11):
    """Synthetic resolved markets: a YES price that drifts toward its outcome,
    with signed flow tracking the drift. Later markets get later end_dates so the
    out-of-time split is meaningful. Returns list of market dicts.
    """
    rng = random.Random(seed)
    markets = []
    base_ts = 1_600_000_000
    for m in range(n_markets):
        label = 1 if rng.random() < 0.5 else 0
        target = 0.97 if label else 0.03
        info = rng.uniform(0.02, 0.06)  # convergence speed varies per market
        n = rng.randint(20, 90)
        created = base_ts + m * 86_400
        end = created + n * 3600 + 86_400
        price = rng.uniform(0.30, 0.70)
        trades = []
        for i in range(n):
            drift = info * (target - price)
            price = min(max(price + drift + rng.gauss(0, 0.03), 0.02), 0.98)
            usd = rng.uniform(10, 500)
            signed = usd * (1 if (drift + rng.gauss(0, 0.02)) > 0 else -1)
            ts = created + i * 3600
            trades.append((ts, price, signed, usd))
        op = ["0.97", "0.03"] if label else ["0.03", "0.97"]
        markets.append({"trades": trades, "created": created, "end": end,
                        "outcome_prices": json.dumps(op)})
    return markets


def run_smoke() -> int:
    print("== FLAGSHIP resolution model — SMOKE (pure-stdlib, synthetic) ==")
    markets = _synth_markets()

    # Derive labels through the real code path and build snapshots.
    built = []
    n_labeled = n_ambiguous = 0
    for mk in markets:
        lab = resolution_label(mk["outcome_prices"])
        if lab is None:
            n_ambiguous += 1
            continue
        n_labeled += 1
        rows = market_snapshots(mk["trades"], mk["created"], mk["end"], lab)
        if rows:
            built.append((mk["end"], rows))
    print(f"markets: {len(markets)}  labeled: {n_labeled}  ambiguous/void: {n_ambiguous}")

    # Out-of-time split by end_date: earliest→train, middle→calibration, latest→test.
    built.sort(key=lambda t: t[0])
    n = len(built)
    tr_end = int(n * 0.55)
    cal_end = int(n * 0.70)

    def flat(chunk):
        rows = []
        for _, rs in chunk:
            rows.extend(rs)
        return rows

    tr_rows = flat(built[:tr_end])
    cal_rows = flat(built[tr_end:cal_end])
    te_rows = flat(built[cal_end:])
    if not (tr_rows and cal_rows and te_rows):
        print("!! not enough data for a 3-way temporal split")
        return 1

    Xtr = [r[0] for r in tr_rows]; ytr = [float(r[1]) for r in tr_rows]
    Xca = [r[0] for r in cal_rows]; yca = [float(r[1]) for r in cal_rows]
    Xte = [r[0] for r in te_rows]; yte = [float(r[1]) for r in te_rows]
    print(f"snapshots — train {len(Xtr)}  calib {len(Xca)}  test {len(Xte)}  "
          f"(train YES-rate {_mean(ytr):.3f})")

    # Normalise on TRAIN only, fit the logistic baseline.
    fmean, fstd = zstats(Xtr)
    w, b = logreg_fit(znorm(Xtr, fmean, fstd), ytr)

    p_raw = logreg_predict(w, b, znorm(Xte, fmean, fstd))

    # Calibrate: fit PAV on the held-out calibration fold, apply to test.
    p_cal_fold = logreg_predict(w, b, znorm(Xca, fmean, fstd))
    curve = isotonic_fit(p_cal_fold, yca)
    p_cal = [isotonic_predict(curve, p) for p in p_raw]

    print("\n-- out-of-time test metrics --")
    print("                 AUC     Brier    LogLoss")
    print(f"  uncalibrated  {auc(p_raw, yte):.4f}  {brier(p_raw, yte):.4f}   {log_loss(p_raw, yte):.4f}")
    print(f"  calibrated    {auc(p_cal, yte):.4f}  {brier(p_cal, yte):.4f}   {log_loss(p_cal, yte):.4f}")

    base_rate = _mean(yte)
    print(f"  baseline (predict {base_rate:.3f} for all)  Brier {brier([base_rate] * len(yte), yte):.4f}")

    print("\n-- calibration / reliability table (calibrated) --")
    print("  bin          n     mean_pred  frac_pos")
    for row in calibration_table(p_cal, yte):
        mp = "  -  " if row["mean_pred"] is None else f"{row['mean_pred']:.3f}"
        fp = "  -  " if row["frac_pos"] is None else f"{row['frac_pos']:.3f}"
        print(f"  {row['bin']:<11} {row['n']:>4}    {mp:>6}     {fp:>6}")

    ok = auc(p_cal, yte) > 0.5 and 0.0 <= brier(p_cal, yte) <= 1.0
    print("\nSMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ─────────────────────────── Modal H100 pipeline ────────────────────────────
# Everything below imports heavy deps INSIDE the function bodies. Registered on
# the Modal App only when modal is installed (see the guard at the bottom).

def _train_resolution(max_markets: int, max_row_groups: int, epochs: int,
                      push: bool, hf_token: str) -> dict:
    """The real job: stream HF → snapshots → GBDT+MLP ensemble → PAV calibration
    → out-of-time eval → artifacts. Runs on the H100 worker."""
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    # 1. markets.parquet (85 MB) — labels + timing, fully in memory.
    mpath = hf_hub_download(REPO, "markets.parquet", repo_type="dataset")
    mt = pq.read_table(mpath, columns=["id", "outcome_prices", "created_at",
                                        "end_date", "volume", "closed"]).to_pydict()

    import datetime as _dt

    def _epoch(x):
        # end_date / created_at arrive as tz-aware datetimes (pyarrow → python),
        # but tolerate date, epoch numbers, or ISO strings too.
        if x is None:
            return None
        if isinstance(x, _dt.datetime):
            return x.timestamp()
        if isinstance(x, _dt.date):
            return _dt.datetime(x.year, x.month, x.day, tzinfo=_dt.timezone.utc).timestamp()
        if isinstance(x, (int, float)):
            return float(x)
        try:
            return float(x.timestamp())  # pandas.Timestamp and friends
        except Exception:
            pass
        try:
            return _dt.datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                return float(x)
            except Exception:
                return None

    meta = {}
    for i in range(len(mt["id"])):
        lab = resolution_label(mt["outcome_prices"][i])
        if lab is None:
            continue
        if mt["closed"][i] not in (1, True, "1"):
            continue
        c = _epoch(mt["created_at"][i])
        e = _epoch(mt["end_date"][i])
        if c is None or e is None or e <= c:
            continue
        vol = float(mt["volume"][i] or 0.0)
        meta[str(mt["id"][i])] = {"label": lab, "created": c, "end": e, "volume": vol}
    print(f"resolved+decisive markets: {len(meta)}", flush=True)

    # Keep the most-traded markets to bound trade-side memory.
    keep = set(sorted(meta, key=lambda m: meta[m]["volume"], reverse=True)[:max_markets])
    print(f"keeping top {len(keep)} markets by volume", flush=True)

    # 2. Stream trades.parquet, fold into per-market YES-frame trade lists.
    tpath = hf_hub_download(REPO, "trades.parquet", repo_type="dataset")
    pf = pq.ParquetFile(tpath)
    ngroups = pf.num_row_groups if max_row_groups <= 0 else min(max_row_groups, pf.num_row_groups)
    per_market = {m: [] for m in keep}
    cols = ["timestamp", "market_id", "price", "usd_amount", "taker_direction", "nonusdc_side"]
    for rg in range(ngroups):
        tb = pf.read_row_group(rg, columns=cols).to_pydict()
        mid = tb["market_id"]
        ts = tb["timestamp"]; pr = tb["price"]; ua = tb["usd_amount"]
        td = tb["taker_direction"]; ns = tb["nonusdc_side"]
        for i in range(len(mid)):
            lst = per_market.get(str(mid[i]))
            if lst is None:
                continue
            p = float(pr[i]); usd = float(ua[i])
            if not (0.0 < p < 1.0) or usd <= 0.0:
                continue
            yp, signed = yes_view(p, ns[i], td[i], usd)
            lst.append((float(ts[i]), yp, signed, usd))
        if (rg + 1) % 20 == 0 or rg == ngroups - 1:
            print(f"  streamed row-group {rg+1}/{ngroups}", flush=True)

    # 3. Snapshots.
    rows = []
    for m, trades in per_market.items():
        info = meta[m]
        rows.extend(market_snapshots(trades, info["created"], info["end"], info["label"]))
    if len(rows) < 500:
        return {"error": f"too few snapshots ({len(rows)}); raise max_markets/row_groups"}
    X = np.array([r[0] for r in rows], np.float32)
    y = np.array([r[1] for r in rows], np.float32)
    ends = np.array([r[2] for r in rows], np.float64)
    print(f"built {len(y)} snapshots, {X.shape[1]} features, YES-rate {y.mean():.3f}", flush=True)

    # 4. Out-of-time split by end_date: train < q60, calib q60..q75, test >= q75.
    q60, q75 = np.quantile(ends, 0.60), np.quantile(ends, 0.75)
    tr = ends < q60
    ca = (ends >= q60) & (ends < q75)
    te = ends >= q75
    if tr.sum() < 200 or ca.sum() < 50 or te.sum() < 100:
        # fall back to a fractional split if end_dates are too clumped
        order = np.argsort(ends)
        tr = np.zeros(len(y), bool); ca = tr.copy(); te = tr.copy()
        tr[order[: int(0.6 * len(y))]] = True
        ca[order[int(0.6 * len(y)): int(0.75 * len(y))]] = True
        te[order[int(0.75 * len(y)):]] = True

    fmean = X[tr].mean(0)
    fstd = X[tr].std(0) + 1e-6
    Xn = (X - fmean) / fstd

    result = {
        "runtime": "modal H100 / resolution", "dataset": REPO,
        "task": "calibrated market resolution probability (final YES/NO from mid-market state)",
        "features": FEATURE_NAMES, "snapshot_fracs": list(SNAP_FRACS), "recent_window": RECENT,
        "markets_used": len(keep), "snapshots": int(len(y)),
        "train": int(tr.sum()), "calib": int(ca.sum()), "test": int(te.sum()),
        "test_yes_rate": round(float(y[te].mean()), 4),
        "baseline_brier": round(float(np.mean((y[tr].mean() - y[te]) ** 2)), 4),
    }

    # ── GBDT ─────────────────────────────────────────────────────────────────
    import lightgbm as lgb
    dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=list(FEATURE_NAMES))
    dva = lgb.Dataset(X[ca], label=y[ca], reference=dtr)
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.02,
              "num_leaves": 47, "min_data_in_leaf": 100, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "seed": 11, "verbose": -1}
    bst = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    gbdt_te = bst.predict(X[te])
    gbdt_ca = bst.predict(X[ca])
    result["gbdt"] = {"auc": round(auc(list(gbdt_te), list(y[te])), 4),
                      "brier": round(brier(list(gbdt_te), list(y[te])), 4),
                      "log_loss": round(log_loss(list(gbdt_te), list(y[te])), 4)}
    result["feature_importance_gbdt"] = sorted(
        zip(FEATURE_NAMES, [round(float(x), 1) for x in bst.feature_importance("gain")]),
        key=lambda t: -t[1])
    print(f"[gbdt] {result['gbdt']}", flush=True)

    # ── torch MLP ─────────────────────────────────────────────────────────────
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self, d, hidden=48):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.3),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
                                     nn.Linear(hidden, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    Xt = torch.tensor(Xn, device=dev)
    yt = torch.tensor(y, device=dev)
    ti = torch.tensor(np.where(tr)[0], device=dev)
    ci = torch.tensor(np.where(ca)[0], device=dev)
    ei = torch.tensor(np.where(te)[0], device=dev)
    torch.manual_seed(11)
    net = Net(Xn.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    best_ll, best_state, mlp_te, mlp_ca = 1e9, None, None, None
    for ep in range(epochs):
        net.train()
        perm = ti[torch.randperm(len(ti), device=dev)]
        for bstart in range(0, len(ti), 4096):
            bi = perm[bstart:bstart + 4096]
            loss = bce(net(Xt[bi]), yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xt[ci])).cpu().numpy()
        ll = log_loss(list(pv), list(y[ca]))
        if ll < best_ll:
            best_ll = ll
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            with torch.no_grad():
                mlp_te = torch.sigmoid(net(Xt[ei])).cpu().numpy()
                mlp_ca = pv
    result["mlp"] = {"auc": round(auc(list(mlp_te), list(y[te])), 4),
                     "brier": round(brier(list(mlp_te), list(y[te])), 4),
                     "log_loss": round(log_loss(list(mlp_te), list(y[te])), 4)}
    print(f"[mlp] {result['mlp']}", flush=True)

    # ── Blend + isotonic (PAV) calibration on the held-out calib fold ─────────
    blend_te = (np.asarray(gbdt_te) + np.asarray(mlp_te)) / 2.0
    blend_ca = (np.asarray(gbdt_ca) + np.asarray(mlp_ca)) / 2.0
    result["ensemble_raw"] = {"auc": round(auc(list(blend_te), list(y[te])), 4),
                              "brier": round(brier(list(blend_te), list(y[te])), 4),
                              "log_loss": round(log_loss(list(blend_te), list(y[te])), 4)}
    curve = isotonic_fit(list(blend_ca), list(y[ca]))
    cal_te = [isotonic_predict(curve, float(p)) for p in blend_te]
    result["ensemble_calibrated"] = {"auc": round(auc(cal_te, list(y[te])), 4),
                                     "brier": round(brier(cal_te, list(y[te])), 4),
                                     "log_loss": round(log_loss(cal_te, list(y[te])), 4)}
    result["calibration_table"] = calibration_table(cal_te, list(y[te]))
    print(f"[ensemble raw] {result['ensemble_raw']}", flush=True)
    print(f"[ensemble calibrated] {result['ensemble_calibrated']}", flush=True)

    # ── Walk-forward over end_date quantile blocks ────────────────────────────
    wf = []
    edges = [float(np.quantile(ends, q)) for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    for kf in range(1, 5):
        tmask = ends < edges[kf]
        vmask = (ends >= edges[kf]) & (ends < edges[kf + 1])
        if tmask.sum() < 200 or vmask.sum() < 50:
            continue
        d1 = lgb.Dataset(X[tmask], label=y[tmask])
        b = lgb.train(params, d1, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
        pv = b.predict(X[vmask])
        wf.append({"fold": kf, "auc": round(auc(list(pv), list(y[vmask])), 4),
                   "brier": round(brier(list(pv), list(y[vmask])), 4), "n": int(vmask.sum())})
    result["walk_forward"] = {"folds": wf,
                              "mean_auc": round(float(np.mean([f["auc"] for f in wf])), 4) if wf else None,
                              "mean_brier": round(float(np.mean([f["brier"] for f in wf])), 4) if wf else None}
    print(f"[walk-forward] {result['walk_forward']}", flush=True)

    # ── Artifacts: safetensors (MLP) + gbdt.txt + normaliser (+ calib curve) ──
    import base64
    norm = {
        "features": FEATURE_NAMES, "fmean": fmean.tolist(), "fstd": fstd.tolist(),
        "winner": "ensemble_calibrated", "blend": "mean(gbdt, mlp)",
        "hidden": 48, "snapshot_fracs": list(SNAP_FRACS), "recent_window": RECENT,
        "isotonic_breakpoints": [[round(float(x), 6), round(float(v), 6)] for x, v in curve],
        "label": "1 = market resolves YES (token1); derived from outcome_prices",
    }
    if best_state is not None:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in best_state.items()}, "/tmp/resolve_mlp.safetensors")
    bst.save_model("/tmp/resolve_gbdt.txt")
    artifacts = {"resolve_normalizer.json": json.dumps(norm).encode()}
    for name, p in [("resolve_mlp.safetensors", "/tmp/resolve_mlp.safetensors"),
                    ("resolve_gbdt.txt", "/tmp/resolve_gbdt.txt")]:
        if os.path.exists(p):
            artifacts[name] = open(p, "rb").read()
    result["_artifacts_b64"] = {k: base64.b64encode(v).decode() for k, v in artifacts.items()}

    if push and hf_token:
        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
            for name, data in artifacts.items():
                api.upload_file(path_or_fileobj=data, path_in_repo=name,
                                repo_id=HF_MODEL_REPO, repo_type="model")
            clean = {k: v for k, v in result.items() if k != "_artifacts_b64"}
            api.upload_file(path_or_fileobj=json.dumps(clean, indent=2).encode(),
                            path_in_repo="metrics/resolve_metrics.json",
                            repo_id=HF_MODEL_REPO, repo_type="model")
            api.upload_file(path_or_fileobj=model_card(clean).encode(),
                            path_in_repo="README.md", repo_id=HF_MODEL_REPO, repo_type="model")
            result["hf_repo"] = f"https://huggingface.co/{HF_MODEL_REPO}"
            print(f"pushed → {result['hf_repo']}", flush=True)
        except Exception as e:
            result["push_error"] = f"{type(e).__name__}: {e}"[:300]
            print("push failed:", result["push_error"], flush=True)

    return result


def model_card(m: dict) -> str:
    cal = m.get("ensemble_calibrated", {})
    raw = m.get("ensemble_raw", {})
    wf = m.get("walk_forward", {})
    top = [f for f, _ in m.get("feature_importance_gbdt", [])[:6]]
    return f"""---
license: apache-2.0
tags:
- tabular-classification
- calibration
- polymarket
- prediction-markets
- resolution-probability
language:
- en
---

# Polymarket resolution-probability model (calibrated)

Given a market's **mid-market state**, this predicts the probability it finally
resolves **YES** (token1). Labels are ground truth — each market's recorded
`outcome_prices`. Trained on `{REPO}`, validated strictly **out-of-time** (split
by market `end_date`, so training markets resolve *before* test ones).

## Method
- Labeled mid-market snapshots at trade-timeline fractions {m.get('snapshot_fracs')}
  ({m.get('snapshots')} snapshots over {m.get('markets_used')} markets).
- Features: {', '.join(m.get('features', []))}.
- Ensemble: LightGBM GBDT + a small torch MLP, mean-blended.
- **Isotonic calibration via PAV** (pool-adjacent-violators) on a held-out fold.

## Out-of-time test metrics
| model | AUC | Brier | log-loss |
|---|---|---|---|
| ensemble (raw) | {raw.get('auc')} | {raw.get('brier')} | {raw.get('log_loss')} |
| ensemble (calibrated) | {cal.get('auc')} | {cal.get('brier')} | {cal.get('log_loss')} |

Baseline (predict train YES-rate for all): Brier {m.get('baseline_brier')}.
Walk-forward mean AUC {wf.get('mean_auc')}, mean Brier {wf.get('mean_brier')}
across time folds. Top GBDT features: {top}.

## Honesty
Resolution from mid-market state is genuinely hard — AUC is modest and *not*
inflated. Calibration (Brier / reliability table in `metrics/resolve_metrics.json`)
is the point: the served probabilities are meant to be trustworthy, not just
rank-ordered. Not financial advice.

## Inference
Load `resolve_gbdt.txt` (LightGBM) and/or `resolve_mlp.safetensors` (torch), mean-
blend, then map through `isotonic_breakpoints` in `resolve_normalizer.json`
(which also carries the z-score `fmean`/`fstd` and feature order).
"""


def _build_modal_app():
    """Construct the Modal App + remote function. Called only when modal is
    installed, so a bare import never touches modal."""
    app = modal.App("pmt-resolve")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install("pyarrow", "numpy", "torch", "lightgbm",
                     "huggingface_hub", "safetensors")
    )

    @app.function(image=image, gpu="H100", cpu=16.0, memory=131072, timeout=10800)
    def run(max_markets: int = 6000, max_row_groups: int = 0, epochs: int = 60,
            push: bool = False, hf_token: str = "") -> dict:
        return _train_resolution(max_markets, max_row_groups, epochs, push, hf_token)

    @app.local_entrypoint()
    def main(max_markets: int = 6000, max_row_groups: int = 0, epochs: int = 60,
             push: bool = False):
        import base64
        token = os.environ.get("HF_TOKEN", "")
        report = run.remote(max_markets=max_markets, max_row_groups=max_row_groups,
                            epochs=epochs, push=push, hf_token=token)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        for name, b64 in report.pop("_artifacts_b64", {}).items():
            with open(os.path.join(data_dir, name), "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"saved data/{name}")
        with open(os.path.join(data_dir, "resolve_metrics.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print(f"\nwrote {data_dir}/resolve_metrics.json")

    return app, run, main


if _HAS_MODAL:
    # Module-level `app` so `modal run ml/modal_resolve.py` discovers it.
    app, run, main = _build_modal_app()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Flagship resolution-probability trainer")
    ap.add_argument("--smoke", action="store_true",
                    help="run the full pipeline on synthetic data with pure stdlib (no GPU/Modal)")
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(run_smoke())
    print("Nothing to do. Use --smoke for the local pipeline, or:\n"
          "  modal run ml/modal_resolve.py --max-markets 6000 --push")
