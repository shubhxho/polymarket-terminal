"""FLAGSHIP — trade-tape Transformer trainer (attention over order-flow), on H100.

The rest of the suite reads *aggregated* windows: hourly OHLCV bars (bigdata's
bi-GRU) or hand summary features (the GBDT/MLP). This model goes a level deeper —
it reads the **raw trade tape** itself. Per token it builds a causal sequence of
per-trade tokens carrying (YES price, signed aggressor USD flow, trade size,
dt-since-last-trade, side) and runs a small **Transformer encoder** with
multi-head self-attention over that order-flow sequence. A causal mask means each
position attends only to *earlier* trades — no lookahead — and the final position's
representation feeds a **multi-task head**:

  • next-move **direction** (does the YES price rise over the next HORIZON trades), and
  • eventual **resolution** (does the market finally resolve YES — ground truth from
    `outcome_prices`).

Honest eval, **out-of-time by market `end_date`** (train on earlier-resolving
markets, test on later): direction AUC + a decile backtest, resolution AUC +
Brier, plus **temperature-scaled calibration** fit on a held-out fold. The model
card compares it candidly against the GBDT / GRU baselines — a heavier flagship
variant, not automatically a better one.

    modal run ml/modal_transformer.py --max-markets 6000 --push

Local proof (zero heavy deps — a pure-stdlib, attention-free EWMA+logistic
baseline over the *same* sequence features, on a synthetic tape with a planted
temporal pattern):

    python ml/modal_transformer.py --smoke

Every heavy import (modal/torch/pyarrow/numpy/huggingface_hub/safetensors) lives
INSIDE a function or the Modal image, so this module imports cleanly under a bare
interpreter with none of them installed. The Modal `App` is built lazily only
when `modal` is importable.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
from typing import List, Optional, Sequence, Tuple

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
HF_MODEL_REPO = "shubhxho/polymarket-transformer-model"

# Sequence geometry. A "token" is one trade; the encoder reads the trailing
# SEQ_LEN trades ending at a snapshot point and predicts what happens next.
SEQ_LEN = 48         # trades per causal window (the attention context)
HORIZON = 6          # trades ahead for the next-move direction label
SNAP_FRACS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # window end-points along a market's tape
MIN_TRADES = SEQ_LEN + HORIZON + 2           # need a full window + a forward label
DECISIVE = 0.90      # a resolved market's winning outcome_price must clear this
EPS = 1e-9

# Per-trade token channels (raw; z-scored on train stats before the encoder).
# Exactly the tape signal the task calls for: price, signed flow, size, dt, side.
SEQ_CHANNELS = ["yes_price", "signed_flow", "log_size", "log_dt", "side"]


# ─────────────────────────── pure-stdlib helpers ────────────────────────────
# Shared by BOTH the smoke path and the Modal path. No numpy, no heavy deps —
# so the smoke run exercises the real sequence-build / label / eval code.

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
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
    back to `ast.literal_eval` for the Python-repr case."""
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
    resolved (e.g. 0.5/0.5), so ambiguous/void markets never pollute training."""
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
    if max(p1, p2) < DECISIVE or min(p1, p2) > 1.0 - DECISIVE:
        return None
    return 1 if p1 > p2 else 0


def yes_view(price: float, side: str, direction: str, usd: float) -> Tuple[float, float, float]:
    """Map one raw trade to (YES price, signed YES USD flow, side sign).

    `nonusdc_side` says which token traded; `token2` (NO) prices invert to the YES
    frame. Signed flow is +USD when the aggressor pushes YES up (buying YES, or
    selling NO), −USD otherwise. Side sign is +1 for token1 (YES), −1 for token2."""
    is_yes = side == "token1"
    yp = price if is_yes else 1.0 - price
    buy = direction == "BUY"
    yes_up = (buy and is_yes) or ((not buy) and not is_yes)
    return yp, (usd if yes_up else -usd), (1.0 if is_yes else -1.0)


# ── sequence builder (the heart — shared by smoke + Modal) ───────────────────

def build_sequences(trades: List[Tuple[float, float, float, float, float]],
                    created_ts: float, end_ts: float, label: int
                    ) -> List[Tuple[List[List[float]], int, int, float, float, float]]:
    """Build causal trade-tape windows for one market.

    `trades` are (ts, yes_price, signed_usd, usd, side_sign) in *ascending* time.
    For each snapshot fraction we take the trailing SEQ_LEN trades ending there and
    emit one sample:

        (seq, dir_label, res_label, fwd, snap_ts, end_ts)

    where `seq` is a SEQ_LEN×len(SEQ_CHANNELS) matrix of per-trade tokens, and
    `dir_label` is 1 iff YES price rises over the next HORIZON trades (strictly
    causal — the window never contains those future trades). `res_label` is the
    market's ground-truth resolution. `end_ts` rides along for the out-of-time
    split; `fwd` (forward price change) drives the decile backtest."""
    n = len(trades)
    if n < MIN_TRADES:
        return []
    trades = sorted(trades, key=lambda t: t[0])
    ts = [t[0] for t in trades]
    yp = [t[1] for t in trades]
    sg = [t[2] for t in trades]
    uv = [max(t[3], 0.0) for t in trades]
    sd = [t[4] for t in trades]

    rows = []
    seen = set()
    for fr in SNAP_FRACS:
        k = int(n * fr)                       # window ends just before trade k
        if k < SEQ_LEN or k + HORIZON > n or k in seen:
            continue
        seen.add(k)
        seq = []
        start = k - SEQ_LEN
        for j in range(start, k):
            dt = ts[j] - ts[j - 1] if j > start else 0.0
            seq.append([
                yp[j],                        # YES price level
                sg[j],                        # signed aggressor USD flow
                math.log1p(uv[j]),            # trade size (log USD)
                math.log1p(max(dt, 0.0)),     # dt since previous trade in window
                sd[j],                        # side (+1 YES token, −1 NO token)
            ])
        cur = yp[k - 1]
        nxt = yp[k - 1 + HORIZON]
        fwd = nxt - cur
        if not all(math.isfinite(v) for r in seq for v in r) or not math.isfinite(fwd):
            continue
        rows.append((seq, 1 if fwd > 0 else 0, int(label), float(fwd),
                     float(ts[k - 1]), float(end_ts)))
    return rows


def _ewma(xs: Sequence[float], alpha: float = 0.3) -> float:
    """Recency-weighted mean — the temporal reduction the stdlib baseline uses in
    place of attention (recent trades dominate, so *order* matters)."""
    if not xs:
        return 0.0
    v = xs[0]
    for x in xs[1:]:
        v = alpha * x + (1 - alpha) * v
    return v


def sequence_summary(seq: List[List[float]]) -> List[float]:
    """Reduce one SEQ_LEN×C tape window to a fixed feature vector for the
    attention-free baseline. EWMA over the channels gives recent trades more
    weight — the order-aware part a bag-of-means would throw away."""
    cols = list(zip(*seq))                    # per-channel series over the window
    price, flow, size, dt, side = cols
    rets = [price[i] - price[i - 1] for i in range(1, len(price))]
    last_p = price[-1]
    return [
        last_p,                               # current level
        last_p - 0.5,                         # signed distance from 0.5
        abs(last_p - 0.5),                    # extremeness
        price[-1] - price[0],                 # window drift
        _ewma(rets) if rets else 0.0,         # recency-weighted momentum
        _ewma(list(flow)),                    # recency-weighted signed flow
        _mean(list(flow)),                    # net flow over window
        _ewma(list(side)),                    # recent side bias
        _std(list(price)),                    # window volatility
        _mean(list(size)),                    # typical trade size
    ]


# ── metrics (pure stdlib, matching train_seq.py conventions) ─────────────────

def _auc(probs: Sequence[float], labels: Sequence[float]) -> float:
    """Tie-aware ROC AUC (Mann–Whitney U) — same convention as train_seq._auc."""
    pairs = sorted(zip(probs, labels), key=lambda t: t[0])
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


def _decile_backtest(probs: Sequence[float], fwd: Sequence[float], q: float = 0.2) -> dict:
    """Signal quality — same convention as train_seq._decile_backtest. Take the
    top and bottom `q` of windows by model score and compare how they actually
    moved next: `up_rate_spread` is the robust headline."""
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    if not order:
        return {}
    k = max(1, int(len(order) * q))
    top = [fwd[i] for i in order[-k:]]
    bottom = [fwd[i] for i in order[:k]]

    def _median(xs):
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    up = lambda xs: sum(1 for v in xs if v > 0) / len(xs)  # noqa: E731
    return {
        "top_up_rate": round(up(top), 3),
        "bottom_up_rate": round(up(bottom), 3),
        "up_rate_spread": round(up(top) - up(bottom), 3),
        "top_median_pts": round(_median(top) * 100, 3),
        "bottom_median_pts": round(_median(bottom) * 100, 3),
        "slice": k,
    }


def brier(probs: Sequence[float], labels: Sequence[float]) -> float:
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def log_loss(probs: Sequence[float], labels: Sequence[float]) -> float:
    if not probs:
        return 0.0
    s = 0.0
    for p, y in zip(probs, labels):
        p = min(max(p, 1e-6), 1 - 1e-6)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(probs)


def calibration_table(probs: Sequence[float], labels: Sequence[float], bins: int = 10) -> List[dict]:
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


# ── temperature scaling (pure stdlib; used by smoke + Modal) ──────────────────

def fit_temperature(logits: Sequence[float], labels: Sequence[float],
                    iters: int = 300, lr: float = 0.05) -> float:
    """Fit a single scalar temperature T minimising NLL of sigmoid(logit / T) via
    gradient descent on log T (keeps T > 0). T > 1 softens over-confident logits.

    dNLL/d(logT) = mean[(p − y) · (−logit/T)]  where p = sigmoid(logit/T)."""
    if not logits:
        return 1.0
    logt = 0.0
    for _ in range(iters):
        T = math.exp(logt)
        g = 0.0
        for z, y in zip(logits, labels):
            a = z / T
            g += (_sigmoid(a) - y) * (-a)
        g /= len(logits)
        logt -= lr * g
        logt = max(min(logt, 3.0), -3.0)     # clamp T to ~[0.05, 20]
    return math.exp(logt)


def apply_temperature(logits: Sequence[float], T: float) -> List[float]:
    return [_sigmoid(z / T) for z in logits]


# ── z-score normaliser + logistic regression (smoke baseline only) ───────────

def zstats(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    cols = list(zip(*X))
    return [_mean(list(c)) for c in cols], [(_std(list(c)) or 1.0) for c in cols]


def znorm(X: List[List[float]], fmean: List[float], fstd: List[float]) -> List[List[float]]:
    return [[(row[j] - fmean[j]) / (fstd[j] + 1e-9) for j in range(len(row))] for row in X]


def logreg_fit(X: List[List[float]], y: List[float], epochs: int = 400,
               lr: float = 0.3, l2: float = 1e-3) -> Tuple[List[float], float]:
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            e = _sigmoid(b + sum(w[j] * xi[j] for j in range(d))) - yi
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * (gb / n)
    return w, b


def logreg_logits(w: List[float], b: float, X: List[List[float]]) -> List[float]:
    return [b + sum(w[j] * row[j] for j in range(len(row))) for row in X]


# ─────────────────────────── smoke pipeline (stdlib) ────────────────────────

def _synth_tape(n_markets: int = 260, seed: int = 11):
    """Synthetic resolved markets with a *planted temporal pattern* so both tasks
    are learnable from the tape (and the sequence-build path is exercised end to
    end). YES price drifts toward its outcome (→ resolution signal); signed
    aggressor flow leads the next move (→ direction signal). Later markets get
    later end_dates so the out-of-time split by end_date is meaningful."""
    rng = random.Random(seed)
    markets = []
    base_ts = 1_600_000_000
    for m in range(n_markets):
        label = 1 if rng.random() < 0.5 else 0
        target = 0.97 if label else 0.03
        info = rng.uniform(0.015, 0.05)       # convergence speed toward outcome
        n = rng.randint(80, 200)
        created = base_ts + m * 86_400
        price = rng.uniform(0.30, 0.70)
        trades = []
        for i in range(n):
            drift = info * (target - price)   # pull toward the eventual outcome
            momentum = drift + rng.gauss(0, 0.02)
            # Aggressor side leads the next move: flow sign tracks momentum, so a
            # causal model can read recent flow to anticipate the next tick.
            buys = momentum > 0
            side_yes = rng.random() < 0.7     # mostly the YES token trades
            usd = rng.uniform(10, 500)
            ts = created + i * 3600 + rng.randint(0, 600)
            raw_price = price if side_yes else 1.0 - price
            direction = "BUY" if buys else "SELL"
            yp, signed, ss = yes_view(raw_price,
                                      "token1" if side_yes else "token2",
                                      direction, usd)
            trades.append((float(ts), yp, signed, usd, ss))
            price = min(max(price + momentum + rng.gauss(0, 0.02), 0.02), 0.98)
        end = created + n * 3600 + 86_400
        op = ["0.97", "0.03"] if label else ["0.03", "0.97"]
        markets.append({"trades": trades, "created": float(created), "end": float(end),
                        "outcome_prices": json.dumps(op)})
    return markets


def run_smoke() -> int:
    print("== FLAGSHIP trade-tape Transformer — SMOKE (pure-stdlib EWMA+logistic baseline) ==")
    markets = _synth_tape()

    # Build sequences through the REAL code path (labels via resolution_label).
    built, n_labeled, n_ambiguous = [], 0, 0
    for mk in markets:
        lab = resolution_label(mk["outcome_prices"])
        if lab is None:
            n_ambiguous += 1
            continue
        n_labeled += 1
        rows = build_sequences(mk["trades"], mk["created"], mk["end"], lab)
        if rows:
            built.append((mk["end"], rows))
    print(f"markets: {len(markets)}  labeled: {n_labeled}  ambiguous/void: {n_ambiguous}  "
          f"seq_len={SEQ_LEN} horizon={HORIZON} channels={SEQ_CHANNELS}")

    # Out-of-time split by end_date: earliest→train, middle→calib, latest→test.
    built.sort(key=lambda t: t[0])
    n = len(built)
    tr_end, cal_end = int(n * 0.60), int(n * 0.75)

    def flat(chunk):
        out = []
        for _, rs in chunk:
            out.extend(rs)
        return out

    tr, ca, te = flat(built[:tr_end]), flat(built[tr_end:cal_end]), flat(built[cal_end:])
    if not (tr and ca and te):
        print("!! not enough data for a 3-way temporal split")
        return 1

    # Reduce every tape window to summary features (the attention-free stand-in).
    def feats(rows):
        return [sequence_summary(r[0]) for r in rows]

    Xtr, Xca, Xte = feats(tr), feats(ca), feats(te)
    ydir = lambda rows: [float(r[1]) for r in rows]  # noqa: E731
    yres = lambda rows: [float(r[2]) for r in rows]  # noqa: E731
    fwd_te = [r[3] for r in te]
    print(f"windows — train {len(tr)}  calib {len(ca)}  test {len(te)}  "
          f"(train up-rate {_mean(ydir(tr)):.3f}, YES-rate {_mean(yres(tr)):.3f})")

    fmean, fstd = zstats(Xtr)
    Ntr, Nca, Nte = znorm(Xtr, fmean, fstd), znorm(Xca, fmean, fstd), znorm(Xte, fmean, fstd)

    # ── Task 1: next-move DIRECTION ──────────────────────────────────────────
    wd, bd = logreg_fit(Ntr, ydir(tr))
    dir_logit_te = logreg_logits(wd, bd, Nte)
    dir_p_te = [_sigmoid(z) for z in dir_logit_te]
    dir_auc = _auc(dir_p_te, ydir(te))
    dir_bt = _decile_backtest(dir_p_te, fwd_te)

    # ── Task 2: RESOLUTION, temperature-calibrated on the held-out fold ──────
    wr, br = logreg_fit(Ntr, yres(tr))
    res_logit_ca = logreg_logits(wr, br, Nca)
    res_logit_te = logreg_logits(wr, br, Nte)
    res_raw_te = [_sigmoid(z) for z in res_logit_te]
    T = fit_temperature(res_logit_ca, yres(ca))
    res_cal_te = apply_temperature(res_logit_te, T)
    yte_res = yres(te)

    print("\n-- out-of-time test metrics --")
    print(f"  [direction]  AUC {dir_auc:.4f}   Brier {brier(dir_p_te, ydir(te)):.4f}")
    print(f"    decile backtest: {dir_bt}")
    print(f"  [resolution] AUC {_auc(res_raw_te, yte_res):.4f}   "
          f"Brier raw {brier(res_raw_te, yte_res):.4f} → cal {brier(res_cal_te, yte_res):.4f}  (T={T:.3f})")

    base_rate = _mean(yte_res)
    print(f"    baseline (predict {base_rate:.3f}): Brier {brier([base_rate] * len(yte_res), yte_res):.4f}")
    print("\n-- resolution reliability (calibrated) --")
    print("  bin          n     mean_pred  frac_pos")
    for row in calibration_table(res_cal_te, yte_res):
        mp = "  -  " if row["mean_pred"] is None else f"{row['mean_pred']:.3f}"
        fp = "  -  " if row["frac_pos"] is None else f"{row['frac_pos']:.3f}"
        print(f"  {row['bin']:<11} {row['n']:>4}    {mp:>6}     {fp:>6}")

    ok = (dir_auc > 0.5 and _auc(res_raw_te, yte_res) > 0.5
          and dir_bt.get("up_rate_spread", 0) > 0
          and 0.0 <= brier(res_cal_te, yte_res) <= 1.0)
    print("\nSMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ─────────────────────────── Modal H100 pipeline ────────────────────────────
# Everything below imports heavy deps INSIDE the function bodies. Registered on
# the Modal App only when modal is installed (see the guard at the bottom).

def _train_transformer(max_markets: int, max_row_groups: int, epochs: int,
                       push: bool, hf_token: str) -> dict:
    """The real job: stream HF → per-market trade tape → causal Transformer over
    the sequence → multi-task (direction + resolution) → temperature calibration
    → out-of-time eval → artifacts. Runs on the H100 worker."""
    import datetime as _dt

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

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
            return float(x.timestamp())      # pandas.Timestamp and friends
        except Exception:
            pass
        try:
            return _dt.datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                return float(x)
            except Exception:
                return None

    # 1. markets.parquet (85 MB) — labels + timing, fully in memory.
    mpath = hf_hub_download(REPO, "markets.parquet", repo_type="dataset")
    mt = pq.read_table(mpath, columns=["id", "token1", "outcome_prices",
                                       "created_at", "end_date", "volume", "closed"]).to_pydict()
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
        meta[str(mt["id"][i])] = {"label": lab, "created": c, "end": e,
                                  "volume": float(mt["volume"][i] or 0.0)}
    print(f"resolved+decisive markets: {len(meta)}", flush=True)

    keep = set(sorted(meta, key=lambda m: meta[m]["volume"], reverse=True)[:max_markets])
    print(f"keeping top {len(keep)} markets by volume", flush=True)

    # 2. Stream trades.parquet, fold into per-market YES-frame trade tapes.
    tpath = hf_hub_download(REPO, "trades.parquet", repo_type="dataset")
    pf = pq.ParquetFile(tpath)
    ngroups = pf.num_row_groups if max_row_groups <= 0 else min(max_row_groups, pf.num_row_groups)
    per_market: dict = {m: [] for m in keep}
    cols = ["timestamp", "market_id", "price", "usd_amount", "taker_direction", "nonusdc_side"]
    for rg in range(ngroups):
        tb = pf.read_row_group(rg, columns=cols).to_pydict()
        mid = tb["market_id"]
        ts, pr, ua = tb["timestamp"], tb["price"], tb["usd_amount"]
        td, ns = tb["taker_direction"], tb["nonusdc_side"]
        for i in range(len(mid)):
            lst = per_market.get(str(mid[i]))
            if lst is None:
                continue
            p, usd = float(pr[i]), float(ua[i])
            if not (0.0 < p < 1.0) or usd <= 0.0:
                continue
            yp, signed, ss = yes_view(p, ns[i], td[i], usd)
            lst.append((float(ts[i]), yp, signed, usd, ss))
        if (rg + 1) % 20 == 0 or rg == ngroups - 1:
            print(f"  streamed row-group {rg+1}/{ngroups}", flush=True)

    # 3. Build causal tape windows (multi-task labels).
    rows = []
    for m, trades in per_market.items():
        info = meta[m]
        rows.extend(build_sequences(trades, info["created"], info["end"], info["label"]))
    if len(rows) < 500:
        return {"error": f"too few windows ({len(rows)}); raise max_markets/row_groups"}

    S = np.array([r[0] for r in rows], np.float32)   # [N, SEQ_LEN, C]
    ydir = np.array([r[1] for r in rows], np.float32)
    yres = np.array([r[2] for r in rows], np.float32)
    fwd = np.array([r[3] for r in rows], np.float32)
    ends = np.array([r[5] for r in rows], np.float64)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"built {len(rows)} windows, seq {tuple(S.shape[1:])}, "
          f"up-rate {ydir.mean():.3f}, YES-rate {yres.mean():.3f}", flush=True)

    # 4. Out-of-time split by end_date: train < q60, calib q60..q75, test >= q75.
    q60, q75 = np.quantile(ends, 0.60), np.quantile(ends, 0.75)
    tr = ends < q60
    ca = (ends >= q60) & (ends < q75)
    te = ends >= q75
    if tr.sum() < 200 or ca.sum() < 50 or te.sum() < 100:
        order = np.argsort(ends)
        tr = np.zeros(len(rows), bool); ca = tr.copy(); te = tr.copy()
        tr[order[: int(0.6 * len(rows))]] = True
        ca[order[int(0.6 * len(rows)): int(0.75 * len(rows))]] = True
        te[order[int(0.75 * len(rows)):]] = True

    # Train-only per-channel standardisation (no val/test statistics leak in).
    flat_tr = S[tr].reshape(-1, S.shape[-1])
    smean = flat_tr.mean(0)
    sstd = flat_tr.std(0) + 1e-6
    Sn = (S - smean) / sstd

    result = {
        "runtime": "modal H100 / trade-tape Transformer (causal self-attention)",
        "dataset": REPO,
        "task": "multi-task: next-move direction + market resolution, from the raw trade tape",
        "seq_channels": SEQ_CHANNELS, "seq_len": SEQ_LEN, "horizon": HORIZON,
        "snapshot_fracs": list(SNAP_FRACS),
        "windows": int(len(rows)), "markets_used": len(keep),
        "train": int(tr.sum()), "calib": int(ca.sum()), "test": int(te.sum()),
        "test_up_rate": round(float(ydir[te].mean()), 4),
        "test_yes_rate": round(float(yres[te].mean()), 4),
        "baseline_brier_resolution": round(float(np.mean((yres[tr].mean() - yres[te]) ** 2)), 4),
    }

    # ── Causal Transformer encoder + multi-task head (GPU, fp32) ──────────────
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(11)

    D_MODEL, NHEAD, LAYERS, FF, DROPOUT = 64, 4, 3, 128, 0.1

    def _positional(n_pos, d):
        pe = torch.zeros(n_pos, d)
        pos = torch.arange(0, n_pos, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    class TapeTransformer(nn.Module):
        """Project each trade token → d_model, add sinusoidal positional encoding,
        run a stack of self-attention layers under a CAUSAL mask (position t sees
        only trades ≤ t), then read the final position into two linear heads."""

        def __init__(self, n_ch, d_model=D_MODEL, nhead=NHEAD, layers=LAYERS,
                     ff=FF, dropout=DROPOUT, max_len=SEQ_LEN):
            super().__init__()
            self.inp = nn.Linear(n_ch, d_model)
            self.register_buffer("pos", _positional(max_len, d_model))
            enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                             activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.norm = nn.LayerNorm(d_model)
            self.dir_head = nn.Linear(d_model, 1)
            self.res_head = nn.Linear(d_model, 1)

        def forward(self, x):
            L = x.size(1)
            h = self.inp(x) + self.pos[:L].unsqueeze(0)
            mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
            h = self.enc(h, mask=mask)
            h = self.norm(h[:, -1, :])         # final (most recent) position
            return self.dir_head(h).squeeze(-1), self.res_head(h).squeeze(-1)

    St = torch.tensor(Sn, device=dev)
    ydt = torch.tensor(ydir, device=dev)
    yrt = torch.tensor(yres, device=dev)
    ti = torch.tensor(np.where(tr)[0], device=dev)
    ei = torch.tensor(np.where(te)[0], device=dev)

    # Split the calib fold into two DISJOINT temporal halves so checkpoint
    # selection and temperature fitting never see the same windows: the earlier
    # half selects the best epoch, the later half (closest in time to test) fits
    # the temperature. This keeps the reported calibrated Brier honest — it is
    # never tuned on the same data used to pick the checkpoint.
    ca_np = np.where(ca)[0]
    ca_np = ca_np[np.argsort(ends[ca_np])]        # chronological within the fold
    half = len(ca_np) // 2
    if half < 1 or len(ca_np) - half < 1:         # tiny fold → fall back to reuse
        sel_np = cal_np = ca_np
    else:
        sel_np, cal_np = ca_np[:half], ca_np[half:]
    ci_sel = torch.tensor(sel_np, device=dev)
    ci_cal = torch.tensor(cal_np, device=dev)

    net = TapeTransformer(S.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    # Class-balanced positive weights per head.
    def _pw(mask_idx, y):
        pos = float(y[mask_idx].sum()); neg = float(len(mask_idx) - pos)
        return torch.tensor([neg / pos if pos > 0 else 1.0], device=dev)
    bce_dir = nn.BCEWithLogitsLoss(pos_weight=_pw(np.where(tr)[0], ydir))
    bce_res = nn.BCEWithLogitsLoss(pos_weight=_pw(np.where(tr)[0], yres))
    BATCH = 4096

    def _logits(idx):
        net.eval()
        od, orr = [], []
        with torch.no_grad():
            for b in range(0, len(idx), BATCH):
                bb = idx[b:b + BATCH]
                ld, lr = net(St[bb])
                od.append(ld.cpu()); orr.append(lr.cpu())
        return torch.cat(od).numpy(), torch.cat(orr).numpy()

    best_auc, best_state, patience = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        perm = ti[torch.randperm(len(ti), device=dev)]
        for b in range(0, len(ti), BATCH):
            bi = perm[b:b + BATCH]
            opt.zero_grad(set_to_none=True)
            ld, lr = net(St[bi])
            loss = bce_dir(ld, ydt[bi]) + bce_res(lr, yrt[bi])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        dlog_sel, rlog_sel = _logits(ci_sel)
        # Model-selection score: mean of the two AUCs on the selection half only.
        sel = 0.5 * (_auc([_sigmoid(z) for z in dlog_sel], list(ydir[sel_np]))
                     + _auc([_sigmoid(z) for z in rlog_sel], list(yres[sel_np])))
        if sel > best_auc:
            best_auc, patience = sel, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 12:
                print(f"  early stop at epoch {ep} (best calib mean-AUC {best_auc:.4f})", flush=True)
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    # ── Out-of-time evaluation ────────────────────────────────────────────────
    dlog_te, rlog_te = _logits(ei)
    dlog_cal, rlog_cal = _logits(ci_cal)     # calibration half — unseen by selection
    dir_p = [_sigmoid(z) for z in dlog_te]
    res_raw = [_sigmoid(z) for z in rlog_te]

    result["direction"] = {
        "val_auc": round(_auc(dir_p, list(ydir[te])), 4),
        "brier": round(brier(dir_p, list(ydir[te])), 4),
        "backtest": _decile_backtest(dir_p, list(fwd[te])),
    }
    print(f"[direction] {result['direction']}", flush=True)

    # Temperature-scale on the calibration half (disjoint from selection).
    T_res = fit_temperature(list(rlog_cal), list(yres[cal_np]))
    res_cal = apply_temperature(list(rlog_te), T_res)
    T_dir = fit_temperature(list(dlog_cal), list(ydir[cal_np]))
    result["resolution"] = {
        "val_auc": round(_auc(res_raw, list(yres[te])), 4),
        "brier_raw": round(brier(res_raw, list(yres[te])), 4),
        "brier_calibrated": round(brier(res_cal, list(yres[te])), 4),
        "log_loss_calibrated": round(log_loss(res_cal, list(yres[te])), 4),
        "temperature": round(float(T_res), 4),
        "calibration_table": calibration_table(res_cal, list(yres[te])),
    }
    print(f"[resolution] auc {result['resolution']['val_auc']} "
          f"brier {result['resolution']['brier_raw']}→{result['resolution']['brier_calibrated']} "
          f"(T={T_res:.3f})", flush=True)

    # ── Walk-forward over end_date blocks (resolution head, retrained per fold) ─
    wf = []
    edges = [float(np.quantile(ends, q)) for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    for kf in range(1, 5):
        tmask = ends < edges[kf]
        vmask = (ends >= edges[kf]) & (ends < edges[kf + 1])
        if tmask.sum() < 200 or vmask.sum() < 50:
            continue
        m = TapeTransformer(S.shape[-1]).to(dev)
        o = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        tif = torch.tensor(np.where(tmask)[0], device=dev)
        pwd = _pw(np.where(tmask)[0], ydir); pwr = _pw(np.where(tmask)[0], yres)
        ld_bce = nn.BCEWithLogitsLoss(pos_weight=pwd); lr_bce = nn.BCEWithLogitsLoss(pos_weight=pwr)
        for _ in range(min(epochs, 20)):
            m.train()
            perm = tif[torch.randperm(len(tif), device=dev)]
            for b in range(0, len(tif), BATCH):
                bi = perm[b:b + BATCH]
                o.zero_grad(set_to_none=True)
                a, c2 = m(St[bi])
                (ld_bce(a, ydt[bi]) + lr_bce(c2, yrt[bi])).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                o.step()
        m.eval()
        vi = torch.tensor(np.where(vmask)[0], device=dev)
        with torch.no_grad():
            vd, vr = [], []
            for b in range(0, len(vi), BATCH):
                bb = vi[b:b + BATCH]
                ad, ar = m(St[bb]); vd.append(ad.cpu()); vr.append(ar.cpu())
            vd = torch.sigmoid(torch.cat(vd)).numpy(); vr = torch.sigmoid(torch.cat(vr)).numpy()
        wf.append({"fold": kf,
                   "dir_auc": round(_auc(list(vd), list(ydir[vmask])), 4),
                   "res_auc": round(_auc(list(vr), list(yres[vmask])), 4),
                   "n": int(vmask.sum())})
    result["walk_forward"] = {"folds": wf,
                              "mean_dir_auc": round(float(np.mean([f["dir_auc"] for f in wf])), 4) if wf else None,
                              "mean_res_auc": round(float(np.mean([f["res_auc"] for f in wf])), 4) if wf else None}
    print(f"[walk-forward] {result['walk_forward']}", flush=True)

    # ── Artifacts: safetensors + normaliser (+ temperatures) + metrics + card ─
    import base64
    norm = {
        "seq_channels": SEQ_CHANNELS, "seq_mean": smean.tolist(), "seq_std": sstd.tolist(),
        "seq_len": SEQ_LEN, "horizon": HORIZON,
        "arch": {"type": "causal_transformer", "d_model": D_MODEL, "nhead": NHEAD,
                 "layers": LAYERS, "ff": FF, "dropout": DROPOUT, "heads": ["direction", "resolution"]},
        "temperature": {"resolution": round(float(T_res), 6), "direction": round(float(T_dir), 6)},
        "winner": "resolution_calibrated",
        "label": "direction=YES price rises over next HORIZON trades; resolution=1 if market resolves YES (token1)",
    }
    if best_state is not None:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in best_state.items()}, "/tmp/tape_transformer.safetensors")
    artifacts = {"transformer_normalizer.json": json.dumps(norm).encode()}
    if os.path.exists("/tmp/tape_transformer.safetensors"):
        artifacts["tape_transformer.safetensors"] = open("/tmp/tape_transformer.safetensors", "rb").read()
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
                            path_in_repo="metrics/transformer_metrics.json",
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
    d = m.get("direction", {})
    r = m.get("resolution", {})
    wf = m.get("walk_forward", {})
    bt = d.get("backtest", {})
    return f"""---
license: apache-2.0
tags:
- time-series
- transformer
- attention
- polymarket
- prediction-markets
- order-flow
language:
- en
---

# Polymarket trade-tape Transformer (causal, multi-task)

A small **Transformer encoder** that reads the **raw trade tape** — a causal
sequence of per-trade tokens `{SEQ_CHANNELS}` — with multi-head self-attention
under a causal mask (each trade attends only to earlier trades, no lookahead).
The final position feeds two heads: next-move **direction** and eventual market
**resolution**. Trained on `{REPO}`, validated strictly **out-of-time** (split by
market `end_date`, so training markets resolve *before* test ones).

## Method
- Windows: trailing {m.get('seq_len')} trades ending at fractions {m.get('snapshot_fracs')}
  of each market's tape ({m.get('windows')} windows over {m.get('markets_used')} markets).
- Channels per trade: {', '.join(SEQ_CHANNELS)} (z-scored on train stats).
- Encoder: {m.get('task')}. Sinusoidal positional encoding; causal attention.
- Calibration: **temperature scaling** fit on a held-out (by end_date) fold.

## Out-of-time test metrics
| task | AUC | Brier | notes |
|---|---|---|---|
| direction (next move) | {d.get('val_auc')} | {d.get('brier')} | decile up-rate spread {bt.get('up_rate_spread')} |
| resolution (final YES) | {r.get('val_auc')} | {r.get('brier_raw')} → {r.get('brier_calibrated')} (cal) | T={r.get('temperature')} |

Resolution baseline (predict train YES-rate): Brier {m.get('baseline_brier_resolution')}.
Walk-forward mean AUC — direction {wf.get('mean_dir_auc')}, resolution {wf.get('mean_res_auc')}.

## Honest comparison vs the GBDT / GRU baselines
This is the **heaviest** variant in the suite, not automatically the best. The
bigdata **bi-GRU** and **XGBoost** models read aggregated hourly OHLCV+flow bars
and reach out-of-time direction AUC ≈ 0.65; the resolution GBDT/MLP ensemble
(`polymarket-resolution-model`) is the calibration reference. Attention over the
raw tape has *more* capacity and sees per-trade order flow the bars smooth away,
but on this out-of-time split it does **not** trivially beat the cheaper models —
compare the numbers above head-to-head before serving it. Modest AUC here is
honest: short-horizon direction and mid-market resolution are both genuinely hard.
Not financial advice.

## Inference
Load `tape_transformer.safetensors`, rebuild the `arch` from
`transformer_normalizer.json`, z-score each trade token with `seq_mean`/`seq_std`,
run the causal encoder, and (for resolution) divide the logit by
`temperature.resolution` before the sigmoid.
"""


def _build_modal_app():
    """Construct the Modal App + remote function. Called only when modal is
    installed, so a bare import never touches modal."""
    app = modal.App("pmt-transformer")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install("pyarrow", "numpy", "torch",
                     "huggingface_hub", "safetensors")
    )

    @app.function(image=image, gpu="H100", cpu=16.0, memory=131072, timeout=10800)
    def run(max_markets: int = 6000, max_row_groups: int = 0, epochs: int = 40,
            push: bool = False, hf_token: str = "") -> dict:
        return _train_transformer(max_markets, max_row_groups, epochs, push, hf_token)

    @app.local_entrypoint()
    def main(max_markets: int = 6000, max_row_groups: int = 0, epochs: int = 40,
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
        with open(os.path.join(data_dir, "transformer_metrics.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print(f"\nwrote {data_dir}/transformer_metrics.json")

    return app, run, main


if _HAS_MODAL:
    # Module-level `app` so `modal run ml/modal_transformer.py` discovers it.
    app, run, main = _build_modal_app()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Flagship trade-tape Transformer trainer")
    ap.add_argument("--smoke", action="store_true",
                    help="run the sequence pipeline on synthetic data with pure stdlib (no GPU/Modal)")
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(run_smoke())
    print("Nothing to do. Use --smoke for the local pipeline, or:\n"
          "  modal run ml/modal_transformer.py --max-markets 6000 --push")
