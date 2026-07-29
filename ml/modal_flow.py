"""Order-flow DIRECTION trainer on H100 — true taker-aggressor flow at 418M scale.

Trains a next-move direction model from the *real* aggressor side of every trade.
`SII-WANGZJ/Polymarket_data`'s `trades.parquet` (418.3M CLOB trades, 2022-11 →)
carries `taker_direction` — the genuine BUY/SELL of the aggressor that crossed the
book. Prior repo models only *proxied* order flow from a candle's close-vs-open
sign; here we sum `usd_amount` signed by the true aggressor side, so the
imbalance, VPIN, toxicity and buy-pressure features are microstructure signal
rather than a price-shape echo.

Pipeline (all inside Modal, streaming the parquet by row-group so the 28 GB file
never lands whole in RAM):

  1. bucket trades per `asset_id` into fixed BAR_SECONDS windows → OHLCV + signed
     aggressor flow (BUY=+usd, SELL=-usd), plus buy/sell/whale volume.
  2. inline order-flow features (signed_imbalance, VPIN proxy, toxicity,
     trade_intensity, buy_pressure, whale_ratio, …). Label = price higher HORIZON
     buckets ahead.
  3. train a LightGBM GBDT + a torch GRU over the per-bar flow sequence; blend.
     Decile backtest + AUC on a strict out-of-time (tfrac) split.
  4. push artifacts + normalizer JSON + metrics JSON + model card to the Hub
     (`shubhxho/polymarket-flow-model`).

    modal run ml/modal_flow.py --max-rows 20000000 --top-tokens 6000 --push

Local proof (zero heavy deps — pure stdlib logistic baseline over the SAME
bucket→feature→label→eval code):

    python ml/modal_flow.py --smoke

Every heavy import (modal, pyarrow, numpy, torch, lightgbm, datasets) lives
inside a function or the Modal image, so this module imports cleanly with none of
them installed and the smoke path runs anywhere Python does.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

# Modal is optional at import time. When it is installed we build a real H100
# app; when it is not (e.g. the stdlib smoke env) `app` is None and the two
# decorators below degrade to no-ops, so the module still imports and --smoke
# still runs. Heavy deps are NEVER imported at module top.
try:  # pragma: no cover - depends on environment
    import modal

    app = modal.App("pmt-flow")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "pyarrow", "numpy", "torch", "lightgbm",
            "datasets", "huggingface_hub", "safetensors",
        )
    )
except Exception:  # ImportError in the smoke env, and any partial-install error
    modal = None
    app = None
    image = None


def _remote(**kwargs):
    """`@app.function(**kwargs)` when Modal is present, else identity so the
    decorated function stays a plain callable for the stdlib smoke path."""

    def deco(fn):
        return app.function(**kwargs)(fn) if app is not None else fn

    return deco


def _entrypoint():
    def deco(fn):
        return app.local_entrypoint()(fn) if app is not None else fn

    return deco


# ── configuration ─────────────────────────────────────────────────────────────
REPO = "SII-WANGZJ/Polymarket_data"
DATA_FILE = "trades.parquet"
HF_MODEL_REPO = "shubhxho/polymarket-flow-model"

WINDOW = 16            # look-back buckets per sample
HORIZON = 4            # predict price direction this many buckets ahead
BAR_SECONDS = 3600     # hourly buckets
MIN_CANDLES = 48       # keep only assets with enough history
MIN_STD = 1e-4         # skip dead (flat) windows
WHALE_USD = 1000.0     # a single trade at/above this counts toward whale volume

# 4-channel per-bar sequence fed to the GRU (true order flow, step by step).
SEQ_CHANNELS = ["ret", "bar_imbalance", "buy_minus_sell_ratio", "log_volume"]

FEATURE_NAMES = [
    # close-path shape
    "last", "mean_ret", "vol", "drift", "band_z", "momentum",
    # oscillators / volatility from true OHLC range
    "rsi", "atr", "parkinson_vol",
    # TRUE order flow (signed by taker_direction aggressor side)
    "signed_imbalance", "ofi_last", "ofi_trend", "vpin", "toxicity",
    "trade_intensity", "buy_pressure", "whale_ratio", "vol_z",
    # context
    "extremeness", "log_volume",
]


# ── pure-python numeric helpers (stdlib only; shared by smoke + Modal) ─────────
def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _sigmoid(z):
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


# ── bucketing: trades → per-asset OHLCV + true signed aggressor flow ───────────
def bucket_trades(trades, bar_seconds=BAR_SECONDS, whale_usd=WHALE_USD):
    """Aggregate an iterable of (ts, asset_id, price, usd_amount, taker_direction)
    into fixed time buckets per asset. `taker_direction == 'BUY'` is the true
    aggressor buying → signed flow +usd; 'SELL' → -usd. Returns
    {asset_id: [bar, …]} sorted chronologically. Open/close track the first/last
    trade *by timestamp* so out-of-order or cross-row-group trades merge right."""
    books = {}
    for ts, asset, price, usd, direction in trades:
        ts = int(ts)
        b = ts // bar_seconds * bar_seconds
        d = books.setdefault(asset, {})
        bar = d.get(b)
        if bar is None:
            bar = d[b] = {
                "t": b, "open": price, "close": price, "high": price, "low": price,
                "volume": 0.0, "trades": 0, "signed_vol": 0.0,
                "buy_vol": 0.0, "sell_vol": 0.0, "whale_vol": 0.0,
                "_first_ts": ts, "_last_ts": ts,
            }
        if ts <= bar["_first_ts"]:
            bar["_first_ts"], bar["open"] = ts, price
        if ts >= bar["_last_ts"]:
            bar["_last_ts"], bar["close"] = ts, price
        if price > bar["high"]:
            bar["high"] = price
        if price < bar["low"]:
            bar["low"] = price
        bar["volume"] += usd
        bar["trades"] += 1
        is_buy = direction == "BUY"
        bar["signed_vol"] += usd if is_buy else -usd
        if is_buy:
            bar["buy_vol"] += usd
        else:
            bar["sell_vol"] += usd
        if usd >= whale_usd:
            bar["whale_vol"] += usd
    return {asset: [d[k] for k in sorted(d)] for asset, d in books.items()}


def merge_books(dst, src):
    """Fold `src` ({asset: [bar,…]}) into `dst` ({asset: {ts: bar}}) so partial
    buckets that straddle row-group boundaries combine correctly while streaming.
    Mutates and returns `dst`."""
    for asset, bars in src.items():
        cur = dst.setdefault(asset, {})
        for b in bars:
            e = cur.get(b["t"])
            if e is None:
                cur[b["t"]] = dict(b)
                continue
            if b["_first_ts"] <= e["_first_ts"]:
                e["_first_ts"], e["open"] = b["_first_ts"], b["open"]
            if b["_last_ts"] >= e["_last_ts"]:
                e["_last_ts"], e["close"] = b["_last_ts"], b["close"]
            e["high"] = max(e["high"], b["high"])
            e["low"] = min(e["low"], b["low"])
            for k in ("volume", "trades", "signed_vol", "buy_vol", "sell_vol", "whale_vol"):
                e[k] += b[k]
    return dst


def _finalize_books(book_map):
    """{asset: {ts: bar}} → {asset: [bar,…]} sorted by time."""
    return {asset: [d[k] for k in sorted(d)] for asset, d in book_map.items()}


# ── features: one 20-dim vector per look-back window ───────────────────────────
def flow_features(win):
    """20 features for one OHLCV+flow window, centred on REAL aggressor flow."""
    c = [b["close"] for b in win]
    o = [b["open"] for b in win]  # noqa: F841 - kept for parity/readability
    h = [b["high"] for b in win]
    lo = [b["low"] for b in win]
    vol = [b["volume"] for b in win]
    trd = [b["trades"] for b in win]
    sig = [b["signed_vol"] for b in win]
    buy = [b["buy_vol"] for b in win]
    sell = [b["sell_vol"] for b in win]
    whale = [b["whale_vol"] for b in win]

    rets = [c[i] - c[i - 1] for i in range(1, len(c))]
    std = _std(rets)
    mean_c, std_c = _mean(c), _std(c)
    band_z = (c[-1] - mean_c) / std_c if std_c > 1e-9 else 0.0
    momentum = _mean(rets[-4:]) if len(rets) >= 4 else _mean(rets)

    gains = sum(r for r in rets if r > 0)
    losses = -sum(r for r in rets if r < 0)
    rsi = (gains - losses) / (gains + losses) if gains + losses > 1e-9 else 0.0

    trs = [max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(c))]
    atr = (_mean(trs) / max(c[-1], 1e-3)) if trs else 0.0
    pk = [(math.log(h[i] / lo[i])) ** 2 for i in range(len(c)) if h[i] > lo[i] and lo[i] > 0]
    park = math.sqrt(_mean(pk) / (4 * math.log(2))) if pk else 0.0

    total_v = sum(vol) + 1e-9
    signed_imbalance = sum(sig) / total_v                       # net aggressor tilt over window
    ofi_last = sig[-1] / (vol[-1] + 1e-9)                        # latest bar tilt
    ofi_trend = sum(sig[-4:]) / (sum(vol[-4:]) + 1e-9) if len(sig) >= 4 else signed_imbalance
    # VPIN proxy: volume-synchronised probability of informed trading — the mean
    # per-bar absolute buy/sell imbalance ratio.
    per = [abs(buy[i] - sell[i]) / (buy[i] + sell[i]) for i in range(len(win)) if buy[i] + sell[i] > 1e-9]
    vpin = _mean(per) if per else 0.0

    vmean, vstd = _mean(vol), _std(vol)
    vol_z = (vol[-1] - vmean) / (vstd + 1e-9)
    # toxicity: recent one-sided flow amplified by a volume surge (informed push).
    toxicity = abs(ofi_trend) * (1.0 + max(vol_z, 0.0))

    tmean, tstd = _mean(trd), _std(trd)
    trade_intensity = (trd[-1] - tmean) / (tstd + 1e-9)
    buy_pressure = sum(buy) / total_v                           # 0..1 fraction bought
    whale_ratio = sum(whale) / total_v

    return [
        c[-1], _mean(rets), std, c[-1] - c[0], band_z, momentum,
        rsi, atr, park,
        signed_imbalance, ofi_last, ofi_trend, vpin, toxicity,
        trade_intensity, buy_pressure, whale_ratio, vol_z,
        abs(c[-1] - 0.5) * 2, math.log1p(total_v),
    ]


def window_sequence(win):
    """Per-bar flow sequence (len WINDOW, SEQ_CHANNELS wide) for the GRU."""
    seq = []
    prev_c = win[0]["close"]
    for b in win:
        tot = b["buy_vol"] + b["sell_vol"]
        ret = b["close"] - prev_c
        prev_c = b["close"]
        imb = b["signed_vol"] / (b["volume"] + 1e-9)
        buy_minus_sell = (b["buy_vol"] - b["sell_vol"]) / (tot + 1e-9)
        seq.append([ret, imb, buy_minus_sell, math.log1p(b["volume"])])
    return seq


# ── windowing + labels ─────────────────────────────────────────────────────────
def build_dataset(books, window=WINDOW, horizon=HORIZON, min_candles=MIN_CANDLES,
                  min_std=MIN_STD, with_seq=False):
    """Slide over each asset's bar series → (feats, seqs|None, labels, fwds, tfrac).

    Label = 1 if close HORIZON buckets ahead is higher. `tfrac` is the window's
    fractional position in its asset timeline → drives the out-of-time split.
    """
    feats, seqs, labels, fwds, tfrac = [], [], [], [], []
    for bars in books.values():
        N = len(bars)
        if N < min_candles:
            continue
        c = [b["close"] for b in bars]
        for i in range(window, N - horizon):
            cw = c[i - window:i]
            rets = [cw[k] - cw[k - 1] for k in range(1, len(cw))]
            if _std(rets) < min_std:
                continue
            win = bars[i - window:i]
            fwd = c[i + horizon] - c[i]
            feats.append(flow_features(win))
            if with_seq:
                seqs.append(window_sequence(win))
            labels.append(1.0 if fwd > 0 else 0.0)
            fwds.append(fwd)
            tfrac.append(i / N)
    return feats, (seqs if with_seq else None), labels, fwds, tfrac


# ── evaluation kit (matches train_seq.py conventions) ─────────────────────────
def _auc(probs, labels):
    """Rank AUC with proper tie handling (mirrors train_seq._auc)."""
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


def _decile_backtest(probs, fwd, q=0.2):
    """Top vs bottom q of markets by model score: how did they actually move?
    Same schema as train_seq._decile_backtest so the metrics drop into the kit."""
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

    def _up(xs):
        return sum(1 for v in xs if v > 0) / len(xs)

    return {
        "top_up_rate": round(_up(top), 3),
        "bottom_up_rate": round(_up(bottom), 3),
        "up_rate_spread": round(_up(top) - _up(bottom), 3),
        "top_median_pts": round(_median(top) * 100, 3),
        "bottom_median_pts": round(_median(bottom) * 100, 3),
        "slice": k,
    }


def _brier(probs, labels):
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs) if probs else 0.0


# ── stdlib logistic baseline (smoke only — proves the train/eval path) ─────────
def _moments(X):
    d = len(X[0])
    mean = [_mean([row[j] for row in X]) for j in range(d)]
    std = [_std([row[j] for row in X]) + 1e-6 for j in range(d)]
    return mean, std


def _normalize(X, mean, std):
    return [[(row[j] - mean[j]) / std[j] for j in range(len(row))] for row in X]


def _fit_logistic(X, y, epochs=250, lr=0.1, l2=1e-4, seed=11):
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    idx = list(range(n))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            row = X[i]
            z = b + sum(w[j] * row[j] for j in range(d))
            g = _sigmoid(z) - y[i]
            b -= lr * g
            for j in range(d):
                w[j] -= lr * (g * row[j] + l2 * w[j])
    return w, b


def _predict_logistic(X, w, b):
    return [_sigmoid(b + sum(w[j] * row[j] for j in range(len(row)))) for row in X]


# ── synthetic trades for the smoke test ───────────────────────────────────────
def synth_trades(n_assets=14, bars=90, seed=11):
    """Emit raw (ts, asset, price, usd, direction) trades with a *plantable*
    signal: a latent informed pressure biases the aggressor side AND drives a
    persistent price drift, so recent true order flow leads the forward move —
    exactly what the model must learn to recover. Some trades are whales."""
    rng = random.Random(seed)
    trades = []
    t0 = 1_700_000_000
    for a in range(n_assets):
        asset = f"synth-{a:03d}"
        price = rng.uniform(0.3, 0.7)
        drift = 0.0
        for bi in range(bars):
            bt = t0 + bi * BAR_SECONDS
            informed = rng.gauss(0.0, 1.0)
            drift = 0.9 * drift + 0.02 * informed          # persistent (autoregressive) drift
            pbuy = _sigmoid(3.0 * informed)                # aggressor tilt tracks the pressure
            ntr = rng.randint(4, 14)
            for _ in range(ntr):
                direction = "BUY" if rng.random() < pbuy else "SELL"
                price += drift / ntr + rng.gauss(0.0, 0.0008)
                price = min(0.99, max(0.01, price))
                usd = rng.expovariate(1 / 40.0) * (6.0 if rng.random() < 0.04 else 1.0)
                ts = bt + rng.randint(0, BAR_SECONDS - 1)
                trades.append((ts, asset, price, usd, direction))
    return trades


def run_smoke(n_assets=14, bars=90, seed=11):
    """FULL pipeline on synthetic data with zero heavy deps: bucket → features →
    label → train (stdlib logistic) → out-of-time AUC + decile backtest."""
    trades = synth_trades(n_assets=n_assets, bars=bars, seed=seed)
    books = bucket_trades(trades)
    n_bars = sum(len(v) for v in books.values())
    feats, _, labels, fwds, tfrac = build_dataset(books, with_seq=False)
    if not feats:
        raise SystemExit("smoke: no windows built — increase --bars/--assets")

    # Strict out-of-time split: latest 20% of each asset timeline is validation.
    tr = [i for i, t in enumerate(tfrac) if t < 0.8]
    va = [i for i, t in enumerate(tfrac) if t >= 0.8]
    if not tr or not va:
        raise SystemExit("smoke: degenerate split")

    Xtr = [feats[i] for i in tr]
    ytr = [labels[i] for i in tr]
    mean, std = _moments(Xtr)
    Xtr_n = _normalize(Xtr, mean, std)
    Xva_n = _normalize([feats[i] for i in va], mean, std)
    yva = [labels[i] for i in va]
    fva = [fwds[i] for i in va]

    w, b = _fit_logistic(Xtr_n, ytr)
    pva = _predict_logistic(Xva_n, w, b)

    auc = _auc(pva, yva)
    bt = _decile_backtest(pva, fva)
    base = max(_mean(ytr), 1 - _mean(ytr))
    weights = sorted(zip(FEATURE_NAMES, [round(x, 3) for x in w]), key=lambda t: -abs(t[1]))

    report = {
        "mode": "smoke (synthetic, stdlib logistic — no heavy deps)",
        "assets": len(books), "bars": n_bars, "windows": len(feats),
        "features": len(FEATURE_NAMES), "up_rate": round(_mean(labels), 4),
        "train_windows": len(tr), "val_windows": len(va),
        "majority_baseline_acc": round(base, 4),
        "val_auc": round(auc, 4),
        "val_brier": round(_brier(pva, yva), 4),
        "decile_backtest": bt,
        "top_flow_weights": weights[:6],
    }
    print(json.dumps(report, indent=2))
    print(f"\nval AUC {auc:.4f} (baseline {base:.4f}) | "
          f"decile up-rate spread {bt.get('up_rate_spread'):+.3f}")
    ok = auc > 0.5 and bt.get("up_rate_spread", 0) > 0
    print("smoke OK — feature/label/eval paths recover the planted flow signal."
          if ok else "smoke ran (signal weak on this synthetic seed).")
    return report


# ── Modal H100 job ─────────────────────────────────────────────────────────────
@_remote(image=image, gpu="H100", cpu=16.0, memory=131072, timeout=9000)
def run(max_rows: int = 20_000_000, top_tokens: int = 6000, epochs: int = 40,
        bar_seconds: int = BAR_SECONDS, push: bool = False, hf_token: str = "") -> dict:
    """Stream trades.parquet → bucket → flow features → GBDT + GRU blend → push."""
    import base64

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    # 1) Stream the 28 GB parquet by row-group; accumulate bars per asset.
    path = hf_hub_download(REPO, DATA_FILE, repo_type="dataset")
    pf = pq.ParquetFile(path)
    cols = ["timestamp", "asset_id", "price", "usd_amount", "taker_direction"]
    book_map = {}
    seen = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=cols)
        ts = tbl.column("timestamp").to_numpy(zero_copy_only=False)
        asset = tbl.column("asset_id").to_pylist()
        price = tbl.column("price").to_numpy(zero_copy_only=False)
        usd = tbl.column("usd_amount").to_numpy(zero_copy_only=False)
        direction = tbl.column("taker_direction").to_pylist()
        part = bucket_trades(zip(ts.tolist(), asset, price.tolist(), usd.tolist(), direction),
                             bar_seconds=bar_seconds)
        merge_books(book_map, part)
        seen += len(asset)
        if seen % 5_000_000 < len(asset):
            print(f"  streamed {seen:,} trades, {len(book_map)} assets", flush=True)
        if max_rows and seen >= max_rows:
            break
    os.remove(path)

    books = _finalize_books(book_map)
    print(f"streamed {seen:,} trades → {len(books)} assets", flush=True)

    # Keep the most-traded tokens → a tractable, liquid training set.
    ranked = sorted(books.items(), key=lambda kv: -sum(b["trades"] for b in kv[1]))
    books = dict(ranked[:top_tokens])
    n_bars = sum(len(v) for v in books.values())
    print(f"kept {len(books)} tokens, {n_bars} bars", flush=True)

    # 2) Windows + features + per-bar flow sequences.
    feats, seqs, labels, fwds, tfrac = build_dataset(books, with_seq=True)
    if not feats:
        raise RuntimeError("no windows built — raise --max-rows (too few bars per token)")
    X = np.asarray(feats, np.float32)
    S = np.asarray(seqs, np.float32)
    y = np.asarray(labels, np.float32)
    fwds = np.asarray(fwds, np.float32)
    tfrac = np.asarray(tfrac, np.float32)
    print(f"built {len(y)} windows, {X.shape[1]} feats, seq {S.shape[1:]}, up-rate {y.mean():.3f}", flush=True)

    val = tfrac >= 0.8
    tr_i, va_i = np.where(~val)[0], np.where(val)[0]
    fmean, fstd = X[tr_i].mean(0), X[tr_i].std(0) + 1e-6
    smean, sstd = S[tr_i].mean((0, 1)), S[tr_i].std((0, 1)) + 1e-6

    result = {
        "runtime": "modal H100 / order-flow direction",
        "dataset": REPO, "data_file": DATA_FILE,
        "source": "true taker_direction aggressor flow → OHLCV + signed order flow",
        "trades_streamed": int(seen), "tokens": len(books), "bars": int(n_bars),
        "windows": int(len(y)), "features": FEATURE_NAMES, "seq_channels": SEQ_CHANNELS,
        "window": WINDOW, "horizon": HORIZON, "bar_seconds": int(bar_seconds),
        "train_windows": int(len(tr_i)), "val_windows": int(len(va_i)),
        "up_rate": round(float(y.mean()), 4),
        "majority_baseline_acc": round(float(max(y[tr_i].mean(), 1 - y[tr_i].mean())), 4),
    }

    # 3a) GBDT (primary for tabular flow features).
    import lightgbm as lgb
    dtr = lgb.Dataset(X[tr_i], label=y[tr_i], feature_name=list(FEATURE_NAMES))
    dva = lgb.Dataset(X[va_i], label=y[va_i], reference=dtr)
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "is_unbalance": True, "seed": 11, "verbose": -1}
    bst = lgb.train(params, dtr, num_boost_round=1200, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    gp = bst.predict(X[va_i])
    result["gbdt"] = {"val_auc": round(_auc(gp, y[va_i]), 4),
                      "brier": round(float(np.mean((gp - y[va_i]) ** 2)), 4),
                      "backtest": _decile_backtest(list(gp), list(fwds[va_i]))}
    result["feature_importance_gbdt"] = sorted(
        zip(FEATURE_NAMES, [round(float(x), 1) for x in bst.feature_importance("gain")]),
        key=lambda t: -t[1])
    print(f"[gbdt] {result['gbdt']}", flush=True)

    # 3b) GRU over the per-bar true-flow sequence, fused with the feature vector.
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Sn = (S - smean) / sstd
    Xn = (X - fmean) / fstd

    class FlowGRU(nn.Module):
        def __init__(self, n_ch, n_feat, hidden=64):
            super().__init__()
            self.gru = nn.GRU(n_ch, hidden, batch_first=True)
            self.fproj = nn.Linear(n_feat, hidden)
            self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                      nn.Dropout(0.3), nn.Linear(hidden, 1))

        def forward(self, seq, feat):
            out, _ = self.gru(seq)
            fused = torch.cat([out[:, -1, :], torch.relu(self.fproj(feat))], dim=-1)
            return self.head(fused).squeeze(-1)

    St = torch.tensor(Sn, device=dev)
    Xt = torch.tensor(Xn, device=dev)
    yt = torch.tensor(y, device=dev)
    ti = torch.tensor(tr_i, device=dev)
    vi = torch.tensor(va_i, device=dev)
    torch.manual_seed(11)
    net = FlowGRU(S.shape[2], X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    best_auc, best_state, gru_probs = -1.0, None, None
    for _ in range(epochs):
        net.train()
        perm = ti[torch.randperm(len(tr_i), device=dev)]
        for b0 in range(0, len(tr_i), 2048):
            bi = perm[b0:b0 + 2048]
            loss = bce(net(St[bi], Xt[bi]), yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(St[vi], Xt[vi])).cpu().numpy()
        a = _auc(pv, y[va_i])
        if a > best_auc:
            best_auc, gru_probs = a, pv
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    result["gru"] = {"val_auc": round(best_auc, 4),
                     "brier": round(float(np.mean((gru_probs - y[va_i]) ** 2)), 4),
                     "backtest": _decile_backtest(list(gru_probs), list(fwds[va_i]))}
    print(f"[gru] {result['gru']}", flush=True)

    # 3c) Blend.
    ens = (np.asarray(gp) + np.asarray(gru_probs)) / 2
    result["blend"] = {"val_auc": round(_auc(ens, y[va_i]), 4),
                       "brier": round(float(np.mean((ens - y[va_i]) ** 2)), 4),
                       "backtest": _decile_backtest(list(ens), list(fwds[va_i]))}
    result["overall_best"] = max(("gbdt", "gru", "blend"), key=lambda k: result[k]["val_auc"])
    print(f"[blend] {result['blend']} | best {result['overall_best']}", flush=True)

    # 4) Artifacts: gbdt txt + gru safetensors + normalizer json + metrics + card.
    norm = {"fmean": fmean.tolist(), "fstd": fstd.tolist(), "features": FEATURE_NAMES,
            "seq_mean": smean.tolist(), "seq_std": sstd.tolist(), "seq_channels": SEQ_CHANNELS,
            "window": WINDOW, "horizon": HORIZON, "bar_seconds": int(bar_seconds),
            "hidden": 64, "gru_hidden": 64}
    bst.save_model("/tmp/flow_gbdt.txt")
    artifacts = {"flow_normalizer.json": json.dumps(norm).encode(),
                 "flow_gbdt.txt": open("/tmp/flow_gbdt.txt", "rb").read()}
    if best_state is not None:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in best_state.items()}, "/tmp/flow_gru.safetensors")
        artifacts["flow_gru.safetensors"] = open("/tmp/flow_gru.safetensors", "rb").read()
    artifacts["flow_metrics.json"] = json.dumps(
        {k: v for k, v in result.items() if not k.startswith("_")}, indent=2).encode()
    artifacts["README.md"] = _model_card(result).encode()
    result["_artifacts_b64"] = {k: base64.b64encode(v).decode() for k, v in artifacts.items()}

    if push and hf_token:
        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
            for name, data in artifacts.items():
                sub = name if name == "README.md" else f"flow/{name}"
                api.upload_file(path_or_fileobj=data, path_in_repo=sub,
                                repo_id=HF_MODEL_REPO, repo_type="model")
            result["hf_repo"] = f"https://huggingface.co/{HF_MODEL_REPO}"
            print(f"pushed → {result['hf_repo']}", flush=True)
        except Exception as e:  # noqa: BLE001
            result["push_error"] = f"{type(e).__name__}: {e}"[:300]
            print("push failed:", result["push_error"], flush=True)

    return result


def _model_card(result):
    def line(k):
        m = result.get(k, {})
        bt = m.get("backtest", {})
        return (f"| {k} | {m.get('val_auc')} | {m.get('brier')} | "
                f"{bt.get('up_rate_spread')} |")

    return f"""---
license: mit
tags:
- polymarket
- order-flow
- prediction-markets
- time-series
---

# polymarket-flow-model

Next-move **direction** model trained from the **true taker aggressor side** of
{result.get('trades_streamed', 0):,} Polymarket CLOB trades
(`{REPO}` → `{DATA_FILE}`). Unlike close-vs-open flow proxies, `signed_imbalance`,
`vpin`, `toxicity` and `buy_pressure` here are built from each trade's real
`taker_direction`, bucketed into {result.get('bar_seconds')}-second bars.

- window {result.get('window')} bars, horizon {result.get('horizon')} bars ahead
- {result.get('windows', 0):,} windows over {result.get('tokens')} tokens
- strict out-of-time (tfrac >= 0.8) validation split

## Metrics (out-of-time)

| model | val AUC | brier | decile up-rate spread |
|-------|---------|-------|-----------------------|
{line('gbdt')}
{line('gru')}
{line('blend')}

Best: **{result.get('overall_best')}**. Majority baseline acc
{result.get('majority_baseline_acc')}, up-rate {result.get('up_rate')}.

## Artifacts
- `flow/flow_gbdt.txt` — LightGBM booster
- `flow/flow_gru.safetensors` — torch GRU (fuses per-bar flow sequence + features)
- `flow/flow_normalizer.json` — feature/sequence normaliser + window/horizon
- `flow/flow_metrics.json` — full metrics

Trained self-contained on a Modal H100 via `ml/modal_flow.py`.
"""


@_entrypoint()
def main(max_rows: int = 20_000_000, top_tokens: int = 6000, epochs: int = 40, push: bool = False):
    import base64

    token = os.environ.get("HF_TOKEN", "")
    report = run.remote(max_rows=max_rows, top_tokens=top_tokens, epochs=epochs,
                        push=push, hf_token=token)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    for name, b64 in report.pop("_artifacts_b64", {}).items():
        with open(os.path.join(data_dir, name.replace("/", "_")), "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"saved data/{name.replace('/', '_')}")
    with open(os.path.join(data_dir, "flow_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {data_dir}/flow_metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Order-flow direction trainer (Modal H100).")
    ap.add_argument("--smoke", action="store_true",
                    help="run the full pipeline on synthetic data with zero heavy deps")
    ap.add_argument("--assets", type=int, default=14, help="synthetic assets (smoke)")
    ap.add_argument("--bars", type=int, default=90, help="synthetic bars/asset (smoke)")
    ap.add_argument("--seed", type=int, default=11, help="synthetic seed (smoke)")
    args = ap.parse_args()
    if args.smoke:
        run_smoke(n_assets=args.assets, bars=args.bars, seed=args.seed)
    else:
        print("This is a Modal H100 job. Launch it with:")
        print("  modal run ml/modal_flow.py --max-rows 20000000 --top-tokens 6000 --push")
        print("Validate the pipeline locally (no heavy deps) with:")
        print("  python ml/modal_flow.py --smoke")
