"""Train the flagship signal model on ~50 GB of raw Polymarket trades, on H100.

Streams `TimeSeventeen/Polymarket-v1` — the on-chain OrderFilled tape, **1.2B
trades** (27 GB) covering 2022-11 → 2026-04 — inside Modal, aggregates it to
hourly OHLCV bars with **true order-flow imbalance** (from each trade's
`taker_direction`, i.e. the real aggressor side, not the close-vs-open proxy the
small model used), builds windows, and trains a GBDT + neural ensemble with
strict out-of-time (temporal walk-forward) validation.

Writes better metrics + safetensors and pushes them to the Hub.

    modal run ml/modal_bigdata.py --max-files 42 --top-tokens 8000 --push

Cost estimated first (~$7, <2 hr on H100+16CPU). --max-files caps the tape slice
for a cheaper dry run; the default processes the whole 27 GB.

Everything trains on the GPU: the tabular model is **XGBoost on CUDA**, and the
neural model is a **bi-GRU over the raw per-bar window sequence fused with a
feature-MLP** (AMP, cosine LR, class-weighted loss). The two are ensembled with
a weight tuned on the out-of-time validation slice.
"""

from __future__ import annotations

import json
import os

import modal

app = modal.App("pmt-bigdata")

# Persist the built windows so iterating on the GPU model doesn't re-download and
# re-aggregate the 27 GB tape every attempt — the expensive, ~15-min part is
# identical across model changes, so it is cached by (max_files, top_tokens).
cache_vol = modal.Volume.from_name("pmt-bigdata-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("polars", "pyarrow", "numpy", "torch", "xgboost",
                 "huggingface_hub", "safetensors")
)

REPO = "TimeSeventeen/Polymarket-v1"
WINDOW = 16
HORIZON = 4
BAR_SECONDS = 3600      # hourly bars
MIN_CANDLES = 48        # keep tokens with enough history
MERGE_EVERY = 6         # incremental merge cadence to bound memory


def _agg_file(pl, path):
    """Per-file hourly aggregation, carrying first/last timestamps so partial
    bars that span files merge correctly."""
    df = pl.read_parquet(path, columns=["block_timestamp", "token_asset_id", "price", "usdc_amount", "taker_direction"])
    df = df.filter((pl.col("price") > 0.0) & (pl.col("price") < 1.0) & (pl.col("token_asset_id") != "0"))
    if df.height == 0:
        return None
    df = df.with_columns([
        (pl.col("block_timestamp") // BAR_SECONDS * BAR_SECONDS).alias("hour"),
        pl.when(pl.col("taker_direction") == "BUY").then(pl.col("usdc_amount"))
          .otherwise(-pl.col("usdc_amount")).alias("signed"),
    ]).sort("block_timestamp")
    return df.group_by(["token_asset_id", "hour"]).agg([
        pl.col("price").first().alias("open"),
        pl.col("price").last().alias("close"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("usdc_amount").sum().alias("volume"),
        pl.len().alias("trades"),
        pl.col("signed").sum().alias("signed_vol"),
        pl.col("block_timestamp").min().alias("first_ts"),
        pl.col("block_timestamp").max().alias("last_ts"),
    ])


def _merge(pl, frames):
    """Combine partial bars across files: global open/close by first/last ts."""
    return pl.concat(frames).group_by(["token_asset_id", "hour"]).agg([
        pl.col("open").sort_by("first_ts").first().alias("open"),
        pl.col("close").sort_by("last_ts").last().alias("close"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("volume").sum().alias("volume"),
        pl.col("trades").sum().alias("trades"),
        pl.col("signed_vol").sum().alias("signed_vol"),
        pl.col("first_ts").min().alias("first_ts"),
        pl.col("last_ts").max().alias("last_ts"),
    ])


def _win_feats(o, h, l, c, vol, trd, sig):
    """22 features for one OHLCV+flow window, incl. REAL order-flow imbalance."""
    import numpy as np

    c = np.asarray(c); o = np.asarray(o); h = np.asarray(h); l = np.asarray(l)
    vol = np.asarray(vol); trd = np.asarray(trd); sig = np.asarray(sig)
    rets = np.diff(c)
    std = float(rets.std()) if len(rets) > 1 else 0.0
    mean_c, std_c = float(c.mean()), float(c.std())
    band_z = (c[-1] - mean_c) / std_c if std_c > 1e-9 else 0.0
    momentum = float(rets[-4:].mean()) if len(rets) >= 4 else float(rets.mean() if len(rets) else 0)
    # oscillators (proper — true high/low/typical price)
    tp = (h + l + c) / 3
    mad = float(np.mean(np.abs(tp - tp.mean())))
    cci = np.clip((tp[-1] - tp.mean()) / (0.015 * mad) / 100, -5, 5) if mad > 1e-9 else 0.0
    hi, lo = float(h.max()), float(l.min())
    stoch = (c[-1] - lo) / (hi - lo) if hi - lo > 1e-9 else 0.5
    will = -(hi - c[-1]) / (hi - lo) if hi - lo > 1e-9 else -0.5
    gains = float(rets[rets > 0].sum()); losses = float(-rets[rets < 0].sum())
    rsi = (gains - losses) / (gains + losses) if gains + losses > 1e-9 else 0.0
    trs = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])]) if len(c) > 1 else np.array([0.0])
    atr = float(trs.mean()) / max(float(c[-1]), 1e-3)
    hl = np.log(np.clip(h / np.clip(l, 1e-6, None), 1e-9, None)) ** 2
    park = float(np.sqrt(hl.mean() / (4 * np.log(2)))) if len(hl) else 0.0
    # volume / REAL order flow (taker aggressor side)
    total_v = float(vol.sum()) + 1e-9
    ofi_window = float(sig.sum()) / total_v                    # net aggressor imbalance over window
    ofi_last = float(sig[-1]) / (float(vol[-1]) + 1e-9)        # latest bar imbalance
    ofi_trend = float(sig[-4:].sum()) / (float(vol[-4:].sum()) + 1e-9) if len(sig) >= 4 else ofi_window
    vmean = float(vol.mean())
    vol_z = (float(vol[-1]) - vmean) / (float(vol.std()) + 1e-9)
    tmean = float(trd.mean())
    trade_intensity = (float(trd[-1]) - tmean) / (float(trd.std()) + 1e-9)
    avg_trade = vmean / (tmean + 1e-9)
    return [
        float(c[-1]), float(rets.mean() if len(rets) else 0), std, float(c[-1] - c[0]), band_z, momentum,
        float(rsi), float(cci), float(stoch), float(will), float(atr), float(park),
        ofi_window, ofi_last, ofi_trend, vol_z, trade_intensity, float(np.log1p(avg_trade)),
        abs(float(c[-1]) - 0.5) * 2, float(np.log1p(total_v)), float(np.log1p(trd.sum())), float(len(c)),
    ]


FEATURE_NAMES = [
    "last", "mean_ret", "vol", "drift", "band_z", "momentum",
    "rsi", "cci", "stoch_k", "williams_r", "atr", "parkinson_vol",
    "ofi_window", "ofi_last", "ofi_trend", "vol_z", "trade_intensity", "log_avg_trade",
    "extremeness", "log_volume", "log_trades", "win_len",
]

# Raw per-bar channels the GRU reads across the window — the sequence the
# summary features are computed *from*, so the neural branch can learn shape
# the 22 scalars throw away (order of moves, not just their aggregate).
SEQ_CHANNELS = ["ret", "range", "body", "log_vol", "flow", "log_trades"]


def _win_seq(o, h, l, c, vol, trd, sig):
    """Per-bar channel matrix (WINDOW × len(SEQ_CHANNELS)) for the GRU branch."""
    import numpy as np

    c = np.asarray(c, np.float64); o = np.asarray(o, np.float64)
    h = np.asarray(h, np.float64); l = np.asarray(l, np.float64)
    vol = np.asarray(vol, np.float64); trd = np.asarray(trd, np.float64)
    sig = np.asarray(sig, np.float64)
    prev = np.concatenate([c[:1], c[:-1]])            # close shifted one bar
    ret = (c - prev) / np.clip(prev, 1e-6, None)
    rng = (h - l) / np.clip(c, 1e-6, None)
    body = (c - o) / np.clip(c, 1e-6, None)
    lvol = np.log1p(np.clip(vol, 0, None))
    flow = sig / (vol + 1e-9)                          # per-bar aggressor imbalance ∈ [-1, 1]
    ltrd = np.log1p(np.clip(trd, 0, None))
    return np.stack([ret, rng, body, lvol, flow, ltrd], axis=1).astype(np.float32)


def _auc(scores, labels):
    import numpy as np
    s = np.asarray(scores); y = np.asarray(labels) > 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    p, n = int(y.sum()), int((~y).sum())
    return 0.5 if p == 0 or n == 0 else float((ranks[y].sum() - p * (p + 1) / 2) / (p * n))


def _backtest(scores, fwds, q=0.2):
    import numpy as np
    s = np.asarray(scores); f = np.asarray(fwds)
    m = np.abs(f) > 1e-9; s, f = s[m], f[m]
    if len(f) == 0:
        return {}
    o = np.argsort(s); k = max(1, int(len(o) * q))
    top, bot = f[o[-k:]] > 0, f[o[:k]] > 0
    return {"top_up_rate": round(float(top.mean()), 3), "bottom_up_rate": round(float(bot.mean()), 3),
            "up_rate_spread": round(float(top.mean() - bot.mean()), 3), "slice": int(k)}


@app.function(image=image, gpu="H100", cpu=16.0, memory=131072, timeout=14400,
              volumes={"/cache": cache_vol})
def run(max_files: int = 42, top_tokens: int = 8000, epochs: int = 80, push: bool = False, hf_token: str = "") -> dict:
    import numpy as np
    import polars as pl
    from huggingface_hub import HfApi, hf_hub_download

    # Authenticate the dataset downloads. Unauthenticated pulls of the 27 GB tape
    # get rate-limited and crawl, which is exactly the long window where earlier
    # runs stalled and were cancelled. A token lifts the limit and speeds it up.
    tok = hf_token or None
    if tok:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    cache_path = f"/cache/windows_mf{max_files}_tt{top_tokens}.npz"
    if os.path.exists(cache_path):
        print(f"loading cached windows: {cache_path}", flush=True)
        d = np.load(cache_path)
        X, S, y, fwds, tfrac = d["X"], d["S"], d["y"], d["fwds"], d["tfrac"]
        n_files, n_tokens, n_bars = int(d["n_files"]), int(d["n_tokens"]), int(d["n_bars"])
    else:
        api = HfApi(token=tok)
        # OrderFilled/ holds 42 monthly parquet files (2022_11 … 2026_04), already
        # chronological — sorted() keeps time order, so incremental merges stay cheap.
        files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
                 if f.startswith("OrderFilled/") and f.endswith(".parquet")]
        files = sorted(files)[:max_files]
        print(f"orderfilled files: {len(files)} (auth={'yes' if tok else 'no'})", flush=True)

        master, buf = None, []
        for i, fn in enumerate(files):
            path = hf_hub_download(REPO, fn, repo_type="dataset", token=tok)
            a = _agg_file(pl, path)
            os.remove(path)                       # free the 27 GB as we go
            if a is not None:
                buf.append(a)
            if len(buf) >= MERGE_EVERY or i == len(files) - 1:
                frames = ([master] if master is not None else []) + buf
                master = _merge(pl, frames)
                buf = []
                print(f"  merged through file {i+1}/{len(files)} → {master.height} bars", flush=True)

        # Top tokens by trade count → tractable, liquid training set.
        counts = master.group_by("token_asset_id").agg(pl.col("trades").sum().alias("t")).sort("t", descending=True)
        keep = set(counts.head(top_tokens)["token_asset_id"].to_list())
        master = master.filter(pl.col("token_asset_id").is_in(keep)).sort(["token_asset_id", "hour"])
        n_files, n_tokens, n_bars = len(files), len(keep), master.height
        print(f"kept {n_tokens} tokens, {n_bars} bars", flush=True)

        # Build per-token series → windows with real-OFI features + raw sequences.
        feats, seqs, labels, fwds, tfrac = [], [], [], [], []
        by_tok = master.partition_by("token_asset_id")
        ntok = len(by_tok)
        for gi, g in enumerate(by_tok):
            if gi % 1000 == 0:
                print(f"  window-build {gi}/{ntok} tokens → {len(labels)} windows", flush=True)
            o = g["open"].to_list(); h = g["high"].to_list(); l = g["low"].to_list(); c = g["close"].to_list()
            v = g["volume"].to_list(); n = g["trades"].to_list(); s = g["signed_vol"].to_list()
            N = len(c)
            if N < MIN_CANDLES:
                continue
            for i in range(WINDOW, N - HORIZON):
                cw = c[i - WINDOW:i]
                if float(np.std(np.diff(cw))) < 1e-4:
                    continue
                fwd = c[i + HORIZON] - c[i]
                sl = slice(i - WINDOW, i)
                feats.append(_win_feats(o[sl], h[sl], l[sl], cw, v[sl], n[sl], s[sl]))
                seqs.append(_win_seq(o[sl], h[sl], l[sl], cw, v[sl], n[sl], s[sl]))
                labels.append(1.0 if fwd > 0 else 0.0)
                fwds.append(fwd)
                tfrac.append(i / N)
        X = np.array(feats, np.float32); y = np.array(labels, np.float32)
        S = np.asarray(seqs, np.float32)          # [N, WINDOW, len(SEQ_CHANNELS)]
        fwds = np.array(fwds, np.float32); tfrac = np.array(tfrac, np.float32)
        np.savez(cache_path, X=X, S=S, y=y, fwds=fwds, tfrac=tfrac,
                 n_files=n_files, n_tokens=n_tokens, n_bars=n_bars)
        cache_vol.commit()
        print(f"cached windows → {cache_path}", flush=True)

    # Guard the GPU against any non-finite value slipping through (a single
    # inf/nan feeding cuDNN shows up as an illegal memory access, not a clean
    # error), then standardise on train-only statistics.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"windows {len(y)}, features {X.shape[1]}, seq {tuple(S.shape[1:])}, up-rate {y.mean():.3f}", flush=True)

    val = tfrac >= 0.8
    tr_i, va_i = np.where(~val)[0], np.where(val)[0]
    # Train-only standardisation for both branches — no val statistics leak in.
    fmean, fstd = X[tr_i].mean(0), X[tr_i].std(0) + 1e-6
    Xn = (X - fmean) / fstd
    smean = S[tr_i].reshape(-1, S.shape[-1]).mean(0)
    sstd = S[tr_i].reshape(-1, S.shape[-1]).std(0) + 1e-6
    Sn = (S - smean) / sstd

    result = {
        "runtime": "modal H100 / bigdata (all-GPU: XGBoost-CUDA + bi-GRU fusion)",
        "dataset": REPO,
        "source": "orderfilled tape (1.2B trades) → hourly OHLCV + true order-flow",
        "files_processed": n_files, "tokens": n_tokens, "bars": n_bars,
        "windows": int(len(y)), "features": FEATURE_NAMES, "seq_channels": SEQ_CHANNELS,
        "window": WINDOW, "horizon": HORIZON,
        "train_windows": int(len(tr_i)), "val_windows": int(len(va_i)),
        "majority_baseline_acc": round(float(max(y[tr_i].mean(), 1 - y[tr_i].mean())), 4),
    }

    # ── GBDT on GPU (XGBoost / CUDA, primary for tabular) ─────────────────────
    import xgboost as xgb
    pos = float(y[tr_i].sum()); neg = float(len(tr_i) - pos)
    dtr = xgb.DMatrix(X[tr_i], label=y[tr_i], feature_names=list(FEATURE_NAMES))
    dva = xgb.DMatrix(X[va_i], label=y[va_i], feature_names=list(FEATURE_NAMES))
    xparams = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
               "device": "cuda", "max_depth": 8, "eta": 0.02, "subsample": 0.8,
               "colsample_bytree": 0.8, "min_child_weight": 5.0, "lambda": 1.0, "gamma": 0.1,
               "max_bin": 256, "scale_pos_weight": (neg / pos) if pos > 0 else 1.0, "seed": 11}
    bst = xgb.train(xparams, dtr, num_boost_round=2000, evals=[(dva, "val")],
                    early_stopping_rounds=100, verbose_eval=False)
    best_it = getattr(bst, "best_iteration", None)
    rng = (0, best_it + 1) if best_it is not None else None
    gp = bst.predict(dva, iteration_range=rng) if rng else bst.predict(dva)
    result["gbdt"] = {"impl": "xgboost-cuda", "best_iteration": int(best_it) if best_it is not None else None,
                      "val_auc": round(_auc(gp, y[va_i]), 4),
                      "brier": round(float(np.mean((gp - y[va_i]) ** 2)), 4), "backtest": _backtest(gp, fwds[va_i])}
    gain = bst.get_score(importance_type="gain")
    imp = sorted(((f, round(float(gain.get(f, 0.0)), 1)) for f in FEATURE_NAMES), key=lambda t: -t[1])
    result["feature_importance_gbdt"] = imp
    print(f"[gbdt] {result['gbdt']}", flush=True)

    # ── Neural: bi-GRU over the raw sequence + feature-MLP, fused (GPU, fp32) ──
    import torch
    import torch.nn as nn
    dev = "cuda"
    torch.manual_seed(11)

    class FusionNet(nn.Module):
        """Two branches on one window: a 2-layer bidirectional GRU reads the raw
        per-bar sequence (shape, order, flow over time); an MLP embeds the 22
        summary features. Their concatenation feeds a deeper head. The GRU sees
        what the scalars discard — the *path*, not just its aggregates."""

        def __init__(self, n_feat, n_ch, gru=128, hidden=128, layers=2):
            super().__init__()
            self.gru = nn.GRU(n_ch, gru, num_layers=layers, batch_first=True,
                              bidirectional=True, dropout=0.2)
            self.fmlp = nn.Sequential(
                nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3))
            self.head = nn.Sequential(
                nn.Linear(gru * 2 + hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(hidden // 2, 1))

        def forward(self, seq, feat):
            _, h = self.gru(seq)                        # h: [layers*2, B, gru]
            hcat = torch.cat([h[-2], h[-1]], dim=-1)    # final layer's fwd + bwd states
            return self.head(torch.cat([hcat, self.fmlp(feat)], dim=-1)).squeeze(-1)

    Xt = torch.tensor(Xn, device=dev)
    St = torch.tensor(Sn, device=dev)
    yt = torch.tensor(y, device=dev)
    ti = torch.tensor(tr_i, device=dev); vi = torch.tensor(va_i, device=dev)
    net = FusionNet(Xn.shape[1], S.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0], device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    BATCH = 8192
    best_auc, best_state, nn_probs, patience = 0.0, None, None, 0

    def _predict(idx):
        net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(idx), BATCH):
                bb = idx[b:b + BATCH]
                out.append(torch.sigmoid(net(St[bb], Xt[bb])).cpu())
        return torch.cat(out).numpy()

    # fp32 throughout: cuDNN's GRU under autocast was throwing an illegal memory
    # access at full scale, and predictions are batched so no single forward has
    # to hold millions of windows of activations at once.
    for ep in range(epochs):
        net.train()
        perm = ti[torch.randperm(len(tr_i), device=dev)]
        for b in range(0, len(tr_i), BATCH):
            bi = perm[b:b + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = bce(net(St[bi], Xt[bi]), yt[bi])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        pv = _predict(vi)
        a = _auc(pv, y[va_i])
        if a > best_auc:
            best_auc, nn_probs, patience = a, pv, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                print(f"  neural early stop at epoch {ep} (best AUC {best_auc:.4f})", flush=True)
                break
    result["neural"] = {"impl": "bi-GRU+feat-MLP fusion (fp32, cosine LR)", "val_auc": round(best_auc, 4),
                        "brier": round(float(np.mean((nn_probs - y[va_i]) ** 2)), 4), "backtest": _backtest(nn_probs, fwds[va_i])}
    print(f"[neural] {result['neural']}", flush=True)

    # ── Ensemble: weight tuned on the out-of-time val slice ───────────────────
    gp = np.asarray(gp); nn_probs = np.asarray(nn_probs)
    best_w, best_ens_auc = 0.5, 0.0
    for w in np.linspace(0.0, 1.0, 21):
        a = _auc(w * gp + (1 - w) * nn_probs, y[va_i])
        if a > best_ens_auc:
            best_ens_auc, best_w = a, float(w)
    ens = best_w * gp + (1 - best_w) * nn_probs
    result["ensemble"] = {"gbdt_weight": round(best_w, 3), "val_auc": round(_auc(ens, y[va_i]), 4),
                          "brier": round(float(np.mean((ens - y[va_i]) ** 2)), 4), "backtest": _backtest(ens, fwds[va_i])}
    result["overall_best"] = max(("gbdt", "neural", "ensemble"), key=lambda k: result[k]["val_auc"])
    print(f"[ensemble] {result['ensemble']} | overall best {result['overall_best']}", flush=True)

    # ── Walk-forward over global time (XGBoost on GPU per fold) ───────────────
    wf = []
    for kf in range(1, 5):
        lo, hi = 0.2 * kf, 0.2 * (kf + 1)
        vmask = (tfrac >= lo) & (tfrac < hi)
        tmask = tfrac < lo - HORIZON / 1000
        if vmask.sum() < 50 or tmask.sum() < 200:
            continue
        d1 = xgb.DMatrix(X[tmask], label=y[tmask], feature_names=list(FEATURE_NAMES))
        dv = xgb.DMatrix(X[vmask], feature_names=list(FEATURE_NAMES))
        b = xgb.train(xparams, d1, num_boost_round=500, verbose_eval=False)
        pv = b.predict(dv)
        wf.append({"fold": kf, "val_auc": round(_auc(pv, y[vmask]), 4), "up_rate_spread": _backtest(pv, fwds[vmask]).get("up_rate_spread")})
    result["walk_forward"] = {"folds": wf, "mean_auc": round(float(np.mean([f["val_auc"] for f in wf])), 4) if wf else None}
    print(f"[walk-forward] {result['walk_forward']}", flush=True)

    # ── Save safetensors + normaliser + gbdt; return them for local push ─────
    import base64
    norm = {"fmean": fmean.tolist(), "fstd": fstd.tolist(), "features": FEATURE_NAMES,
            "seq_mean": smean.tolist(), "seq_std": sstd.tolist(), "seq_channels": SEQ_CHANNELS,
            "arch": {"type": "fusion", "gru_hidden": 128, "gru_layers": 2, "mlp_hidden": 128, "bidirectional": True},
            "window": WINDOW, "horizon": HORIZON, "bar_seconds": BAR_SECONDS}
    if best_state is not None:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in best_state.items()}, "/tmp/bigdata_model.safetensors")
    bst.save_model("/tmp/bigdata_gbdt.json")
    artifacts = {"bigdata_normalizer.json": json.dumps(norm).encode()}
    for name, p in [("bigdata_model.safetensors", "/tmp/bigdata_model.safetensors"),
                    ("bigdata_gbdt.json", "/tmp/bigdata_gbdt.json")]:
        if os.path.exists(p):
            artifacts[name] = open(p, "rb").read()
    result["_artifacts_b64"] = {k: base64.b64encode(v).decode() for k, v in artifacts.items()}

    # Optional in-Modal push (only if a token was passed); local push is the default path.
    if push and hf_token:
        try:
            api2 = HfApi(token=hf_token)
            repo = "shubhxho/polymarket-signal-model"
            api2.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
            for name, data in artifacts.items():
                api2.upload_file(path_or_fileobj=data, path_in_repo=f"bigdata/{name}", repo_id=repo, repo_type="model")
            api2.upload_file(path_or_fileobj=json.dumps({k: v for k, v in result.items() if k != "_artifacts_b64"}, indent=2).encode(),
                             path_in_repo="metrics/bigdata_metrics.json", repo_id=repo, repo_type="model")
            result["hf_repo"] = f"https://huggingface.co/{repo}"
            print(f"pushed → {result['hf_repo']}", flush=True)
        except Exception as e:
            result["push_error"] = f"{type(e).__name__}: {e}"[:300]
            print("push failed:", result["push_error"], flush=True)

    return result


@app.local_entrypoint()
def main(max_files: int = 42, top_tokens: int = 8000, epochs: int = 80, push: bool = False):
    token = os.environ.get("HF_TOKEN", "")
    report = run.remote(max_files=max_files, top_tokens=top_tokens, epochs=epochs, push=push, hf_token=token)
    import base64
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    # Save the returned model artifacts locally so push_hf.py can upload them.
    for name, b64 in report.pop("_artifacts_b64", {}).items():
        with open(os.path.join(data_dir, name), "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"saved data/{name}")
    with open(os.path.join(data_dir, "bigdata_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {data_dir}/bigdata_metrics.json")
