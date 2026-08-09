"""Distil Chronos-Bolt (teacher) into a tiny direction student, on an H100.

The terminal serves a small MLX model for latency. A 205M-param foundation
model (`amazon/chronos-bolt-base`, top of the fev-bench / GIFT-Eval board) knows
far more about time-series shape than our student can learn from labels alone.
Knowledge distillation transfers that: the teacher emits a *soft* up-probability
for every window — read straight from where the current price sits in Chronos's
predictive quantile distribution — and the student is trained to match the soft
target as well as the hard 0/1 outcome.

To prove the distillation actually buys something, the job trains the *same*
student twice — once with the KD term, once on hard labels only — and reports
both. The KD student and its metrics are pushed to the HF Hub.

    modal run ml/modal_distill.py --n-markets 1200 --push

Feature extraction is imported from the sibling `features.py`, which Modal mounts
automatically — so the student sees the exact 13 features (incl. RSI/CCI/MACD)
the local model and MCP server use. Nothing drifts.
"""

from __future__ import annotations

import json
import os

import modal

app = modal.App("pmt-distill")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "chronos-forecasting",
        "huggingface_hub",
        "numpy",
        "torch",
    )
    # Mount the real feature code so the student's inputs match the terminal's.
    .add_local_python_source("features")
)

# Same public-API fetch the terminal + fetch_data.py use, embedded so the remote
# container needs no extra local files beyond features.py.
FETCH_SRC = r'''
import json, urllib.parse, urllib.request, concurrent.futures as cf
GAMMA="https://gamma-api.polymarket.com"; CLOB="https://clob.polymarket.com"
H={"accept":"application/json","user-agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
def _get(url,tries=3):
    last=None
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=H),timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e: last=e
    raise RuntimeError(f"{url} :: {last}")
def _tokens(limit):
    out=[]; seen=set(); off=0
    while len(out)<limit:
        q=urllib.parse.urlencode({"limit":100,"offset":off,"order":"volume24hr","ascending":"false","active":"true","closed":"false"})
        rows=_get(f"{GAMMA}/markets?{q}")
        if not rows: break
        for m in rows:
            raw=m.get("clobTokenIds"); ids=json.loads(raw) if isinstance(raw,str) else raw
            if ids and str(ids[0]) not in seen: seen.add(str(ids[0])); out.append(str(ids[0]))
        off+=100
        if len(rows)<100: break
    return out[:limit]
def _hist(tok,interval):
    q=urllib.parse.urlencode({"market":tok,"interval":interval,"fidelity":"60"})
    d=_get(f"{CLOB}/prices-history?{q}")
    pts=d.get("history",[]) if isinstance(d,dict) else []
    return [float(p["p"]) for p in pts if "p" in p]
def fetch(limit, interval="max"):
    toks=_tokens(limit); print(f"tokens {len(toks)}; pulling history…", flush=True); series=[]
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        futs={ex.submit(_hist,t,interval):t for t in toks}
        for i,f in enumerate(cf.as_completed(futs)):
            try: s=f.result()
            except Exception: s=[]
            if len(s)>=40: series.append(s)
            if (i+1)%100==0: print(f"  {i+1}/{len(toks)} kept {len(series)}", flush=True)
    return series
'''

WINDOW = 16
HORIZON = 4
CTX = 128          # context length fed to the teacher
STRIDE = 3         # subsample windows so the teacher pass stays bounded
MAX_CONTEXTS = 40000


def _signal_metrics(scores, fwds):
    """Direction accuracy + AUC + decile backtest from predicted up-scores."""
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    fwds = np.asarray(fwds, dtype=float)
    mask = np.abs(fwds) > 1e-9
    scores, fwds = scores[mask], fwds[mask]
    if len(fwds) == 0:
        return {"direction_acc": 0.0, "auc": 0.5, "n": 0, "backtest": {}}
    up = fwds > 0
    acc = float(((scores > 0.5) == up).mean())
    # AUC via Mann-Whitney on the scores.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(up.sum())
    n_neg = len(up) - n_pos
    auc = 0.5 if n_pos == 0 or n_neg == 0 else float((ranks[up].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    k = max(1, int(len(order) * 0.2))
    so = np.argsort(scores)
    top = up[so[-k:]]
    bot = up[so[:k]]
    return {
        "direction_acc": round(acc, 4),
        "auc": round(auc, 4),
        "n": int(len(fwds)),
        "backtest": {
            "top_up_rate": round(float(top.mean()), 3),
            "bottom_up_rate": round(float(bot.mean()), 3),
            "up_rate_spread": round(float(top.mean() - bot.mean()), 3),
            "slice": int(k),
        },
    }


@app.function(image=image, gpu="H100", timeout=5400, cpu=8.0)
def run(n_markets: int = 1200, interval: str = "max", epochs: int = 60,
        alpha: float = 0.5, push: bool = False, hf_token: str = "") -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    from features import FEATURE_NAMES, window_features  # mounted alongside

    ns: dict = {}
    exec(FETCH_SRC, ns)
    series = [s for s in ns["fetch"](n_markets, interval) if len(s) > WINDOW + HORIZON + 4]
    print(f"fetched {len(series)} usable series", flush=True)

    # ── Build windows (temporal, per-series), carrying context for the teacher ─
    # rec = (series_idx, ctx_prices, seq_rets, feat, hard_label, fwd, t_frac)
    recs = []
    for si, s in enumerate(series):
        for i in range(WINDOW, len(s) - HORIZON, STRIDE):
            w = s[i - WINDOW:i]
            rets = [w[k] - w[k - 1] for k in range(1, len(w))]
            if float(np.std(rets)) < 1e-4:
                continue
            fwd = s[i + HORIZON] - s[i]
            ctx = s[max(0, i - CTX):i]
            recs.append((si, ctx, rets, window_features(w), 1.0 if fwd > 0 else 0.0, fwd, i / len(s)))
    if len(recs) > MAX_CONTEXTS:
        idx = np.linspace(0, len(recs) - 1, MAX_CONTEXTS).astype(int)
        recs = [recs[j] for j in idx]
    print(f"built {len(recs)} windows", flush=True)

    result = {
        "runtime": "modal H100 / distillation",
        "teacher": "amazon/chronos-bolt-base",
        "student": "gru+features (torch)",
        "series": len(series),
        "windows": len(recs),
        "features": FEATURE_NAMES,
        "horizon": HORIZON,
        "alpha_kd": alpha,
    }

    # ── Teacher: soft up-probability from Chronos's predictive quantiles ───────
    QLEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    soft = np.full(len(recs), 0.5, dtype=np.float32)
    try:
        from chronos import BaseChronosPipeline

        pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map="cuda", torch_dtype=torch.bfloat16
        )
        B = 256
        for b in range(0, len(recs), B):
            chunk = recs[b:b + B]
            ctxs = [torch.tensor(r[1], dtype=torch.float32) for r in chunk]
            q, _ = pipe.predict_quantiles(ctxs, prediction_length=HORIZON, quantile_levels=QLEVELS)
            # q: (batch, HORIZON, n_quantiles). Distribution at the horizon step.
            qh = q[:, HORIZON - 1, :].float().cpu().numpy()
            ql = np.asarray(QLEVELS, dtype=np.float64)
            for j, r in enumerate(chunk):
                last = float(r[1][-1])
                qs = np.sort(qh[j].astype(np.float64))  # enforce monotone grid
                # CDF(last) = P(forecast <= last), read off the quantile grid by
                # linear interpolation; np.interp clamps to [ql[0], ql[-1]] at the
                # tails. P(up) = 1 - CDF(last).
                cdf_at_last = float(np.interp(last, qs, ql))
                soft[b + j] = float(min(0.999, max(0.001, 1.0 - cdf_at_last)))
            if (b // B) % 10 == 0:
                print(f"  teacher {b + len(chunk)}/{len(recs)}", flush=True)
        result["teacher_signal"] = _signal_metrics(soft.tolist(), [r[5] for r in recs])
        print(f"[teacher] {result['teacher_signal']}", flush=True)
    except Exception as e:  # graceful: student still trains on hard labels
        result["teacher_error"] = f"{type(e).__name__}: {e}"[:400]
        print("teacher failed, KD disabled:", result["teacher_error"], flush=True)
        alpha = 1.0

    # ── Tensors + temporal split (val = each series' latest windows) ──────────
    seqs = np.array([[[x] for x in r[2]] for r in recs], dtype=np.float32)
    feats = np.array([r[3] for r in recs], dtype=np.float32)
    hard = np.array([r[4] for r in recs], dtype=np.float32)
    fwds = np.array([r[5] for r in recs], dtype=np.float32)
    tfrac = np.array([r[6] for r in recs], dtype=np.float32)
    val_mask = tfrac >= 0.8
    tr_i = np.where(~val_mask)[0]
    va_i = np.where(val_mask)[0]

    fmean = feats[tr_i].mean(0)
    fstd = feats[tr_i].std(0) + 1e-6
    rstd = seqs[tr_i].std() + 1e-6
    feats = (feats - fmean) / fstd
    seqs = seqs / rstd

    dev = "cuda"
    S = torch.tensor(seqs, device=dev)
    F = torch.tensor(feats, device=dev)
    H = torch.tensor(hard, device=dev)
    T = torch.tensor(soft, device=dev)

    class Student(nn.Module):
        def __init__(self, n_feat, hidden=48):
            super().__init__()
            self.gru = nn.GRU(1, hidden, batch_first=True)
            self.fproj = nn.Linear(n_feat, hidden)
            self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

        def forward(self, s, f):
            _, h = self.gru(s)
            x = torch.cat([h[-1], torch.relu(self.fproj(f))], -1)
            return self.head(x).squeeze(-1)

    def train_student(use_kd: bool):
        torch.manual_seed(11)
        m = Student(F.shape[1]).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=3e-4)
        bce = nn.BCEWithLogitsLoss()
        n = len(tr_i)
        best_auc, best_state = 0.0, None
        idx_t = torch.tensor(tr_i, device=dev)
        idx_v = torch.tensor(va_i, device=dev)
        for ep in range(epochs):
            m.train()
            perm = idx_t[torch.randperm(n, device=dev)]
            for b in range(0, n, 256):
                bi = perm[b:b + 256]
                logit = m(S[bi], F[bi])
                loss = bce(logit, H[bi])
                if use_kd and alpha < 1.0:
                    loss = alpha * loss + (1 - alpha) * bce(logit, T[bi])
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pv = torch.sigmoid(m(S[idx_v], F[idx_v])).cpu().numpy()
            a = _signal_metrics(pv.tolist(), fwds[va_i].tolist())["auc"]
            if a > best_auc:
                best_auc = a
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_state)
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(S[torch.tensor(va_i, device=dev)], F[torch.tensor(va_i, device=dev)])).cpu().numpy()
        return m, _signal_metrics(pv.tolist(), fwds[va_i].tolist())

    print(f"training student WITH distillation (alpha={alpha}) …", flush=True)
    kd_model, kd_metrics = train_student(use_kd=True)
    result["student_distilled"] = kd_metrics
    print(f"[distilled] {kd_metrics}", flush=True)

    print("training student WITHOUT distillation (ablation) …", flush=True)
    _, base_metrics = train_student(use_kd=False)
    result["student_no_kd"] = base_metrics
    result["kd_auc_gain"] = round(kd_metrics["auc"] - base_metrics["auc"], 4)
    result["kd_acc_gain"] = round(kd_metrics["direction_acc"] - base_metrics["direction_acc"], 4)
    print(f"KD auc gain {result['kd_auc_gain']:+.4f}  acc gain {result['kd_acc_gain']:+.4f}", flush=True)

    # ── Push distilled student + normaliser + metrics ─────────────────────────
    if push and hf_token:
        try:
            import io
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            repo = "shubhxho/polymarket-signal-distilled"
            api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
            buf = io.BytesIO()
            torch.save({"state_dict": kd_model.state_dict(),
                        "fmean": fmean.tolist(), "fstd": fstd.tolist(), "rstd": float(rstd),
                        "features": FEATURE_NAMES, "window": WINDOW, "horizon": HORIZON}, buf)
            api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo="student.pt", repo_id=repo, repo_type="model")
            card = (
                "---\nlicense: apache-2.0\nbase_model: amazon/chronos-bolt-base\n"
                "tags:\n- time-series-forecasting\n- polymarket\n- knowledge-distillation\n- trading-signals\n---\n"
                "# Polymarket signal student, distilled from Chronos-Bolt\n\n"
                "A tiny GRU+features direction model distilled from `amazon/chronos-bolt-base`. "
                "The teacher's predictive quantiles become soft up-probabilities; the student learns "
                "them alongside realised outcomes. Ablation (with vs without KD) and signal metrics below.\n\n"
                f"```json\n{json.dumps(result, indent=2)}\n```\n"
            )
            api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="model")
            api.upload_file(path_or_fileobj=json.dumps(result, indent=2).encode(), path_in_repo="metrics.json", repo_id=repo, repo_type="model")
            result["hf_repo"] = f"https://huggingface.co/{repo}"
            print(f"pushed → {result['hf_repo']}", flush=True)
        except Exception as e:
            result["push_error"] = f"{type(e).__name__}: {e}"[:300]
            print("push failed:", result["push_error"], flush=True)

    return result


@app.local_entrypoint()
def main(n_markets: int = 1200, interval: str = "max", epochs: int = 60,
         alpha: float = 0.5, push: bool = False):
    token = ""
    if push:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token() or os.environ.get("HF_TOKEN", "")
        except Exception:
            token = os.environ.get("HF_TOKEN", "")
    report = run.remote(n_markets=n_markets, interval=interval, epochs=epochs, alpha=alpha, push=push, hf_token=token)
    out = os.path.join(os.path.dirname(__file__), "data", "distill_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
