"""Train the signal model on Modal, over the full Polymarket dataset.

MLX is Apple-silicon only (no Linux wheel), so it cannot run on Modal's Linux
containers. This is the honest cloud counterpart: the *same* architecture (a GRU
over the return sequence fused with the hand features vs a feature-MLP baseline)
and the *same* methodology (market-disjoint split, train-only standardisation,
class-weighted loss, best-val-AUC, decile backtest) ported to PyTorch, which is
the right tool for Modal's CUDA/CPU workers. The local MLX path in train_seq.py
stays the Apple-silicon story; this scales the data and the compute.

The whole job is self-contained so the remote worker needs no local files:
fetching (paged Gamma + threaded CLOB history), features, model and training all
live here.

    modal run ml/modal_train.py                 # default ~800 markets
    modal run ml/modal_train.py --n-markets 1500 # more of the dataset
"""

from __future__ import annotations

import json
import os

import modal

app = modal.App("pmt-signal")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "numpy")

# ── Everything below runs on the Modal worker ────────────────────────────────

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
        except Exception as e:
            last=e
    raise RuntimeError(f"{url} :: {last}")
def tokens(limit):
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
def history(tok,interval="1m",fidelity="60"):
    q=urllib.parse.urlencode({"market":tok,"interval":interval,"fidelity":fidelity})
    d=_get(f"{CLOB}/prices-history?{q}")
    pts=d.get("history",[]) if isinstance(d,dict) else []
    return [float(p["p"]) for p in pts if "p" in p]
def fetch(limit, interval):
    toks=tokens(limit); print(f"tokens {len(toks)}; pulling history threaded…", flush=True)
    series=[]
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        futs={ex.submit(history,t,interval):t for t in toks}
        for i,f in enumerate(cf.as_completed(futs)):
            try:
                s=f.result()
            except Exception: s=[]
            if len(s)>=24: series.append(s)
            if (i+1)%100==0: print(f"  {i+1}/{len(toks)} kept {len(series)}", flush=True)
    return series
'''

FEATURES_SRC = r'''
import math
WINDOW=16; HORIZON=4; MIN_STD=1e-4
FEATURE_NAMES=["last","mean_ret","vol","drift","band_z","momentum","autocorr","activity"]
def _mean(xs): return sum(xs)/len(xs) if xs else 0.0
def _std(xs):
    if len(xs)<2: return 0.0
    m=_mean(xs); return (sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5
def _ac(xs):
    if len(xs)<3: return 0.0
    m=_mean(xs); num=sum((xs[i]-m)*(xs[i-1]-m) for i in range(1,len(xs))); den=sum((x-m)**2 for x in xs)
    return max(-1.0,min(1.0,num/den)) if den>1e-12 else 0.0
def feats(w):
    r=[w[i]-w[i-1] for i in range(1,len(w))]; sw=_std(w); mw=_mean(w)
    bz=(w[-1]-mw)/sw if sw>1e-9 else 0.0; mo=_mean(r[-4:]) if len(r)>=4 else _mean(r)
    return [w[-1],_mean(r),_std(r),w[-1]-w[0],bz,mo,_ac(r),sum(abs(x) for x in r)]
def rich(prices):
    out=[]; n=len(prices)
    for i in range(WINDOW,n-HORIZON):
        w=prices[i-WINDOW:i]; r=[w[k]-w[k-1] for k in range(1,len(w))]
        if _std(r)<MIN_STD: continue
        fwd=prices[i+HORIZON]-prices[i]
        out.append((r,feats(w),1 if fwd>0 else 0,fwd))
    return out
'''


@app.function(image=image, timeout=3600, cpu=8.0)
def train_remote(n_markets: int = 800, interval: str = "1m") -> dict:
    ns: dict = {}
    exec(FETCH_SRC, ns)
    exec(FEATURES_SRC, ns)
    fetch, rich, FEATURE_NAMES = ns["fetch"], ns["rich"], ns["FEATURE_NAMES"]

    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(11)
    series = fetch(n_markets, interval)
    print(f"fetched {len(series)} series ({sum(len(s) for s in series)} points)", flush=True)

    def build(split):
        S, F, Y, W = [], [], [], []
        for prices in split:
            for r, f, y, fwd in rich(prices):
                S.append(r)
                F.append(f)
                Y.append(y)
                W.append(fwd)
        return S, F, Y, W

    n_val = max(1, int(len(series) * 0.2))
    Str, Ftr, Ytr, _ = build(series[n_val:])
    Sva, Fva, Yva, Wva = build(series[:n_val])
    if not Str or not Sva:
        raise SystemExit("not enough data")

    Ftr_a = np.array(Ftr, dtype=np.float32)
    fmean, fstd = Ftr_a.mean(0), Ftr_a.std(0) + 1e-6
    rstd = float(np.array([v for row in Str for v in row], dtype=np.float32).std()) + 1e-6

    def tens(S, F, Y):
        s = (torch.tensor(np.array(S, dtype=np.float32)) / rstd).unsqueeze(-1)
        f = (torch.tensor(np.array(F, dtype=np.float32)) - torch.tensor(fmean)) / torch.tensor(fstd)
        y = torch.tensor(np.array(Y, dtype=np.float32))
        return s, f, y

    str_, ftr_, ytr_ = tens(Str, Ftr, Ytr)
    sva_, fva_, yva_ = tens(Sva, Fva, Yva)

    class FeatureMLP(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n, 32), nn.ReLU(), nn.Dropout(0.25), nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.25), nn.Linear(32, 1))

        def forward(self, s, f):
            return self.net(f)

    class SeqGRU(nn.Module):
        def __init__(self, n, h=32):
            super().__init__()
            self.gru = nn.GRU(1, h, batch_first=True)
            self.fp = nn.Linear(n, h)
            self.h1 = nn.Linear(h * 2, h)
            self.h2 = nn.Linear(h, 1)
            self.drop = nn.Dropout(0.25)

        def forward(self, s, f):
            _, hn = self.gru(s)
            x = torch.cat([hn[-1], torch.relu(self.fp(f))], dim=-1)
            return self.h2(self.drop(torch.relu(self.h1(x))))

    def auc(p, y):
        order = np.argsort(p)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(p) + 1)
        npos = float(y.sum())
        nneg = len(y) - npos
        if npos == 0 or nneg == 0:
            return 0.5
        return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

    def backtest(p, fwd, q=0.2):
        order = np.argsort(p)
        k = max(1, int(len(p) * q))
        top, bot = fwd[order[-k:]], fwd[order[:k]]
        up = lambda a: float((a > 0).mean())
        return {"top_up_rate": round(up(top), 3), "bottom_up_rate": round(up(bot), 3),
                "up_rate_spread": round(up(top) - up(bot), 3),
                "top_median_pts": round(float(np.median(top)) * 100, 3),
                "bottom_median_pts": round(float(np.median(bot)) * 100, 3), "slice": k}

    pos = ytr_.mean().item()
    pw = torch.tensor([(1 - pos) / max(pos, 1e-3)])
    fwd_va = np.array(Wva, dtype=np.float32)
    results = {}
    best_name, best_auc, best_state = None, 0.0, None

    for name, model in {"feature_mlp": FeatureMLP(len(FEATURE_NAMES)), "seq_gru": SeqGRU(len(FEATURE_NAMES))}.items():
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        n = str_.shape[0]
        b_auc, b_state = 0.0, None
        for epoch in range(40):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i : i + 256]
                opt.zero_grad()
                out = model(str_[idx], ftr_[idx]).squeeze(-1)
                lossf(out, ytr_[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(sva_, fva_).squeeze(-1)).numpy()
            a = auc(pv, yva_.numpy())
            if a > b_auc:
                b_auc, b_state = a, {k: v.clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 10 == 0:
                print(f"  [{name}] epoch {epoch+1} val auc {a:.4f} (best {b_auc:.4f})", flush=True)
        model.load_state_dict(b_state)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(sva_, fva_).squeeze(-1)).numpy()
        results[name] = {"val_auc": round(float(b_auc), 4), "backtest": backtest(pv, fwd_va)}
        if b_auc > best_auc:
            best_name, best_auc, best_state = name, b_auc, b_state

    base = max(float(ytr_.mean()), 1 - float(ytr_.mean()))
    return {
        "runtime": "modal / pytorch",
        "series": len(series),
        "train_windows": int(str_.shape[0]),
        "val_windows": int(sva_.shape[0]),
        "majority_baseline_acc": round(base, 4),
        "models": results,
        "winner": best_name,
        "features": FEATURE_NAMES,
    }


@app.local_entrypoint()
def main(n_markets: int = 800, interval: str = "1m"):
    report = train_remote.remote(n_markets=n_markets, interval=interval)
    out = os.path.join(os.path.dirname(__file__), "data", "modal_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
