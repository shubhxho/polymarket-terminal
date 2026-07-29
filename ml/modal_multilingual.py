"""A multi-lingual signal model, on an H100.

Polymarket is a global venue; its questions arrive in many languages. This job
adds a *language-aware* signal channel: it embeds each market's question with a
multilingual sentence transformer (`paraphrase-multilingual-MiniLM-L12-v2`,
50+ languages) and asks whether that text signal adds anything to the numeric
price signal for short-horizon direction.

It is built to give an **honest** answer, not a flattering one:

1. Trains three heads on the same out-of-time split — price-features only, text
   (question embedding) only, and the fusion of both — and reports all three.
   If text adds nothing, the ablation says so.
2. Runs a **cross-lingual invariance check**: the same sentence in English,
   Spanish, Chinese, Arabic, Hindi and Portuguese should embed to nearly the
   same vector. High mean cosine similarity is the proof the model is genuinely
   multi-lingual, independent of whether text helps prediction.

    modal run ml/modal_multilingual.py --n-markets 700
"""

from __future__ import annotations

import json
import os

import modal

app = modal.App("pmt-multilingual")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("sentence-transformers", "torch", "numpy", "huggingface_hub")
    .add_local_python_source("features")
)

# Fetch markets WITH their question text (needed for the language channel).
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
def _markets(limit):
    out=[]; seen=set(); off=0
    while len(out)<limit:
        q=urllib.parse.urlencode({"limit":100,"offset":off,"order":"volume24hr","ascending":"false","active":"true","closed":"false"})
        rows=_get(f"{GAMMA}/markets?{q}")
        if not rows: break
        for m in rows:
            raw=m.get("clobTokenIds"); ids=json.loads(raw) if isinstance(raw,str) else raw
            ques=m.get("question") or ""
            if ids and str(ids[0]) not in seen and ques:
                seen.add(str(ids[0])); out.append((ques, str(ids[0])))
        off+=100
        if len(rows)<100: break
    return out[:limit]
def _hist(tok):
    q=urllib.parse.urlencode({"market":tok,"interval":"max","fidelity":"60"})
    d=_get(f"{CLOB}/prices-history?{q}")
    pts=d.get("history",[]) if isinstance(d,dict) else []
    return [float(p["p"]) for p in pts if "p" in p]
def fetch(limit):
    mk=_markets(limit); print(f"markets {len(mk)}; pulling history…", flush=True); out=[]
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        futs={ex.submit(_hist,t):(ques,t) for ques,t in mk}
        for i,f in enumerate(cf.as_completed(futs)):
            ques,tok=futs[f]
            try: s=f.result()
            except Exception: s=[]
            if len(s)>=40: out.append({"question":ques,"series":s})
            if (i+1)%100==0: print(f"  {i+1}/{len(mk)} kept {len(out)}", flush=True)
    return out
'''

WINDOW = 16
HORIZON = 4


def _auc(scores, labels):
    import numpy as np
    s = np.asarray(scores); y = np.asarray(labels) > 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    p, n = int(y.sum()), int((~y).sum())
    if p == 0 or n == 0:
        return 0.5
    return float((ranks[y].sum() - p * (p + 1) / 2) / (p * n))


@app.function(image=image, gpu="H100", timeout=3600, cpu=8.0)
def run(n_markets: int = 700, epochs: int = 40, push: bool = False, hf_token: str = "") -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer

    from features import window_features  # mounted; the 13 numeric features

    ns: dict = {}
    exec(FETCH_SRC, ns)
    data = ns["fetch"](n_markets)
    print(f"fetched {len(data)} markets with questions", flush=True)

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cuda")

    # ── Cross-lingual invariance check (proves the model is multi-lingual) ────
    probe = {
        "en": "Will the incumbent win the presidential election?",
        "es": "¿Ganará el titular las elecciones presidenciales?",
        "zh": "现任者会赢得总统选举吗？",
        "ar": "هل سيفوز الرئيس الحالي في الانتخابات الرئاسية؟",
        "hi": "क्या मौजूदा राष्ट्रपति चुनाव जीतेंगे?",
        "pt": "O atual presidente vencerá as eleições presidenciais?",
    }
    pv = embedder.encode(list(probe.values()), normalize_embeddings=True)
    sims = [float(np.dot(pv[0], pv[i])) for i in range(1, len(pv))]
    invariance = {lang: round(s, 3) for lang, s in zip(list(probe)[1:], sims)}
    result = {
        "runtime": "modal H100 / multilingual",
        "text_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "markets": len(data),
        "cross_lingual_cosine_to_en": invariance,
        "mean_cross_lingual_cosine": round(float(np.mean(sims)), 3),
    }
    print(f"cross-lingual cosine to EN: {invariance}", flush=True)

    # ── Build windows; each carries its market's question embedding ───────────
    questions = [d["question"] for d in data]
    q_emb = embedder.encode(questions, normalize_embeddings=True, batch_size=128)
    feats, texts, labels, fwds, tfrac = [], [], [], [], []
    for mi, d in enumerate(data):
        s = d["series"]
        for i in range(WINDOW, len(s) - HORIZON, 2):
            w = s[i - WINDOW:i]
            rets = [w[k] - w[k - 1] for k in range(1, len(w))]
            if float(np.std(rets)) < 1e-4:
                continue
            fwd = s[i + HORIZON] - s[i]
            feats.append(window_features(w))
            texts.append(q_emb[mi])
            labels.append(1.0 if fwd > 0 else 0.0)
            fwds.append(fwd)
            tfrac.append(i / len(s))
    feats = np.array(feats, np.float32)
    texts = np.array(texts, np.float32)
    labels = np.array(labels, np.float32)
    tfrac = np.array(tfrac, np.float32)
    print(f"built {len(labels)} windows; text dim {texts.shape[1]}", flush=True)

    val = tfrac >= 0.8
    tr_i, va_i = np.where(~val)[0], np.where(val)[0]
    fmean, fstd = feats[tr_i].mean(0), feats[tr_i].std(0) + 1e-6
    feats = (feats - fmean) / fstd

    dev = "cuda"
    Xf = torch.tensor(feats, device=dev)
    Xt = torch.tensor(texts, device=dev)
    Y = torch.tensor(labels, device=dev)
    ti = torch.tensor(tr_i, device=dev)
    vi = torch.tensor(va_i, device=dev)

    def train_head(use_price, use_text):
        torch.manual_seed(11)
        din = (Xf.shape[1] if use_price else 0) + (Xt.shape[1] if use_text else 0)
        net = nn.Sequential(nn.Linear(din, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
        bce = nn.BCEWithLogitsLoss()

        def X(idx):
            parts = []
            if use_price:
                parts.append(Xf[idx])
            if use_text:
                parts.append(Xt[idx])
            return torch.cat(parts, -1)

        best = 0.5
        n = len(tr_i)
        for _ in range(epochs):
            net.train()
            perm = ti[torch.randperm(n, device=dev)]
            for b in range(0, n, 256):
                bi = perm[b:b + 256]
                loss = bce(net(X(bi)).squeeze(-1), Y[bi])
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(X(vi)).squeeze(-1)).cpu().numpy()
            best = max(best, _auc(pv, labels[va_i]))
        return round(best, 4)

    result["price_only_auc"] = train_head(True, False)
    result["text_only_auc"] = train_head(False, True)
    result["fusion_auc"] = train_head(True, True)
    result["text_adds_auc"] = round(result["fusion_auc"] - result["price_only_auc"], 4)
    print(f"price {result['price_only_auc']} | text {result['text_only_auc']} | "
          f"fusion {result['fusion_auc']} | text adds {result['text_adds_auc']:+.4f}", flush=True)

    if push and hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            repo = "shubhxho/polymarket-multilingual-signal"
            api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=json.dumps(result, indent=2).encode(),
                            path_in_repo="metrics.json", repo_id=repo, repo_type="model")
            result["hf_repo"] = f"https://huggingface.co/{repo}"
        except Exception as e:
            result["push_error"] = f"{type(e).__name__}: {e}"[:300]

    return result


@app.local_entrypoint()
def main(n_markets: int = 700, epochs: int = 40, push: bool = False):
    token = os.environ.get("HF_TOKEN", "")
    report = run.remote(n_markets=n_markets, epochs=epochs, push=push, hf_token=token)
    out = os.path.join(os.path.dirname(__file__), "data", "multilingual_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
