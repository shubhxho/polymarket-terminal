"""Chronos-Bolt-Base on Polymarket, on an H100 — the best time-series foundation
model, applied to real market data and pushed to the HF Hub.

`amazon/chronos-bolt-base` leads the GIFT-Eval foundation-model leaderboard.
This job runs it on real Polymarket price series two ways:

1. **Zero-shot** via the native `chronos-forecasting` API — I control the context
   tensors directly, so it's robust (AutoGluon's wrapper has a Chronos-Bolt patch
   bug on variable-length inputs). This is the guaranteed signal benchmark.
2. **Fine-tuned** via AutoGluon-TimeSeries, best-effort — if the library
   cooperates the fine-tuned signal is reported and its checkpoint pushed; if the
   known bug bites, the run degrades gracefully to the zero-shot result.

Either way it evaluates the *signal* (direction accuracy + decile backtest) and
pushes a model repo to the HF Hub.

    modal run ml/modal_chronos.py --n-markets 800 --push
"""

from __future__ import annotations

import json
import os

import modal

app = modal.App("pmt-chronos")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "chronos-forecasting",
        "autogluon.timeseries",
        "huggingface_hub",
        "pandas",
        "numpy",
        "torch",
    )
)

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
            if len(s)>=32: series.append(s)
            if (i+1)%100==0: print(f"  {i+1}/{len(toks)} kept {len(series)}", flush=True)
    return series
'''

HORIZON = 6      # steps ahead the signal looks
CTX = 256        # context points fed to the model


def _signal_metrics(scores, fwds):
    """direction accuracy + decile backtest from predicted-move `scores` and
    realised forward moves `fwds`."""
    import numpy as np

    scores = np.asarray(scores)
    fwds = np.asarray(fwds)
    mask = np.abs(fwds) > 1e-9
    scores, fwds = scores[mask], fwds[mask]
    if len(fwds) == 0:
        return {"direction_acc": 0.0, "n": 0, "backtest": {}}
    acc = float(((scores > 0) == (fwds > 0)).mean())
    order = np.argsort(scores)
    k = max(1, int(len(order) * 0.2))
    top = fwds[order[-k:]] > 0
    bot = fwds[order[:k]] > 0
    return {
        "direction_acc": round(acc, 4),
        "n": int(len(fwds)),
        "backtest": {
            "top_up_rate": round(float(top.mean()), 3),
            "bottom_up_rate": round(float(bot.mean()), 3),
            "up_rate_spread": round(float(top.mean() - bot.mean()), 3),
            "slice": int(k),
        },
    }


@app.function(image=image, gpu="H100", timeout=3600, cpu=8.0)
def run(n_markets: int = 800, steps: int = 1000, interval: str = "max", push: bool = False, hf_token: str = "") -> dict:
    ns: dict = {}
    exec(FETCH_SRC, ns)
    series = [s for s in ns["fetch"](n_markets, interval) if len(s) > HORIZON + 8]
    print(f"fetched {len(series)} usable series", flush=True)

    import numpy as np
    import torch

    result = {
        "runtime": "modal H100 / chronos-bolt-base",
        "base_model": "amazon/chronos-bolt-base",
        "series": len(series),
        "horizon": HORIZON,
    }

    # ── 1. Zero-shot via native chronos (robust) ─────────────────────────────
    try:
        from chronos import BaseChronosPipeline

        pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map="cuda", torch_dtype=torch.bfloat16
        )
        contexts, lasts, actuals = [], [], []
        for s in series:
            ctx = s[:-HORIZON][-CTX:]
            contexts.append(torch.tensor(ctx, dtype=torch.float32))
            lasts.append(s[-HORIZON - 1])
            actuals.append(s[-1])
        scores, fwds = [], []
        B = 256
        for i in range(0, len(contexts), B):
            # `inputs` is the first positional arg; returns (quantiles, mean).
            _, mean = pipe.predict_quantiles(
                contexts[i : i + B], prediction_length=HORIZON, quantile_levels=[0.5]
            )
            m = mean[:, HORIZON - 1].float().cpu().numpy()
            for j, fc in enumerate(m):
                scores.append(float(fc) - lasts[i + j])
                fwds.append(actuals[i + j] - lasts[i + j])
        result["zero_shot"] = _signal_metrics(scores, fwds)
        print(f"[zero-shot] {result['zero_shot']}", flush=True)
    except Exception as e:
        result["zero_shot_error"] = f"{type(e).__name__}: {e}"[:400]
        print("zero-shot failed:", result["zero_shot_error"], flush=True)

    # ── 2. Fine-tune via AutoGluon (best-effort) ─────────────────────────────
    ck = None
    try:
        import pandas as pd
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        rows = []
        for i, s in enumerate(series):
            start = pd.Timestamp("2020-01-01")
            for t, p in enumerate(s[-(CTX + HORIZON) :]):
                rows.append((f"m{i}", start + pd.Timedelta(hours=t), float(p)))
        df = pd.DataFrame(rows, columns=["item_id", "timestamp", "target"])
        tsdf = TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")
        tr = tsdf.slice_by_timestep(None, -HORIZON)
        ft = TimeSeriesPredictor(prediction_length=HORIZON, target="target", verbosity=1, path="/tmp/ag_ft")
        ft.fit(
            tr,
            hyperparameters={"Chronos": {"model_path": "bolt_base", "context_length": CTX, "fine_tune": True, "fine_tune_steps": steps}},
            skip_model_selection=True,
            time_limit=1800,
        )
        pred = ft.predict(tr)
        fscores, ffwds = [], []
        for item in tr.item_ids:
            last = float(tr.loc[item]["target"].iloc[-1])
            fc = float(pred.loc[item]["mean"].iloc[HORIZON - 1])
            actual = float(tsdf.loc[item]["target"].iloc[-1])
            fscores.append(fc - last)
            ffwds.append(actual - last)
        result["fine_tuned"] = _signal_metrics(fscores, ffwds)
        result["improvement_acc"] = round(
            result["fine_tuned"]["direction_acc"] - result["zero_shot"]["direction_acc"], 4
        )
        print(f"[fine-tuned] {result['fine_tuned']}", flush=True)
        import glob

        for c in glob.glob("/tmp/ag_ft/models/**/*.safetensors", recursive=True):
            d = os.path.dirname(c)
            if os.path.exists(os.path.join(d, "config.json")):
                ck = d
                break
    except Exception as e:
        result["fine_tune_error"] = f"{type(e).__name__}: {e}"[:400]
        print("fine-tune failed (keeping zero-shot):", result["fine_tune_error"], flush=True)

    # ── 3. Push to HF Hub ────────────────────────────────────────────────────
    if push and hf_token:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            repo = "shubhxho/chronos-bolt-polymarket-signals"
            api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
            card = (
                "---\nlicense: apache-2.0\nbase_model: amazon/chronos-bolt-base\n"
                "tags:\n- time-series-forecasting\n- polymarket\n- chronos\n- trading-signals\n---\n"
                "# Chronos-Bolt-Base on Polymarket signals\n\n"
                "The SOTA Chronos-Bolt foundation model applied to live Polymarket price "
                "series for short-horizon direction signals. Metrics below.\n\n"
                f"```json\n{json.dumps(result, indent=2)}\n```\n"
            )
            api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="model")
            api.upload_file(path_or_fileobj=json.dumps(result, indent=2).encode(), path_in_repo="metrics.json", repo_id=repo, repo_type="model")
            if ck:
                api.upload_folder(folder_path=ck, repo_id=repo, repo_type="model")
                result["pushed_checkpoint"] = True
            result["hf_repo"] = f"https://huggingface.co/{repo}"
            print(f"pushed → {result['hf_repo']}", flush=True)
        except Exception as e:
            result["push_error"] = f"{type(e).__name__}: {e}"[:300]
            print("push failed:", result["push_error"], flush=True)

    return result


@app.local_entrypoint()
def main(n_markets: int = 800, steps: int = 1000, interval: str = "max", push: bool = False):
    token = ""
    if push:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token() or os.environ.get("HF_TOKEN", "")
        except Exception:
            token = os.environ.get("HF_TOKEN", "")
    report = run.remote(n_markets=n_markets, steps=steps, interval=interval, push=push, hf_token=token)
    out = os.path.join(os.path.dirname(__file__), "data", "chronos_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
