"""Push the trained Polymarket signal models to the HuggingFace Hub.

Uploads the served close-only MLX model and the richer OHLCV model (+ the GBDT/
ensemble metrics), their normalisers, every metrics report, the feature/training
code and the research notes, plus a generated model card. Uses the cached HF
login (huggingface-cli login) — no token is printed or stored.

    python ml/push_hf.py                       # default repo
    python ml/push_hf.py --repo you/name       # custom repo
"""

from __future__ import annotations

import argparse
import json
import os

from huggingface_hub import HfApi

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")

# (local path, path in repo)
ARTIFACTS = [
    # served close-only MLX model
    ("data/seq_model.safetensors", "seq_model.safetensors"),
    ("data/seq_normalizer.json", "seq_normalizer.json"),
    ("data/seq_metrics.json", "metrics/seq_metrics.json"),
    # richer OHLCV model
    ("data/ohlcv_model.safetensors", "ohlcv_model.safetensors"),
    ("data/ohlcv_normalizer.json", "ohlcv_normalizer.json"),
    ("data/ohlcv_metrics.json", "metrics/ohlcv_metrics.json"),
    # flagship big-data model (1.2B trades, true order-flow) — from modal_bigdata.py
    ("data/bigdata_model.safetensors", "bigdata/bigdata_model.safetensors"),
    ("data/bigdata_gbdt.json", "bigdata/bigdata_gbdt.json"),
    ("data/bigdata_normalizer.json", "bigdata/bigdata_normalizer.json"),
    ("data/bigdata_metrics.json", "metrics/bigdata_metrics.json"),
    # H100 experiment reports
    ("data/distill_metrics.json", "metrics/distill_metrics.json"),
    ("data/multilingual_metrics.json", "metrics/multilingual_metrics.json"),
    ("data/chronos_metrics.json", "metrics/chronos_metrics.json"),
    ("data/series_100kb_metrics.json", "metrics/series_100kb_metrics.json"),
    # code (reproducibility)
    ("features.py", "code/features.py"),
    ("features_ohlcv.py", "code/features_ohlcv.py"),
    ("train_seq.py", "code/train_seq.py"),
    ("train_ohlcv.py", "code/train_ohlcv.py"),
    ("mcp_server.py", "code/mcp_server.py"),
    ("RESEARCH.md", "RESEARCH.md"),
]


def _load(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def _card(repo: str) -> str:
    seq = _load("seq_metrics.json")
    ohlcv = _load("ohlcv_metrics.json")
    distill = _load("distill_metrics.json")
    ml = _load("multilingual_metrics.json")
    big = _load("bigdata_metrics.json")
    seq_auc = seq.get("models", {}).get(seq.get("winner", ""), {}).get("val_auc")
    wf = seq.get("walk_forward", {})
    ob = {k: v.get("val_auc") for k, v in ohlcv.get("models", {}).items()}
    big_line = ""
    if big:
        bb = {k: big.get(k, {}).get("val_auc") for k in ("gbdt", "neural", "ensemble")}
        big_line = (
            f"\n**`bigdata/bigdata_model.safetensors` + `bigdata_gbdt.txt`** — the "
            f"flagship, trained on **{big.get('source','the 1.2B-trade tape')}** "
            f"({big.get('windows','?')} windows from {big.get('tokens','?')} tokens, "
            f"{big.get('bars','?')} hourly bars). Real order-flow imbalance from each "
            f"trade's aggressor side. Out-of-time AUC {json.dumps(bb)}; walk-forward "
            f"mean {big.get('walk_forward',{}).get('mean_auc')}. Top features: "
            f"{[f for f,_ in big.get('feature_importance_gbdt',[])[:6]]}.\n"
        )
    return f"""---
license: apache-2.0
tags:
- time-series-forecasting
- tabular-classification
- polymarket
- prediction-markets
- trading-signals
- mlx
language:
- en
- multilingual
---

# Polymarket short-horizon direction signals

Small, **honest** models that score a prediction market's short-horizon price
direction from microstructure. Trained on real Polymarket data, validated
strictly **out-of-time** (temporal split + walk-forward), and benchmarked against
a distilled time-series foundation model. Methodology, sources and ablations:
see `RESEARCH.md` in this repo.

## Models

**`seq_model.safetensors`** — the served model. An MLX feature-MLP / GRU over 13
close-price features (incl. proper Wilder RSI, CCI, MACD-hist, stochastic %K).
Winner **{seq.get('winner','?')}**, out-of-time **val AUC {seq_auc}**; walk-forward
mean AUC **{wf.get('mean_auc')}** (min {wf.get('min_auc')}) across 4 time folds.

**`ohlcv_model.safetensors`** — richer model on true OHLCV (open/high/low/close/
volume/trade_count) with 20 features (typical-price CCI, stochastic/Williams on
true high-low, ATR, Parkinson vol, order-flow imbalance). Head-to-head:
`{json.dumps(ob)}` — a **LightGBM** GBDT beats the small MLP/GRU, and the 3-way
ensemble is best (matching published tabular-vs-deep evidence).
{big_line}
## H100 experiments

- **Distillation** (`metrics/distill_metrics.json`): Chronos-Bolt-Base teacher →
  tiny student, with a no-KD ablation. KD gain **{distill.get('kd_auc_gain')}**
  (neutral) — Chronos zero-shot is near-chance on this task, so it can't transfer.
- **Multilingual** (`metrics/multilingual_metrics.json`): multilingual question
  embeddings; cross-lingual cosine to EN mean **{ml.get('mean_cross_lingual_cosine')}**
  (6 languages). Text-only AUC {ml.get('text_only_auc')} but redundant with price
  for short-horizon direction (adds {ml.get('text_adds_auc')}).

## Honesty notes

Metrics are out-of-time, not on an inflated split. `up_rate_spread` (top vs
bottom quintile hit-rate) is the trade-relevant headline. Chronos/foundation
models are baselines here, not the primary signal. Not financial advice.

Inference: load the safetensors with the matching `*_normalizer.json` (feature
order + z-score stats). `code/mcp_server.py` serves the close-only model live
over the Model Context Protocol.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="shubhxho/polymarket-signal-model")
    args = ap.parse_args()

    api = HfApi()
    who = api.whoami()
    print(f"authed as: {who.get('name')}")
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)

    api.upload_file(path_or_fileobj=_card(args.repo).encode(),
                    path_in_repo="README.md", repo_id=args.repo, repo_type="model")
    n = 1
    for local, remote in ARTIFACTS:
        p = os.path.join(HERE, local)
        if not os.path.exists(p):
            print(f"  skip (missing): {local}")
            continue
        api.upload_file(path_or_fileobj=p, path_in_repo=remote, repo_id=args.repo, repo_type="model")
        n += 1
        print(f"  uploaded {remote}")
    print(f"\npushed {n} files → https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
