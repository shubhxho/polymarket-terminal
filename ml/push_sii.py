"""Push the NEW SII model suite (resolution / order-flow / smart-money) to the Hub.

Separate from `push_hf.py` (which serves the short-horizon *direction* models).
This uploader ships the models trained on **`SII-WANGZJ/Polymarket_data`** — the
first public Polymarket dump that carries market linkage, a TRUE taker aggressor
direction, market RESOLUTION outcomes (labels) and a user-level tape. See
`RESEARCH_SII.md` for the methodology and the dataset ranking.

Each model group targets its own repo (resolution is the primary). Artifacts are
produced by the other units / real training runs, so anything missing is skipped
gracefully. Uses the cached HF login (`huggingface-cli login`) — no token is
printed or stored.

    python ml/push_sii.py --dry-run      # list intended uploads + card preview, no network
    python ml/push_sii.py                # push every present artifact to its repo
    python ml/push_sii.py --only resolution
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")

PRIMARY_REPO = "shubhxho/polymarket-resolution-model"

DATASET = "SII-WANGZJ/Polymarket_data"

# The three new models trained on the SII dataset. Each group is uploaded to its
# own repo; `artifacts` are (local path relative to ml/, path in the repo).
MODELS = {
    "resolution": {
        "repo": PRIMARY_REPO,
        "title": "Polymarket resolution model (which side pays out)",
        "tag": "resolution-prediction",
        "blurb": (
            "Scores the probability a market resolves **YES** from its price path "
            "and microstructure, using the real `outcome_prices` labels in "
            "`markets.parquet`. A resolution model lives or dies on **calibration**, "
            "not raw accuracy — a 0.70 forecast should hit ~70% of the time."
        ),
        "metrics": "resolve_metrics.json",
        "artifacts": [
            ("data/resolve_model.safetensors", "resolve_model.safetensors"),
            ("data/resolve_normalizer.json", "resolve_normalizer.json"),
            ("data/resolve_metrics.json", "metrics/resolve_metrics.json"),
        ],
    },
    "order-flow": {
        "repo": "shubhxho/polymarket-flow-model",
        "title": "Polymarket order-flow model (true taker aggressor)",
        "tag": "time-series-forecasting",
        "blurb": (
            "Short-horizon direction from **true** order-flow imbalance — the "
            "`taker_direction` aggressor side of every processed trade in "
            "`trades.parquet` (418.3M trades), not a candle-sign proxy. This is the "
            "signal §1 of `RESEARCH.md` said needs tick-level aggressor data to exploit."
        ),
        "metrics": "flow_metrics.json",
        "artifacts": [
            ("data/flow_model.safetensors", "flow_model.safetensors"),
            ("data/flow_normalizer.json", "flow_normalizer.json"),
            ("data/flow_metrics.json", "metrics/flow_metrics.json"),
        ],
    },
    "smart-money": {
        "repo": "shubhxho/polymarket-smart-money-model",
        "title": "Polymarket smart-money model (user-level tape)",
        "tag": "tabular-classification",
        "blurb": (
            "Follows the informed traders. Built on `users.parquet` (340.6M "
            "user-level maker/taker rows with signed `token_amount`): ranks markets "
            "by the net positioning of historically profitable wallets. An "
            "information-flow signal, not a price-only one."
        ),
        "metrics": "smart_metrics.json",
        "artifacts": [
            ("data/smart_model.safetensors", "smart_model.safetensors"),
            ("data/smart_normalizer.json", "smart_normalizer.json"),
            ("data/smart_metrics.json", "metrics/smart_metrics.json"),
        ],
    },
}


def _load(name):
    p = os.path.join(DATA, name)
    try:
        return json.load(open(p)) if os.path.exists(p) else {}
    except (ValueError, OSError):
        return {}


def _fmt(x):
    return "n/a" if x is None else x


def _card(key: str) -> str:
    """Generate a model card describing the SII dataset provenance + honest metrics."""
    m = MODELS[key]
    metrics = _load(m["metrics"])
    wf = metrics.get("walk_forward", {})
    # Honest, calibration-first headline metrics (present only after a real run).
    auc = metrics.get("val_auc", metrics.get("auc"))
    brier = metrics.get("brier")
    logloss = metrics.get("log_loss", metrics.get("logloss"))
    cal = metrics.get("calibration_error", metrics.get("ece"))
    n_markets = metrics.get("n_markets")

    if metrics:
        metric_line = (
            f"- Out-of-time **AUC {_fmt(auc)}** · **Brier {_fmt(brier)}** · "
            f"**log-loss {_fmt(logloss)}** · calibration error {_fmt(cal)}\n"
            f"- Walk-forward mean AUC **{_fmt(wf.get('mean_auc'))}** "
            f"(min {_fmt(wf.get('min_auc'))})"
            + (f" across {len(wf.get('folds', []))} folds" if wf.get("folds") else "")
            + (f"\n- Evaluated on {n_markets} markets" if n_markets else "")
        )
    else:
        metric_line = (
            "- _Metrics pending a real training run — this repo currently ships "
            "the card + code only._"
        )

    files = os.linesep.join(f"- `{r}`" for _, r in m["artifacts"])

    return f"""---
license: mit
tags:
- prediction-markets
- polymarket
- {m['tag']}
- trading-signals
- calibration
datasets:
- {DATASET}
language:
- en
---

# {m['title']}

{m['blurb']}

## Data provenance

Trained on **[`{DATASET}`](https://huggingface.co/datasets/{DATASET})** (MIT,
Shanghai Innovation Institute et al.) — **163 GB, 1.9 B records, 538,587 markets**,
CLOB history **2022-11-21 → 2026-03-04**. Unlike the raw OrderFilled dumps this
set is *processed and linked*: trades carry market linkage and a **true taker
aggressor direction**, `markets.parquet` carries **resolution outcomes**
(`outcome_prices`, the labels), and `users.parquet` gives a **user-level tape**.
This is what makes a resolution / smart-money model possible at all. Full ranking
and methodology: `RESEARCH_SII.md`.

## Honest evaluation

Validation is strictly **out-of-time**, split **walk-forward by market
`end_date`** so no resolved market leaks backward into training. For a resolution
model we report **calibration first** — Brier score, log-loss and calibration
error — because a well-calibrated 0.65 is worth more than an over-confident,
mis-calibrated 0.75. Expect a **modest out-of-time AUC (~0.6–0.7)**; anything
that prints 0.9+ on this task is leakage, not edge.

{metric_line}

## Files

{files}

Load the `*.safetensors` weights with the matching `*_normalizer.json` (feature
order + z-score stats). Not financial advice.
"""


def _upload_group(key: str, dry_run: bool) -> int:
    m = MODELS[key]
    repo = m["repo"]
    present = [(l, r) for l, r in m["artifacts"] if os.path.exists(os.path.join(HERE, l))]

    print(f"\n[{key}] → https://huggingface.co/{repo}")
    if dry_run:
        print("  up    README.md                              (generated model card)")
        for local, remote in m["artifacts"]:
            exists = os.path.exists(os.path.join(HERE, local))
            print(f"  {'up  ' if exists else 'skip'}  {local:38s} → {remote}"
                  + ("" if exists else "  (missing)"))
        print("  --- model-card preview ---")
        preview = _card(key).splitlines()
        for line in preview[:24]:
            print(f"  | {line}")
        if len(preview) > 24:
            print(f"  | … ({len(preview) - 24} more lines)")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=_card(key).encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="model")
    n = 1
    for local, remote in present:
        api.upload_file(path_or_fileobj=os.path.join(HERE, local), path_in_repo=remote,
                        repo_id=repo, repo_type="model")
        n += 1
        print(f"  uploaded {remote}")
    for local, remote in m["artifacts"]:
        if (local, remote) not in present:
            print(f"  skip (missing): {local}")
    # Ship the research doc alongside the primary repo for provenance.
    if repo == PRIMARY_REPO:
        research = os.path.join(HERE, "RESEARCH_SII.md")
        if os.path.exists(research):
            api.upload_file(path_or_fileobj=research, path_in_repo="RESEARCH_SII.md",
                            repo_id=repo, repo_type="model")
            n += 1
            print("  uploaded RESEARCH_SII.md")
    print(f"  pushed {n} files")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="list intended uploads + card preview WITHOUT any network call")
    ap.add_argument("--only", choices=sorted(MODELS), default=None,
                    help="push a single model group instead of all three")
    args = ap.parse_args()

    keys = [args.only] if args.only else list(MODELS)

    if not args.dry_run:
        from huggingface_hub import HfApi

        who = HfApi().whoami()
        print(f"authed as: {who.get('name')}")

    total = 0
    for key in keys:
        total += _upload_group(key, args.dry_run)

    if args.dry_run:
        print("\n[dry-run] no network calls made.")
    else:
        print(f"\ndone — {total} files pushed across {len(keys)} repo(s).")


if __name__ == "__main__":
    main()
