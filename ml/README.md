# Direction classifier (MLX)

A small MLP, trained in [MLX](https://github.com/ml-explore/mlx) on Apple
Silicon, that predicts a market's **short-horizon price direction** from the
same microstructure the terminal already computes — drift, realised vol, a
Bollinger-style band z, short momentum, lag-1 autocorrelation and activity.

It is deliberately an honest model, not a demo that prints 99%. On live
Polymarket data it lands around **val AUC ≈ 0.66** — real, modest predictive
signal well above the 0.5 no-skill line — with train AUC ≈ val AUC, i.e. it
isn't overfit.

## Pipeline

```
fetch_data.py   Gamma (market list) + CLOB (price history) → data/series.json   [stdlib only]
features.py     price series → (features, label) sliding windows                [stdlib only]
train.py        MLX MLP: class-weighted BCE, AdamW + dropout, best-val-AUC       [needs mlx]
                → data/model.safetensors, data/normalizer.json, data/metrics.json
```

## Run

```bash
cd ml
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python fetch_data.py     # ~1–2 min; N_MARKETS / INTERVAL / FIDELITY via env
python train.py          # trains, evals, writes the model + metrics
python features.py       # self-check on a synthetic series
```

## Why it's done properly

- **Market-disjoint split** — every window from one market lands wholly in train
  or wholly in val, so val accuracy can't be inflated by leakage between
  neighbouring windows of the same series.
- **Train-only standardisation** — features are z-scored by train statistics;
  val never informs the scaler.
- **Class-weighted loss** — the up/down base rate is skewed by drift (~66% up),
  so the loss reweights the classes; the model can't win by always predicting
  "up".
- **AUC + balanced accuracy** — the honest metrics for a skewed binary target.
  Raw accuracy and the majority baseline are reported alongside so the edge is
  never overstated.
- **Regularisation** — AdamW weight decay + dropout, with early stopping on the
  best validation AUC snapshot.

The committed `data/metrics.json` is the report from the last real training run.
`data/series.json` and `.venv/` are regenerable and gitignored.
