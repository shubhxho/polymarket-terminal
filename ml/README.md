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
fetch_data.py   Gamma (paged market list) + CLOB (price history) → data/series.json   [stdlib only]
features.py     price series → sliding windows: hand features + raw return sequence   [stdlib only]
train.py        MLX MLP over the hand features (baseline)                              [needs mlx]
train_seq.py    MLX GRU over the return sequence, fused with the features (best model) [needs mlx]
                → data/seq_model.safetensors, data/seq_metrics.json
```

## Two models, and why the GRU

`train.py`'s MLP sees only *summary statistics* of each window — it throws the
order away. `train_seq.py` adds an **MLX GRU** that reads the return sequence
step by step and fuses its final state with the same hand features, which is the
natural way to pull more signal from the same data. It trains both and keeps the
one with the higher validation AUC.

### Signal quality, not just accuracy

The headline metric a trader cares about isn't accuracy — it's whether ranking
markets by the model's score *separates the ones that go up from the ones that
go down*. `train_seq.py` runs a **decile backtest**: take the top and bottom 20%
by model score and compare how they actually moved next. `up_rate_spread` (the
gap in the fraction that rose) is the robust, outlier-proof measure of edge.
`data/seq_metrics.json` holds the last real run's numbers for both models.

## On "HuggingFace MLX"

MLX is the Apple-silicon runtime from `ml-explore`, and it pulls weights and
quantised models straight from the HuggingFace Hub — that *is* the HF + MLX
stack. `mlx-lm` (LoRA-fine-tuning `mlx-community/*` LLMs) is the right tool for
**text** tasks; this is a **tabular / sequence** signal-prediction problem, so
the right, honest choice is a purpose-built core-MLX model, not an LLM bent to
fit.

## Run

```bash
cd ml
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

N_MARKETS=300 INTERVAL=1m python fetch_data.py   # paged fetch; a few minutes
python train_seq.py      # trains GRU + MLP, picks best by val AUC, backtests
python train.py          # the feature-MLP on its own (baseline)
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
