# Polymarket signal model (MLX + Chronos distillation)

A small model — an [MLX](https://github.com/ml-explore/mlx) GRU on Apple silicon
— that scores a market's **short-horizon price direction** from the same
microstructure the terminal computes, now widened with the classic oscillators
traders actually watch. Served live over MCP; verified **out-of-time**, not on
an inflated split; and benchmarked against a distilled foundation model on an
H100.

## Features (13)

The window features mirror the terminal's quant lib and add the standard
technical oscillators:

| group       | features                                                                                                                                                                           |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| path        | `last`, `mean_ret`, `vol`, `drift`, `band_z`, `momentum`, `autocorr`, `activity`                                                                                                   |
| oscillators | `rsi` (Wilder, recentred to ±1), `cci` (Commodity Channel Index / 100), `macd_hist` (MACD 3/8 − 4-EMA signal), `stoch_k` (%K in 0..1), `downside_vol` (std of negative increments) |

All are pure stdlib in `features.py`, finite on degenerate/flat windows, and
covered by `test_ml.py`.

## Pipeline

```
fetch_data.py   Gamma (paged list) + CLOB (history) → data/series.json        [stdlib]
features.py     price series → windows: 13 features + raw return sequence      [stdlib]
train_seq.py    MLX GRU + feature-MLP, ensemble, temporal walk-forward         [mlx]
                → data/seq_model.safetensors, data/seq_metrics.json, seq_normalizer.json
mcp_server.py   serves the trained model over MCP (signals + oscillator read)  [mlx]
modal_distill.py  Chronos-Bolt teacher → distilled student on H100 (+ ablation) [modal]

# Best-dataset / GBDT path (see RESEARCH.md — driven by a verified deep-research pass)
fetch_hf.py     ImpliedData real 1h OHLCV (Polymarket+Manifold) → data/ohlcv.json   [hf_hub, pyarrow]
features_ohlcv.py  20 features: proper CCI/stoch/ATR + order-flow imbalance         [stdlib]
train_ohlcv.py  MLP + GRU + LightGBM + ensemble, Brier calibration, walk-forward    [mlx, lightgbm]
                → data/ohlcv_model.safetensors, data/ohlcv_metrics.json
```

**Deep research → concrete changes** are documented in
[`RESEARCH.md`](RESEARCH.md): order-flow imbalance features (verified strongest
short-horizon signal), a **LightGBM** model that beats the small MLP/GRU on
tabular market data (GBDT AUC 0.603 > MLP 0.599 > GRU 0.596; **ensemble 0.608**),
Brier-score calibration, and the ranked list of best public datasets
(`TimeSeventeen/Polymarket-v1`, 1.2B trades with on-chain aggressor direction, is
the upgrade path for true order-flow).

## Verified over time, not over-fit

The honest question isn't val accuracy on a random split — it's whether a signal
**survives into the future**. So validation is strictly **out-of-time**:

- **Temporal split** — each market's earlier windows train, its _later_ windows
  validate, with a `HORIZON` purge between so no forward label peeks across the
  boundary (`_split` in `train_seq.py`).
- **Walk-forward** — each series is cut into time blocks; fold _k_ trains on
  blocks `0..k` and validates on `k+1`. Reporting AUC per fold shows whether the
  edge is stable across epochs or a one-slice fluke. See `walk_forward` in
  `data/seq_metrics.json`.

This is a stricter, more honest bar than a market-index split — and it lands the
edge where reality is: a **real but modest** separation, not a demo that prints
99%.

**Last real run** (430 markets, 192k points → 107k train / 27k val windows): the
feature-MLP won at **val AUC 0.650** (GRU 0.647, ensemble 0.650 — all within
noise), and its decile backtest is the story — the top-quintile by score rose
**40.2%** of the time next versus **12.8%** for the bottom, a **+0.27 up-rate
spread** on 27k out-of-time windows. Walk-forward confirms it holds across time
rather than lucking into one slice:

| fold (train → val, later in time) | val AUC | up-rate spread |
| --------------------------------- | ------- | -------------- |
| 1                                 | 0.603   | +0.220         |
| 2                                 | 0.621   | +0.228         |
| 3                                 | 0.640   | +0.268         |
| 4                                 | 0.649   | +0.270         |

Mean walk-forward AUC **0.628**, min **0.603** — the signal persists (and even
strengthens) as the training window expands. `up_rate_spread` is the outlier-proof
headline; full numbers in `data/seq_metrics.json`.

## Distilling Chronos-Bolt on an H100

`amazon/chronos-bolt-base` (205M, T5; top of the fev-bench / GIFT-Eval board) is
the teacher. `modal_distill.py` reads a **soft up-probability** for every window
straight from where the current price sits in Chronos's predictive quantile
distribution, then trains a tiny GRU student to match the soft target _and_ the
realised outcome. Feature extraction is imported from `features.py` (Modal mounts
it), so the student sees the exact same 13 features.

```bash
modal run ml/modal_distill.py --n-markets 1000 --epochs 60 --alpha 0.5
modal run ml/modal_distill.py --push          # also push the student to the HF Hub
```

To prove the distillation earns its keep, the job trains the same student
**twice** — with and without the KD term — and reports both plus the gap
(`kd_auc_gain`, `kd_acc_gain`).

**Last H100 run** (840 series, 40k windows, strict out-of-time val): the Chronos
zero-shot teacher is near-chance on this specific short-horizon direction task
(**AUC 0.527**), so it has little to transfer — the ablation honestly shows KD is
**neutral** here (`kd_auc_gain ≈ +0.003`). The lesson is real, not a failure: a
foundation model that tops general forecasting boards does _not_ automatically
beat purpose-built microstructure features on Polymarket's noisy, short-horizon
prints. The hand features carry the signal. Full numbers in
`data/distill_metrics.json`.

### H100 cost, estimated before spending

| stage                               | wall       | USD @ $3.95/hr |
| ----------------------------------- | ---------- | -------------- |
| chronos zero-shot (800 mkts)        | ~4 min     | $0.26          |
| chronos fine-tune 1000 steps        | ~22 min    | $1.45          |
| distillation (teacher + 2 students) | ~30 min    | $1.98          |
| **full pipeline**                   | **< 2 hr** | **~$6.65**     |

## Serving over MCP

`mcp_server.py` loads the trained MLX model and serves it over the Model Context
Protocol, so Claude Code / the terminal / an agent can pull live signals:

```
market_signal(token_id)   up-probability, direction, conviction, oscillator read
scan_signals(limit)       top markets ranked by model conviction
model_info()              which model is loaded + walk-forward stability
```

`market_signal` now returns an `oscillators` block (RSI/CCI states, %K, MACD
histogram) so a caller sees _why_, and `model_info` surfaces the walk-forward
AUCs so the persistence of the signal is inspectable, not just asserted.

Register:

```bash
claude mcp add pmt-signals -- /path/to/ml/.venv/bin/python /path/to/ml/mcp_server.py
```

## Run

```bash
cd ml
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

N_MARKETS=450 INTERVAL=max python fetch_data.py   # paged fetch; a few minutes
python features.py       # self-check on a synthetic series
python test_ml.py        # 8 stdlib tests (features + temporal split)
python train_seq.py      # GRU + MLP + ensemble + walk-forward, picks best by val AUC
```

## Why it's done properly

- **Out-of-time validation** — temporal split + walk-forward with a purge, so the
  reported edge is what survives into the future, not leakage between neighbouring
  windows.
- **Train-only standardisation** — features z-scored by train statistics only.
- **Class-weighted loss** — the up/down base rate is drift-skewed (~70% up on
  this set), so the loss reweights classes; the model can't win by always
  predicting "up".
- **AUC + up-rate spread** — the honest metrics for a skewed binary target;
  majority baseline reported alongside so the edge is never overstated.
- **Adversarial ablation on distillation** — KD is only claimed to help when the
  with/without comparison says so. Here it doesn't, and the README says so.

The small `data/*_metrics.json` + trained weights are committed as the report of
the last real run; `data/series.json` (regenerable) and `.venv/` are gitignored.
