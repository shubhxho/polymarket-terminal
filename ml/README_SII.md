# SII model suite — resolution, order-flow, smart-money, and the stack on top

A second, larger model family than the one in [`README.md`](README.md). Where that
suite scores **short-horizon price direction** from public OHLCV, this one is built
on **[`SII-WANGZJ/Polymarket_data`](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data)**
— the first public Polymarket dump that carries market linkage, a **true taker
aggressor direction**, market **resolution labels**, and a **user-level tape**.
Those four things unlock three problems the old data could not pose (resolution,
true order-flow, smart-money) and a set of advanced models that compose them.

The methodology and the dataset ranking that motivated all of this live in
[`RESEARCH_SII.md`](RESEARCH_SII.md). This file is the map: what each file is, how
to reproduce locally, where the models live, and how to serve them over MCP.

## The dataset in one table

`SII-WANGZJ/Polymarket_data` — **MIT license**, **163 GB, 1.9 B records, 538,587
markets**, CLOB history **2022-11-21 → 2026-03-04** (the `OrderFilled` era; no
pre-Nov-2022 FPMM/AMM trades).

| file                  | size  | rows    | what it gives us                                                                     |
| --------------------- | ----- | ------- | ------------------------------------------------------------------------------------ |
| `trades.parquet`      | 28 GB | 418.3 M | processed trades, market-linked, `taker_direction` (true aggressor) + `nonusdc_side` |
| `orderfilled.parquet` | 84 GB | 689 M   | raw `OrderFilled` fills — the finest microstructure view                             |
| `markets.parquet`     | 85 MB | 538,587 | market metadata incl. `outcome_prices` (**resolution labels**)                       |
| `quant.parquet`       | 28 GB | 418.2 M | YES-normalized trade series                                                          |
| `users.parquet`       | 23 GB | 340.6 M | per-user maker/taker split, signed `token_amount` (**who** traded)                   |

## Architecture at a glance

```
                 SII-WANGZJ/Polymarket_data  (trades / orderfilled / markets / users)
                                    │
              ┌─────────────────────┼──────────────────────────────┐
              │                     │                               │
   ┌── loaders ──┐        ┌── six feature families ──┐     ┌── base trainers (H100) ──┐
   fetch_sii.py           features_flow   ─ true flow       modal_resolve.py  → resolution
                          features_resolve ─ snapshot        modal_flow.py     → order-flow
                          features_smart  ─ user tape        modal_smart.py    → smart-money
                          features_crossmarket ─ neg-risk
                          features_event  ─ co-movement
                          features_micro  ─ book-free micro
                                    │
                          features_all.py  (66 unified, namespaced features)
                                    │
        ┌───────────── advanced models ─────────────┐
        modal_ensemble.py + ensemble.py  — meta-stacker over the 3 base models
        modal_mega.py                    — one trainer over all six families
        modal_transformer.py             — attention over the raw trade tape
                                    │
        ┌───────────── tooling / serving ───────────┐
        signal_engine.py   — best-signal composer (blend → edge → BUY/HOLD)
        backtest.py        — realized-PnL / Sharpe / calibration harness
        evaluate_all.py    — leaderboard: every model on one holdout
        train_resolve.py   — local MLX resolution trainer (laptop-sized)
        mcp_signals_pro.py — MCP server `pmt-signals-pro` (resolution)
        mcp_ensemble.py    — MCP server `pmt-ensemble` (fused best-signal)
        push_sii.py        — upload the suite to the HF Hub
```

## File → role

### Loader

| file           | role                                                                                                                                                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fetch_sii.py` | Stream `trades` + `markets` from HF into `ml/data/`: `sii_series.json` (time-bucketed OHLCV + flow per token) and `sii_resolve.json` (labeled resolution snapshots). `--sample` for a fast ~2000-row pull used by tests. |

### Feature families (6) → unified builder

| file                      | family                                                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `features_flow.py`        | true-aggressor order flow (bucketed `taker_direction`): imbalance, VPIN-style toxicity, buy-pressure                                             |
| `features_resolve.py`     | resolution snapshot: mid-life price / time-remaining / momentum / flow → terminal YES/NO                                                         |
| `features_smart.py`       | smart-money / behavioral features from the `users.parquet` tape (who is on each side)                                                            |
| `features_crossmarket.py` | neg-risk basket / cross-market arb — sibling legs of one event should sum to ~1                                                                  |
| `features_event.py`       | event-level relative-value / co-movement — a market's move _relative to its peers_                                                               |
| `features_micro.py`       | book-free microstructure from raw `orderfilled` fills (effective spread, Kyle's λ, toxicity)                                                     |
| `features_all.py`         | composes all six into **66** namespaced features (`flow.vpin`, `micro.kyle_lambda`, …) with per-family spans; the seam the mega trainer consumes |

### Base models (Modal H100, one specialist each)

| file               | model                                                                                                | HF repo                                |
| ------------------ | ---------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `modal_resolve.py` | calibrated **resolution probability** — P(market resolves YES) from mid-market state (flagship base) | `shubhxho/polymarket-resolution-model` |
| `modal_flow.py`    | short-horizon **direction** from true order-flow imbalance at 418 M-trade scale                      | `shubhxho/polymarket-flow-model`       |
| `modal_smart.py`   | **smart-money** lean — forward-return signal from the profitable-wallet cohort                       | `shubhxho/polymarket-smartmoney-model` |

### Advanced models

| file                                | model                                                                                                                             | HF repo                                 |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `modal_ensemble.py` + `ensemble.py` | calibrated **meta-stacker** blending the three base models; `ensemble.py` is the pure-stdlib combiner (`blend`) reused everywhere | `shubhxho/polymarket-ensemble-model`    |
| `modal_mega.py`                     | **mega unified trainer** over the union of all six feature families; reports which family earns its place                         | `shubhxho/polymarket-mega-model`        |
| `modal_transformer.py`              | **trade-tape Transformer** — causal self-attention over the raw per-trade tape (multi-task direction head)                        | `shubhxho/polymarket-transformer-model` |

### Tooling

| file               | role                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signal_engine.py` | best-signal composer: `ensemble.blend` → edge vs. live price → BUY_YES / BUY_NO / HOLD gated by `MIN_EDGE`, with confidence fused from backtested reliability    |
| `backtest.py`      | signal-quality harness — realized PnL, hit-rate, Sharpe, drawdown, calibration/decile, no-lookahead threshold sweep                                              |
| `evaluate_all.py`  | model leaderboard — runs every model's reported metrics _and_ re-reproduces them through the same `backtest.run_backtest` holdout (AUC / Brier / log-loss / ECE) |
| `train_resolve.py` | laptop-sized **local MLX** counterpart to `modal_resolve.py`; out-of-time split by market end time, reuses `train_seq` model + eval                              |

### Serving

| file                 | role                                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `mcp_signals_pro.py` | MCP server **`pmt-signals-pro`** — the resolution model. Tools: `resolution_signal`, `scan_resolution`, `model_info`     |
| `mcp_ensemble.py`    | MCP server **`pmt-ensemble`** — the fused best-signal. Tools: `best_signal`, `scan_best`, `model_info`                   |
| `push_sii.py`        | upload the suite to the Hub (resolution is primary; ships `RESEARCH_SII.md` alongside it). `--dry-run`, `--only <model>` |

## Honest-evaluation stance (shared by every model above)

- **Walk-forward split by market `end_date`** — training markets resolve strictly
  _before_ test markets, so no resolved outcome leaks backward.
- **Calibration over accuracy** — Brier, log-loss and an ECE / reliability table
  are the product for the resolution and ensemble models; a well-calibrated 0.65
  beats an over-confident 0.75. Resolution predictions are isotonic-calibrated
  (PAV) on a held-out fold.
- **Expect modest real out-of-time AUC ~0.6–0.7.** Prediction markets are
  near-efficient; the price already encodes most of the resolution probability.
  Anything printing 0.9+ is leakage, not edge.
- **Smoke AUCs are not edge.** Every `modal_*.py` has a `--smoke` path that runs
  the full pipeline on **synthetic, planted-signal** data with pure stdlib (no
  GPU, no Modal). Those numbers prove the _code path_, not real predictive power.
  Real metrics are **pending the H100 runs** and land in `ml/data/*_metrics.json`
  and on the model cards.

## Reproduce locally

There is no single `run_all.py`; the suite reproduces per step. Everything below
is pure stdlib (or MLX for the local trainer) — no GPU needed. The heavy trainers
run on Modal; their smoke paths run anywhere.

```bash
cd ml
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# 0. unit tests (stdlib features + temporal-split invariants)
python test_ml.py

# 1. pull a fast sample of the SII dataset into ml/data/
python fetch_sii.py --sample

# 2. exercise each base/advanced trainer end-to-end on synthetic data (no GPU)
python modal_resolve.py    --smoke
python modal_flow.py       --smoke
python modal_smart.py      --smoke
python modal_ensemble.py   --smoke
python modal_mega.py       --smoke
python modal_transformer.py --smoke

# 3. tooling selfchecks
python ensemble.py           # combiner selfcheck
python signal_engine.py      # best-signal selfcheck
python backtest.py           # backtester selfcheck
python evaluate_all.py       # leaderboard selfcheck  (--report to read data/*_metrics.json)

# 4. local resolution trainer (MLX; --fixture trains on the bundled synthetic set)
python train_resolve.py --fixture
```

Real training (H100 on Modal, pushes to the Hub):

```bash
modal run ml/modal_resolve.py    --max-markets 6000 --push
modal run ml/modal_flow.py       --push
modal run ml/modal_smart.py      --push
modal run ml/modal_ensemble.py   --push
modal run ml/modal_mega.py       --push
modal run ml/modal_transformer.py --push
python  ml/push_sii.py           # or push any present artifacts to their repos
```

## Model repos (HF Hub)

| model                  | repo                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| resolution (primary)   | [`shubhxho/polymarket-resolution-model`](https://huggingface.co/shubhxho/polymarket-resolution-model)   |
| order-flow direction   | [`shubhxho/polymarket-flow-model`](https://huggingface.co/shubhxho/polymarket-flow-model)               |
| smart-money            | [`shubhxho/polymarket-smartmoney-model`](https://huggingface.co/shubhxho/polymarket-smartmoney-model)   |
| meta-ensemble          | [`shubhxho/polymarket-ensemble-model`](https://huggingface.co/shubhxho/polymarket-ensemble-model)       |
| mega unified           | [`shubhxho/polymarket-mega-model`](https://huggingface.co/shubhxho/polymarket-mega-model)               |
| trade-tape transformer | [`shubhxho/polymarket-transformer-model`](https://huggingface.co/shubhxho/polymarket-transformer-model) |

## Serving over MCP

```bash
# resolution model
claude mcp add pmt-signals-pro -- /path/to/ml/.venv/bin/python /path/to/ml/mcp_signals_pro.py
# fused best-signal
claude mcp add pmt-ensemble    -- /path/to/ml/.venv/bin/python /path/to/ml/mcp_ensemble.py
```

`pmt-signals-pro` serves the calibrated resolution model (`resolution_signal`,
`scan_resolution`, `model_info`); `pmt-ensemble` serves the fused best-signal
(`best_signal`, `scan_best`, `model_info`). Both fetch live price history from
Polymarket's public Gamma + CLOB APIs, build features inline, and return a
probability, its edge versus the live market price, and a confidence read. These
are separate from `pmt-signals` (`mcp_server.py`), which serves the short-horizon
direction model from the OHLCV suite.
