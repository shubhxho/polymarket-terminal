# SII-WANGZJ/Polymarket_data → the #1 dataset, and the model suite it unlocks

`RESEARCH.md` ranked the best _public_ datasets for a short-horizon **direction**
model and picked `ImpliedData/prediction-markets` as the best _immediately usable_
one, with `TimeSeventeen/Polymarket-v1` (1.2 B raw OrderFilled) as the tick
upgrade path. This doc re-runs that ranking against a dataset that did not exist
when the first pass ran — **[`SII-WANGZJ/Polymarket_data`](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data)**
— and it is not close: SII is now **#1**, and it is the first public dump that
lets us build **resolution** and **smart-money** models at all, not just another
direction model.

## Why SII beats the current best

| dataset                        | scale                                    | direction          | aggressor                            | resolution labels                        | user tape                                 |
| ------------------------------ | ---------------------------------------- | ------------------ | ------------------------------------ | ---------------------------------------- | ----------------------------------------- |
| **SII-WANGZJ/Polymarket_data** | 163 GB, 1.9 B records, 538,587 markets   | ✅ market-linked   | ✅ **true `taker_direction`**        | ✅ `outcome_prices` in `markets.parquet` | ✅ `users.parquet`, signed `token_amount` |
| TimeSeventeen/Polymarket-v1    | ~1.2 B OrderFilled (~2.64 B rows, 49 GB) | on-chain sign only | on-chain sign, **no market linkage** | ❌                                       | ❌                                        |
| ImpliedData/prediction-markets | 272k-row sample (of 404 M fills)         | 1 h OHLCV          | candle-sign proxy only               | ❌                                       | ❌                                        |

The two datasets `RESEARCH.md` leaned on are both **microstructure-only**:

- `TimeSeventeen/Polymarket-v1` is 1.2 B raw `OrderFilled` events with an on-chain
  buyer/seller sign, but the events are **not linked to a market/outcome** and
  there is **no resolution label** — you can build order-flow features but you
  cannot say which side _won_, and joining trades to markets is left to you.
- `ImpliedData/prediction-markets` is clean 1 h OHLCV (used in the current
  pipeline) but it is a **small sample**, has **no true aggressor** (order-flow is
  a candle-sign proxy), and **no resolution outcome**.

SII adds four things neither has, in one linked schema:

1. **Market linkage** — `trades.parquet` is already joined to markets, so trade
   flow, price and outcome sit in the same key space. No fragile join to build.
2. **True taker aggressor direction** — `taker_direction` is the real aggressor
   side of each processed trade (plus `nonusdc_side`), the ground truth
   `RESEARCH.md` §1 said order-flow imbalance needs and the candle-sign proxy only
   approximates.
3. **Resolution outcomes (labels)** — `markets.parquet` carries `outcome_prices`,
   i.e. how each of 538,587 markets actually resolved. That is a **supervised
   label**, which is what turns "predict the next tick" into "predict which side
   pays out".
4. **A user-level tape** — `users.parquet` is 340.6 M rows of per-user maker/taker
   activity with a **signed `token_amount`**, so you can measure _who_ is on each
   side and follow the informed money.

## Dataset facts (verbatim)

- HF id **`SII-WANGZJ/Polymarket_data`**, **MIT license**, authors Shanghai
  Innovation Institute et al.
- **163 GB, 1.9 B records, 538,587 markets.** CLOB history
  **2022-11-21 → 2026-03-04** — the CLOB `OrderFilled` era only; pre-Nov-2022
  FPMM/AMM trades are excluded.

| file                  | size  | rows    | what it is                                                                               |
| --------------------- | ----- | ------- | ---------------------------------------------------------------------------------------- |
| `trades.parquet`      | 28 GB | 418.3 M | processed trades w/ market linkage + `taker_direction` (true aggressor) + `nonusdc_side` |
| `orderfilled.parquet` | 84 GB | 689 M   | raw `OrderFilled` events                                                                 |
| `markets.parquet`     | 85 MB | 538,587 | market metadata incl. `outcome_prices` (**resolution**)                                  |
| `quant.parquet`       | 28 GB | 418.2 M | YES-normalized trade series                                                              |
| `users.parquet`       | 23 GB | 340.6 M | user-level maker/taker split, signed `token_amount`                                      |

> **Companion doc:** [`README_SII.md`](README_SII.md) is the architecture map and
> quick-start (file→role table, local reproduce path, HF repos, MCP registration).
> This doc is the _why_: the dataset ranking, the model designs, and the
> honest-evaluation stance. The suite has grown well past the three base models
> below — see **[The full suite](#the-full-suite-base--advanced--tooling)**.

## The three base models

Where `RESEARCH.md` produced one short-horizon direction model, the SII labels +
tape support a small **suite**. The three base models below are the foundation;
`modal_ensemble` / `modal_mega` / `modal_transformer` build on top of them (see
[The full suite](#the-full-suite-base--advanced--tooling)). All are uploaded by
`push_sii.py`.

### 1. Resolution model — `shubhxho/polymarket-resolution-model` (primary)

Predicts the probability a market resolves **YES** from its price path and
microstructure, supervised on the real `outcome_prices` labels. This is a genuinely
different target from next-hour direction: the horizon is the market's _lifetime_,
and the metric that matters is **calibration** — a "70%" forecast should resolve
YES ~70% of the time — far more than raw accuracy. Trainers:
`modal_resolve.py` (flagship H100 job — LightGBM + torch MLP blend, **isotonic
PAV calibration** on a held-out fold, out-of-time split by `end_date`) and
`train_resolve.py` (laptop-sized MLX counterpart). Features live in
`features_resolve.py`; the 12-feature snapshot state (price level, dist-from-0.5,
time-to-resolution, recent momentum/vol, signed flow imbalance, volume maturity)
is sampled several times along each market's life.

### 2. Order-flow model — `shubhxho/polymarket-flow-model`

Short-horizon direction from **true** order-flow imbalance built on
`taker_direction`, not the candle-sign proxy the OHLCV pipeline used.
`RESEARCH.md` §1 found order-flow imbalance is the strongest microstructure signal
but that its edge **decays in seconds-to-minutes**, so it needs tick-level
aggressor data to exploit — which `trades.parquet` (418.3 M linked trades with a
real aggressor side) is exactly. Trainer `modal_flow.py` streams the 28 GB parquet
by row-group and sums `usd_amount` signed by the aggressor into imbalance / VPIN /
toxicity / buy-pressure features (`features_flow.py`).

### 3. Smart-money model — `shubhxho/polymarket-smartmoney-model`

Ranks markets by the net positioning of historically profitable wallets, using
`users.parquet` (signed `token_amount`, maker/taker split). An **information-flow**
signal rather than a price-only one: it asks _who_ is buying, not just _that_
buying happened. Trainer `modal_smart.py` builds a per-wallet PnL leaderboard,
tags the top/bottom cohorts, and turns their net flow into a forward-return signal
(`features_smart.py`: `smart_net_flow`, `smart_share`, `herding`, `crowding`,
`whale_concentration`, `smart_minus_dumb`). _(The trainer, ensemble and MCP
servers all target `polymarket-smartmoney-model`; the `push_sii.py` uploader
currently labels the same model `polymarket-smart-money-model` — one hyphen apart,
same artifacts.)_

## The full suite (base + advanced + tooling)

The three base models above are the foundation. On top of them the repo now
builds a **six-family feature space**, three **advanced** models that compose it,
and a **tooling + serving** layer. `README_SII.md` is the file→role map; this
section is the design rationale.

### Six feature families → one 66-feature space

Every base trainer had to look through _one_ lens. `features_all.py` composes all
six into a single **66-feature**, namespaced vector (`flow.vpin`,
`micro.kyle_lambda`, …) with per-family spans, so a model can see every angle at
once. The families:

| module                    | family                    | what it sees that a single-market OHLCV model cannot                                                     |
| ------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| `features_flow.py`        | true-aggressor order flow | genuine imbalance / VPIN / toxicity from `taker_direction`, not a candle-sign echo                       |
| `features_resolve.py`     | resolution snapshot       | mid-life state → _terminal_ YES/NO label                                                                 |
| `features_smart.py`       | smart-money / behavioral  | _who_ trades, from the `users.parquet` tape                                                              |
| `features_crossmarket.py` | neg-risk basket / arb     | sibling legs of one event should sum to ~1; the drift is a signal a per-market model literally can't see |
| `features_event.py`       | event co-movement         | a market's move _relative to its event peers_ (the informative part)                                     |
| `features_micro.py`       | book-free microstructure  | effective spread / Kyle's λ / toxicity derived from 689 M raw `orderfilled` fills, with no order book    |

### Three advanced models

- **Meta-ensemble stacker** — `modal_ensemble.py` (+ `ensemble.py`), repo
  `shubhxho/polymarket-ensemble-model`. Trains a **stacker** whose inputs are the
  three base models' out-of-fold predictions (fixed order `resolution, flow,
smartmoney`), then isotonic-calibrates the blend. `ensemble.py` is the
  pure-stdlib combiner (`blend`, `StackWeights`) reused verbatim by the signal
  engine and the `pmt-ensemble` MCP server — missing base models drop out and the
  weights renormalise over what is present. Each specialist is good at a different
  thing; the stacker learns when to trust which.
- **Mega unified trainer** — `modal_mega.py`, repo
  `shubhxho/polymarket-mega-model`. One H100 model over the **union of all six
  feature families** via a single `unified_features(ctx)` seam, reporting which
  family actually earns its place. The highest-quality signal the suite can
  produce from hand features.
- **Trade-tape Transformer** — `modal_transformer.py`, repo
  `shubhxho/polymarket-transformer-model`. Goes a level deeper than aggregated
  windows: a small **Transformer encoder** with **causal** multi-head attention
  over the raw per-trade tape (48-trade context; per-token channels: YES price,
  signed aggressor USD flow, log size, log dt, side). The causal mask forbids
  lookahead; the final position feeds a multi-task head (next-move direction +
  auxiliary targets).

### Tooling

- **Signal engine** — `signal_engine.py`: the single best-signal entry point for
  the terminal / MCP. Folds the base probabilities via `ensemble.blend`, computes
  the **edge** against the live market price, and emits a `BUY_YES` / `BUY_NO` /
  `HOLD` call gated by `MIN_EDGE`, with a confidence that fuses the ensemble's own
  agreement with the signal's _backtested_ reliability.
- **Signal-quality backtester** — `backtest.py`: scores a signal by **realized
  trading quality** (PnL, hit-rate, Sharpe, drawdown, calibration/decile), not
  just AUC, and picks the edge threshold _without looking ahead_. It is the source
  of truth reused by the engine and the leaderboard.
- **Model leaderboard** — `evaluate_all.py`: runs every model through the **same**
  `backtest.run_backtest` on the **same** holdout, adds AUC / Brier / log-loss /
  ECE, and flags where a model's _reported_ metrics disagree with _reproduced_
  ones — "which model gives the best signal", with evidence rather than vibes.
- **Local MLX resolution trainer** — `train_resolve.py`: the runnable,
  laptop-sized counterpart to `modal_resolve.py`, out-of-time split by whole
  market end time (a market's snapshots share one label, so per-market is the only
  leakage-safe partition).
- **Loader** — `fetch_sii.py`: streams `trades` + `markets` from HF into
  `ml/data/` (`sii_series.json`, `sii_resolve.json`); `--sample` for the fast pull
  the tests use.

### Serving

- **`pmt-signals-pro`** (`mcp_signals_pro.py`) — serves the calibrated resolution
  model. Tools: `resolution_signal`, `scan_resolution`, `model_info`.
- **`pmt-ensemble`** (`mcp_ensemble.py`) — serves the fused best-signal, scoring
  every available base model and folding them via `ensemble.blend`. Tools:
  `best_signal`, `scan_best`, `model_info`.
- Both are separate from `pmt-signals` (`mcp_server.py`, the OHLCV direction
  model), and both fetch live price history from Polymarket's public Gamma + CLOB
  APIs and return a probability, its edge vs. the live market, and a confidence.
- **HF push** — `push_sii.py` ships the suite to the Hub (resolution primary) and
  uploads this doc alongside it for provenance.

### A note on smoke tests vs. real edge

Every `modal_*.py` trainer has a `--smoke` path that runs the _whole_ pipeline —
labels, split, model, calibration, metrics — on **synthetic, planted-signal** data
with pure stdlib (no GPU, no Modal). A high smoke AUC only proves the plumbing is
correct on data where the signal was inserted by hand; it is **not** evidence of
real predictive edge. The tooling selfchecks (`ensemble.py`, `signal_engine.py`,
`backtest.py`, `evaluate_all.py` run as `__main__`) are the same kind of check.
Real out-of-time numbers land in `ml/data/*_metrics.json` and the model cards
after the H100 runs — and per the stance below, they are expected to be modest.

## Honest-evaluation stance

Carried over from `RESEARCH.md`, tightened for resolution labels, and applied
uniformly across the **whole suite** (base, advanced and tooling):

- **Walk-forward split by market `end_date`.** Because we now have resolution
  labels, the dominant leakage risk is temporal: a market that resolved must never
  train a fold that predicts an earlier market. Folds are cut by market `end_date`
  (expanding window), so every validation market resolves strictly _after_ its
  training set. This is the resolution-model analogue of the temporal + walk-forward
  purge already used for the direction models.
- **Report Brier + log-loss + calibration, not just accuracy.** For a resolution
  model, calibration is the product. A well-calibrated 0.65 beats a mis-calibrated,
  over-confident 0.75. We report Brier score, log-loss and a calibration/ECE curve
  alongside AUC, with the base rate stated so no number is read in a vacuum.
- **Expect modest out-of-time AUC ~0.6–0.7.** Prediction markets are near-efficient;
  the price already encodes most of the resolution probability. An honest
  out-of-time AUC lands in the **0.6–0.7** band. Anything printing 0.9+ on this
  task is leakage (usually a feature computed after resolution), not edge — the
  same lesson `RESEARCH.md` learned when Chronos zero-shot came in near-chance.
- **Calibration > raw accuracy** for the resolution model specifically: a trader
  sizing a position cares that the probability is _true_, not that the argmax is
  right.
- **Realized-PnL, not just AUC.** AUC says a signal _ranks_ markets; it does not
  say trading it makes money after fees. `backtest.py` settles each call at the
  realized 0/1 outcome and reports PnL / Sharpe / drawdown with a no-lookahead
  threshold sweep, and `evaluate_all.py` re-reproduces every model on the same
  holdout so "reported" is checked against "reproduced".
- **Smoke ≠ edge.** The `--smoke` paths and `__main__` selfchecks run on
  synthetic planted-signal data to prove the code path only; treat their AUCs as
  plumbing checks, never as claimed performance. **All real benchmark numbers are
  pending the H100 runs** and will be written to `ml/data/*_metrics.json` and the
  model cards.

## Net effect

`RESEARCH.md` moved out-of-time AUC from 0.601 → 0.608 by adding order-flow
features and a GBDT, and noted the biggest remaining lever was **data resolution**
(an hourly bar can't see a signal that decays in minutes). SII is that lever plus
two new problems the old data could not pose at all: it supplies tick-level true
aggressor flow _and_ the resolution labels and user tape that turn a single
direction model into a resolution / flow / smart-money suite. That is why it ranks
#1 for this repo.

And the suite has since grown past those three: a **66-feature** unified space over
six families, a **meta-ensemble stacker**, a **mega** unified trainer, and a
**trade-tape Transformer**, all served over MCP (`pmt-signals-pro`, `pmt-ensemble`)
and reproducible locally via the `--smoke` paths. See [`README_SII.md`](README_SII.md)
for the full map and the quick-start.
