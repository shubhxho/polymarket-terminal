# SII-WANGZJ/Polymarket_data → the #1 dataset, and the model suite it unlocks

`RESEARCH.md` ranked the best *public* datasets for a short-horizon **direction**
model and picked `ImpliedData/prediction-markets` as the best *immediately usable*
one, with `TimeSeventeen/Polymarket-v1` (1.2 B raw OrderFilled) as the tick
upgrade path. This doc re-runs that ranking against a dataset that did not exist
when the first pass ran — **[`SII-WANGZJ/Polymarket_data`](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data)**
— and it is not close: SII is now **#1**, and it is the first public dump that
lets us build **resolution** and **smart-money** models at all, not just another
direction model.

## Why SII beats the current best

| dataset | scale | direction | aggressor | resolution labels | user tape |
|---|---|---|---|---|---|
| **SII-WANGZJ/Polymarket_data** | 163 GB, 1.9 B records, 538,587 markets | ✅ market-linked | ✅ **true `taker_direction`** | ✅ `outcome_prices` in `markets.parquet` | ✅ `users.parquet`, signed `token_amount` |
| TimeSeventeen/Polymarket-v1 | ~1.2 B OrderFilled (~2.64 B rows, 49 GB) | on-chain sign only | on-chain sign, **no market linkage** | ❌ | ❌ |
| ImpliedData/prediction-markets | 272k-row sample (of 404 M fills) | 1 h OHLCV | candle-sign proxy only | ❌ | ❌ |

The two datasets `RESEARCH.md` leaned on are both **microstructure-only**:

- `TimeSeventeen/Polymarket-v1` is 1.2 B raw `OrderFilled` events with an on-chain
  buyer/seller sign, but the events are **not linked to a market/outcome** and
  there is **no resolution label** — you can build order-flow features but you
  cannot say which side *won*, and joining trades to markets is left to you.
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
   activity with a **signed `token_amount`**, so you can measure *who* is on each
   side and follow the informed money.

## Dataset facts (verbatim)

- HF id **`SII-WANGZJ/Polymarket_data`**, **MIT license**, authors Shanghai
  Innovation Institute et al.
- **163 GB, 1.9 B records, 538,587 markets.** CLOB history
  **2022-11-21 → 2026-03-04** — the CLOB `OrderFilled` era only; pre-Nov-2022
  FPMM/AMM trades are excluded.

| file | size | rows | what it is |
|---|---|---|---|
| `trades.parquet` | 28 GB | 418.3 M | processed trades w/ market linkage + `taker_direction` (true aggressor) + `nonusdc_side` |
| `orderfilled.parquet` | 84 GB | 689 M | raw `OrderFilled` events |
| `markets.parquet` | 85 MB | 538,587 | market metadata incl. `outcome_prices` (**resolution**) |
| `quant.parquet` | 28 GB | 418.2 M | YES-normalized trade series |
| `users.parquet` | 23 GB | 340.6 M | user-level maker/taker split, signed `token_amount` |

## The three new models

Where `RESEARCH.md` produced one short-horizon direction model, the SII labels +
tape support a small **suite**. All three are uploaded by `push_sii.py`.

### 1. Resolution model — `shubhxho/polymarket-resolution-model` (primary)

Predicts the probability a market resolves **YES** from its price path and
microstructure, supervised on the real `outcome_prices` labels. This is a genuinely
different target from next-hour direction: the horizon is the market's *lifetime*,
and the metric that matters is **calibration** — a "70%" forecast should resolve
YES ~70% of the time — far more than raw accuracy.

### 2. Order-flow model — `shubhxho/polymarket-flow-model`

Short-horizon direction from **true** order-flow imbalance built on
`taker_direction`, not the candle-sign proxy the OHLCV pipeline used.
`RESEARCH.md` §1 found order-flow imbalance is the strongest microstructure signal
but that its edge **decays in seconds-to-minutes**, so it needs tick-level
aggressor data to exploit — which `trades.parquet` (418.3 M linked trades with a
real aggressor side) is exactly.

### 3. Smart-money model — `shubhxho/polymarket-smart-money-model`

Ranks markets by the net positioning of historically profitable wallets, using
`users.parquet` (signed `token_amount`, maker/taker split). An **information-flow**
signal rather than a price-only one: it asks *who* is buying, not just *that*
buying happened.

## Honest-evaluation stance

Carried over from `RESEARCH.md` and tightened for resolution labels:

- **Walk-forward split by market `end_date`.** Because we now have resolution
  labels, the dominant leakage risk is temporal: a market that resolved must never
  train a fold that predicts an earlier market. Folds are cut by market `end_date`
  (expanding window), so every validation market resolves strictly *after* its
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
  sizing a position cares that the probability is *true*, not that the argmax is
  right.

## Net effect

`RESEARCH.md` moved out-of-time AUC from 0.601 → 0.608 by adding order-flow
features and a GBDT, and noted the biggest remaining lever was **data resolution**
(an hourly bar can't see a signal that decays in minutes). SII is that lever plus
two new problems the old data could not pose at all: it supplies tick-level true
aggressor flow *and* the resolution labels and user tape that turn a single
direction model into a resolution / flow / smart-money suite. That is why it ranks
#1 for this repo.
