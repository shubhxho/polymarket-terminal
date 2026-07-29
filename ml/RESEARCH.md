# Deep research → what changed in the pipeline

A 97-agent deep-research pass (fan-out web search → source fetch → 3-vote
adversarial verification → synthesis) on *how to predict short-horizon direction
in prediction markets*. Findings below are the ones that survived verification,
each tagged with confidence and the vote, and mapped to the code change it drove.

## 1. Features — order-flow imbalance is the strongest microstructure signal `[high, 3-0]`

Order-flow / order-book imbalance (OBI) and signed trade-sign pressure add real
predictive value beyond RSI/CCI/MACD — but the edge **decays within seconds to a
few minutes**.

- arXiv 2507.22712v2 — OBI is "the net pressure between buy and sell interest at
  or near the best quotes… strongly associated with short-horizon price
  formation"; introduces a trade-based variant from signed buyer/seller trades.
- Cont, Kukanov & Stoikov (arXiv 1011.6402) — order-flow imbalance has a
  near-linear relationship with short-horizon price change, stronger than trade
  imbalance.
- Caveat (verified): all evidence is equities/futures/crypto, not thin
  prediction-market books — transfer is by analogy.

**→ Done:** `features_ohlcv.py` adds `signed_flow` (volume-weighted order-flow
imbalance, sign from each candle's close-vs-open) and `intrabar_pressure`.
**Honest result:** at the **1-hour** horizon of our best public OHLCV data,
order-flow is *not* the top feature by GBDT gain — `extremeness`, `stoch_k`,
`last`, `vol` dominate. That's consistent with the research caveat: OFI's edge
lives at seconds-minutes; by 1h it has decayed and price-level / vol-regime
effects take over. To exploit OFI properly needs the tick datasets in §3.

## 2. Model class — GBDTs beat small MLP/GRU on tabular market data `[high, 3-0]`

Gradient-boosted trees (XGBoost/LightGBM/CatBoost) match or beat small MLPs/GRUs
and bespoke deep-tabular nets — including on the deep models' *own* benchmark
datasets — and win especially on the skewed, heavy-tailed feature distributions
typical of markets.

- Shwartz-Ziv & Armon, *Information Fusion* 2022 (arXiv 2106.03253): "XGBoost
  outperforms these deep models across the datasets, including the datasets used
  in the papers that proposed the deep models."
- McElfresh et al., NeurIPS 2023 (arXiv 2305.02997): across 19 algos × 176
  datasets, "GBDTs excel when datasets contain skewed or heavy-tailed feature
  distributions"; light GBDT tuning often matters more than model class.
- Grinsztajn et al., NeurIPS 2022 (arXiv 2207.08815).

**→ Done:** added **LightGBM** to `train_ohlcv.py` head-to-head. Result on the
OHLCV set (out-of-time val): **GBDT AUC 0.6026 > feature_mlp 0.5993 > seq_gru
0.5961**, and the **3-model ensemble is best at 0.6082** (up-rate spread +0.277,
best Brier 0.2314). GBDT is now the recommended primary; the GRU stays as an
ensemble member — exactly what the evidence prescribes.

## 3. Time-series foundation models are a baseline, not the primary `[medium, 2-1]`

Chronos/TimesFM/Moirai were **not** shown to beat strong baselines. One MDPI
benchmark ranked the Chronos family top *internally* (Chronos-Bolt-Base/Large >
Tiny > TimesFM), but the "FMs beat baselines" claim was **refuted 0-3**.

**→ Confirms our own result:** the H100 distillation ablation
(`modal_distill.py`) found Chronos-Bolt zero-shot near-chance on this task
(AUC 0.527) and KD **neutral** (+0.003). Independent evidence says: keep Chronos
as a zero-shot baseline / candidate feature, not the core model. We do.

## 4. Validation — calibration + risk metrics, not accuracy alone `[high, 3-0]`

Pair probabilistic scores (CRPS, quantile/pinball loss) and calibration with
backtest/PnL diagnostics, under rolling-origin (walk-forward) evaluation.

- ACM 2025 TSFM-in-finance protocol (10.1145/3785706.3785728); Gneiting &
  Raftery 2007 (CRPS); Kupiec/Christoffersen VaR/ES backtesting.

**→ Done:** every model now reports **Brier score** (calibration) alongside AUC
and the up-rate-spread backtest, all under the existing temporal + walk-forward
split. *Evidence gap (honest):* no verified source in this corpus directly
benchmarked López de Prado **purging/embargo** or **triple-barrier** labeling for
prediction markets — so those remain standard-practice recommendations, not
things this pass proved. The temporal split already purges HORIZON windows; a
full embargo + triple-barrier is the next step, not a claimed win.

## 5. Best public datasets (verified)

Ranked by usefulness for a microstructure direction model:

| dataset | size | what it adds | access |
|---|---|---|---|
| **TimeSeventeen/Polymarket-v1** (HF) | ~1.2B OrderFilled trades (~2.64B rows, 49 GB) | **ground-truth on-chain aggressor direction** → real trade sign for true OFI | HF parquet |
| **ImpliedData/prediction-markets** (HF) | 272k rows sample (of 404M fills) | 1h **OHLCV** + volume + trade_count, Polymarket + Manifold, polarity-aligned | HF parquet — **used here** |
| BrockMisner / Mindbyte-89 crypto up/down (HF) | ~27M rows | 5m/15m crypto markets with **L1 order-book + tick fills** | HF parquet |
| Mithilss/polymarket_minute_parquet (HF) | 192M rows | minute trade prints with **side** (buy/sell) → OFI | HF parquet |
| thomaswmitch/kalshi-prediction-markets-betting (HF) | 1–10M | Kalshi trade tape (microstructure/price-impact) | HF parquet |
| Oddpool (institutional) | full tape | millisecond orderbook deltas + trades, Kalshi + Polymarket, one schema | commercial |

**→ Done:** `fetch_hf.py` pulls the ImpliedData OHLCV set (best *immediately
usable* one — true OHLCV, small, permissive). The tick sets (TimeSeventeen with
aggressor direction; the 5m/15m order-book sets) are the upgrade path to exploit
§1's order-flow edge at the horizon where it actually lives.

---

### Net effect on the model

Best out-of-time AUC moved from 0.601 (NN-only OHLCV) to **0.608** (GBDT +
ensemble), with calibration now reported, order-flow features in, and every claim
above traceable to a verified source. The biggest remaining lever is data
resolution: an hourly bar can't see a signal that decays in minutes.

## 6. Multi-lingual signal channel (H100) — `modal_multilingual.py`

Polymarket is global; questions arrive in many languages. This job embeds each
market's question with a multilingual sentence transformer
(`paraphrase-multilingual-MiniLM-L12-v2`, 50+ languages) and asks — honestly —
whether that text channel adds signal.

- **Cross-lingual invariance (the multi-lingual proof):** the same election
  question in EN/ES/ZH/AR/HI/PT embeds to nearly the same vector — mean cosine to
  English **0.892** (zh/ar 0.93, pt 0.92, hi 0.89, es 0.79). The channel is
  genuinely language-agnostic.
- **Ablation (text vs price vs fusion), out-of-time on 596 markets / 93k
  windows:** text-only AUC **0.610** — the question *alone* beats chance (it
  encodes category / favorite-longshot base rates). But price-only is **0.636**
  and fusion **0.634**, so **text adds −0.002**: for *short-horizon direction*
  the price features already contain what the text knows. Kept as an honest
  negative — the multilingual model works, the text signal is real standalone but
  redundant with price here. It would matter more for longer-horizon / resolution
  prediction than next-hour direction.

## 7. The ~100kB Polymarket dataset

The original committed dataset was ~100 kB of Polymarket price series;
`data/series_100kb.json` reconstructs that size (103,654 bytes, 26 markets,
14k points). Trained for parity: feature-MLP AUC **0.639**, GRU **0.642**,
up-rate spread **+0.28–0.30** (`data/series_100kb_metrics.json`) — the signal is
already there at 100 kB; the 14× larger fetch mainly stabilises the walk-forward.
