/**
 * The trained model, in the browser.
 *
 * `ml/train_seq.py` fits a small MLP over thirteen hand features of a market's
 * recent probability path and keeps whichever architecture won out of sample —
 * here that is the feature MLP, at ~0.65 walk-forward AUC. That is a weak edge,
 * not an oracle, which is exactly why it is wired in as a *nudge* to the rule
 * engine rather than a replacement for it (see `blendedScore` in `signals.ts`).
 *
 * The Python inference path is MLX and only runs on Apple silicon, so instead of
 * shipping a service the terminal has to reach at request time, the winning
 * network is tiny (13→32→32→1) and its weights are frozen into `mlModel.json`.
 * The forward pass and the feature block below are a line-for-line port of
 * `features.py`; the two must stay in lockstep or the frozen weights see inputs
 * they were never normalised against. Everything here is pure and allocation-cheap
 * so it can run per-market inside the scan without a second network round-trip.
 */

import type { Direction } from "./signals";
import type { PricePoint } from "./types";
import model from "./mlModel.json";

/** Look-back the model was trained on. Must equal `WINDOW` in `features.py`. */
export const MODEL_WINDOW = model.window;

/** The model's own out-of-sample AUC, carried through so the blend can weight it. */
export const MODEL_AUC = model.val_auc;

/**
 * Temperature that calibrates the raw logits (fit by `ml/calibrate.py` on the
 * strictly out-of-time validation split; see `model.calibration` for the ECE
 * it bought). Applied as `logit / T` before the sigmoid. Because the shift is
 * zero, the 0.5 decision boundary — and therefore the direction and the AUC —
 * is untouched; only the confidence is corrected. Defaults to 1 (a no-op) for
 * an un-calibrated export.
 */
export const MODEL_TEMPERATURE: number = (model as { temperature?: number }).temperature ?? 1;

/** Reliability of the calibrated probabilities, straight from validation. */
export const MODEL_CALIBRATION = (model as { calibration?: Record<string, number> }).calibration;

export type ModelRead = {
  /** Probability that YES drifts up over the next few hours, 0..1. */
  readonly prob: number;
  /** Sign of `prob` around the coin-flip line. */
  readonly direction: Direction;
  /** |prob − 0.5| · 2, in 0..1 — how far off the fence the model is willing to go. */
  readonly conviction: number;
  /** Walk-forward AUC of the network that produced this read. */
  readonly auc: number;
};

// ── Feature block (port of features.py) ──────────────────────────────────────

function mean(xs: readonly number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

/** Sample standard deviation (n−1), matching `_std`. */
function std(xs: readonly number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1));
}

function autocorr(xs: readonly number[]): number {
  if (xs.length < 3) return 0;
  const m = mean(xs);
  let num = 0;
  let den = 0;
  for (let i = 0; i < xs.length; i++) {
    den += (xs[i] - m) ** 2;
    if (i >= 1) num += (xs[i] - m) * (xs[i - 1] - m);
  }
  return den > 1e-12 ? Math.max(-1, Math.min(1, num / den)) : 0;
}

/** Wilder RSI over the increments, recentred to −1..1 — `(RSI−50)/50`. */
function rsi(rets: readonly number[]): number {
  let gains = 0;
  let losses = 0;
  for (const r of rets) {
    if (r > 0) gains += r;
    else if (r < 0) losses -= r;
  }
  const denom = gains + losses;
  return denom < 1e-9 ? 0 : (gains - losses) / denom;
}

function emaSeries(xs: readonly number[], span: number): number[] {
  const a = 2 / (span + 1);
  const out = [xs[0]];
  for (let i = 1; i < xs.length; i++) out.push(a * xs[i] + (1 - a) * out[i - 1]);
  return out;
}

/** CCI / 100, clipped to ±5. Typical price is the probability itself. */
function cci(window: readonly number[]): number {
  const sma = mean(window);
  const mad = mean(window.map((p) => Math.abs(p - sma)));
  if (mad < 1e-9) return 0;
  const raw = (window[window.length - 1] - sma) / (0.015 * mad);
  return Math.max(-5, Math.min(5, raw / 100));
}

/** MACD(3,8) histogram minus its 4-EMA signal. */
function macdHist(window: readonly number[]): number {
  if (window.length < 4) return 0;
  const fast = emaSeries(window, 3);
  const slow = emaSeries(window, 8);
  const macd = fast.map((f, i) => f - slow[i]);
  const signal = emaSeries(macd, 4);
  return macd[macd.length - 1] - signal[signal.length - 1];
}

/** Stochastic %K: where the last price sits in the window's range, 0..1. */
function stochK(window: readonly number[]): number {
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of window) {
    if (p < lo) lo = p;
    if (p > hi) hi = p;
  }
  return hi - lo < 1e-9 ? 0.5 : (window[window.length - 1] - lo) / (hi - lo);
}

/** The thirteen features, in the exact order `FEATURE_NAMES` declares them. */
function windowFeatures(window: readonly number[]): number[] {
  const rets: number[] = [];
  for (let i = 1; i < window.length; i++) rets.push(window[i] - window[i - 1]);
  const last = window[window.length - 1];
  const stdW = std(window);
  const bandZ = stdW > 1e-9 ? (last - mean(window)) / stdW : 0;
  const momentum = rets.length >= 4 ? mean(rets.slice(-4)) : mean(rets);
  const neg = rets.filter((r) => r < 0);
  return [
    last,
    mean(rets),
    std(rets),
    last - window[0],
    bandZ,
    momentum,
    autocorr(rets),
    rets.reduce((a, r) => a + Math.abs(r), 0),
    rsi(rets),
    cci(window),
    macdHist(window),
    stochK(window),
    neg.length >= 2 ? std(neg) : 0,
  ];
}

// ── Forward pass (13 → 32 → 32 → 1, ReLU, dropout off at inference) ───────────

function relu(x: number): number {
  return x > 0 ? x : 0;
}

/** y = W·x + b for a `[out][in]` weight matrix. */
function dense(x: readonly number[], w: number[][], b: number[]): number[] {
  return w.map((row, i) => {
    let acc = b[i];
    for (let j = 0; j < row.length; j++) acc += row[j] * x[j];
    return acc;
  });
}

/**
 * Score the tail of a probability series.
 *
 * Returns `null` — never a made-up 0.5 — when there is not a full window to
 * stand on, so a market with no history simply carries no model read rather than
 * a fake coin flip that the blend would then treat as real signal.
 */
export function modelSignal(history: readonly PricePoint[] | undefined): ModelRead | null {
  if (!history || history.length < MODEL_WINDOW) return null;
  return modelSignalFromPrices(history.map((p) => p.p));
}

/**
 * Same read, straight off a probability array.
 *
 * The live path on the client keeps a rolling window of raw prices (not
 * `PricePoint`s) and appends the socket's last trade to it on every tick, so it
 * wants to score without first re-wrapping each number — this is that entry
 * point, and `modelSignal` is a thin adapter over it. Only the last
 * `MODEL_WINDOW` prices matter; anything earlier is ignored exactly as the
 * trained model ignored it.
 */
export function modelSignalFromPrices(prices: readonly number[]): ModelRead | null {
  if (prices.length < MODEL_WINDOW) return null;
  const window = prices.slice(-MODEL_WINDOW);

  const raw = windowFeatures(window);
  // Standardise against the training distribution before the frozen weights.
  const x = raw.map((v, i) => (v - model.fmean[i]) / (model.fstd[i] || 1));

  const h1 = dense(x, model.l1.w, model.l1.b).map(relu);
  const h2 = dense(h1, model.l2.w, model.l2.b).map(relu);
  const logit = dense(h2, model.out.w, model.out.b)[0];
  // Temperature-calibrated before the sigmoid — see MODEL_TEMPERATURE.
  const prob = 1 / (1 + Math.exp(-logit / MODEL_TEMPERATURE));

  return {
    prob,
    direction: prob > 0.5 ? "bullish" : prob < 0.5 ? "bearish" : "neutral",
    conviction: Math.min(1, Math.abs(prob - 0.5) * 2),
    auc: MODEL_AUC,
  };
}
