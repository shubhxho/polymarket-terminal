/**
 * Statistical primitives for the signal engine.
 *
 * Pure, dependency-free, and deliberately separated from `signals.ts`: the
 * detectors there decide *what is interesting*, these decide *what is true*.
 * Keeping them apart means a threshold can be argued about without anyone
 * re-deriving a variance.
 *
 * Every function here degrades to a defined value on short or degenerate
 * input rather than returning NaN, because a NaN leaking into a score silently
 * poisons a whole ranking.
 */

import type { BookLevel, OrderBook, PricePoint } from "./types";

// ── Descriptive statistics ─────────────────────────────────────────────────

export function mean(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/** Sample standard deviation (n−1). Returns 0 for fewer than two points. */
export function stdev(xs: readonly number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  let acc = 0;
  for (const x of xs) acc += (x - m) ** 2;
  return Math.sqrt(acc / (xs.length - 1));
}

export function median(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/**
 * Median absolute deviation, scaled to be comparable with a standard
 * deviation on normal data.
 *
 * Used instead of stdev wherever the input is a cross-section of markets: a
 * single market doing 200x its normal volume would inflate a standard
 * deviation enough to hide every other outlier behind it.
 */
export function mad(xs: readonly number[]): number {
  if (xs.length < 2) return 0;
  const m = median(xs);
  return 1.4826 * median(xs.map((x) => Math.abs(x - m)));
}

/**
 * Robust z-score of `x` against a population, using median/MAD.
 *
 * Falls back to the classical mean/stdev when the MAD collapses to zero,
 * which happens when more than half the population share one value.
 */
export function robustZ(x: number, population: readonly number[]): number {
  if (population.length < 4) return 0;
  const m = median(population);
  const scale = mad(population);
  if (scale > 1e-9) return (x - m) / scale;
  const sd = stdev(population);
  return sd > 1e-9 ? (x - mean(population)) / sd : 0;
}

/** Fraction of the population at or below `x`, in 0..1. */
export function percentileOf(x: number, population: readonly number[]): number {
  if (population.length === 0) return 0.5;
  let below = 0;
  for (const p of population) if (p <= x) below++;
  return below / population.length;
}

export function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

/** Maps an unbounded non-negative value into 0..1 with diminishing returns. */
export function saturate(value: number, halfPoint: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0;
  return value / (value + halfPoint);
}

/** Logistic squash centred on zero — turns a z-score into a 0..1 weight. */
export function logistic(z: number, steepness = 1): number {
  return 1 / (1 + Math.exp(-steepness * z));
}

// ── Time series ────────────────────────────────────────────────────────────

/**
 * Simple returns between consecutive points.
 *
 * Prices here are probabilities, so a *log* return is the wrong transform: a
 * market moving 2¢→4¢ has doubled in log terms but only moved two points of
 * probability, and treating that as a 69% move would let every longshot
 * dominate the volatility ranking.
 */
export function diffs(points: readonly PricePoint[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < points.length; i++) out.push(points[i].p - points[i - 1].p);
  return out;
}

/**
 * Realised volatility in probability points per day.
 *
 * Scaled by the actual sampling interval so a 1h series and a 1w series are
 * directly comparable — without that, "volatility" would mostly measure which
 * fidelity the caller happened to request.
 */
export function realisedVol(points: readonly PricePoint[]): number {
  if (points.length < 3) return 0;
  const d = diffs(points);
  const sd = stdev(d);
  const spanSeconds =
    (points[points.length - 1].t - points[0].t) / Math.max(1, points.length - 1);
  if (spanSeconds <= 0) return 0;
  const samplesPerDay = 86400 / spanSeconds;
  return sd * Math.sqrt(samplesPerDay) * 100;
}

/** Ordinary-least-squares slope of price against time, in points per day. */
export function driftPerDay(points: readonly PricePoint[]): number {
  if (points.length < 3) return 0;
  const t0 = points[0].t;
  const xs = points.map((p) => (p.t - t0) / 86400);
  const ys = points.map((p) => p.p * 100);
  const mx = mean(xs);
  const my = mean(ys);
  let num = 0;
  let den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  return den > 1e-12 ? num / den : 0;
}

/**
 * Lag-1 autocorrelation of increments.
 *
 * Positive means moves persist (trending); negative means they reverse
 * (choppy). This is what separates a market genuinely repricing from one
 * oscillating inside a range with the same total movement.
 */
export function autocorrelation(points: readonly PricePoint[]): number {
  const d = diffs(points);
  if (d.length < 8) return 0;
  const m = mean(d);
  let num = 0;
  let den = 0;
  for (let i = 0; i < d.length; i++) {
    den += (d[i] - m) ** 2;
    if (i > 0) num += (d[i] - m) * (d[i - 1] - m);
  }
  return den > 1e-12 ? clamp(num / den, -1, 1) : 0;
}

/**
 * Position of the latest price within its own recent band, as a z-score.
 *
 * The classic Bollinger reading: beyond ±2 the price is stretched relative to
 * how it has been trading, which is either a breakout or an overreaction.
 */
export function bandZ(points: readonly PricePoint[], window = 60): number {
  if (points.length < 12) return 0;
  const tail = points.slice(-window).map((p) => p.p);
  const sd = stdev(tail);
  if (sd < 1e-6) return 0;
  return (tail[tail.length - 1] - mean(tail)) / sd;
}

/**
 * Ratio of recent volatility to the earlier baseline.
 *
 * Below ~0.6 the market has gone quiet relative to itself — a coiled range,
 * which historically precedes expansion.
 */
export function volCompression(points: readonly PricePoint[]): number {
  if (points.length < 24) return 1;
  const cut = Math.floor(points.length / 2);
  const early = stdev(diffs(points.slice(0, cut)));
  const late = stdev(diffs(points.slice(cut)));
  if (early < 1e-9) return 1;
  return late / early;
}

// ── Order book microstructure ──────────────────────────────────────────────

export type BookStats = {
  bestBid?: number;
  bestAsk?: number;
  mid?: number;
  /**
   * Size-weighted mid. Weighting each side by the *opposite* side's size is
   * the standard micro-price: when the bid is much larger than the ask, the
   * fair value sits closer to the ask, because that is where it will trade.
   */
  microPrice?: number;
  /** Micro-price minus mid, in probability points. Signed short-horizon lean. */
  microLean: number;
  spread?: number;
  /** Cost-weighted imbalance of resting capital, −1..1. */
  imbalance: number;
  bidNotional: number;
  askNotional: number;
  /** Dollars needed to sweep the book one full cent from mid. */
  costToMoveOneCent: number;
};

function notional(levels: readonly BookLevel[], side: "bid" | "ask"): number {
  // A bid risks `price` per share; an ask risks `1 - price`, since that is what
  // the seller has locked up. Without this, a wall of 2¢ asks would read as a
  // far bigger commitment than it is.
  let sum = 0;
  for (const l of levels) sum += (side === "bid" ? l.price : 1 - l.price) * l.size;
  return sum;
}

export function bookStats(book: OrderBook | undefined, depth = 10): BookStats {
  const empty: BookStats = {
    microLean: 0,
    imbalance: 0,
    bidNotional: 0,
    askNotional: 0,
    costToMoveOneCent: 0,
  };
  if (!book || book.bids.length === 0 || book.asks.length === 0) return empty;

  const bestBid = book.bids[0].price;
  const bestAsk = book.asks[0].price;
  const bidSize = book.bids[0].size;
  const askSize = book.asks[0].size;
  const mid = (bestBid + bestAsk) / 2;

  const totalTop = bidSize + askSize;
  const microPrice =
    totalTop > 0 ? (bidSize * bestAsk + askSize * bestBid) / totalTop : mid;

  const bidNotional = notional(book.bids.slice(0, depth), "bid");
  const askNotional = notional(book.asks.slice(0, depth), "ask");
  const total = bidNotional + askNotional;

  // Cost of lifting every offer within one cent above mid.
  let costToMoveOneCent = 0;
  for (const l of book.asks) {
    if (l.price > mid + 0.01) break;
    costToMoveOneCent += l.price * l.size;
  }

  return {
    bestBid,
    bestAsk,
    mid,
    microPrice,
    microLean: (microPrice - mid) * 100,
    spread: bestAsk - bestBid,
    imbalance: total > 0 ? (bidNotional - askNotional) / total : 0,
    bidNotional,
    askNotional,
    costToMoveOneCent,
  };
}
