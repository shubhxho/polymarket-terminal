import { describe, expect, it } from "vitest";
import {
  autocorrelation,
  bookStats,
  clamp,
  diffs,
  driftPerDay,
  logistic,
  mad,
  mean,
  median,
  percentileOf,
  realisedVol,
  robustZ,
  saturate,
  stdev,
} from "@/lib/quant";
import type { OrderBook, PricePoint } from "@/lib/types";

/** A price series sampled at a fixed cadence. `p` in 0..1, `t` in seconds. */
function series(prices: number[], stepSeconds = 86_400): PricePoint[] {
  return prices.map((p, i) => ({ t: i * stepSeconds, p }));
}

const deep = (price: number, size: number) => ({ price, size });

describe("descriptive statistics", () => {
  it("means, and returns 0 on empty rather than NaN", () => {
    expect(mean([2, 4, 6])).toBe(4);
    expect(mean([])).toBe(0);
  });

  it("uses a sample (n-1) standard deviation and needs two points", () => {
    expect(stdev([2, 4])).toBeCloseTo(1.4142, 4);
    expect(stdev([5])).toBe(0);
    expect(stdev([])).toBe(0);
  });

  it("medians both odd and even length series", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([1, 2, 3, 4])).toBe(2.5);
    expect(median([])).toBe(0);
  });

  it("collapses MAD to 0 on degenerate input", () => {
    expect(mad([1])).toBe(0);
    expect(mad([5, 5, 5, 5])).toBe(0);
  });
});

describe("robustZ", () => {
  it("is 0 for populations too small to be robust", () => {
    expect(robustZ(10, [1, 2, 3])).toBe(0);
  });

  it("falls back to mean/stdev when MAD collapses", () => {
    // Majority share one value → MAD is 0 → classical z is used.
    const pop = [1, 1, 1, 5];
    expect(robustZ(5, pop)).not.toBe(0);
  });
});

describe("percentileOf", () => {
  it("is the fraction at or below x", () => {
    expect(percentileOf(2, [1, 2, 3])).toBeCloseTo(2 / 3, 6);
    expect(percentileOf(0, [])).toBe(0.5);
  });
});

describe("bounded maps", () => {
  it("clamps to the interval", () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(11, 0, 10)).toBe(10);
  });

  it("saturates non-negative input into 0..1", () => {
    expect(saturate(0, 5)).toBe(0);
    expect(saturate(-3, 5)).toBe(0);
    expect(saturate(5, 5)).toBe(0.5);
  });

  it("squashes a z-score through the logistic centred on zero", () => {
    expect(logistic(0)).toBe(0.5);
    expect(logistic(100)).toBeCloseTo(1, 6);
    expect(logistic(-100)).toBeCloseTo(0, 6);
  });
});

describe("time series", () => {
  it("diffs consecutive points", () => {
    expect(diffs(series([0.1, 0.2, 0.15]))).toEqual([
      expect.closeTo(0.1, 6),
      expect.closeTo(-0.05, 6),
    ]);
  });

  it("recovers a linear drift in points per day", () => {
    // 0.1 → 0.2 → 0.3 over one day each = +10 probability points/day.
    expect(driftPerDay(series([0.1, 0.2, 0.3]))).toBeCloseTo(10, 6);
  });

  it("reports zero drift on too-few or flat series", () => {
    expect(driftPerDay(series([0.1, 0.2]))).toBe(0);
    expect(driftPerDay(series([0.4, 0.4, 0.4]))).toBeCloseTo(0, 6);
  });

  it("realised vol is positive when increments vary, zero when they don't", () => {
    expect(realisedVol(series([0.1, 0.25, 0.3]))).toBeGreaterThan(0);
    expect(realisedVol(series([0.2, 0.2, 0.2]))).toBe(0); // flat → no vol
    expect(realisedVol(series([0.2, 0.3]))).toBe(0); // < 3 points
  });

  it("autocorrelation stays within [-1, 1] and is 0 on short input", () => {
    expect(autocorrelation(series([0.1, 0.2]))).toBe(0);
    const ac = autocorrelation(series([1, 2, 3, 2, 4, 3, 5, 4, 6, 5].map((x) => x / 10)));
    expect(ac).toBeGreaterThanOrEqual(-1);
    expect(ac).toBeLessThanOrEqual(1);
  });
});

describe("bookStats", () => {
  it("returns a neutral shape for an empty or missing book", () => {
    const s = bookStats(undefined);
    expect(s.imbalance).toBe(0);
    expect(s.mid).toBeUndefined();
  });

  it("derives mid, spread and a signed imbalance", () => {
    const book: OrderBook = {
      tokenId: "t",
      timestamp: 0,
      bids: [{ price: 0.4, size: 100 }],
      asks: [{ price: 0.42, size: 100 }],
    };
    const s = bookStats(book);
    expect(s.bestBid).toBe(0.4);
    expect(s.bestAsk).toBe(0.42);
    expect(s.mid).toBeCloseTo(0.41, 6);
    expect(s.spread).toBeCloseTo(0.02, 6);
    // Ask side risks (1-price) per share, so equal sizes lean bid-negative.
    expect(s.imbalance).toBeLessThan(0);
  });

  it("reports one-cent depth on both sides and takes the thinner as the honest cost", () => {
    const book: OrderBook = {
      tokenId: "t",
      timestamp: 0,
      // mid = 0.50. Bids within a cent: only the 0.49 level (0.495 is inside).
      bids: [
        { price: 0.49, size: 1000 },
        { price: 0.4, size: 9999 }, // outside the cent band → excluded
      ],
      // Asks within a cent: only the 0.51 level.
      asks: [
        { price: 0.51, size: 100 },
        { price: 0.6, size: 9999 }, // outside → excluded
      ],
    };
    const s = bookStats(book);
    expect(s.costDownOneCent).toBeCloseTo(0.49 * 1000, 6);
    expect(s.costUpOneCent).toBeCloseTo(0.51 * 100, 6);
    // The offer is far thinner, so the honest two-way cost is the ask side.
    expect(s.costToMoveOneCent).toBeCloseTo(0.51 * 100, 6);
  });

  it("scores a tight, two-sided book above a wide or one-sided one", () => {
    const tight = bookStats({
      tokenId: "t",
      timestamp: 0,
      bids: [deep(0.499, 60_000), deep(0.498, 60_000)],
      asks: [deep(0.501, 60_000), deep(0.502, 60_000)],
    });
    const wide = bookStats({
      tokenId: "t",
      timestamp: 0,
      bids: [deep(0.45, 60_000)],
      asks: [deep(0.55, 60_000)],
    });
    const oneSided = bookStats({
      tokenId: "t",
      timestamp: 0,
      bids: [deep(0.499, 60_000)], // deep bid
      asks: [deep(0.501, 5)], // hollow ask — the thinner side caps the score
    });
    expect(tight.liquidityScore).toBeGreaterThan(wide.liquidityScore);
    expect(tight.liquidityScore).toBeGreaterThan(oneSided.liquidityScore);
    expect(tight.liquidityScore).toBeGreaterThan(0.5);
    expect(tight.liquidityScore).toBeLessThanOrEqual(1);
  });

  it("is a zero-liquidity book when a side is empty", () => {
    const s = bookStats({ tokenId: "t", timestamp: 0, bids: [], asks: [] });
    expect(s.liquidityScore).toBe(0);
    expect(s.costToMoveOneCent).toBe(0);
    expect(s.costUpOneCent).toBe(0);
    expect(s.costDownOneCent).toBe(0);
  });
});
