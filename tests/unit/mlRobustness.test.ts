import { describe, expect, test } from "vitest";
import { MODEL_AUC, MODEL_WINDOW, modelSignalFromPrices } from "@/lib/mlSignal";

/**
 * Numerical-robustness fuzz for the model port.
 *
 * The parity fixtures pin a handful of exact outputs; this pins the invariants
 * that must hold for *every* input — no NaN or Infinity leaking out of a feature
 * divide, probabilities always a real number strictly inside (0, 1), conviction
 * and direction always consistent with that probability. A regression in the
 * feature block (a zero denominator, a wrong clamp) tends to surface here long
 * before it shows up as a subtly-wrong ranking.
 *
 * Randomness is a seeded LCG so the "fuzz" is fully deterministic and replays
 * identically in CI — `Math.random` is deliberately avoided.
 */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/** A probability path of `n` points in (lo, hi), seeded and reproducible. */
function series(seed: number, n = 20, lo = 0.02, hi = 0.98): number[] {
  const rand = lcg(seed);
  return Array.from({ length: n }, () => lo + (hi - lo) * rand());
}

describe("model numerical robustness", () => {
  test("every seeded path yields a finite probability strictly inside (0, 1)", () => {
    for (let seed = 1; seed <= 400; seed++) {
      const read = modelSignalFromPrices(series(seed));
      expect(read).not.toBeNull();
      const p = read!.prob;
      expect(Number.isFinite(p)).toBe(true);
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThan(1);
    }
  });

  test("conviction and direction always follow from the probability", () => {
    for (let seed = 1; seed <= 400; seed++) {
      const read = modelSignalFromPrices(series(seed))!;
      expect(read.conviction).toBeCloseTo(Math.min(1, Math.abs(read.prob - 0.5) * 2), 12);
      expect(read.conviction).toBeGreaterThanOrEqual(0);
      expect(read.conviction).toBeLessThanOrEqual(1);
      expect(read.direction).toBe(
        read.prob > 0.5 ? "bullish" : read.prob < 0.5 ? "bearish" : "neutral"
      );
      expect(read.auc).toBe(MODEL_AUC);
    }
  });

  test("degenerate paths (flat, pinned to a bound, single spike) never leak NaN", () => {
    const flat = Array(MODEL_WINDOW).fill(0.5);
    const lowPinned = Array(MODEL_WINDOW).fill(0.02);
    const highPinned = Array(MODEL_WINDOW).fill(0.98);
    const spike = [...Array(MODEL_WINDOW - 1).fill(0.5), 0.97];
    for (const path of [flat, lowPinned, highPinned, spike]) {
      const read = modelSignalFromPrices(path);
      expect(read).not.toBeNull();
      expect(Number.isFinite(read!.prob)).toBe(true);
      expect(read!.prob).toBeGreaterThan(0);
      expect(read!.prob).toBeLessThan(1);
    }
  });

  test("scoring only ever reads the last MODEL_WINDOW points", () => {
    const tail = series(7);
    const withNoisyPrefix = [...series(99, 40), ...tail];
    expect(modelSignalFromPrices(withNoisyPrefix)!.prob).toBeCloseTo(
      modelSignalFromPrices(tail)!.prob,
      12
    );
  });

  test("is a pure function — same input, same output across repeated calls", () => {
    const path = series(123);
    const a = modelSignalFromPrices(path)!;
    const b = modelSignalFromPrices(path.slice())!;
    expect(a.prob).toBe(b.prob);
  });
});
