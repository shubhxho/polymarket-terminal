import { describe, expect, test } from "vitest";
import { MODEL_CALIBRATION, MODEL_TEMPERATURE, MODEL_WINDOW, modelSignal } from "@/lib/mlSignal";
import type { PricePoint } from "@/lib/types";

/**
 * Parity harness for the in-browser model port.
 *
 * The forward pass in `mlSignal.ts` is a hand port of the MLX network in
 * `ml/`; the frozen weights only mean anything if the TypeScript features and
 * matmuls reproduce what Python computed. These expectations were captured by
 * running the actual trained `FeatureMLP` (data/seq_model.safetensors) over the
 * same series WITH the fitted temperature applied (`logit / T` before the
 * sigmoid) — regenerate them with `ml/calibrate.py` if the model is retrained or
 * recalibrated. The tolerance is loose enough to absorb float32-vs-float64 drift
 * and nothing more.
 */
const REFERENCE: Record<string, { prices: number[]; prob: number }> = {
  rising: { prices: Array.from({ length: 20 }, (_, i) => 0.3 + 0.01 * i), prob: 0.368915 },
  falling: { prices: Array.from({ length: 18 }, (_, i) => 0.7 - 0.012 * i), prob: 0.376239 },
  chop: { prices: Array.from({ length: 16 }, (_, i) => 0.5 + 0.03 * (-1) ** i), prob: 0.391598 },
  drifted: {
    prices: [
      0.4, 0.41, 0.39, 0.42, 0.44, 0.43, 0.46, 0.48, 0.47, 0.5, 0.52, 0.51, 0.55, 0.57, 0.56, 0.6,
    ],
    prob: 0.368915,
  },
};

const toHistory = (prices: number[]): PricePoint[] => prices.map((p, i) => ({ t: i, p }));

describe("modelSignal parity with the trained network", () => {
  for (const [name, { prices, prob }] of Object.entries(REFERENCE)) {
    test(`${name} reproduces the Python up-probability`, () => {
      const read = modelSignal(toHistory(prices));
      expect(read).not.toBeNull();
      expect(read!.prob).toBeCloseTo(prob, 3);
    });
  }
});

describe("temperature calibration", () => {
  test("ships a fitted temperature and an out-of-sample reliability report", () => {
    expect(MODEL_TEMPERATURE).toBeGreaterThan(0);
    expect(MODEL_CALIBRATION).toBeTruthy();
    expect(MODEL_CALIBRATION!.val_n).toBeGreaterThan(0);
  });

  test("calibration improved reliability — post-fit ECE is no worse than raw", () => {
    expect(MODEL_CALIBRATION!.ece_after).toBeLessThanOrEqual(MODEL_CALIBRATION!.ece_before);
  });

  // The parity fixtures above already encode the temperature: they were captured
  // WITH `logit / T`, so if the forward pass ever stopped applying it every one
  // of them would fail. This is the guard that the calibration is actually live.
});

describe("modelSignal contract", () => {
  test("returns null below a full window rather than a fake coin flip", () => {
    const short = toHistory(Array.from({ length: MODEL_WINDOW - 1 }, (_, i) => 0.5 + 0.001 * i));
    expect(modelSignal(short)).toBeNull();
    expect(modelSignal(undefined)).toBeNull();
    expect(modelSignal([])).toBeNull();
  });

  test("direction and conviction follow from prob", () => {
    const read = modelSignal(toHistory(REFERENCE.chop.prices));
    expect(read).not.toBeNull();
    const { prob, direction, conviction } = read!;
    expect(direction).toBe(prob > 0.5 ? "bullish" : "bearish");
    expect(conviction).toBeCloseTo(Math.min(1, Math.abs(prob - 0.5) * 2), 10);
    expect(conviction).toBeGreaterThanOrEqual(0);
    expect(conviction).toBeLessThanOrEqual(1);
  });

  test("only the last window drives the read — older points are ignored", () => {
    const tail = REFERENCE.chop.prices;
    const withPrefix = toHistory([0.1, 0.9, 0.2, 0.8, ...tail]);
    const bare = modelSignal(toHistory(tail));
    const prefixed = modelSignal(withPrefix);
    expect(prefixed!.prob).toBeCloseTo(bare!.prob, 10);
  });

  test("runs at exactly one window and no earlier", () => {
    const prices = Array.from({ length: MODEL_WINDOW }, (_, i) => 0.4 + 0.005 * i);
    expect(modelSignal(toHistory(prices))).not.toBeNull();
    expect(modelSignal(toHistory(prices.slice(1)))).toBeNull();
  });

  test("a dead-flat window never divides by zero — prob stays a real number in (0,1)", () => {
    const flat = toHistory(Array(MODEL_WINDOW).fill(0.5));
    const read = modelSignal(flat);
    expect(read).not.toBeNull();
    expect(Number.isFinite(read!.prob)).toBe(true);
    expect(read!.prob).toBeGreaterThan(0);
    expect(read!.prob).toBeLessThan(1);
  });

  test("carries the model's out-of-sample AUC through unchanged", () => {
    const read = modelSignal(toHistory(REFERENCE.rising.prices));
    expect(read!.auc).toBeCloseTo(0.6502, 6);
  });
});
