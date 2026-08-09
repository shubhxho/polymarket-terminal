import { describe, expect, test } from "vitest";
import { blendedScore, modelAgreement, type MarketSignals } from "@/lib/signals";
import type { ModelRead } from "@/lib/mlSignal";

/**
 * Unit coverage for the model×rule-engine blend.
 *
 * `blendedScore` and `modelAgreement` only ever read `heat`, `bias` and
 * `model`, so the fixtures below are minimal projections of `MarketSignals`
 * rather than full scans — the point is the arithmetic of the nudge and the
 * gating of the verdict, not the detectors that produced the inputs.
 */
const read = (prob: number, auc = 0.65): ModelRead => ({
  prob,
  direction: prob > 0.5 ? "bullish" : prob < 0.5 ? "bearish" : "neutral",
  conviction: Math.min(1, Math.abs(prob - 0.5) * 2),
  auc,
});

const ms = (heat: number, bias: number, model?: ModelRead): MarketSignals =>
  ({ heat, bias, model }) as unknown as MarketSignals;

describe("blendedScore", () => {
  test("falls back to raw heat when the market carried no model read", () => {
    expect(blendedScore(ms(40, 60))).toBe(40);
  });

  test("lifts a market the model agrees with, versus the same market without it", () => {
    const base = ms(50, 60);
    const withModel = ms(50, 60, read(0.85)); // model bullish, bias bullish
    expect(blendedScore(withModel)).toBeGreaterThan(blendedScore(base));
  });

  test("holds back a market the model fights", () => {
    const withModel = ms(50, 60, read(0.15)); // model bearish, bias bullish
    expect(blendedScore(withModel)).toBeLessThan(50);
  });

  test("the nudge is capped — full agreement cannot more than ~+22% at 0.65 AUC", () => {
    const maxUp = blendedScore(ms(50, 100, read(1))); // conviction 1, aligned
    // trust = (0.65-0.5)/0.2 = 0.75; factor ≤ 1 + 0.75·1·1·0.3 = 1.225
    expect(maxUp).toBeLessThanOrEqual(50 * 1.225 + 1e-9);
    expect(maxUp).toBeGreaterThan(50);
  });

  test("a stronger model (higher AUC) is trusted to move the score more", () => {
    const weak = blendedScore(ms(50, 80, read(0.85, 0.55)));
    const strong = blendedScore(ms(50, 80, read(0.85, 0.7)));
    expect(strong).toBeGreaterThan(weak);
  });

  test("a coin-flip model (0.5 AUC) earns zero trust and does not move the score", () => {
    expect(blendedScore(ms(50, 80, read(0.9, 0.5)))).toBeCloseTo(50, 6);
  });

  test("a fence-sitting model (prob 0.5) leaves heat untouched", () => {
    expect(blendedScore(ms(50, 80, read(0.5)))).toBeCloseTo(50, 6);
  });

  test("stays within 0..100 under extreme inputs", () => {
    expect(blendedScore(ms(100, 100, read(1)))).toBeLessThanOrEqual(100);
    expect(blendedScore(ms(0, -100, read(0)))).toBeGreaterThanOrEqual(0);
  });
});

describe("modelAgreement", () => {
  test("neutral when there is no model", () => {
    expect(modelAgreement(ms(50, 60))).toBe("neutral");
  });

  test("confirms when model and bias point the same way", () => {
    expect(modelAgreement(ms(50, 60, read(0.8)))).toBe("confirms");
    expect(modelAgreement(ms(50, -60, read(0.2)))).toBe("confirms");
  });

  test("conflicts when they point opposite ways", () => {
    expect(modelAgreement(ms(50, 60, read(0.2)))).toBe("conflicts");
    expect(modelAgreement(ms(50, -60, read(0.8)))).toBe("conflicts");
  });

  test("neutral when the model has too little conviction to commit", () => {
    expect(modelAgreement(ms(50, 60, read(0.52)))).toBe("neutral"); // conviction 0.04 < 0.08
  });

  test("neutral when the rule engine itself has no clear bias", () => {
    expect(modelAgreement(ms(50, 4, read(0.9)))).toBe("neutral"); // |bias| 4 < 8
  });
});
