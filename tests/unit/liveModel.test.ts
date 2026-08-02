import { describe, expect, test } from "vitest";
import { appendTick, liveModelFor } from "@/lib/liveModel";
import { MODEL_WINDOW, modelSignal, modelSignalFromPrices } from "@/lib/mlSignal";
import type { MarketSignals } from "@/lib/signals";
import { blendedScore } from "@/lib/signals";
import type { PricePoint } from "@/lib/types";

const window16 = Array.from({ length: MODEL_WINDOW }, (_, i) => 0.4 + 0.005 * i);
const row = (over: Partial<MarketSignals> = {}): MarketSignals =>
  ({
    market: { id: "m", outcomes: [{ tokenId: "t" }] },
    heat: 50,
    bias: 60,
    ...over,
  }) as MarketSignals;

describe("appendTick", () => {
  test("drops an unchanged tick and returns the same array by reference", () => {
    const tail = [0.5, 0.51, 0.52];
    expect(appendTick(tail, 0.52)).toBe(tail);
  });

  test("appends a moved price as the newest element", () => {
    expect(appendTick([0.5, 0.51], 0.53)).toEqual([0.5, 0.51, 0.53]);
  });

  test("grows from empty", () => {
    expect(appendTick([], 0.5)).toEqual([0.5]);
  });

  test("never exceeds the model window — oldest print slides off the front", () => {
    const full = Array.from({ length: MODEL_WINDOW }, (_, i) => i / 100);
    const next = appendTick(full, 0.99);
    expect(next.length).toBe(MODEL_WINDOW);
    expect(next[next.length - 1]).toBe(0.99);
    expect(next[0]).toBe(full[1]); // full[0] dropped
  });
});

describe("modelSignalFromPrices", () => {
  test("matches modelSignal over the same series", () => {
    const history: PricePoint[] = window16.map((p, i) => ({ t: i, p }));
    expect(modelSignalFromPrices(window16)!.prob).toBeCloseTo(modelSignal(history)!.prob, 12);
  });

  test("returns null below a full window", () => {
    expect(modelSignalFromPrices(window16.slice(1))).toBeNull();
  });
});

describe("liveModelFor", () => {
  test("returns null until the window is full", () => {
    expect(liveModelFor(row(), window16.slice(1))).toBeNull();
  });

  test("blended equals blendedScore of the row with the live model spliced in", () => {
    const r = row({ heat: 70, bias: 80 });
    const live = liveModelFor(r, window16)!;
    expect(live.blended).toBeCloseTo(blendedScore({ ...r, model: live.model }), 12);
  });

  test("a stream of ticks keeps the buffer capped and the read valid throughout", () => {
    let tail = window16.slice();
    for (let i = 0; i < 100; i++) {
      tail = appendTick(tail, 0.3 + 0.004 * i);
      expect(tail.length).toBeLessThanOrEqual(MODEL_WINDOW);
      const read = liveModelFor(row(), tail);
      expect(read).not.toBeNull();
      expect(read!.model.prob).toBeGreaterThan(0);
      expect(read!.model.prob).toBeLessThan(1);
    }
    expect(tail.length).toBe(MODEL_WINDOW);
  });

  test("the read tracks the price path — a different live window scores differently", () => {
    // A steady climb and an alternating chop are materially different paths; the
    // live read must reflect that rather than returning a frozen snapshot.
    const climb = Array.from({ length: MODEL_WINDOW }, (_, i) => 0.3 + 0.01 * i);
    const chop = Array.from({ length: MODEL_WINDOW }, (_, i) => 0.5 + 0.03 * (-1) ** i);
    const a = liveModelFor(row(), climb)!;
    const b = liveModelFor(row(), chop)!;
    expect(a.model.prob).not.toBeCloseTo(b.model.prob, 4);
  });
});
