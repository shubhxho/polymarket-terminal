import { describe, expect, it } from "vitest";
import { buildCells, gridLayout, heatColor, intensity } from "@/lib/heatmap";

describe("gridLayout", () => {
  it("degenerates safely on empty or zero-size input", () => {
    expect(gridLayout(0, 100, 100).cols).toBe(0);
    expect(gridLayout(10, 0, 100).cols).toBe(0);
    const l = gridLayout(0, 100, 100);
    expect(l.rect(0)).toEqual({ x: 0, y: 0, w: 0, h: 0 });
  });

  it("squares up cells for the aspect ratio", () => {
    // 100 cells in a square box → ~10×10.
    const l = gridLayout(100, 400, 400);
    expect(l.cols).toBe(10);
    expect(l.rows).toBe(10);
  });

  it("uses more columns in a wide box", () => {
    const wide = gridLayout(100, 1600, 100);
    const tall = gridLayout(100, 100, 1600);
    expect(wide.cols).toBeGreaterThan(tall.cols);
  });

  it("packs every cell inside the box", () => {
    const w = 300;
    const h = 200;
    const l = gridLayout(37, w, h, 2);
    for (let i = 0; i < 37; i++) {
      const r = l.rect(i);
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeGreaterThanOrEqual(0);
      expect(r.x + r.w).toBeLessThanOrEqual(w + 0.001);
      expect(r.y + r.h).toBeLessThanOrEqual(h + 0.001);
    }
  });
});

describe("intensity", () => {
  it("is signed and scaled to the field max, compressing the tail", () => {
    expect(intensity(0, 10)).toBe(0);
    expect(intensity(10, 10)).toBeCloseTo(1, 6);
    expect(intensity(-10, 10)).toBeCloseTo(-1, 6);
    // sqrt compression: a quarter of the peak reads at half intensity.
    expect(intensity(2.5, 10)).toBeCloseTo(0.5, 6);
  });

  it("is 0 for a degenerate field", () => {
    expect(intensity(5, 0)).toBe(0);
    expect(intensity(Number.NaN, 10)).toBe(0);
  });
});

describe("heatColor", () => {
  it("is neutral at zero and diverges up/down", () => {
    const [nr, ng, nb] = heatColor(0);
    expect([nr, ng, nb]).toEqual([0.5, 0.5, 0.48]); // neutral grey
    // Green channel dominates on the up side, red on the down side.
    expect(heatColor(1)[1]).toBeGreaterThan(heatColor(1)[0]);
    expect(heatColor(-1)[0]).toBeGreaterThan(heatColor(-1)[1]);
  });

  it("clamps out-of-range intensity", () => {
    expect(heatColor(5)).toEqual(heatColor(1));
    expect(heatColor(-5)).toEqual(heatColor(-1));
  });
});

describe("buildCells", () => {
  it("returns one cell per value with a rect and signed intensity", () => {
    const cells = buildCells([3, -6, 0], 300, 100, 2);
    expect(cells).toHaveLength(3);
    expect(cells[1].t).toBeLessThan(0); // -6 is the peak → -1
    expect(cells[1].t).toBeCloseTo(-1, 6);
    expect(cells[2].t).toBe(0);
    expect(cells[0].i).toBe(0);
  });
});
