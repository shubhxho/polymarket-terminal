/**
 * Pure layout + colour maths for the market heatmap.
 *
 * Kept separate from the renderer so the same numbers drive both the WebGPU
 * path and the Canvas2D fallback — one source of truth, and unit-testable
 * without a GPU or a DOM. The renderer only uploads what this produces.
 */

export type HeatCell = {
  /** Index into the source array. */
  i: number;
  /** Pixel rect, top-left origin. */
  x: number;
  y: number;
  w: number;
  h: number;
  /** Signed intensity in −1..1: the value mapped against the field's own max. */
  t: number;
};

/**
 * Packs `count` cells into a near-square grid that fills `width × height`.
 *
 * Column count is chosen to make cells as square as possible for the aspect
 * ratio, which is what keeps a 200-cell board readable rather than a row of
 * slivers.
 */
export function gridLayout(
  count: number,
  width: number,
  height: number,
  gap = 2
): {
  cols: number;
  rows: number;
  rect: (i: number) => { x: number; y: number; w: number; h: number };
} {
  const n = Math.max(0, Math.floor(count));
  if (n === 0 || width <= 0 || height <= 0) {
    return { cols: 0, rows: 0, rect: () => ({ x: 0, y: 0, w: 0, h: 0 }) };
  }
  // Aspect-aware column count: √(n · width/height) squares up the cells.
  const cols = Math.max(1, Math.min(n, Math.round(Math.sqrt((n * width) / height))));
  const rows = Math.ceil(n / cols);
  const cellW = (width - gap * (cols - 1)) / cols;
  const cellH = (height - gap * (rows - 1)) / rows;

  const rect = (i: number) => {
    const c = i % cols;
    const r = Math.floor(i / cols);
    return { x: c * (cellW + gap), y: r * (cellH + gap), w: cellW, h: cellH };
  };
  return { cols, rows, rect };
}

/**
 * Signed, magnitude-scaled intensity for a value against the field's peak.
 *
 * A square-root compresses the long tail so a single 40-point mover doesn't
 * wash every other cell to flat — the same reason the movers list is capped.
 */
export function intensity(value: number, maxAbs: number): number {
  if (!Number.isFinite(value) || maxAbs <= 0) return 0;
  const sign = value < 0 ? -1 : 1;
  const mag = Math.min(1, Math.sqrt(Math.abs(value) / maxAbs));
  return sign * mag;
}

export function buildCells(
  values: readonly number[],
  width: number,
  height: number,
  gap = 2
): HeatCell[] {
  const layout = gridLayout(values.length, width, height, gap);
  let maxAbs = 0;
  for (const v of values) {
    const a = Math.abs(v);
    if (Number.isFinite(a) && a > maxAbs) maxAbs = a;
  }
  return values.map((v, i) => {
    const r = layout.rect(i);
    return { i, ...r, t: intensity(v, maxAbs) };
  });
}

/**
 * Direction palette, mirroring the `--c-up` / `--c-down` tokens (green up, red
 * down, neutral grey at zero) as linear-space RGB in 0..1 so both the WGSL
 * shader and the Canvas2D path can read the same numbers. Green and red carry
 * price direction here exactly as they do in the tables.
 */
const UP: [number, number, number] = [0.07, 0.573, 0.306]; // #12924e
const DOWN: [number, number, number] = [0.816, 0.231, 0.231]; // #d03b3b
const NEUTRAL: [number, number, number] = [0.5, 0.5, 0.48];

/** Maps signed intensity −1..1 to an RGB triple in 0..1. */
export function heatColor(t: number): [number, number, number] {
  const c = Math.max(-1, Math.min(1, t));
  const target = c >= 0 ? UP : DOWN;
  const k = Math.abs(c);
  return [
    NEUTRAL[0] + (target[0] - NEUTRAL[0]) * k,
    NEUTRAL[1] + (target[1] - NEUTRAL[1]) * k,
    NEUTRAL[2] + (target[2] - NEUTRAL[2]) * k,
  ];
}

/** `rgb(...)` string for the Canvas2D fallback. */
export function heatColorCss(t: number): string {
  const [r, g, b] = heatColor(t);
  return `rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)})`;
}
