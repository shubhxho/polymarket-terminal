"use client";

import { useMemo } from "react";
import type { PricePoint } from "@/lib/types";

/**
 * Inline SVG micro-chart for grid rows. Sized in a fixed viewBox and scaled by
 * CSS so it stays crisp at any row height without re-measuring.
 */
export function Sparkline({
  points,
  width = 72,
  height = 16,
  className = "",
}: {
  points: PricePoint[];
  width?: number;
  height?: number;
  className?: string;
}) {
  const model = useMemo(() => {
    if (points.length < 2) return null;
    const ps = points.length > 120 ? decimate(points, 120) : points;
    const ys = ps.map((p) => p.p);
    let lo = Math.min(...ys);
    let hi = Math.max(...ys);
    // A perfectly flat series would divide by zero; give it a hairline band.
    if (hi - lo < 1e-6) {
      lo -= 0.005;
      hi += 0.005;
    }
    const dx = width / (ps.length - 1);
    const y = (v: number) => height - ((v - lo) / (hi - lo)) * (height - 2) - 1;
    const d = ps
      .map((p, i) => `${i === 0 ? "M" : "L"}${(i * dx).toFixed(2)},${y(p.p).toFixed(2)}`)
      .join(" ");
    // Latest point, so the eye can find "now" without tracing the whole line.
    return { d, endX: (ps.length - 1) * dx, endY: y(ps[ps.length - 1].p) };
  }, [points, width, height]);

  if (!model) {
    return <div style={{ width, height }} className={className} />;
  }

  const first = points[0].p;
  const last = points[points.length - 1].p;
  const stroke =
    last > first ? "var(--color-up)" : last < first ? "var(--color-down)" : "var(--color-faint)";

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      className={`overflow-visible ${className}`}
      aria-hidden
    >
      <path
        d={model.d}
        fill="none"
        stroke={stroke}
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
      <circle cx={model.endX} cy={model.endY} r={1.4} fill={stroke} />
    </svg>
  );
}

/** Uniform stride sampling — exactly `target` points, first and last inclusive. */
function decimate(points: PricePoint[], target: number): PricePoint[] {
  if (target < 2 || points.length <= target) return points.slice();
  const out: PricePoint[] = [];
  // Span the full range so i=0 → first and i=target-1 → last, with no duplicated
  // final sample (the old `length/target` stride plus an extra push overshot to
  // target+1 points and repeated the last).
  const step = (points.length - 1) / (target - 1);
  for (let i = 0; i < target; i++) out.push(points[Math.round(i * step)]);
  return out;
}
