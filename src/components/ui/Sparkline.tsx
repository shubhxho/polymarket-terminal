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
  const path = useMemo(() => {
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
    return ps
      .map((p, i) => `${i === 0 ? "M" : "L"}${(i * dx).toFixed(2)},${y(p.p).toFixed(2)}`)
      .join(" ");
  }, [points, width, height]);

  if (!path) {
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
        d={path}
        fill="none"
        stroke={stroke}
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

/** Uniform stride sampling — keeps first and last so endpoints stay honest. */
function decimate(points: PricePoint[], target: number): PricePoint[] {
  const step = points.length / target;
  const out: PricePoint[] = [];
  for (let i = 0; i < target; i++) out.push(points[Math.floor(i * step)]);
  out.push(points[points.length - 1]);
  return out;
}
