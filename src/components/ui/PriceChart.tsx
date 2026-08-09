"use client";

import { useCallback, useMemo, useState } from "react";
import { useElementSize } from "@/hooks/useElementSize";
import { cents } from "@/lib/format";
import type { PricePoint } from "@/lib/types";

export type Series = {
  tokenId: string;
  label: string;
  color: string;
  points: PricePoint[];
};

/**
 * Categorical slots for multi-outcome events, in fixed order — never cycled.
 *
 * These resolve to CSS custom properties so each theme supplies its own step
 * for its own surface, rather than one set being auto-flipped into the other.
 * The ordering is CVD-optimised (worst adjacent pair ΔE 13.3 light / 23.6 dark)
 * and deliberately puts green and red last, so a chart with a handful of legs
 * never paints a line in a hue that means "up" or "down" everywhere else.
 *
 * A ninth series is never a generated hue — the chart caps at eight and the
 * remaining legs stay in the table below it.
 */
export const SERIES_COLORS = [
  "var(--c-series-1)",
  "var(--c-series-2)",
  "var(--c-series-3)",
  "var(--c-series-4)",
  "var(--c-series-5)",
  "var(--c-series-6)",
  "var(--c-series-7)",
  "var(--c-series-8)",
];

// Left padding leaves room for the first x-axis label, which is centred on the
// plot origin and would otherwise be clipped by the panel edge.
const PAD = { top: 8, right: 44, bottom: 18, left: 18 };

/**
 * Multi-series probability chart with a snapping crosshair.
 *
 * The y-axis is always probability, so it is clamped to a padded window around
 * the data rather than 0-100: a market oscillating between 61¢ and 64¢ should
 * fill the panel, not sit as a flat line across the middle.
 */
export function PriceChart({ series, height = 200 }: { series: Series[]; height?: number }) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [hoverX, setHoverX] = useState<number | null>(null);

  const w = Math.max(size.width, 120);
  const h = height;
  const plotW = Math.max(10, w - PAD.left - PAD.right);
  const plotH = Math.max(10, h - PAD.top - PAD.bottom);

  const model = useMemo(() => {
    const live = series.filter((s) => s.points.length > 1);
    if (live.length === 0) return null;

    let tMin = Infinity;
    let tMax = -Infinity;
    let pMin = Infinity;
    let pMax = -Infinity;
    for (const s of live) {
      for (const pt of s.points) {
        if (pt.t < tMin) tMin = pt.t;
        if (pt.t > tMax) tMax = pt.t;
        if (pt.p < pMin) pMin = pt.p;
        if (pt.p > pMax) pMax = pt.p;
      }
    }
    if (!Number.isFinite(tMin) || tMax === tMin) return null;

    // Pad the value axis by 8% of range, floor the band at 2 points so a
    // pinned market still shows a readable line.
    const span = Math.max(pMax - pMin, 0.02);
    const padY = span * 0.08;
    const lo = Math.max(0, pMin - padY);
    const hi = Math.min(1, pMax + padY);
    const range = Math.max(hi - lo, 1e-6);

    const x = (t: number) => PAD.left + ((t - tMin) / (tMax - tMin)) * plotW;
    const y = (p: number) => PAD.top + (1 - (p - lo) / range) * plotH;

    const paths = live.map((s) => ({
      ...s,
      d: s.points
        .map((pt, i) => `${i === 0 ? "M" : "L"}${x(pt.t).toFixed(1)},${y(pt.p).toFixed(1)}`)
        .join(" "),
      lastPoint: s.points[s.points.length - 1],
    }));

    const yTicks = niceTicks(lo, hi, 4);
    const xTicks = timeTicks(tMin, tMax, 5);

    return { live, tMin, tMax, lo, hi, x, y, paths, yTicks, xTicks };
  }, [series, plotW, plotH]);

  const onMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHoverX(e.clientX - rect.left);
  }, []);

  if (!model) {
    return (
      <div
        ref={ref}
        style={{ height }}
        className="flex items-center justify-center text-[10px] tracking-widest text-faint uppercase"
      >
        no price history
      </div>
    );
  }

  const { tMin, tMax, x, y, paths, yTicks, xTicks } = model;

  // Snap the crosshair to the nearest sample of each series.
  const cursor =
    hoverX !== null && hoverX >= PAD.left && hoverX <= PAD.left + plotW
      ? (() => {
          const t = tMin + ((hoverX - PAD.left) / plotW) * (tMax - tMin);
          return {
            t,
            px: x(t),
            readings: paths.map((s) => {
              const pt = nearest(s.points, t);
              return {
                tokenId: s.tokenId,
                label: s.label,
                color: s.color,
                price: pt.p,
                y: y(pt.p),
              };
            }),
          };
        })()
      : null;

  return (
    <div ref={ref} className="relative w-full" style={{ height }}>
      <svg
        width="100%"
        height={h}
        viewBox={`0 0 ${w} ${h}`}
        preserveAspectRatio="none"
        onMouseMove={onMove}
        onMouseLeave={() => setHoverX(null)}
        className="block"
      >
        {yTicks.map((tick) => (
          <g key={`y${tick}`}>
            <line
              x1={PAD.left}
              x2={PAD.left + plotW}
              y1={y(tick)}
              y2={y(tick)}
              stroke="var(--color-grid)"
              strokeDasharray="1 3"
            />
            <text x={PAD.left + plotW + 5} y={y(tick) + 3} fill="var(--color-faint)" fontSize={9}>
              {cents(tick, 0)}¢
            </text>
          </g>
        ))}

        {xTicks.map((tick) => (
          <g key={`x${tick.t}`}>
            <line
              x1={x(tick.t)}
              x2={x(tick.t)}
              y1={PAD.top}
              y2={PAD.top + plotH}
              stroke="var(--color-grid)"
              strokeDasharray="1 3"
            />
            <text
              x={x(tick.t)}
              y={h - 5}
              fill="var(--color-faint)"
              fontSize={9}
              textAnchor="middle"
            >
              {tick.label}
            </text>
          </g>
        ))}

        {paths.map((s) => (
          <path
            key={s.tokenId}
            d={s.d}
            fill="none"
            stroke={s.color}
            strokeWidth={1.25}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        ))}

        {paths.map((s) => (
          <g key={`mark-${s.tokenId}`}>
            <line
              x1={PAD.left}
              x2={PAD.left + plotW}
              y1={y(s.lastPoint.p)}
              y2={y(s.lastPoint.p)}
              stroke={s.color}
              strokeWidth={0.5}
              strokeDasharray="2 3"
              opacity={0.5}
            />
            <circle cx={x(s.lastPoint.t)} cy={y(s.lastPoint.p)} r={1.8} fill={s.color} />
          </g>
        ))}

        {cursor ? (
          <g>
            <line
              x1={cursor.px}
              x2={cursor.px}
              y1={PAD.top}
              y2={PAD.top + plotH}
              stroke="var(--color-accent)"
              strokeWidth={0.6}
            />
            {cursor.readings.map((r) => (
              <circle key={r.tokenId} cx={cursor.px} cy={r.y} r={2.4} fill={r.color} />
            ))}
          </g>
        ) : null}
      </svg>

      {cursor ? (
        <div
          className="pointer-events-none absolute top-1 z-10 border border-edge-strong bg-canvas px-1.5 py-1 text-[10px] whitespace-nowrap"
          style={{
            // Flip the tooltip to the left of the cursor near the right edge.
            left: cursor.px > w * 0.6 ? undefined : cursor.px + 8,
            right: cursor.px > w * 0.6 ? w - cursor.px + 8 : undefined,
          }}
        >
          <div className="mb-0.5 text-muted">
            {new Date(cursor.t * 1000).toISOString().slice(5, 16).replace("T", " ")}Z
          </div>
          {cursor.readings.map((r) => (
            <div key={r.tokenId} className="flex items-center gap-1.5">
              <span className="inline-block h-1.5 w-1.5" style={{ background: r.color }} />
              <span className="text-muted">{r.label}</span>
              <span className="ml-auto text-ink">{cents(r.price)}¢</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function nearest(points: PricePoint[], t: number): PricePoint {
  // Points are time-ordered, so binary search beats a linear scan on the
  // several-hundred-point series the CLOB returns.
  let lo = 0;
  let hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid].t < t) lo = mid;
    else hi = mid;
  }
  return Math.abs(points[lo].t - t) <= Math.abs(points[hi].t - t) ? points[lo] : points[hi];
}

/** Axis ticks on 1/2/5 x 10^n boundaries. */
function niceTicks(lo: number, hi: number, count: number): number[] {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    out.push(Number(v.toFixed(6)));
  }
  return out;
}

function timeTicks(tMin: number, tMax: number, count: number) {
  const spanHours = (tMax - tMin) / 3600;
  const out: { t: number; label: string }[] = [];
  for (let i = 0; i <= count; i++) {
    const t = tMin + ((tMax - tMin) * i) / count;
    const d = new Date(t * 1000);
    // Within a couple of days the time of day is the useful axis; beyond that
    // the calendar date is.
    const label =
      spanHours <= 48
        ? d.toISOString().slice(11, 16)
        : d.toISOString().slice(5, 10).replace("-", "/");
    out.push({ t, label });
  }
  return out;
}
