"use client";

import { useRef, useState } from "react";
import type { PricePoint } from "@/lib/polymarket";

export interface ChartSeries {
  /** Stable unique identity for React keys (e.g. CLOB token id). */
  id: string;
  label: string;
  points: PricePoint[];
}

const SERIES_COLORS = [
  "var(--accent)",
  "var(--amber)",
  "var(--cyan)",
  "var(--red)",
  "#c084fc",
];

const W = 900;
const H = 280;
const PAD_L = 40; // left padding for Y-axis labels
const PAD_R = 10;
const PAD_T = 12;
const PAD_B = 8;

const CHART_LEFT = PAD_L;
const CHART_RIGHT = W - PAD_R;
const CHART_TOP = PAD_T;
const CHART_BOTTOM = H - PAD_B;
const CHART_W = CHART_RIGHT - CHART_LEFT;
const CHART_H = CHART_BOTTOM - CHART_TOP;

interface HoverState {
  svgX: number;
  time: number;
  values: { label: string; price: number; color: string }[];
}

function xOf(t: number, tMin: number, tSpan: number) {
  return CHART_LEFT + ((t - tMin) / tSpan) * CHART_W;
}

function yOf(p: number) {
  return CHART_TOP + (1 - p) * CHART_H;
}

function shortStamp(unixSec: number): string {
  const d = new Date(unixSec * 1000);
  const mo = d.toLocaleString("en-US", { month: "short", timeZone: "UTC" });
  const day = d.getUTCDate();
  const hr = String(d.getUTCHours()).padStart(2, "0");
  const mn = String(d.getUTCMinutes()).padStart(2, "0");
  return `${mo} ${day} ${hr}:${mn}`;
}

function fullStamp(unixSec: number): string {
  return new Date(unixSec * 1000)
    .toISOString()
    .slice(0, 16)
    .replace("T", " ")
    .concat(" UTC");
}

export function PriceChart({ series }: { series: ChartSeries[] }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<HoverState | null>(null);

  const drawn = series.filter((s) => s.points.length > 1);

  if (drawn.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center border border-edge bg-panel text-muted">
        NO PRICE HISTORY AVAILABLE
      </div>
    );
  }

  const allT = drawn.flatMap((s) => s.points.map((p) => p.t));
  const tMin = Math.min(...allT);
  const tMax = Math.max(...allT);
  const tSpan = Math.max(tMax - tMin, 1);

  function handleMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const relX = (e.clientX - rect.left) / rect.width;
    const svgX = relX * W;
    const clampedX = Math.max(CHART_LEFT, Math.min(CHART_RIGHT, svgX));
    const t = tMin + ((clampedX - CHART_LEFT) / CHART_W) * tSpan;

    const values = drawn.map((s, i) => {
      let closest = s.points[0];
      let minDist = Math.abs(s.points[0].t - t);
      for (const p of s.points) {
        const d = Math.abs(p.t - t);
        if (d < minDist) {
          minDist = d;
          closest = p;
        }
      }
      return {
        label: s.label,
        price: closest.p,
        color: SERIES_COLORS[i % SERIES_COLORS.length],
      };
    });

    setHover({ svgX: clampedX, time: t, values });
  }

  const gridLines = [0, 0.25, 0.5, 0.75, 1.0];
  const N_TIME_TICKS = 5;
  const timeTicks = Array.from(
    { length: N_TIME_TICKS },
    (_, i) => tMin + (i / (N_TIME_TICKS - 1)) * tSpan,
  );

  return (
    <div className="border border-edge bg-panel panel-lit">
      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-b border-edge px-3 py-2">
        {drawn.map((s, i) => {
          const color = SERIES_COLORS[i % SERIES_COLORS.length];
          const firstP = s.points[0].p;
          const displayP = hover
            ? (hover.values[i]?.price ?? s.points[s.points.length - 1].p)
            : s.points[s.points.length - 1].p;
          const change = displayP - firstP;
          return (
            <span
              key={s.id}
              className="flex items-center gap-1.5 text-[11px]"
            >
              <span
                className="inline-block h-[3px] w-3 shrink-0 rounded-sm"
                style={{ background: color }}
              />
              <span
                className="truncate text-muted"
                style={{ maxWidth: "9rem" }}
              >
                {s.label.toUpperCase()}
              </span>
              <span
                className="shrink-0 tabular-nums font-bold"
                style={{ color }}
              >
                {(displayP * 100).toFixed(1)}%
              </span>
              <span
                className={`shrink-0 tabular-nums text-[10px] ${change >= 0 ? "text-accent" : "text-red"}`}
              >
                {change >= 0 ? "▲" : "▼"}
                {Math.abs(change * 100).toFixed(1)}pp
              </span>
            </span>
          );
        })}
        {hover && (
          <span className="ml-auto shrink-0 text-[10px] text-muted tabular-nums">
            {fullStamp(hover.time)}
          </span>
        )}
      </div>

      {/* SVG Chart */}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full cursor-crosshair"
        preserveAspectRatio="none"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label="Price history chart"
      >
        <defs>
          {/* Phosphor glow for the plotted lines */}
          <filter id="line-glow" x="-5%" y="-20%" width="110%" height="140%">
            <feGaussianBlur stdDeviation="1.6" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {drawn.map((_, i) => {
            const color = SERIES_COLORS[i % SERIES_COLORS.length];
            return (
              <linearGradient
                key={i}
                id={`chart-area-${i}`}
                x1="0"
                y1={CHART_TOP}
                x2="0"
                y2={CHART_BOTTOM}
                gradientUnits="userSpaceOnUse"
              >
                <stop offset="0%" stopColor={color} stopOpacity="0.20" />
                <stop offset="85%" stopColor={color} stopOpacity="0.02" />
              </linearGradient>
            );
          })}
        </defs>

        {/* Y-axis grid lines and labels */}
        {gridLines.map((g) => (
          <g key={g}>
            <line
              x1={CHART_LEFT}
              x2={CHART_RIGHT}
              y1={yOf(g)}
              y2={yOf(g)}
              stroke="var(--border)"
              strokeWidth={g === 0 || g === 1 ? "0.5" : "0.8"}
              strokeDasharray={g !== 0 && g !== 1 ? "3 8" : undefined}
              opacity={g === 0 || g === 1 ? 0.5 : 0.8}
            />
            <text
              x={CHART_LEFT - 4}
              y={yOf(g) + 4}
              textAnchor="end"
              fontSize="10"
              fill="var(--muted)"
            >
              {g === 0 ? "0%" : g === 1 ? "100%" : `${g * 100}%`}
            </text>
          </g>
        ))}

        {/* Area fills */}
        {drawn.map((s, i) => {
          const first = s.points[0];
          const last = s.points[s.points.length - 1];
          const linePath = s.points
            .map(
              (p, j) =>
                `${j === 0 ? "M" : "L"}${xOf(p.t, tMin, tSpan).toFixed(1)},${yOf(p.p).toFixed(1)}`,
            )
            .join("");
          const areaPath = `${linePath}L${xOf(last.t, tMin, tSpan).toFixed(1)},${CHART_BOTTOM}L${xOf(first.t, tMin, tSpan).toFixed(1)},${CHART_BOTTOM}Z`;
          return (
            <path
              key={`area-${i}`}
              d={areaPath}
              fill={`url(#chart-area-${i})`}
            />
          );
        })}

        {/* Lines */}
        {drawn.map((s, i) => {
          const color = SERIES_COLORS[i % SERIES_COLORS.length];
          const last = s.points[s.points.length - 1];
          const linePath = s.points
            .map(
              (p, j) =>
                `${j === 0 ? "M" : "L"}${xOf(p.t, tMin, tSpan).toFixed(1)},${yOf(p.p).toFixed(1)}`,
            )
            .join("");
          return (
            <g key={s.id} filter="url(#line-glow)">
              <path
                d={linePath}
                fill="none"
                stroke={color}
                strokeWidth="1.5"
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
              />
              {!hover && (
                <>
                  <circle
                    cx={xOf(last.t, tMin, tSpan)}
                    cy={yOf(last.p)}
                    r="5"
                    fill={color}
                    opacity="0.22"
                  />
                  <circle
                    cx={xOf(last.t, tMin, tSpan)}
                    cy={yOf(last.p)}
                    r="3"
                    fill={color}
                  />
                </>
              )}
            </g>
          );
        })}

        {/* Hover crosshair */}
        {hover && (
          <>
            <line
              x1={hover.svgX}
              x2={hover.svgX}
              y1={CHART_TOP}
              y2={CHART_BOTTOM}
              stroke="var(--border-bright)"
              strokeWidth="1"
              strokeDasharray="2 3"
            />
            {hover.values.map((v, i) => (
              <circle
                key={i}
                cx={hover.svgX}
                cy={yOf(v.price)}
                r="3.5"
                fill={v.color}
                stroke="var(--panel)"
                strokeWidth="1.5"
              />
            ))}
          </>
        )}
      </svg>

      {/* Time axis labels */}
      <div
        className="flex items-center justify-between border-t border-edge py-1 text-[9px] text-muted/70"
        style={{
          paddingLeft: `${(CHART_LEFT / W) * 100}%`,
          paddingRight: `${(PAD_R / W) * 100}%`,
        }}
      >
        {timeTicks.map((t, i) => (
          <span
            key={i}
            className={
              i === 0
                ? "text-left"
                : i === timeTicks.length - 1
                  ? "text-right"
                  : "text-center"
            }
          >
            {shortStamp(t)}
          </span>
        ))}
      </div>
    </div>
  );
}
