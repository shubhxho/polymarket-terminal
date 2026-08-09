"use client";

import {
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  useCallback,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { PricePoint } from "@/lib/polymarket";

export interface ChartSeries {
  /** Stable unique identity for React keys (e.g. CLOB token id). */
  id: string;
  label: string;
  points: PricePoint[];
}

/** `full` pins the Y axis to 0–100%; `auto` fits the plotted range. */
export type PriceScale = "auto" | "full";

const SERIES_COLORS = ["var(--accent)", "var(--amber)", "var(--cyan)", "var(--red)", "#c084fc"];

/** Fallback width used for SSR and the first paint, before measurement. */
const FALLBACK_W = 900;
const COMPACT_W = 520;
const H_FULL = 280;
const H_COMPACT = 200;

const PAD_L = 44; // room for Y-axis labels
const PAD_R = 14;
const PAD_T = 14;
const PAD_B = 22; // room for the in-SVG time axis

/** Narrowest Y window `auto` will zoom to, in probability units. */
const MIN_Y_SPAN = 0.06;
/** Arrow-key scrub granularity across the full time domain. */
const KEY_STEPS = 60;
const N_TIME_TICKS = 5;
/** Area fills stack into mud past this many series. */
const MAX_AREA_SERIES = 2;

interface PreparedSeries {
  id: string;
  label: string;
  color: string;
  points: PricePoint[];
}

interface Geometry {
  width: number;
  height: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
  chartW: number;
  chartH: number;
}

interface HoverValue {
  id: string;
  label: string;
  color: string;
  price: number;
  /** False when `t` falls outside this series' own time range. */
  inRange: boolean;
}

interface HoverState {
  x: number;
  t: number;
  values: HoverValue[];
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
  return new Date(unixSec * 1000).toISOString().slice(0, 16).replace("T", " ").concat(" UTC");
}

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Drop garbage points (the CLOB occasionally returns nulls / out-of-range
 * prices), clamp to 0..1, sort by time and collapse duplicate timestamps.
 * Everything downstream may assume points are finite and ascending.
 */
function sanitize(points: PricePoint[]): PricePoint[] {
  const clean: PricePoint[] = [];
  for (const p of points) {
    if (!Number.isFinite(p?.t) || !Number.isFinite(p?.p)) continue;
    clean.push({ t: p.t, p: clamp(p.p, 0, 1) });
  }
  clean.sort((a, b) => a.t - b.t);
  const deduped: PricePoint[] = [];
  for (const p of clean) {
    const prev = deduped[deduped.length - 1];
    if (prev && prev.t === p.t) deduped[deduped.length - 1] = p;
    else deduped.push(p);
  }
  return deduped;
}

/**
 * Collapse points to at most ~4 per horizontal pixel (first / min / max /
 * last per column), which keeps every visible spike while cutting multi-
 * thousand-point histories down to something the SVG renderer likes.
 */
function decimate(points: PricePoint[], chartW: number, tMin: number, tSpan: number): PricePoint[] {
  const cols = Math.max(1, Math.floor(chartW));
  if (points.length <= cols * 4) return points;

  const out: PricePoint[] = [];
  let col = -1;
  let first: PricePoint | null = null;
  let last: PricePoint | null = null;
  let lo: PricePoint | null = null;
  let hi: PricePoint | null = null;

  const flush = () => {
    if (!first || !last || !lo || !hi) return;
    const bucket = [first, lo, hi, last]
      .filter((p, i, arr) => arr.indexOf(p) === i)
      .toSorted((a, b) => a.t - b.t);
    out.push(...bucket);
  };

  for (const p of points) {
    const c = Math.floor(((p.t - tMin) / tSpan) * (cols - 1));
    if (c !== col) {
      flush();
      col = c;
      first = p;
      lo = p;
      hi = p;
    }
    if (lo && p.p < lo.p) lo = p;
    if (hi && p.p > hi.p) hi = p;
    last = p;
  }
  flush();
  return out;
}

/** Linear interpolation at `t`, clamped to the series' own endpoints. */
function priceAt(points: PricePoint[], t: number): number {
  const n = points.length;
  if (n === 0) return Number.NaN;
  if (t <= points[0].t) return points[0].p;
  if (t >= points[n - 1].t) return points[n - 1].p;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const a = points[lo];
  const b = points[hi];
  const span = b.t - a.t;
  return span <= 0 ? b.p : a.p + ((t - a.t) / span) * (b.p - a.p);
}

const TICK_STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1];

/** Round tick values (1/2/5/10/25%) covering [min, max]. */
function niceTicks(min: number, max: number, target = 5): number[] {
  const raw = (max - min) / Math.max(1, target - 1);
  const step = TICK_STEPS.find((s) => s >= raw) ?? 1;
  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + step / 1000; v += step) {
    ticks.push(Math.round(v * 1000) / 1000);
  }
  if (ticks.length === 0) ticks.push(min, max);
  return ticks;
}

function fmtPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

/** Track the rendered width so 1 SVG unit === 1 CSS pixel (no distortion). */
function useElementWidth(ref: RefObject<HTMLElement | null>, fallback: number) {
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const next = Math.round(entries[0]?.contentRect.width ?? 0);
      if (next > 0) setWidth((prev) => (prev === next ? prev : next));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}

export function PriceChart({
  series,
  scale: initialScale = "auto",
}: {
  series: ChartSeries[];
  scale?: PriceScale;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const uid = useId().replace(/[^a-zA-Z0-9-]/g, "");

  const [hover, setHover] = useState<HoverState | null>(null);
  const [muted, setMuted] = useState<ReadonlySet<string>>(() => new Set());
  const [scale, setScale] = useState<PriceScale>(initialScale);

  const measured = useElementWidth(wrapRef, FALLBACK_W);

  // Colors bind to the caller's ordering so a series with no history never
  // shifts the palette of the ones after it.
  const prepared = useMemo<PreparedSeries[]>(
    () =>
      series
        .map((s, i) => ({
          id: s.id,
          label: s.label,
          color: SERIES_COLORS[i % SERIES_COLORS.length],
          points: sanitize(s.points),
        }))
        .filter((s) => s.points.length > 0),
    [series],
  );

  const visible = useMemo(() => prepared.filter((s) => !muted.has(s.id)), [prepared, muted]);

  const geo = useMemo<Geometry>(() => {
    const width = Math.max(320, measured);
    const height = width < COMPACT_W ? H_COMPACT : H_FULL;
    const left = PAD_L;
    const right = width - PAD_R;
    const top = PAD_T;
    const bottom = height - PAD_B;
    return {
      width,
      height,
      left,
      right,
      top,
      bottom,
      chartW: Math.max(1, right - left),
      chartH: Math.max(1, bottom - top),
    };
  }, [measured]);

  const domain = useMemo(() => {
    let tMin = Number.POSITIVE_INFINITY;
    let tMax = Number.NEGATIVE_INFINITY;
    let pMin = Number.POSITIVE_INFINITY;
    let pMax = Number.NEGATIVE_INFINITY;
    for (const s of visible) {
      for (const p of s.points) {
        if (p.t < tMin) tMin = p.t;
        if (p.t > tMax) tMax = p.t;
        if (p.p < pMin) pMin = p.p;
        if (p.p > pMax) pMax = p.p;
      }
    }
    if (!Number.isFinite(tMin)) {
      return { tMin: 0, tSpan: 1, yMin: 0, ySpan: 1 };
    }
    const tSpan = Math.max(tMax - tMin, 1);
    if (scale === "full") return { tMin, tSpan, yMin: 0, ySpan: 1 };

    const pad = Math.max((pMax - pMin) * 0.12, MIN_Y_SPAN / 2);
    let yMin = Math.max(0, pMin - pad);
    let yMax = Math.min(1, pMax + pad);
    if (yMax - yMin < MIN_Y_SPAN) {
      const mid = (yMin + yMax) / 2;
      yMin = Math.max(0, mid - MIN_Y_SPAN / 2);
      yMax = Math.min(1, yMin + MIN_Y_SPAN);
      yMin = Math.max(0, yMax - MIN_Y_SPAN);
    }
    return { tMin, tSpan, yMin, ySpan: Math.max(yMax - yMin, MIN_Y_SPAN) };
  }, [visible, scale]);

  const { tMin, tSpan, yMin, ySpan } = domain;

  const xOf = useCallback(
    (t: number) => geo.left + ((t - tMin) / tSpan) * geo.chartW,
    [geo, tMin, tSpan],
  );
  const yOf = useCallback(
    (p: number) => geo.bottom - ((p - yMin) / ySpan) * geo.chartH,
    [geo, yMin, ySpan],
  );
  const tOf = useCallback(
    (x: number) => tMin + ((clamp(x, geo.left, geo.right) - geo.left) / geo.chartW) * tSpan,
    [geo, tMin, tSpan],
  );

  /**
   * Path geometry is the expensive part, so it is memoized away from `hover`
   * — scrubbing re-renders only the crosshair and markers.
   */
  const paths = useMemo(
    () =>
      visible.map((s) => {
        const pts = decimate(s.points, geo.chartW, tMin, tSpan);
        const line = pts
          .map((p, i) => `${i === 0 ? "M" : "L"}${xOf(p.t).toFixed(1)},${yOf(p.p).toFixed(1)}`)
          .join("");
        const first = pts[0];
        const last = pts[pts.length - 1];
        return {
          ...s,
          line: pts.length > 1 ? line : "",
          area:
            pts.length > 1
              ? `${line}L${xOf(last.t).toFixed(1)},${geo.bottom}L${xOf(first.t).toFixed(1)},${geo.bottom}Z`
              : "",
          last,
        };
      }),
    [visible, geo, tMin, tSpan, xOf, yOf],
  );

  const yTicks = useMemo(
    () => niceTicks(yMin, yMin + ySpan, geo.height < H_FULL ? 4 : 5),
    [yMin, ySpan, geo.height],
  );

  const timeTicks = useMemo(
    () => Array.from({ length: N_TIME_TICKS }, (_, i) => tMin + (i / (N_TIME_TICKS - 1)) * tSpan),
    [tMin, tSpan],
  );

  const readAt = useCallback(
    (t: number): HoverState => ({
      x: xOf(t),
      t,
      values: visible.map((s) => ({
        id: s.id,
        label: s.label,
        color: s.color,
        price: priceAt(s.points, t),
        inRange: t >= s.points[0].t && t <= s.points[s.points.length - 1].t,
      })),
    }),
    [visible, xOf],
  );

  const handlePointer = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect || rect.width === 0) return;
      const svgX = ((e.clientX - rect.left) / rect.width) * geo.width;
      setHover(readAt(tOf(svgX)));
    },
    [geo.width, readAt, tOf],
  );

  const handleKey = useCallback(
    (e: ReactKeyboardEvent<SVGSVGElement>) => {
      const step = tSpan / KEY_STEPS;
      const cur = hover?.t ?? tMin + tSpan;
      let next: number | null = null;
      if (e.key === "ArrowLeft") next = cur - step;
      else if (e.key === "ArrowRight") next = cur + step;
      else if (e.key === "Home") next = tMin;
      else if (e.key === "End") next = tMin + tSpan;
      else if (e.key === "Escape") {
        setHover(null);
        return;
      }
      if (next === null) return;
      e.preventDefault();
      setHover(readAt(clamp(next, tMin, tMin + tSpan)));
    },
    [hover, readAt, tMin, tSpan],
  );

  const toggleSeries = useCallback((id: string) => {
    setHover(null);
    setMuted((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  if (prepared.length === 0) {
    return (
      <div
        data-testid="chart-empty"
        className="flex h-48 items-center justify-center border border-edge bg-panel text-muted"
      >
        NO PRICE HISTORY AVAILABLE
      </div>
    );
  }

  const showAreas = visible.length <= MAX_AREA_SERIES;
  const hoveredById = new Map(hover?.values.map((v) => [v.id, v]) ?? []);
  const tooltipRight = hover ? hover.x > geo.width * 0.62 : false;

  return (
    <div ref={wrapRef} className="border border-edge bg-panel panel-lit">
      {/* Legend — each entry toggles its series */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-edge px-3 py-2">
        {prepared.map((s) => {
          const isMuted = muted.has(s.id);
          const hovered = hoveredById.get(s.id);
          const firstP = s.points[0].p;
          const lastP = s.points[s.points.length - 1].p;
          const displayP =
            hovered?.inRange && Number.isFinite(hovered.price) ? hovered.price : lastP;
          const change = displayP - firstP;
          return (
            <button
              key={s.id}
              type="button"
              data-testid="legend-item"
              data-muted={isMuted ? "true" : "false"}
              aria-pressed={!isMuted}
              onClick={() => toggleSeries(s.id)}
              title={`${s.label} — click to ${isMuted ? "show" : "hide"}`}
              className={`flex items-center gap-1.5 text-[11px] transition-opacity hover:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-edge-bright ${
                isMuted ? "opacity-35" : "opacity-100"
              }`}
            >
              <span
                className="inline-block h-[3px] w-3 shrink-0 rounded-sm"
                style={{
                  background: isMuted ? "var(--muted)" : s.color,
                }}
              />
              <span className="max-w-36 truncate text-muted">{s.label.toUpperCase()}</span>
              <span
                className="shrink-0 font-bold tabular-nums"
                style={{ color: isMuted ? "var(--muted)" : s.color }}
              >
                {fmtPct(displayP)}
              </span>
              <span
                className={`shrink-0 tabular-nums text-[10px] ${change >= 0 ? "text-accent" : "text-red"}`}
              >
                {change >= 0 ? "▲" : "▼"}
                {Math.abs(change * 100).toFixed(1)}pp
              </span>
            </button>
          );
        })}

        <div className="ml-auto flex shrink-0 items-center gap-3">
          {hover && (
            <span data-testid="hover-stamp" className="text-[10px] text-muted tabular-nums">
              {fullStamp(hover.t)}
            </span>
          )}
          <button
            type="button"
            data-testid="scale-toggle"
            onClick={() => setScale((s) => (s === "auto" ? "full" : "auto"))}
            title="Toggle Y-axis between fitted and full 0–100% range"
            className="border border-edge px-1.5 py-0.5 text-[10px] text-muted tabular-nums hover:border-edge-bright hover:text-foreground focus-visible:outline focus-visible:outline-1 focus-visible:outline-edge-bright"
          >
            {scale === "auto" ? "FIT" : "0–100"}
          </button>
        </div>
      </div>

      <div className="relative">
        {/* oxlint-disable-next-line no-noninteractive-element-interactions */}
        <svg
          ref={svgRef}
          viewBox={`0 0 ${geo.width} ${geo.height}`}
          width="100%"
          height={geo.height}
          className="block w-full cursor-crosshair touch-pan-y focus-visible:outline focus-visible:outline-1 focus-visible:outline-edge-bright"
          onPointerMove={handlePointer}
          onPointerDown={handlePointer}
          onPointerLeave={() => setHover(null)}
          onKeyDown={handleKey}
          onBlur={() => setHover(null)}
          // The rule below doesn't treat `application` as an interactive role,
          // but this element genuinely is one: it owns a keyboard handler and
          // must be reachable by Tab for that handler to ever run.
          // oxlint-disable-next-line no-noninteractive-tabindex
          tabIndex={0}
          // `img` is a non-interactive role: it hides the chart's children from
          // assistive tech AND makes a focusable, arrow-key-scrubbable element
          // announce as a static image. `application` is the honest role for a
          // widget that handles its own keys — it tells the screen reader to
          // pass ArrowLeft/Right through instead of using them for browse mode,
          // which is what the scrubbing in `handleKey` needs. The live region
          // below this SVG is what actually voices the scrubbed values.
          role="application"
          aria-roledescription="Interactive price chart. Arrow keys scrub, Escape clears."
          aria-label="Price history chart"
        >
          <defs>
            {/* Phosphor glow for the plotted lines */}
            <filter id={`glow-${uid}`} x="-5%" y="-20%" width="110%" height="140%">
              <feGaussianBlur stdDeviation="1.6" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            {showAreas &&
              paths.map((s, i) => (
                <linearGradient
                  key={s.id}
                  id={`area-${uid}-${i}`}
                  x1="0"
                  y1={geo.top}
                  x2="0"
                  y2={geo.bottom}
                  gradientUnits="userSpaceOnUse"
                >
                  <stop offset="0%" stopColor={s.color} stopOpacity="0.20" />
                  <stop offset="85%" stopColor={s.color} stopOpacity="0.02" />
                </linearGradient>
              ))}
          </defs>

          {/* Y-axis grid + labels */}
          {yTicks.map((g) => {
            const edge = g <= yMin + 1e-9 || g >= yMin + ySpan - 1e-9;
            return (
              <g key={g}>
                <line
                  x1={geo.left}
                  x2={geo.right}
                  y1={yOf(g)}
                  y2={yOf(g)}
                  stroke="var(--border)"
                  strokeWidth={edge ? "0.5" : "0.8"}
                  strokeDasharray={edge ? undefined : "3 8"}
                  opacity={edge ? 0.5 : 0.8}
                />
                <text
                  x={geo.left - 6}
                  y={yOf(g) + 3.5}
                  textAnchor="end"
                  fontSize="10"
                  fill="var(--muted)"
                >
                  {`${Math.round(g * 100)}%`}
                </text>
              </g>
            );
          })}

          {showAreas &&
            paths.map(
              (s, i) =>
                s.area && (
                  <path
                    key={`area-${s.id}`}
                    data-testid="series-area"
                    d={s.area}
                    fill={`url(#area-${uid}-${i})`}
                  />
                ),
            )}

          {paths.map((s) => (
            <g key={s.id} filter={`url(#glow-${uid})`}>
              {s.line ? (
                <path
                  data-testid="series-line"
                  d={s.line}
                  fill="none"
                  stroke={s.color}
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  vectorEffect="non-scaling-stroke"
                />
              ) : null}
              {!hover && (
                <>
                  <circle
                    cx={xOf(s.last.t)}
                    cy={yOf(s.last.p)}
                    r="5"
                    fill={s.color}
                    opacity="0.22"
                  />
                  <circle
                    data-testid="series-endpoint"
                    cx={xOf(s.last.t)}
                    cy={yOf(s.last.p)}
                    r="3"
                    fill={s.color}
                  />
                </>
              )}
            </g>
          ))}

          {/* Time axis */}
          {timeTicks.map((t, i) => (
            <text
              key={t}
              x={clamp(xOf(t), geo.left, geo.right)}
              y={geo.height - 7}
              textAnchor={i === 0 ? "start" : i === N_TIME_TICKS - 1 ? "end" : "middle"}
              fontSize="9"
              fill="var(--muted)"
              opacity="0.7"
            >
              {shortStamp(t)}
            </text>
          ))}

          {/* Hover crosshair */}
          {hover && (
            <>
              <line
                data-testid="crosshair"
                x1={hover.x}
                x2={hover.x}
                y1={geo.top}
                y2={geo.bottom}
                stroke="var(--border-bright)"
                strokeWidth="1"
                strokeDasharray="2 3"
              />
              {hover.values
                .filter((v) => v.inRange && Number.isFinite(v.price))
                .map((v) => (
                  <circle
                    key={v.id}
                    data-testid="hover-marker"
                    cx={hover.x}
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

        {/* Cursor tooltip */}
        {hover && hover.values.length > 0 && (
          <div
            data-testid="chart-tooltip"
            className="pointer-events-none absolute top-2 z-10 min-w-40 border border-edge-bright bg-panel-raised/95 px-2 py-1.5 text-[10px] shadow-lg"
            style={
              tooltipRight
                ? {
                    right: `${((geo.width - hover.x) / geo.width) * 100}%`,
                    marginRight: 8,
                  }
                : { left: `${(hover.x / geo.width) * 100}%`, marginLeft: 8 }
            }
          >
            <div className="mb-1 text-muted tabular-nums">{fullStamp(hover.t)}</div>
            {hover.values
              .toSorted((a, b) => (b.price || 0) - (a.price || 0))
              .map((v) => (
                <div key={v.id} className="flex items-center gap-2 whitespace-nowrap">
                  <span
                    className="inline-block h-[3px] w-2.5 shrink-0 rounded-sm"
                    style={{ background: v.color }}
                  />
                  <span className="max-w-40 truncate text-muted">{v.label.toUpperCase()}</span>
                  <span
                    className="ml-auto shrink-0 font-bold tabular-nums"
                    style={{ color: v.color }}
                  >
                    {v.inRange && Number.isFinite(v.price) ? fmtPct(v.price) : "—"}
                  </span>
                </div>
              ))}
          </div>
        )}
      </div>

      {/* Screen-reader readout of the scrubbed values */}
      <output className="sr-only" aria-live="polite">
        {hover
          ? `${fullStamp(hover.t)}: ${hover.values
              .filter((v) => v.inRange)
              .map((v) => `${v.label} ${fmtPct(v.price)}`)
              .join(", ")}`
          : ""}
      </output>
    </div>
  );
}
