import type { PricePoint } from "@/lib/polymarket";

const W = 58;
const H = 20;
const PAD = 2;

/**
 * A tiny inline price trace for a table row. Pure SVG, server-rendered — no
 * client JS. Auto-scales to the series' own min/max so small moves stay
 * legible, and colors itself by the net move over the window.
 */
export function Sparkline({ points }: { points: PricePoint[] }) {
  if (points.length < 2) {
    return <span className="inline-block w-[58px] text-center text-[10px] text-muted/30">—</span>;
  }

  const first = points[0].p;
  const last = points[points.length - 1].p;
  const up = last >= first;
  const color = up ? "var(--accent)" : "var(--red)";

  const tMin = points[0].t;
  const tSpan = Math.max(points[points.length - 1].t - tMin, 1);
  let pMin = Infinity;
  let pMax = -Infinity;
  for (const pt of points) {
    if (pt.p < pMin) pMin = pt.p;
    if (pt.p > pMax) pMax = pt.p;
  }
  const pSpan = Math.max(pMax - pMin, 0.0001);

  const x = (t: number) => PAD + ((t - tMin) / tSpan) * (W - PAD * 2);
  const y = (p: number) => PAD + (1 - (p - pMin) / pSpan) * (H - PAD * 2);

  const line = points
    .map((pt, i) => `${i === 0 ? "M" : "L"}${x(pt.t).toFixed(1)},${y(pt.p).toFixed(1)}`)
    .join("");
  const area = `${line}L${x(points[points.length - 1].t).toFixed(1)},${H}L${x(points[0].t).toFixed(1)},${H}Z`;
  const gradId = `spark-${up ? "u" : "d"}`;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width={W}
      height={H}
      className="inline-block align-middle"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.28" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gradId})`} />
      <path
        d={line}
        fill="none"
        stroke={color}
        strokeWidth="1.25"
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
      <circle cx={x(points[points.length - 1].t)} cy={y(last)} r="1.6" fill={color} />
    </svg>
  );
}
