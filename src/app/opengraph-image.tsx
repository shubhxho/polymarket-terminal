import { ImageResponse } from "next/og";

// Social share card for the terminal. Generated at build time (no request-time
// APIs), self-contained (no external fonts or images), and styled to read like
// the app: near-black, monospace, a green accent, a rising sparkline motif and a
// faux tape strip. Everything sits inside a 72px safe margin so Twitter's
// summary_large_image crop (2:1) never clips the content.
export const alt = "PMT — Polymarket Terminal · signals, calibrated model, P2P mesh";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const BG = "#07090c";
const INK = "#e6edf3";
const MUTED = "#7d8590";
const UP = "#3fb950";
const DOWN = "#f85149";
const ACCENT = "#e3b341";
const EDGE = "#1b2027";

const mono =
  'ui-monospace, "SFMono-Regular", "SF Mono", Menlo, Consolas, "Liberation Mono", monospace';

// A gently-rising sparkline built from a normalised walk, drawn as an SVG path.
const SPARK = [30, 34, 28, 40, 44, 38, 52, 60, 56, 72, 80, 76, 92, 104, 118];
function sparkPath(w: number, h: number) {
  const max = Math.max(...SPARK);
  const min = Math.min(...SPARK);
  const step = w / (SPARK.length - 1);
  return SPARK.map((v, i) => {
    const x = (i * step).toFixed(1);
    const y = (h - ((v - min) / (max - min)) * h).toFixed(1);
    return `${i === 0 ? "M" : "L"}${x},${y}`;
  }).join(" ");
}

export default function Image() {
  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        background: BG,
        backgroundImage:
          "radial-gradient(900px 500px at 12% -5%, rgba(63,185,80,0.12), transparent 60%), radial-gradient(700px 500px at 100% 110%, rgba(227,179,65,0.08), transparent 55%)",
        color: INK,
        fontFamily: mono,
        padding: 72,
        justifyContent: "space-between",
      }}
    >
      {/* top: brand + status */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              width: 16,
              height: 16,
              borderRadius: 8,
              background: UP,
              boxShadow: `0 0 24px ${UP}`,
            }}
          />
          <div style={{ fontSize: 26, letterSpacing: 8, color: MUTED }}>POLYMARKET TERMINAL</div>
        </div>
        <div style={{ fontSize: 22, color: MUTED, display: "flex", gap: 20 }}>
          <span style={{ color: UP }}>● LIVE</span>
          <span>WSS · CLOB V2</span>
        </div>
      </div>

      {/* middle: wordmark + sparkline + tagline */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 40 }}>
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              fontSize: 200,
              fontWeight: 800,
              lineHeight: 1,
              letterSpacing: -6,
            }}
          >
            PMT<span style={{ color: ACCENT }}>.</span>
          </div>
          <svg width={300} height={140} viewBox="0 0 300 140" style={{ opacity: 0.9 }}>
            <path d={sparkPath(300, 130)} fill="none" stroke={UP} strokeWidth={5} />
            <circle cx={300} cy={0} r={7} fill={UP} />
          </svg>
        </div>
        <div style={{ display: "flex", gap: 12, fontSize: 40, color: INK }}>
          <span>Signals ·</span>
          <span style={{ color: UP }}>calibrated model</span>
          <span>· P2P mesh</span>
        </div>
        <div style={{ fontSize: 25, color: MUTED }}>
          Rule-engine detectors blended with a temperature-calibrated model, re-scored live.
        </div>
      </div>

      {/* bottom: a faux tape strip */}
      <div
        style={{
          display: "flex",
          gap: 12,
          borderTop: `1px solid ${EDGE}`,
          paddingTop: 24,
          fontSize: 24,
        }}
      >
        {[
          ["SIG", UP, "+6.2σ"],
          ["ARB", ACCENT, "2.4pt"],
          ["WHALE", INK, "$1.2M"],
          ["DRIFT", DOWN, "-1.8×"],
          ["MESH", UP, "desk 3"],
        ].map(([k, c, v]) => (
          <div
            key={k}
            style={{
              display: "flex",
              gap: 10,
              border: `1px solid ${EDGE}`,
              borderRadius: 6,
              padding: "8px 16px",
            }}
          >
            <span style={{ color: MUTED }}>{k}</span>
            <span style={{ color: c }}>{v}</span>
          </div>
        ))}
      </div>
    </div>,
    { ...size }
  );
}
