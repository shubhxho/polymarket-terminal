import { ImageResponse } from "next/og";

// Social share card for the terminal. Generated at build time (no request-time
// APIs), self-contained (no external fonts or images), and styled to read like
// the app: near-black, monospace, a green accent and a faux tape strip.
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

export default function Image() {
  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        background: BG,
        color: INK,
        fontFamily: mono,
        padding: 64,
        justifyContent: "space-between",
      }}
    >
      {/* top: brand + status pills */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              width: 18,
              height: 18,
              borderRadius: 9,
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

      {/* middle: wordmark + tagline */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            fontSize: 210,
            fontWeight: 800,
            lineHeight: 1,
            letterSpacing: -6,
          }}
        >
          PMT<span style={{ color: ACCENT }}>.</span>
        </div>
        <div style={{ display: "flex", gap: 12, fontSize: 40, color: INK }}>
          <span>Signals ·</span>
          <span style={{ color: UP }}>calibrated model</span>
          <span>· P2P mesh</span>
        </div>
        <div style={{ fontSize: 26, color: MUTED, marginTop: 6 }}>
          Rule-engine detectors blended with a temperature-calibrated model, re-scored live.
        </div>
      </div>

      {/* bottom: a faux tape strip */}
      <div
        style={{
          display: "flex",
          gap: 12,
          borderTop: `1px solid ${EDGE}`,
          paddingTop: 22,
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
