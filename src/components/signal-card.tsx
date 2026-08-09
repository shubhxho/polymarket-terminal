import Link from "next/link";
import { fmtUsd } from "@/lib/polymarket";
import type { EdgeSignal } from "@/lib/signals-plus";

/** Signed basis points → percent string, e.g. -305 → "-3.05%". */
export function fmtEdge(bps: number): string {
  const pct = bps / 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

interface KindStyle {
  kindColor: string;
  edgeColor: string;
  barColor: string;
  accentBar: string;
  chip: string | null;
  chipClass: string;
  edgeLabel: string;
}

/** Per-kind colour + label logic, kept honest (green only when truly good). */
function styleFor(s: EdgeSignal): KindStyle {
  const positive = s.edgeBps > 0;
  if (s.kind === "ARB") {
    return {
      kindColor: "text-cyan",
      // Green only for a buyable underround; an overround (vig) is amber.
      edgeColor: positive ? "text-accent" : "text-amber",
      barColor: "bg-cyan/60",
      accentBar: "before:bg-cyan/60",
      chip: positive ? "BUYABLE" : null,
      chipClass: "bg-accent/15 text-accent",
      edgeLabel: "EDGE",
    };
  }
  if (s.kind === "LIQUIDITY") {
    return {
      kindColor: "text-amber",
      edgeColor: "text-amber",
      barColor: "bg-amber/60",
      accentBar: "before:bg-amber/60",
      chip: "MAKER",
      chipClass: "bg-amber/15 text-amber",
      edgeLabel: "SPREAD",
    };
  }
  if (s.kind === "RESOLUTION") {
    return {
      kindColor: "text-foreground",
      edgeColor: "text-foreground",
      barColor: "bg-foreground/50",
      accentBar: "before:bg-foreground/40",
      chip: "SOON",
      chipClass: "bg-foreground/10 text-foreground/80",
      edgeLabel: "LIVE",
    };
  }
  // MOMENTUM — directional
  return {
    kindColor: positive ? "text-accent" : "text-red",
    edgeColor: positive ? "text-accent" : "text-red",
    barColor: positive ? "bg-accent/60" : "bg-red/60",
    accentBar: positive ? "before:bg-accent/60" : "before:bg-red/60",
    chip: null,
    chipClass: "",
    edgeLabel: "MOVE",
  };
}

/** Score at which a signal is worth drawing the eye to. */
const PULSE_SCORE = 70;

/**
 * One edge-scanner result. Shared by the on-board EDGE RADAR strip and the
 * full-board /signals scanner so their look stays in lockstep.
 */
export function SignalCard({ signal: s, rank }: { signal: EdgeSignal; rank?: number }) {
  const st = styleFor(s);
  // Pulse only when the signal is BOTH strong and actually liftable. Pulsing a
  // high score that dies at the touch would be drawing the eye to nothing —
  // which is worse than not drawing it at all.
  const live = s.score >= PULSE_SCORE && (s.executableBps ?? 0) > 0;

  // Spread as a rough execution-quality read: tight books are cheaper to trade.
  const spreadPct = s.spreadBps != null ? s.spreadBps / 100 : null;
  const spreadTone =
    spreadPct == null
      ? "text-muted/40"
      : spreadPct <= 1
        ? "text-accent/70"
        : spreadPct <= 3
          ? "text-amber/70"
          : "text-red/70";
  // |edge| relative to a 6% reference, purely for the inline magnitude bar.
  const edgeMag = Math.min((Math.abs(s.edgeBps) / 600) * 100, 100);

  return (
    <Link
      href={`/event/${s.slug}`}
      className={`group relative flex flex-col gap-1.5 bg-panel px-3 py-2.5 pl-4 transition-colors hover:bg-panel-raised before:absolute before:inset-y-0 before:left-0 before:w-0.5 before:transition-all group-hover:before:w-1 ${st.accentBar}`}
    >
      <div className="flex items-center justify-between gap-2 text-[10px] tracking-widest">
        <span className="flex items-center gap-1.5">
          {live && (
            <span
              className="signal-pulse inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-accent"
              title="Strong score and still liftable after crossing the spread"
            />
          )}
          {rank != null && (
            <span className="tabular-nums text-muted/40">{String(rank).padStart(2, "0")}</span>
          )}
          <span className={`rounded-sm border border-current/30 px-1 py-0.5 ${st.kindColor}`}>
            {s.kind}
          </span>
          {st.chip && (
            <span className={`rounded-sm px-1 py-0.5 text-[9px] ${st.chipClass}`}>{st.chip}</span>
          )}
        </span>
        <span className="flex items-center gap-1.5 text-muted/60">
          SCORE
          <span className="font-bold tabular-nums text-foreground">{Math.round(s.score)}</span>
        </span>
      </div>

      <p className="truncate text-xs text-foreground group-hover:text-accent">{s.title}</p>

      {/* Score bar */}
      <div className="h-1 w-full overflow-hidden rounded-sm bg-panel-raised ring-1 ring-inset ring-edge">
        <div className={`h-full ${st.barColor}`} style={{ width: `${Math.min(s.score, 100)}%` }} />
      </div>

      <div className="flex items-center justify-between gap-2 text-[10px] tabular-nums">
        <span className="flex items-center gap-1.5">
          <span className={`font-bold ${st.edgeColor}`}>
            {st.edgeLabel} {fmtEdge(s.edgeBps)}
          </span>
          {/* Inline edge-magnitude bar */}
          <span className="hidden h-1 w-10 overflow-hidden rounded-sm bg-panel-raised sm:inline-block">
            <span className={`block h-full ${st.barColor}`} style={{ width: `${edgeMag}%` }} />
          </span>
        </span>
        <span className={spreadTone}>
          {spreadPct != null ? `${spreadPct.toFixed(1)}% SPR` : "NO BOOK"}
        </span>
      </div>

      {/* Does the edge survive crossing the spread? This is the line that
          separates a mispricing from a trade. */}
      {s.executableBps != null && (
        <div className="flex items-center justify-between gap-2 text-[10px] tabular-nums">
          <span className={s.executableBps > 0 ? "text-accent" : "text-red/80"}>
            {s.executableBps > 0 ? "LIFTABLE" : "DEAD AT TOUCH"} {fmtEdge(s.executableBps)}
          </span>
          <span className="text-muted/40">AFTER SPREAD</span>
        </div>
      )}

      {/* Execution context — how tradeable this actually is */}
      <div className="flex items-center justify-between gap-2 text-[10px] tabular-nums text-muted/50">
        <span>LIQ {fmtUsd(s.liquidity)}</span>
        <span>VOL {fmtUsd(s.volume24h)}</span>
      </div>

      <p className="truncate text-[10px] text-muted/70">{s.detail}</p>
    </Link>
  );
}
