"use client";

import { motion } from "motion/react";
import type { Summary } from "@/app/api/summary/route";
import { useClock } from "@/hooks/useClock";
import { useTerminal } from "@/components/TerminalProvider";
import { WalletButton } from "@/components/WalletButton";
import { cn } from "@/lib/cn";
import { screenTitle } from "@/lib/commands";
import { clock, compact } from "@/lib/format";
import { tapScale, transition } from "@/lib/motion";
import type { PollState } from "@/hooks/usePoll";

/**
 * Masthead: wordmark and current screen on the left, market breadth in the
 * middle, feed health and clock on the right.
 *
 * Fixed height so the workspace never reflows, and everything in it is
 * secondary by design — the tables below are the product.
 */
export function TopBar({
  summary,
  theme,
  onToggleTheme,
}: {
  summary: PollState<Summary>;
  theme: "light" | "dark";
  onToggleTheme: () => void;
}) {
  const { screen } = useTerminal();
  const now = useClock();
  const s = summary.data;

  const breadth = s ? s.advancers + s.decliners + s.unchanged : 0;
  const advPct = s && breadth > 0 ? (s.advancers / breadth) * 100 : 0;
  const decPct = s && breadth > 0 ? (s.decliners / breadth) * 100 : 0;

  return (
    <motion.header
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={transition}
      className="flex h-[46px] shrink-0 items-center gap-3 border-b border-edge bg-canvas px-3"
    >
      <div className="flex shrink-0 items-center gap-2">
        <span className="flex h-[22px] w-[22px] items-center justify-center rounded-md bg-ink text-[11px] font-bold text-canvas">
          P
        </span>
        <span className="text-sm2 font-semibold tracking-[-0.01em]">Polymarket Terminal</span>
      </div>

      <span className="h-4 w-px shrink-0 bg-edge" />

      <span className="shrink-0 truncate text-tiny text-muted">{screenTitle(screen)}</span>

      <div className="hidden min-w-0 flex-1 items-center justify-center gap-5 lg:flex">
        <Stat label="24h volume" value={s ? `$${compact(s.volume24h)}` : "—"} />
        <Stat label="Liquidity" value={s ? `$${compact(s.liquidity)}` : "—"} />
        <div className="flex items-center gap-2" title="Advancers vs decliners">
          <span className="text-[11px] text-faint">A/D</span>
          <span className="text-tiny font-medium text-up">{s?.advancers ?? "—"}</span>
          <span className="flex h-[6px] w-[72px] overflow-hidden rounded-full bg-edge">
            <span className="bg-up" style={{ width: `${advPct}%` }} />
            <span style={{ width: `${100 - advPct - decPct}%` }} />
            <span className="bg-down" style={{ width: `${decPct}%` }} />
          </span>
          <span className="text-tiny font-medium text-down">{s?.decliners ?? "—"}</span>
        </div>
      </div>

      <div className="ml-auto flex shrink-0 items-center gap-3">
        <FeedLight state={summary} />
        <span className="hidden text-tiny text-muted sm:inline">
          {now ? `${clock(now)} UTC` : "—:—:— UTC"}
        </span>
        <span className="h-4 w-px shrink-0 bg-edge" />
        <WalletButton />
        <motion.button
          whileTap={tapScale}
          onClick={onToggleTheme}
          title={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          aria-label="Toggle theme"
          className="flex h-[24px] w-[24px] items-center justify-center rounded-md border border-edge text-muted hover:border-edge-strong hover:text-ink"
        >
          {theme === "light" ? "☾" : "☀"}
        </motion.button>
      </div>
    </motion.header>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-[11px] text-faint">{label}</span>
      <span className="text-tiny font-medium">{value}</span>
    </span>
  );
}

function FeedLight({ state }: { state: PollState<Summary> }) {
  const tone = state.error
    ? { color: "text-down", label: "Feed down" }
    : state.loading
      ? { color: "text-warn", label: "Connecting" }
      : { color: "text-up", label: "Live" };
  return (
    <span className="flex items-center gap-1.5" title={state.error ?? "Market data feed"}>
      <span className={cn("dot", tone.color, state.refreshing && "animate-pulse")} />
      <span className="text-[11px] text-muted">{tone.label}</span>
    </span>
  );
}
