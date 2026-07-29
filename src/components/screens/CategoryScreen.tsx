"use client";

import { motion } from "motion/react";
import { MarketGrid, type GridColumn } from "@/components/MarketGrid";
import { useTerminal } from "@/components/TerminalProvider";
import { ErrorBox, Loading, Panel } from "@/components/ui/Panel";
import { usePoll } from "@/hooks/usePoll";
import { SECTORS } from "@/lib/commands";
import { clock } from "@/lib/format";
import { panelVariants, staggerContainer, tapScale } from "@/lib/motion";
import type { Market } from "@/lib/types";

/** Sector view drops SPRD in favour of the full 1h/24h/1w change ladder. */
const COLUMNS: GridColumn[] = [
  "last",
  "bid",
  "ask",
  "chg1h",
  "chg24h",
  "chg1w",
  "vol24h",
  "liquidity",
  "expiry",
];

/**
 * Sector browser: one tag's markets, ranked by turnover.
 *
 * The chip strip is the whole navigation model — a sector switch is one click
 * and re-enters through `go` so it lands in history like any typed command.
 */
export default function CategoryScreen({ tag, label }: { tag: string; label: string }) {
  const { go } = useTerminal();
  const { data, error, loading, updatedAt } = usePoll<Market[]>(
    `/api/markets?tag=${encodeURIComponent(tag)}&limit=80&order=volume24hr`,
    10000
  );

  const markets = data ?? [];
  // A failed refresh keeps the last good grid up; the header flags it instead.
  const stale = !!error && !!data;

  return (
    <motion.div
      className="flex h-full min-h-0 flex-col gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      <motion.div
        variants={panelVariants}
        className="flex shrink-0 flex-wrap items-center gap-1 border border-edge bg-surface px-1 py-[3px]"
      >
        {SECTORS.map((s) => {
          const active = s.tag === tag;
          return (
            <motion.button
              key={s.key}
              whileTap={tapScale}
              onClick={() => go({ fn: "CAT", tag: s.tag, label: s.label }, `CAT ${s.key}`)}
              title={`CAT ${s.key}`}
              className={`border px-1.5 py-[1px] text-[10px] tracking-wide uppercase ${
                active
                  ? "border-accent bg-accent/8 font-medium text-accent"
                  : "border-edge text-muted hover:border-edge-strong hover:text-ink"
              }`}
            >
              {s.key}
            </motion.button>
          );
        })}
      </motion.div>

      <motion.div variants={panelVariants} className="flex min-h-0 min-w-0 flex-1 flex-col">
        <Panel
          title={`${label} · Markets`}
          right={
            <>
              {markets.length} mkt · as of {updatedAt ? clock(new Date(updatedAt)) : "--:--:--"}
              {stale ? " · stale" : ""}
            </>
          }
          flush
          className="min-h-0 flex-1"
        >
          {loading ? (
            <Loading text={`loading ${label}`} />
          ) : error && !data ? (
            <div className="p-1.5">
              <ErrorBox message={error} />
            </div>
          ) : (
            <MarketGrid
              markets={markets}
              columns={COLUMNS}
              emptyText={`no ${label.toLowerCase()} markets`}
            />
          )}
        </Panel>
      </motion.div>
    </motion.div>
  );
}
