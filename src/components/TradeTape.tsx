"use client";

import { AnimatePresence, motion } from "motion/react";
import { useTerminal } from "@/components/TerminalProvider";
import { Empty } from "@/components/ui/Panel";
import { cents, compact, shortAddr, timeOfDay, truncate } from "@/lib/format";
import { tapeRowVariants } from "@/lib/motion";
import type { Trade } from "@/lib/types";

/**
 * Time & sales print tape.
 *
 * Notional (size x price) is the headline number rather than share count —
 * 10,000 shares at 2¢ is a $200 print and should not read as a whale. Rows are
 * tinted by side and by size so a large print is visible in peripheral vision.
 */
export function TradeTape({
  trades,
  showMarket = false,
  dense = false,
}: {
  trades: Trade[];
  /** Include the market title column — used on the consolidated tape. */
  showMarket?: boolean;
  dense?: boolean;
}) {
  const { go } = useTerminal();

  if (trades.length === 0) return <Empty text="no prints" />;

  // Scale the per-row size bar to the largest print in view, so magnitude reads
  // down the tape continuously rather than only crossing the $10k whale line.
  const maxNotional = Math.max(...trades.map((t) => t.size * t.price), 1);

  return (
    <div className="text-tiny">
      <div className="sticky top-0 z-10 flex items-center gap-1 border-b border-edge-strong bg-surface-2 px-1 py-[3px] text-[10px] tracking-wide text-accent-weak uppercase">
        <span className="w-[52px] shrink-0">Time</span>
        <span className="w-[34px] shrink-0">Side</span>
        {showMarket ? <span className="min-w-0 flex-1">Market</span> : null}
        <span className={`${showMarket ? "w-[74px]" : "min-w-0 flex-1"} shrink-0`}>Outcome</span>
        <span className="w-[48px] shrink-0 text-right">Price</span>
        <span className="w-[60px] shrink-0 text-right">Shares</span>
        <span className="w-[64px] shrink-0 text-right">Notional</span>
        <span className="hidden w-[86px] shrink-0 text-right md:inline">Trader</span>
      </div>

      <AnimatePresence initial={false}>
        {trades.map((t) => {
          const notional = t.size * t.price;
          const buy = t.side === "BUY";
          return (
            <motion.div
              key={t.id}
              variants={tapeRowVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              onClick={() => {
                if (t.slug) go({ fn: "DES", slug: t.slug, kind: "market" }, `DES ${t.slug}`);
              }}
              role={t.slug ? "button" : undefined}
              tabIndex={t.slug ? 0 : undefined}
              onKeyDown={(e) => {
                if ((e.key === "Enter" || e.key === " ") && t.slug) {
                  e.preventDefault();
                  go({ fn: "DES", slug: t.slug, kind: "market" }, `DES ${t.slug}`);
                }
              }}
              className={`relative isolate flex items-center gap-1 overflow-hidden border-b border-edge/30 px-1 ${
                dense ? "py-0" : "py-[2px]"
              } ${t.slug ? "cursor-pointer hover:bg-surface-2" : ""} ${
                // Prints above $10k get a persistent wash, not just a flash.
                notional >= 10_000 ? (buy ? "bg-up/8" : "bg-down/8") : ""
              }`}
            >
              {/* Continuous size bar, behind the row via negative z; neutral fill
                  so it encodes magnitude without spending a direction hue. */}
              <span
                aria-hidden
                className="pointer-events-none absolute inset-y-0 left-0 -z-10 bg-edge-strong/50"
                style={{ width: `${(notional / maxNotional) * 100}%` }}
              />
              <span className="w-[52px] shrink-0 text-muted">{timeOfDay(t.timestamp)}</span>
              <span className={`w-[34px] shrink-0 font-bold ${buy ? "text-up" : "text-down"}`}>
                {buy ? "BUY" : "SELL"}
              </span>
              {showMarket ? (
                <span className="min-w-0 flex-1 truncate text-ink/80" title={t.title}>
                  {truncate(t.title, 52)}
                </span>
              ) : null}
              <span
                className={`${showMarket ? "w-[74px]" : "min-w-0 flex-1"} shrink-0 truncate text-info`}
                title={t.outcome}
              >
                {t.outcome}
              </span>
              <span className="w-[48px] shrink-0 text-right text-ink">{cents(t.price)}</span>
              <span className="w-[60px] shrink-0 text-right text-muted">{compact(t.size)}</span>
              <span
                className={`w-[64px] shrink-0 text-right ${
                  notional >= 10_000 ? "font-bold text-accent" : "text-ink/85"
                }`}
              >
                ${compact(notional)}
              </span>
              <span className="hidden w-[86px] shrink-0 truncate text-right text-faint md:inline">
                {t.name ? truncate(t.name, 11) : shortAddr(t.wallet)}
              </span>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
