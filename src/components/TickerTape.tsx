"use client";

import { useMemo } from "react";
import type { Summary } from "@/app/api/summary/route";
import { useTerminal } from "@/components/TerminalProvider";
import { cents, signed } from "@/lib/format";

/**
 * Scrolling headline tape.
 *
 * The track renders the item list twice and translates by exactly -50%, which
 * makes the loop seamless without measuring anything. Hovering pauses it so a
 * name can actually be clicked.
 */
export function TickerTape({ summary }: { summary: Summary | null }) {
  const { go } = useTerminal();

  const items = useMemo(() => {
    if (!summary?.tape?.length) return [];
    // Lead with the biggest movers — a tape of unchanged markets is dead air.
    return [...summary.tape].sort((a, b) => Math.abs(b.chg24h) - Math.abs(a.chg24h));
  }, [summary]);

  if (items.length === 0) {
    return (
      <div className="flex h-[30px] shrink-0 items-center border-t border-edge bg-surface px-2 text-[10px] tracking-widest text-faint uppercase">
        tape offline
      </div>
    );
  }

  return (
    <div className="flex h-[30px] shrink-0 items-center overflow-hidden border-t border-edge bg-surface">
      <span className="z-10 h-full shrink-0 bg-accent-soft px-2 text-[10px] leading-[30px] font-medium tracking-wide text-[#1a0e1e]">
        MOVERS
      </span>
      <div className="min-w-0 flex-1 overflow-hidden whitespace-nowrap">
        <div className="marquee-track">
          {[0, 1].map((copy) => (
            <div key={copy} className="flex shrink-0" aria-hidden={copy === 1}>
              {items.map((t) => (
                <button
                  key={`${copy}-${t.tokenId}`}
                  onClick={() => go({ fn: "DES", slug: t.slug, kind: "event" }, `DES ${t.slug}`)}
                  className="flex shrink-0 items-baseline gap-1.5 px-2.5 text-tiny hover:bg-surface-2"
                >
                  <span className="text-muted">{t.label}</span>
                  <span className="text-ink">{cents(t.price)}¢</span>
                  <span
                    className={t.chg24h > 0 ? "text-up" : t.chg24h < 0 ? "text-down" : "text-muted"}
                  >
                    {t.chg24h > 0 ? "▲" : t.chg24h < 0 ? "▼" : "•"}
                    {signed(t.chg24h)}
                  </span>
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
