"use client";

import { motion } from "motion/react";
import { useMemo, useState } from "react";
import { MarketGrid } from "@/components/MarketGrid";
import { MarketHeatmap } from "@/components/MarketHeatmap";
import { useTerminal } from "@/components/TerminalProvider";
import { Empty, ErrorBox, Loading, Panel, Refreshing, Segmented } from "@/components/ui/Panel";
import { usePoll } from "@/hooks/usePoll";
import { clock, compact, truncate } from "@/lib/format";
import { panelVariants, staggerContainer, tapScale } from "@/lib/motion";
import type { EventSummary, Market, Trade } from "@/lib/types";

const SORTS = [
  { key: "volume24hr", label: "24H VOL" },
  { key: "volume", label: "TOT VOL" },
  { key: "liquidity", label: "LIQUIDITY" },
  { key: "startDate", label: "NEWEST" },
] as const;

/** Home screen: the full board on the left, context rails on the right. */
export default function MonitorScreen() {
  const { go } = useTerminal();
  const [sort, setSort] = useState<(typeof SORTS)[number]["key"]>("volume24hr");
  const [view, setView] = useState<"grid" | "heat">("grid");

  const markets = usePoll<Market[]>(`/api/markets?limit=120&order=${sort}`, 8000);
  const events = usePoll<EventSummary[]>("/api/events?limit=14&order=volume24hr", 30000);
  // $25k floor keeps the block rail to genuine size rather than retail chop.
  const blocks = usePoll<Trade[]>("/api/trades?limit=40&min=25000", 6000);

  const rows = markets.data ?? [];
  // A board that stopped updating but still shows numbers must say so — an
  // operator can't tell frozen prices from live ones otherwise.
  const stale = !!markets.error && rows.length > 0;

  const asOf = useMemo(
    () => (markets.updatedAt ? clock(new Date(markets.updatedAt)) : "--:--:--"),
    [markets.updatedAt]
  );

  // Top event's 24h turnover, so the leaders rail can size a magnitude bar the
  // way the board's other ranked lists do.
  const eventMaxVol = Math.max(...(events.data ?? []).map((e) => e.volume24h), 1);

  return (
    <motion.div
      className="flex h-full min-h-0 gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      <motion.div variants={panelVariants} className="flex min-h-0 min-w-0 flex-1 flex-col gap-2">
        <div className="flex shrink-0 flex-wrap items-center gap-x-1 gap-y-1 border border-edge bg-surface px-1.5 py-[3px]">
          <span className="mr-1 text-[10px] tracking-wide text-info uppercase">Rank by</span>
          {SORTS.map((s) => (
            <motion.button
              key={s.key}
              whileTap={tapScale}
              onClick={() => setSort(s.key)}
              className={`border px-1.5 py-[1px] text-[10px] tracking-wide ${
                sort === s.key
                  ? "border-accent bg-accent/8 font-medium text-accent"
                  : "border-edge text-muted hover:border-edge-strong hover:text-ink"
              }`}
            >
              {s.label}
            </motion.button>
          ))}
          <span className="ml-auto text-[10px] text-faint">
            {rows.length} markets · as of {asOf} UTC
          </span>
        </div>

        <Panel
          title="MARKET MONITOR"
          right={
            <span className="flex items-center gap-2">
              {stale ? (
                <span className="text-warn">stale</span>
              ) : markets.refreshing ? (
                <Refreshing show />
              ) : (
                <span className="text-faint">{rows.length} rows</span>
              )}
              <Segmented
                size="xs"
                value={view}
                onChange={setView}
                options={[
                  { value: "grid", label: "Grid", title: "Sortable market table" },
                  { value: "heat", label: "Heat", title: "GPU heatmap of the whole board" },
                ]}
              />
            </span>
          }
          className="min-h-0 flex-1"
          flush
        >
          {markets.loading ? (
            <Loading text="loading board" />
          ) : markets.error && rows.length === 0 ? (
            <ErrorBox message={markets.error} />
          ) : view === "heat" ? (
            <div className="h-full p-1">
              <MarketHeatmap markets={rows} />
            </div>
          ) : (
            <MarketGrid markets={rows} />
          )}
        </Panel>
      </motion.div>

      <motion.aside
        variants={panelVariants}
        className="hidden w-[286px] shrink-0 flex-col gap-2 xl:flex"
      >
        <Panel title="EVENT LEADERS" className="min-h-0 flex-1" flush>
          {events.loading ? (
            <Loading />
          ) : events.error && !events.data ? (
            <div className="p-1.5">
              <ErrorBox message={events.error} />
            </div>
          ) : (events.data?.length ?? 0) === 0 ? (
            <Empty />
          ) : (
            <div className="text-tiny">
              {events.data!.map((ev, i) => (
                <button
                  key={ev.id}
                  onClick={() => go({ fn: "DES", slug: ev.slug, kind: "event" }, `DES ${ev.slug}`)}
                  className="relative isolate flex w-full items-baseline gap-1.5 overflow-hidden border-b border-edge/40 px-1.5 py-[3px] text-left hover:bg-surface-2"
                >
                  {/* Turnover bar, scaled to the top event — same ranked-magnitude
                      idiom the order book, tape and holder lists use. */}
                  <span
                    aria-hidden
                    className="pointer-events-none absolute inset-y-0 left-0 -z-10 bg-info/8"
                    style={{ width: `${(ev.volume24h / eventMaxVol) * 100}%` }}
                  />
                  <span className="w-[16px] shrink-0 text-right text-faint">{i + 1}</span>
                  <span className="min-w-0 flex-1 truncate text-ink" title={ev.title}>
                    {truncate(ev.title, 34)}
                  </span>
                  <span className="shrink-0 text-info-weak">{ev.markets.length}m</span>
                  <span className="w-[46px] shrink-0 text-right text-ink/80">
                    ${compact(ev.volume24h)}
                  </span>
                </button>
              ))}
            </div>
          )}
        </Panel>

        <Panel title="BLOCK PRINTS · $25K+" className="min-h-0 flex-1" flush>
          {blocks.loading ? (
            <Loading />
          ) : blocks.error && !blocks.data ? (
            <div className="p-1.5">
              <ErrorBox message={blocks.error} />
            </div>
          ) : (blocks.data?.length ?? 0) === 0 ? (
            <Empty text="no blocks in window" />
          ) : (
            <div className="text-tiny">
              {blocks.data!.map((t) => (
                <button
                  key={t.id}
                  onClick={() =>
                    t.slug && go({ fn: "DES", slug: t.slug, kind: "market" }, `DES ${t.slug}`)
                  }
                  className="flex w-full items-baseline gap-1.5 border-b border-edge/40 px-1.5 py-[3px] text-left hover:bg-surface-2"
                >
                  <span
                    className={`w-[26px] shrink-0 text-[10px] font-bold ${
                      t.side === "BUY" ? "text-up" : "text-down"
                    }`}
                  >
                    {t.side === "BUY" ? "BUY" : "SEL"}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-ink/85" title={t.title}>
                    {truncate(t.title, 28)}
                  </span>
                  <span className="w-[48px] shrink-0 text-right font-bold text-accent">
                    ${compact(t.size * t.price)}
                  </span>
                </button>
              ))}
            </div>
          )}
        </Panel>
      </motion.aside>
    </motion.div>
  );
}
