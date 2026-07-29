"use client";

import { useMemo, useState } from "react";
import { MarketGrid } from "@/components/MarketGrid";
import { useTerminal } from "@/components/TerminalProvider";
import { Empty, ErrorBox, Loading, Panel } from "@/components/ui/Panel";
import { usePoll } from "@/hooks/usePoll";
import { clock, compact, truncate } from "@/lib/format";
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

  return (
    <div className="flex h-full min-h-0 gap-2">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-2">
        <div className="flex shrink-0 flex-wrap items-center gap-x-1 gap-y-1 border border-edge bg-surface px-1.5 py-[3px]">
          <span className="mr-1 text-[10px] tracking-wide text-info uppercase">Rank by</span>
          {SORTS.map((s) => (
            <button
              key={s.key}
              onClick={() => setSort(s.key)}
              className={`border px-1.5 py-[1px] text-[10px] tracking-wide ${
                sort === s.key
                  ? "border-accent bg-accent/8 font-medium text-accent"
                  : "border-edge text-muted hover:border-edge-strong hover:text-ink"
              }`}
            >
              {s.label}
            </button>
          ))}
          <span className="ml-auto text-[10px] text-faint">
            {rows.length} markets · as of {asOf} UTC
          </span>
        </div>

        <Panel
          title="MARKET MONITOR"
          right={
            stale ? (
              <span className="text-warn">stale</span>
            ) : markets.refreshing ? (
              "sync…"
            ) : (
              `${rows.length} rows`
            )
          }
          className="min-h-0 flex-1"
          flush
        >
          {markets.loading ? (
            <Loading text="loading board" />
          ) : markets.error && rows.length === 0 ? (
            <ErrorBox message={markets.error} />
          ) : (
            <MarketGrid markets={rows} />
          )}
        </Panel>
      </div>

      <aside className="hidden w-[286px] shrink-0 flex-col gap-2 xl:flex">
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
                  className="flex w-full items-baseline gap-1.5 border-b border-edge/40 px-1.5 py-[3px] text-left hover:bg-surface-2"
                >
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
      </aside>
    </div>
  );
}
