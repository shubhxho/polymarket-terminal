"use client";

import { motion } from "motion/react";
import type { ReactNode } from "react";
import { MarketGrid } from "@/components/MarketGrid";
import { useTerminal } from "@/components/TerminalProvider";
import { Empty, ErrorBox, Loading, Panel, Refreshing } from "@/components/ui/Panel";
import { usePoll } from "@/hooks/usePoll";
import { compact, timeToExpiry } from "@/lib/format";
import { fuzzyMatch, highlight } from "@/lib/fuzzy";
import { panelVariants, staggerContainer } from "@/lib/motion";
import type { EventSummary, Market } from "@/lib/types";

type SearchResult = { events: EventSummary[]; markets: Market[] };

/**
 * Full-text search across events and markets.
 *
 * Events and markets are two different answers to the same question — "where is
 * this theme traded?" — so they sit side by side rather than in one merged list:
 * the left rail is the coarse index, the right pane is the quotable instrument.
 */
export default function SearchScreen({ q }: { q: string }) {
  const query = q.trim();
  const url = query ? `/api/search?q=${encodeURIComponent(query)}&limit=20` : null;
  const { data, error, loading, refreshing } = usePoll<SearchResult>(url, 20000);

  const events = data?.events ?? [];
  const markets = data?.markets ?? [];
  // A failed refresh keeps the last good hits on screen; the header says so.
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
        className="flex shrink-0 items-center gap-2 border border-edge bg-surface px-1.5 py-[3px] text-[10px] tracking-wide uppercase"
      >
        <span className="shrink-0 text-info">Query</span>
        <span className="min-w-0 truncate text-accent">{query ? `"${query}"` : "—"}</span>
        <span className="ml-auto flex shrink-0 items-center gap-3 text-muted">
          <span>
            <span className="text-info">EVT</span> {events.length}
          </span>
          <span>
            <span className="text-info">MKT</span> {markets.length}
          </span>
          {stale ? <span className="text-warn">stale</span> : null}
          <Refreshing show={refreshing} />
        </span>
      </motion.div>

      <motion.div
        variants={staggerContainer}
        className="grid min-h-0 flex-1 grid-cols-1 gap-2 overflow-y-auto lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)] lg:overflow-visible"
      >
        <Panel title="Events" right={`${events.length}`} flush className="min-h-0" animate>
          <Body loading={loading} error={error} hasData={!!data}>
            {!query ? (
              <Empty text="enter a query" />
            ) : events.length === 0 ? (
              <Empty text="no events" />
            ) : (
              <EventList events={events} query={query} />
            )}
          </Body>
        </Panel>

        <Panel title="Markets" right={`${markets.length}`} flush className="min-h-0" animate>
          <Body loading={loading} error={error} hasData={!!data}>
            {!query ? <Empty text="enter a query" /> : <MarketGrid markets={markets} />}
          </Body>
        </Panel>
      </motion.div>
    </motion.div>
  );
}

/** The three non-data states, resolved once so both panels agree. */
function Body({
  loading,
  error,
  hasData,
  children,
}: {
  loading: boolean;
  error: string | null;
  hasData: boolean;
  children: ReactNode;
}) {
  if (loading) return <Loading text="searching" />;
  if (error && !hasData)
    return (
      <div className="p-1.5">
        <ErrorBox message={error} />
      </div>
    );
  return <>{children}</>;
}

function EventList({ events, query }: { events: EventSummary[]; query: string }) {
  const { go } = useTerminal();

  return (
    <div className="min-w-[320px] text-tiny">
      <div className="sticky top-0 z-10 flex items-center gap-1 border-b border-edge-strong bg-surface-2 px-1 py-[3px] text-[10px] tracking-wide text-accent-weak uppercase">
        <span className="min-w-0 flex-1">Event</span>
        <span className="w-[30px] shrink-0 text-right" title="Markets in the event">
          MKT
        </span>
        <span className="w-[56px] shrink-0 text-right" title="24-hour notional volume">
          VOL 24H
        </span>
        <span className="w-[52px] shrink-0 text-right" title="Resting book liquidity">
          LIQ
        </span>
        <span className="w-[62px] shrink-0 text-right" title="Time to resolution">
          EXPIRY
        </span>
      </div>

      {events.map((ev) => (
        <div
          key={ev.id || ev.slug}
          onClick={() => go({ fn: "DES", slug: ev.slug, kind: "event" }, `DES ${ev.slug}`)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              go({ fn: "DES", slug: ev.slug, kind: "event" }, `DES ${ev.slug}`);
            }
          }}
          role="button"
          tabIndex={0}
          className="flex cursor-pointer items-center gap-1 border-b border-edge/40 px-1 py-[2px] hover:bg-surface-2"
        >
          <span className="min-w-0 flex-1 truncate text-ink" title={ev.title}>
            {/* Bold the matched characters, the way every search result list
                (and our own command palette) shows why a row was returned. */}
            {highlight(ev.title, fuzzyMatch(query, ev.title)?.positions ?? []).map((part, k) => (
              <span key={k} className={part.hit ? "font-semibold text-accent" : undefined}>
                {part.text}
              </span>
            ))}
          </span>
          <span className="w-[30px] shrink-0 text-right text-muted">{ev.markets.length}</span>
          <span className="w-[56px] shrink-0 text-right text-ink/80">{compact(ev.volume24h)}</span>
          <span className="w-[52px] shrink-0 text-right text-info-weak">
            {compact(ev.liquidity)}
          </span>
          <span className="w-[62px] shrink-0 text-right text-muted">
            {timeToExpiry(ev.endDate)}
          </span>
        </div>
      ))}
    </div>
  );
}
