"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";
import { useMarketSocket, type Quote } from "@/hooks/useMarketSocket";
import { useTerminal } from "@/components/TerminalProvider";
import { Empty } from "@/components/ui/Panel";
import { cents, compact, dirClass, signed, timeToExpiry, truncate } from "@/lib/format";
import { rowVariants, staggerContainer, tapScale } from "@/lib/motion";
import type { Market } from "@/lib/types";

export type GridColumn =
  | "last"
  | "bid"
  | "ask"
  | "spread"
  | "chg1h"
  | "chg24h"
  | "chg1w"
  | "vol24h"
  | "volume"
  | "liquidity"
  | "expiry";

const DEFAULT_COLUMNS: GridColumn[] = [
  "last",
  "bid",
  "ask",
  "spread",
  "chg1h",
  "chg24h",
  "vol24h",
  "liquidity",
  "expiry",
];

/**
 * Column geometry, including when a column earns its width.
 *
 * The `width` string is consumed by the header AND by every row cell, so the
 * responsive visibility rides along with it and there is exactly one place to
 * change a column's behaviour.
 *
 * Below `sm` the grid keeps only price and change. Everything else is context:
 * useful on a desk, not worth the horizontal scroll on a phone. Before this,
 * the grid forced a 720px floor at every width, so on a 390px screen a "Top
 * Gainers" table showed the market name and nothing else — the move itself,
 * which is the entire point of the list, sat outside the scroller.
 */
const DESK_ONLY = "hidden sm:block";

const COLUMN_META: Record<GridColumn, { label: string; width: string; title: string }> = {
  last: { label: "LAST", width: "w-[52px]", title: "Last traded probability, in cents" },
  bid: { label: "BID", width: `w-[48px] ${DESK_ONLY}`, title: "Best bid" },
  ask: { label: "ASK", width: `w-[48px] ${DESK_ONLY}`, title: "Best ask" },
  spread: { label: "SPRD", width: `w-[44px] ${DESK_ONLY}`, title: "Bid/ask spread in cents" },
  chg1h: { label: "1H", width: "w-[48px]", title: "Change over 1 hour, in probability points" },
  chg24h: { label: "24H", width: "w-[52px]", title: "Change over 24 hours, in probability points" },
  chg1w: { label: "1W", width: "w-[52px]", title: "Change over 1 week, in probability points" },
  vol24h: { label: "VOL 24H", width: `w-[64px] ${DESK_ONLY}`, title: "24-hour notional volume" },
  volume: { label: "VOL TOT", width: `w-[64px] ${DESK_ONLY}`, title: "Lifetime notional volume" },
  liquidity: { label: "LIQ", width: `w-[60px] ${DESK_ONLY}`, title: "Resting book liquidity" },
  expiry: { label: "EXPIRY", width: `w-[64px] ${DESK_ONLY}`, title: "Time to resolution" },
};

/**
 * The terminal's workhorse table.
 *
 * Rows are quoted from REST on load and then overlaid with live websocket
 * top-of-book, so the grid keeps ticking between the parent's poll cycles.
 * Keyboard navigation is vi-ish and works without the table ever taking focus
 * away from the command line.
 */
// Which mounted grid receives global key events. Most screens show exactly one
// grid, so a lone grid is always active — this keeps the "keys work without
// focusing the table" model. MoversScreen shows two at once; there the hovered
// grid wins, falling back to the first-mounted one so keyboard-only use still
// drives a single, predictable grid instead of both in lockstep.
let gridSeq = 0;
const gridsMounted = new Set<number>();
let hoveredGridId: number | null = null;

export function MarketGrid({
  markets,
  columns = DEFAULT_COLUMNS,
  live = true,
  showRank = true,
  emptyText = "no markets",
  minWidthClass = "lg:min-w-[720px]",
}: {
  markets: Market[];
  columns?: GridColumn[];
  live?: boolean;
  showRank?: boolean;
  emptyText?: string;
  /**
   * Desk-width floor for the row. The default suits the full column set in a
   * full-width panel; a caller showing fewer columns, or placing two grids side
   * by side, needs a smaller one. A viewport breakpoint cannot see how wide the
   * grid's *container* is, so the caller has to say.
   *
   * Gated at `lg`, not `sm`: the sidebar takes ~200px, so an 820px tablet
   * leaves the grid 618px and a 720px floor overflowed it on Sectors, Search,
   * Monitor and Portfolio alike. `lg` is the first width where the panel is
   * genuinely wider than the floor it is being given.
   */
  minWidthClass?: string;
}) {
  const { go, toggleWatch, isWatched } = useTerminal();
  const [sel, setSel] = useState(0);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  // Stable per-instance id; register this grid while mounted so the key handler
  // can tell whether it is the one that should respond.
  const gridIdRef = useRef(-1);
  if (gridIdRef.current < 0) gridIdRef.current = gridSeq++;
  const gridId = gridIdRef.current;
  useEffect(() => {
    gridsMounted.add(gridId);
    return () => {
      gridsMounted.delete(gridId);
      if (hoveredGridId === gridId) hoveredGridId = null;
    };
  }, [gridId]);

  // Only the first outcome of each market is streamed; the grid quotes "Yes",
  // and subscribing to every leg of a 30-way event would flood the socket.
  const tokenIds = useMemo(
    () => markets.map((m) => m.outcomes[0]?.tokenId).filter(Boolean) as string[],
    [markets]
  );
  const feed = useMarketSocket(tokenIds, live);

  // A shrinking result set must not leave the cursor past the last row. Clamped
  // during render so no frame highlights a row that no longer exists.
  const [prevCount, setPrevCount] = useState(markets.length);
  if (prevCount !== markets.length) {
    setPrevCount(markets.length);
    if (sel > markets.length - 1) setSel(Math.max(0, markets.length - 1));
  }

  const openMarket = useCallback(
    (m: Market) => {
      const slug = m.eventSlug || m.slug;
      go({ fn: "DES", slug, kind: m.eventSlug ? "event" : "market" }, `DES ${slug}`);
    },
    [go]
  );

  // Bound to the window rather than the table so the command line keeps focus.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Never act on a modifier chord — ⌘W/Ctrl+W closes a tab, ⌘↑ etc. are the
      // browser's; only bare keys drive the grid.
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      // Only the active grid responds. A lone grid is always active; with several
      // mounted (MoversScreen) the hovered one wins, else the first-mounted.
      const active =
        gridsMounted.size <= 1 ||
        hoveredGridId === gridId ||
        (hoveredGridId === null && Math.min(...gridsMounted) === gridId);
      if (!active) return;
      const target = e.target as HTMLElement | null;
      const typing = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA";
      // Arrow keys still drive the grid while typing; letters must not.
      const isNav = e.key === "ArrowDown" || e.key === "ArrowUp";
      if (typing && !isNav && e.key !== "Enter") return;
      if (markets.length === 0) return;

      if (isNav) {
        e.preventDefault();
        setSel((s) => {
          const next = e.key === "ArrowDown" ? s + 1 : s - 1;
          return Math.max(0, Math.min(markets.length - 1, next));
        });
        return;
      }
      if (e.key === "Enter" && !typing) {
        e.preventDefault();
        openMarket(markets[sel]);
        return;
      }
      if (!typing && (e.key === "w" || e.key === "W")) {
        const m = markets[sel];
        const tokenId = m.outcomes[0]?.tokenId;
        if (!tokenId) return;
        e.preventDefault();
        toggleWatch({
          slug: m.eventSlug || m.slug,
          label: m.groupItemTitle || m.question,
          tokenId,
          marketId: m.id,
          addedAt: Date.now(),
        });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [markets, sel, openMarket, toggleWatch, gridId]);

  // Keep the highlighted row inside the scroll viewport.
  useEffect(() => {
    const el = bodyRef.current?.querySelector<HTMLElement>(`[data-row="${sel}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  if (markets.length === 0) return <Empty text={emptyText} />;

  return (
    <motion.div
      ref={bodyRef}
      // The 720px floor is what the full column set needs; below `sm` the
      // grid drops to price and change, which fit a phone without scrolling.
      className={`min-w-0 text-tiny ${minWidthClass}`}
      variants={staggerContainer}
      initial="initial"
      animate="animate"
      onMouseEnter={() => {
        hoveredGridId = gridId;
      }}
      onMouseLeave={() => {
        if (hoveredGridId === gridId) hoveredGridId = null;
      }}
    >
      <div className="sticky top-0 z-10 flex items-center gap-1 border-b border-edge-strong bg-surface-2 px-1 py-[3px] text-[10px] tracking-wide text-accent-weak uppercase">
        {showRank ? <span className="w-[22px] shrink-0 text-right">#</span> : null}
        <span className="min-w-0 flex-1">Market</span>
        {columns.map((c) => (
          <span
            key={c}
            title={COLUMN_META[c].title}
            className={`${COLUMN_META[c].width} shrink-0 text-right`}
          >
            {COLUMN_META[c].label}
          </span>
        ))}
        <span className="w-[16px] shrink-0" />
      </div>

      {markets.map((m, i) => (
        <Row
          key={m.id || m.slug || i}
          index={i}
          market={m}
          columns={columns}
          showRank={showRank}
          selected={i === sel}
          quote={feed.quotes.get(m.outcomes[0]?.tokenId ?? "")}
          // The feed mutates maps in place; version forces the row to re-read.
          feedVersion={feed.version}
          watched={isWatched(m.outcomes[0]?.tokenId ?? "")}
          onSelect={() => setSel(i)}
          onOpen={() => openMarket(m)}
          onWatch={() => {
            const tokenId = m.outcomes[0]?.tokenId;
            if (!tokenId) return;
            toggleWatch({
              slug: m.eventSlug || m.slug,
              label: m.groupItemTitle || m.question,
              tokenId,
              marketId: m.id,
              addedAt: Date.now(),
            });
          }}
        />
      ))}
    </motion.div>
  );
}

function Row({
  index,
  market,
  columns,
  showRank,
  selected,
  quote,
  feedVersion,
  watched,
  onSelect,
  onOpen,
  onWatch,
}: {
  index: number;
  market: Market;
  columns: GridColumn[];
  showRank: boolean;
  selected: boolean;
  quote: Quote | undefined;
  feedVersion: number;
  watched: boolean;
  onSelect: () => void;
  onOpen: () => void;
  onWatch: () => void;
}) {
  // Take both sides from the same source. A price_change event often carries
  // only one side, so falling back per-side would pair a live bid with a REST
  // ask minutes older and render a crossed book that never existed.
  const live = quote?.bid !== undefined && quote?.ask !== undefined;
  const bid = live ? quote.bid : market.bestBid;
  const ask = live ? quote.ask : market.bestAsk;
  // Mid is a better mark than a stale last print once both sides are quoted.
  const last =
    quote?.last ?? (bid !== undefined && ask !== undefined ? (bid + ask) / 2 : market.last);
  const spread = bid !== undefined && ask !== undefined ? ask - bid : market.spread;
  const flash = useFlash(last, feedVersion);

  const label = market.groupItemTitle ? `${market.groupItemTitle}` : market.question;
  const context = market.groupItemTitle ? market.eventTitle : market.eventTitle;

  const cell = (c: GridColumn) => {
    switch (c) {
      case "last":
        return (
          <span
            className={`${COLUMN_META.last.width} shrink-0 text-right font-bold text-ink ${flash}`}
          >
            {cents(last)}
          </span>
        );
      case "bid":
        return (
          <span className={`${COLUMN_META.bid.width} shrink-0 text-right text-up/85`}>
            {cents(bid)}
          </span>
        );
      case "ask":
        return (
          <span className={`${COLUMN_META.ask.width} shrink-0 text-right text-down/85`}>
            {cents(ask)}
          </span>
        );
      case "spread":
        return (
          <span className={`${COLUMN_META.spread.width} shrink-0 text-right text-muted`}>
            {spread === undefined ? "--" : cents(spread)}
          </span>
        );
      case "chg1h":
      case "chg24h":
      case "chg1w": {
        const v = c === "chg1h" ? market.chg1h : c === "chg24h" ? market.chg24h : market.chg1w;
        return (
          <span className={`${COLUMN_META[c].width} shrink-0 text-right ${dirClass(v)}`}>
            {signed(v)}
          </span>
        );
      }
      case "vol24h":
        return (
          <span className={`${COLUMN_META.vol24h.width} shrink-0 text-right text-ink/80`}>
            {compact(market.volume24h)}
          </span>
        );
      case "volume":
        return (
          <span className={`${COLUMN_META.volume.width} shrink-0 text-right text-muted`}>
            {compact(market.volume)}
          </span>
        );
      case "liquidity":
        return (
          <span className={`${COLUMN_META.liquidity.width} shrink-0 text-right text-info-weak`}>
            {compact(market.liquidity)}
          </span>
        );
      case "expiry": {
        // Polymarket sets endDate to the scheduled event time, but resolution
        // can lag it by hours while the book stays open. Labelling that
        // "EXPIRED" reads as untradeable when it isn't — call it PENDING.
        const raw = timeToExpiry(market.endDate);
        const pending = raw === "EXPIRED" && market.acceptingOrders;
        return (
          <span
            className={`${COLUMN_META.expiry.width} shrink-0 text-right ${
              pending ? "text-accent-weak" : "text-muted"
            }`}
            title={pending ? "past scheduled end date, awaiting resolution" : undefined}
          >
            {pending ? "PENDING" : raw}
          </span>
        );
      }
    }
  };

  return (
    <motion.div
      data-row={index}
      variants={rowVariants}
      whileTap={tapScale}
      onClick={onSelect}
      onDoubleClick={onOpen}
      role="button"
      tabIndex={-1}
      className={`flex cursor-pointer items-center gap-1 border-b border-edge/40 px-1 py-[2px] hover:bg-surface-2 ${
        selected ? "row-sel" : ""
      }`}
    >
      {showRank ? (
        <span className="w-[22px] shrink-0 text-right text-faint">{index + 1}</span>
      ) : null}
      <span className="flex min-w-0 flex-1 items-baseline gap-1.5">
        <span className="truncate text-ink" title={market.question}>
          {truncate(label, 64)}
        </span>
        {context && context !== label ? (
          <span className="hidden truncate text-[10px] text-faint lg:inline">
            {truncate(context, 40)}
          </span>
        ) : null}
      </span>
      {columns.map((c) => (
        <span key={c} className="contents">
          {cell(c)}
        </span>
      ))}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onWatch();
        }}
        title={watched ? "Remove from watchlist (W)" : "Add to watchlist (W)"}
        className={`w-[16px] shrink-0 text-center ${
          watched ? "text-accent hover:text-accent-weak" : "text-faint hover:text-accent"
        }`}
      >
        {watched ? "★" : "☆"}
      </button>
    </motion.div>
  );
}

/** Returns a one-shot flash class whenever `value` changes. */
function useFlash(value: number | undefined, version: number): string {
  const prev = useRef(value);
  const [cls, setCls] = useState("");

  useEffect(() => {
    if (value === undefined) return;
    const before = prev.current;
    prev.current = value;
    if (before === undefined || before === value) return;
    setCls(value > before ? "flash-up" : "flash-down");
    const t = setTimeout(() => setCls(""), 460);
    return () => clearTimeout(t);
    // `version` is included so in-place feed mutations still trigger the check.
  }, [value, version]);

  return cls;
}
