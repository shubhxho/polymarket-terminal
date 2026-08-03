"use client";

import { motion } from "motion/react";
import { useCallback, useMemo } from "react";
import { MarketGrid } from "@/components/MarketGrid";
import { useTerminal } from "@/components/TerminalProvider";
import { Empty, ErrorBox, Loading, Panel, Refreshing } from "@/components/ui/Panel";
import { Sparkline } from "@/components/ui/Sparkline";
import { dirClass, signed, truncate } from "@/lib/format";
import { usePoll } from "@/hooks/usePoll";
import { staggerContainer, tapScale } from "@/lib/motion";
import type { Market, PricePoint } from "@/lib/types";

type Series = { tokenId: string; points: PricePoint[] };

/**
 * `/api/history` fans out one upstream request per token and hard-caps the
 * `tokens` param at 8, so asking for more silently drops the tail. The trend
 * panel therefore charts the 8 oldest pins — the grid above still quotes every
 * entry, and a watchlist deep enough to truncate is being used as a list, not
 * as a chart wall.
 */
const TREND_LIMIT = 8;

/**
 * The user's pinned book.
 *
 * Two reads of the same set: a full quote grid for the numbers, and a stack of
 * 1-day sparklines underneath for the shape of how those numbers got there.
 */
export default function WatchlistScreen() {
  const { watchlist, toggleWatch, toast } = useTerminal();

  // Keys, not the array itself: the poller must not be rebuilt just because a
  // toast or a nav change re-rendered the provider.
  const idKey = watchlist
    .map((w) => w.marketId)
    .filter(Boolean)
    .join(",");
  const tokenKey = watchlist
    .slice(0, TREND_LIMIT)
    .map((w) => w.tokenId)
    .filter(Boolean)
    .join(",");

  const marketsUrl = useMemo(() => {
    if (!idKey) return null;
    const params = new URLSearchParams();
    for (const id of idKey.split(",")) params.append("id", id);
    return `/api/markets?${params.toString()}`;
  }, [idKey]);

  const historyUrl = useMemo(() => {
    if (!tokenKey) return null;
    const params = new URLSearchParams({ tokens: tokenKey, interval: "1d" });
    return `/api/history?${params.toString()}`;
  }, [tokenKey]);

  const { data, error, loading, refreshing } = usePoll<Market[]>(marketsUrl, 8000);
  const history = usePoll<Series[]>(historyUrl, 8000);

  const markets = useMemo(() => {
    const byId = new Map((data ?? []).map((m) => [m.id, m]));
    // Render in pin order, not upstream order, so the grid stops reshuffling.
    return watchlist.map((w) => byId.get(w.marketId)).filter((m): m is Market => m !== undefined);
  }, [data, watchlist]);

  const seriesByToken = useMemo(() => {
    const map = new Map<string, PricePoint[]>();
    for (const s of history.data ?? []) map.set(s.tokenId, s.points);
    return map;
  }, [history.data]);

  const clearAll = useCallback(() => {
    const n = watchlist.length;
    if (n === 0) return;
    // `toggleWatch` is a toggle keyed on tokenId — replaying every pinned item
    // through it removes each one, and its functional setState makes the
    // sequential calls compose.
    for (const w of watchlist) toggleWatch(w);
    toast(`watchlist cleared · ${n} ${n === 1 ? "entry" : "entries"}`);
  }, [watchlist, toggleWatch, toast]);

  if (watchlist.length === 0) {
    return (
      <motion.div
        className="flex h-full min-h-0 flex-col gap-2"
        variants={staggerContainer}
        initial="initial"
        animate="animate"
      >
        <Panel title="Watchlist" flush className="min-h-0 flex-1" animate>
          <Empty text="watchlist empty — press W on any market row to pin it" />
        </Panel>
      </motion.div>
    );
  }

  const trend = watchlist.slice(0, TREND_LIMIT);

  return (
    <motion.div
      className="flex h-full min-h-0 flex-col gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      <Panel
        title="Watchlist"
        flush
        className="min-h-0 flex-1"
        animate
        right={
          <span className="flex items-center gap-2">
            <span>
              {watchlist.length} {watchlist.length === 1 ? "item" : "items"}
            </span>
            <Refreshing show={refreshing} />
            <motion.button
              whileTap={tapScale}
              onClick={clearAll}
              title="Unpin every market"
              className="rounded-sm border border-edge px-1.5 text-[10px] font-medium uppercase hover:border-accent-weak hover:text-accent"
            >
              Clear All
            </motion.button>
          </span>
        }
      >
        {loading ? (
          <Loading text="quoting" />
        ) : error && !data ? (
          <div className="p-1.5">
            <ErrorBox message={error} />
          </div>
        ) : (
          <MarketGrid
            markets={markets}
            columns={[
              "last",
              "bid",
              "ask",
              "spread",
              "chg1h",
              "chg24h",
              "chg1w",
              "vol24h",
              "liquidity",
              "expiry",
            ]}
            showRank={false}
            emptyText="pinned markets unavailable"
          />
        )}
      </Panel>

      <Panel
        title="Trend · 1D"
        flush
        className="max-h-[38%] min-h-0 shrink-0"
        animate
        right={`${trend.length}/${watchlist.length}`}
      >
        {history.loading ? (
          <Loading text="loading history" />
        ) : history.error && !history.data ? (
          <div className="p-1.5">
            <ErrorBox message={history.error} />
          </div>
        ) : (
          <div className="text-tiny">
            {trend.map((w) => {
              const points = seriesByToken.get(w.tokenId) ?? [];
              const net =
                points.length >= 2
                  ? // Probability points, matching the grid's change columns.
                    (points[points.length - 1].p - points[0].p) * 100
                  : undefined;
              return (
                <div
                  key={w.tokenId}
                  className="flex items-center gap-2 border-b border-edge/40 px-1 py-[2px] last:border-0"
                >
                  <span className="min-w-0 flex-1 truncate text-ink" title={w.label}>
                    {truncate(w.label, 56)}
                  </span>
                  <Sparkline points={points} className="shrink-0" />
                  <span className={`w-[52px] shrink-0 text-right ${dirClass(net)}`}>
                    {signed(net)}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </Panel>
    </motion.div>
  );
}
