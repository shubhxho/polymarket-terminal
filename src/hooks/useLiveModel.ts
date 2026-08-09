"use client";

import { useEffect, useRef, useState } from "react";
import type { MarketFeed } from "./useMarketSocket";
import { appendTick, liveModelFor, type LiveRead } from "@/lib/liveModel";
import type { MarketSignals } from "@/lib/signals";

/** Live model reads keyed by market id. Absent for a market means "no live tick
 *  yet" — the caller falls back to the value the poll already carried. */
export type LiveModelMap = ReadonlyMap<string, LiveRead>;

const EMPTY: LiveModelMap = new Map();

/**
 * Re-score every ranked market's model live off the socket, between polls.
 *
 * Seeds a rolling price buffer per market from the window the scan shipped
 * (`row.recent`), appends the live last-trade on each socket tick, and re-runs
 * the forward pass. The socket already coalesces bursts behind a single
 * `version` bump (one commit per animation frame), so reacting to `version` is
 * all the batching this needs — no timer, no worker.
 *
 * Order is intentionally left to the poll: the returned reads update the model
 * column and the header counts in place, but the table does not reflow under the
 * cursor every time a price moves. A fresh scan re-seeds the buffers and clears
 * the live overlay so a market that dropped out of the ranking drops its buffer
 * with it.
 */
export function useLiveModel(rows: readonly MarketSignals[], feed: MarketFeed): LiveModelMap {
  const tailsRef = useRef<Map<string, number[]>>(new Map());
  const [live, setLive] = useState<LiveModelMap>(EMPTY);

  // Re-seed whenever a new scan lands (rows identity changes on each poll).
  useEffect(() => {
    const seeded = new Map<string, number[]>();
    for (const r of rows) {
      if (r.recent && r.recent.length > 0) seeded.set(r.market.id, r.recent.slice());
    }
    tailsRef.current = seeded;
    setLive(EMPTY);
  }, [rows]);

  // Recompute on every coalesced socket tick.
  useEffect(() => {
    if (rows.length === 0) return;
    const tails = tailsRef.current;
    const next = new Map<string, LiveRead>();
    for (const r of rows) {
      const token = r.market.outcomes[0]?.tokenId;
      const seed = token ? tails.get(r.market.id) : undefined;
      if (!token || !seed) continue;
      const px = feed.quotes.get(token)?.last;
      if (px === undefined) continue;
      const advanced = appendTick(seed, px);
      if (advanced !== seed) tails.set(r.market.id, advanced);
      const read = liveModelFor(r, advanced);
      if (read) next.set(r.market.id, read);
    }
    if (next.size > 0) setLive(next);
    // `feed.quotes` is a fresh Map on every coalesced commit, so it and
    // `feed.version` move together; both are listed for the exhaustive-deps rule.
  }, [feed.version, feed.quotes, rows]);

  return live;
}
