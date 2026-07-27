"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { BookLevel, OrderBook } from "@/lib/types";

const WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market";

export type Quote = {
  tokenId: string;
  bid?: number;
  ask?: number;
  last?: number;
  /** Direction of the most recent change, for tick-flash rendering. */
  dir: 0 | 1 | -1;
  ts: number;
};

export type SocketStatus = "idle" | "connecting" | "live" | "retrying" | "closed";

export type MarketFeed = {
  status: SocketStatus;
  books: Map<string, OrderBook>;
  quotes: Map<string, Quote>;
  /** Increments on every flushed batch; use as a render trigger. */
  version: number;
};

type RawLevel = { price?: string; size?: string };

function levels(raw: unknown, dir: "bid" | "ask"): BookLevel[] {
  const arr = Array.isArray(raw) ? (raw as RawLevel[]) : [];
  return arr
    .map((l) => ({ price: parseFloat(l.price ?? "0"), size: parseFloat(l.size ?? "0") }))
    .filter((l) => Number.isFinite(l.price) && l.size > 0)
    .sort((a, b) => (dir === "bid" ? b.price - a.price : a.price - b.price));
}

/**
 * Subscribes to the Polymarket CLOB market channel for a set of outcome tokens
 * and maintains live books + top-of-book quotes.
 *
 * Design notes:
 * - The feed can burst dozens of messages per second. Mutations land in refs
 *   and are flushed on an animation frame, so React re-renders at most once per
 *   frame no matter how hot the market is.
 * - The upstream socket has no unsubscribe verb, so changing the token set
 *   means tearing down and reopening the connection. `tokenIds` is joined into
 *   a string key to make that dependency stable across array identity changes.
 * - Reconnects use exponential backoff capped at 15s to avoid hammering the
 *   endpoint when it is down.
 */
export function useMarketSocket(tokenIds: string[], enabled = true): MarketFeed {
  const key = useMemo(
    () => Array.from(new Set(tokenIds.filter(Boolean))).sort().join(","),
    [tokenIds]
  );

  const active = enabled && key.length > 0;

  // Mutable accumulators. Bursts land here; React only ever sees the frozen
  // snapshots published on an animation frame.
  const booksRef = useRef(new Map<string, OrderBook>());
  const quotesRef = useRef(new Map<string, Quote>());

  const [snapshot, setSnapshot] = useState<{
    books: Map<string, OrderBook>;
    quotes: Map<string, Quote>;
    version: number;
  }>({ books: new Map(), quotes: new Map(), version: 0 });

  const [status, setStatus] = useState<SocketStatus>("connecting");

  // A new token set is a new subscription: drop the old status during render
  // so no frame shows "live" for a socket that is being torn down.
  const [prevKey, setPrevKey] = useState(key);
  if (prevKey !== key) {
    setPrevKey(key);
    setStatus("connecting");
  }

  const dirtyRef = useRef(false);
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    if (!active) return;

    const assets = key.split(",");
    // Quotes for the previous subscription are meaningless now.
    booksRef.current = new Map();
    quotesRef.current = new Map();
    let ws: WebSocket | null = null;
    let closed = false;
    let attempt = 0;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let pingTimer: ReturnType<typeof setInterval> | null = null;

    const markDirty = () => {
      if (dirtyRef.current) return;
      dirtyRef.current = true;
      frameRef.current = requestAnimationFrame(() => {
        dirtyRef.current = false;
        frameRef.current = null;
        // Publish copies: consumers get stable values they can safely memoize
        // against, instead of maps that mutate under them between renders.
        setSnapshot((prev) => ({
          books: new Map(booksRef.current),
          quotes: new Map(quotesRef.current),
          version: prev.version + 1,
        }));
      });
    };

    const touchQuote = (tokenId: string, patch: Partial<Quote>) => {
      const prev = quotesRef.current.get(tokenId);
      const nextLast = patch.last ?? prev?.last;
      let dir: 0 | 1 | -1 = 0;
      if (
        nextLast !== undefined &&
        prev?.last !== undefined &&
        nextLast !== prev.last
      ) {
        dir = nextLast > prev.last ? 1 : -1;
      }
      quotesRef.current.set(tokenId, {
        tokenId,
        bid: patch.bid ?? prev?.bid,
        ask: patch.ask ?? prev?.ask,
        last: nextLast,
        dir,
        ts: Date.now(),
      });
    };

    const handle = (raw: unknown) => {
      const msgs = Array.isArray(raw) ? raw : [raw];
      for (const m of msgs) {
        if (!m || typeof m !== "object") continue;
        const msg = m as Record<string, unknown>;
        const type = String(msg.event_type ?? "");

        if (type === "book" || (msg.bids && msg.asks)) {
          const tokenId = String(msg.asset_id ?? "");
          if (!tokenId) continue;
          const bids = levels(msg.bids, "bid");
          const asks = levels(msg.asks, "ask");
          booksRef.current.set(tokenId, {
            tokenId,
            bids,
            asks,
            timestamp: Number(msg.timestamp ?? Date.now()),
          });
          touchQuote(tokenId, { bid: bids[0]?.price, ask: asks[0]?.price });
          markDirty();
          continue;
        }

        if (type === "price_change" || Array.isArray(msg.price_changes)) {
          const changes = Array.isArray(msg.price_changes) ? msg.price_changes : [];
          for (const c of changes as Record<string, unknown>[]) {
            const tokenId = String(c.asset_id ?? "");
            if (!tokenId) continue;
            const bid = c.best_bid !== undefined ? parseFloat(String(c.best_bid)) : undefined;
            const ask = c.best_ask !== undefined ? parseFloat(String(c.best_ask)) : undefined;
            touchQuote(tokenId, {
              bid: Number.isFinite(bid) ? bid : undefined,
              ask: Number.isFinite(ask) ? ask : undefined,
            });

            // Apply the delta to the cached book so depth stays fresh between
            // snapshots. size 0 means the level was consumed or cancelled.
            const book = booksRef.current.get(tokenId);
            const price = parseFloat(String(c.price ?? ""));
            const size = parseFloat(String(c.size ?? ""));
            if (book && Number.isFinite(price) && Number.isFinite(size)) {
              const side = String(c.side ?? "").toUpperCase() === "SELL" ? "asks" : "bids";
              const rows = book[side].filter((l) => l.price !== price);
              if (size > 0) rows.push({ price, size });
              rows.sort((a, b) => (side === "bids" ? b.price - a.price : a.price - b.price));
              booksRef.current.set(tokenId, { ...book, [side]: rows, timestamp: Date.now() });
            }
          }
          markDirty();
          continue;
        }

        if (type === "last_trade_price" || msg.price !== undefined) {
          const tokenId = String(msg.asset_id ?? "");
          const price = parseFloat(String(msg.price ?? ""));
          if (tokenId && Number.isFinite(price)) {
            touchQuote(tokenId, { last: price });
            markDirty();
          }
        }
      }
    };

    const connect = () => {
      if (closed) return;
      // Status starts at "connecting" from the initial state, and every later
      // transition comes from a socket event — nothing is set synchronously
      // here, which keeps the effect free of cascading renders.
      try {
        ws = new WebSocket(WS_URL);
      } catch {
        schedule();
        return;
      }

      ws.onopen = () => {
        if (closed) return;
        attempt = 0;
        setStatus("live");
        ws?.send(JSON.stringify({ assets_ids: assets, type: "market" }));
        // The server drops idle connections; a periodic ping keeps it warm.
        pingTimer = setInterval(() => {
          if (ws?.readyState === WebSocket.OPEN) ws.send("PING");
        }, 10_000);
      };

      ws.onmessage = (ev) => {
        const text = typeof ev.data === "string" ? ev.data : "";
        if (!text || text === "PONG") return;
        try {
          handle(JSON.parse(text));
        } catch {
          // Non-JSON keepalive frame.
        }
      };

      ws.onerror = () => ws?.close();

      ws.onclose = () => {
        if (pingTimer) clearInterval(pingTimer);
        pingTimer = null;
        if (closed) return;
        schedule();
      };
    };

    const schedule = () => {
      if (closed) return;
      setStatus("retrying");
      const delay = Math.min(15_000, 500 * 2 ** attempt);
      attempt++;
      reconnectTimer = setTimeout(connect, delay);
    };

    connect();

    return () => {
      closed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (pingTimer) clearInterval(pingTimer);
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
      dirtyRef.current = false;
      // Detach handlers first: a close during teardown must not reconnect.
      if (ws) {
        ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
        if (ws.readyState <= WebSocket.OPEN) ws.close();
      }
    };
  }, [key, active]);

  return {
    status: active ? status : "idle",
    books: snapshot.books,
    quotes: snapshot.quotes,
    version: snapshot.version,
  };
}
