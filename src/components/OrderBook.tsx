"use client";

import { useMemo } from "react";
import { cents, compact } from "@/lib/format";
import { Empty } from "@/components/ui/Panel";
import type { BookLevel, OrderBook as Book } from "@/lib/types";

type Row = {
  price: number;
  size: number;
  /** Cumulative size from the top of book down to this level. */
  cum: number;
};

function ladder(levels: BookLevel[], depth: number): Row[] {
  const out: Row[] = [];
  let cum = 0;
  for (const l of levels.slice(0, depth)) {
    cum += l.size;
    out.push({ price: l.price, size: l.size, cum });
  }
  return out;
}

/**
 * Depth ladder with cumulative-size shading.
 *
 * Asks are rendered top-down descending so the spread sits in the middle of the
 * panel, matching how a trader reads a real book. Bar widths encode *cumulative*
 * depth, which is what actually determines fill price on a market order.
 */
export function OrderBookLadder({
  book,
  depth = 10,
  outcomeLabel,
}: {
  book: Book | undefined;
  depth?: number;
  outcomeLabel?: string;
}) {
  const model = useMemo(() => {
    if (!book) return null;
    const bids = ladder(book.bids, depth);
    const asks = ladder(book.asks, depth);
    const maxCum = Math.max(
      bids[bids.length - 1]?.cum ?? 0,
      asks[asks.length - 1]?.cum ?? 0,
      1
    );
    const bestBid = book.bids[0]?.price;
    const bestAsk = book.asks[0]?.price;
    const spread = bestBid !== undefined && bestAsk !== undefined ? bestAsk - bestBid : undefined;
    const mid = bestBid !== undefined && bestAsk !== undefined ? (bestBid + bestAsk) / 2 : undefined;

    // Sum of resting notional on each side — a crude but useful skew read.
    const bidNotional = book.bids.reduce((s, l) => s + l.price * l.size, 0);
    const askNotional = book.asks.reduce((s, l) => s + (1 - l.price) * l.size, 0);
    const imbalance =
      bidNotional + askNotional > 0
        ? (bidNotional - askNotional) / (bidNotional + askNotional)
        : 0;

    return { bids, asks, maxCum, spread, mid, imbalance, bidNotional, askNotional };
  }, [book, depth]);

  if (!model) return <Empty text="waiting for book" />;
  if (model.bids.length === 0 && model.asks.length === 0) return <Empty text="book is empty" />;

  return (
    <div className="text-tiny">
      {outcomeLabel ? (
        <div className="mb-1 truncate text-[10px] tracking-wide text-info uppercase">
          {outcomeLabel}
        </div>
      ) : null}

      <div className="flex items-center gap-1 border-b border-edge px-1 py-[2px] text-[10px] tracking-wide text-accent-weak uppercase">
        <span className="w-[52px] text-right">Price</span>
        <span className="w-[64px] text-right">Size</span>
        <span className="w-[64px] text-right">Cum</span>
        <span className="flex-1 text-right">Depth</span>
      </div>

      {/* Asks, worst price first so best ask sits adjacent to the spread. */}
      {[...model.asks].reverse().map((r) => (
        <LadderRow key={`a${r.price}`} row={r} maxCum={model.maxCum} side="ask" />
      ))}

      <div className="my-[2px] flex items-center justify-between gap-2 border-y border-edge-strong bg-surface-2 px-1 py-[3px]">
        <span className="text-[10px] tracking-wide text-muted uppercase">Mid</span>
        <span className="font-bold text-accent">{cents(model.mid)}¢</span>
        <span className="text-[10px] tracking-wide text-muted uppercase">Spread</span>
        <span className="text-ink">{model.spread === undefined ? "--" : `${cents(model.spread)}¢`}</span>
      </div>

      {model.bids.map((r) => (
        <LadderRow key={`b${r.price}`} row={r} maxCum={model.maxCum} side="bid" />
      ))}

      <div className="mt-1.5 border-t border-edge pt-1">
        <div className="mb-[3px] flex justify-between text-[10px] text-muted">
          <span>BID ${compact(model.bidNotional)}</span>
          <span>IMBALANCE {(model.imbalance * 100).toFixed(0)}%</span>
          <span>ASK ${compact(model.askNotional)}</span>
        </div>
        <div className="flex h-[6px] w-full overflow-hidden bg-edge">
          <div
            className="bg-up/70"
            style={{ width: `${((model.imbalance + 1) / 2) * 100}%` }}
          />
          <div className="flex-1 bg-down/70" />
        </div>
      </div>
    </div>
  );
}

function LadderRow({
  row,
  maxCum,
  side,
}: {
  row: Row;
  maxCum: number;
  side: "bid" | "ask";
}) {
  const pct = Math.min(100, (row.cum / maxCum) * 100);
  return (
    <div className="relative flex items-center gap-1 px-1 py-[1px]">
      <div
        className={`absolute inset-y-0 right-0 ${side === "bid" ? "bg-up/12" : "bg-down/12"}`}
        style={{ width: `${pct}%` }}
        aria-hidden
      />
      <span
        className={`relative w-[52px] text-right font-bold ${
          side === "bid" ? "text-up" : "text-down"
        }`}
      >
        {cents(row.price)}
      </span>
      <span className="relative w-[64px] text-right text-ink/85">{compact(row.size)}</span>
      <span className="relative w-[64px] text-right text-muted">{compact(row.cum)}</span>
      <span className="relative flex-1 text-right text-[10px] text-faint">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}
