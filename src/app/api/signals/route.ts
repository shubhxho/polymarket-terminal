import type { NextRequest } from "next/server";
import { type GammaEvent, getTopEvents, searchEvents } from "@/lib/polymarket";

/**
 * Feeds the client-side edge scanner: returns the raw events for the current
 * board so the Web Worker can compute signals without re-implementing the
 * Gamma fetch/caching. Trimmed to the fields scanSignals actually reads to keep
 * the payload — and the postMessage clone into the worker — small.
 */
export async function GET(request: NextRequest) {
  const tag = request.nextUrl.searchParams.get("tag") ?? "";
  const q = request.nextUrl.searchParams.get("q") ?? "";

  let events: GammaEvent[];
  try {
    events = q
      ? await searchEvents(q)
      : await getTopEvents(tag || undefined, 200, 0);
  } catch {
    return Response.json(
      { error: "Failed to fetch market data" },
      { status: 502 },
    );
  }

  const trimmed = events
    .filter((e) => e.markets.length > 0)
    .map((e) => ({
      slug: e.slug,
      title: e.title,
      negRisk: e.negRisk,
      liquidity: e.liquidity,
      volume24hr: e.volume24hr,
      endDate: e.endDate,
      markets: e.markets.map((m) => ({
        id: m.id,
        question: m.question,
        groupItemTitle: m.groupItemTitle,
        outcomes: m.outcomes,
        outcomePrices: m.outcomePrices,
        clobTokenIds: m.clobTokenIds,
        lastTradePrice: m.lastTradePrice,
        bestBid: m.bestBid,
        bestAsk: m.bestAsk,
        spread: m.spread,
        oneDayPriceChange: m.oneDayPriceChange,
        oneWeekPriceChange: m.oneWeekPriceChange,
        volume24hr: m.volume24hr,
        active: m.active,
        closed: m.closed,
      })),
    }));

  return Response.json({ events: trimmed });
}
