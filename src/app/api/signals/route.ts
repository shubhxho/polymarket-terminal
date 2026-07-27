import { fetchBooks, fetchEvents, fetchMarkets, fetchTrades } from "@/lib/polymarket";
import {
  aggregateFlow,
  findArbitrage,
  scoreMarket,
  WHALE_NOTIONAL,
  type ArbOpportunity,
  type MarketSignals,
  type SignalKind,
} from "@/lib/signals";
import { fail, ok } from "@/lib/api-util";

export type SignalsPayload = {
  arbs: ArbOpportunity[];
  markets: MarketSignals[];
  stats: {
    scanned: number;
    flagged: number;
    bullish: number;
    bearish: number;
    byKind: Partial<Record<SignalKind, number>>;
    /** Notional of block prints in the scanned window. */
    blockNotional: number;
  };
};

/** Books are the slowest input, so only the busiest markets get depth. */
const BOOK_DEPTH_LIMIT = 60;

/**
 * The scanner behind the SIG screen.
 *
 * Runs the whole signal engine server-side: one client request replaces what
 * would otherwise be four upstream calls plus per-market book fetches from
 * every connected browser.
 */
export async function GET() {
  try {
    const [markets, events, trades] = await Promise.all([
      fetchMarkets({ limit: 200, order: "volume24hr" }),
      fetchEvents({ limit: 80, order: "volume24hr" }),
      // The tape is the only source of block flow; `min` filters server-side so
      // we aren't paging thousands of $5 retail prints to find the size.
      fetchTrades({ limit: 500, minSize: WHALE_NOTIONAL }),
    ]);

    const flow = aggregateFlow(trades);
    const arbs = findArbitrage(events);

    // Depth only for the busiest names. A failure here degrades the IMBALANCE
    // detector to silence rather than failing the whole scan.
    const bookTokens = markets
      .slice(0, BOOK_DEPTH_LIMIT)
      .map((m) => m.outcomes[0]?.tokenId)
      .filter((t): t is string => Boolean(t));

    const books = await fetchBooks(bookTokens).catch(() => []);
    const bookByToken = new Map(books.map((b) => [b.tokenId, b]));

    const scored = markets
      .map((m) =>
        scoreMarket(m, {
          flow: flow.get(m.conditionId),
          book: bookByToken.get(m.outcomes[0]?.tokenId ?? ""),
        })
      )
      .filter((s): s is MarketSignals => s !== null)
      .sort((a, b) => b.heat - a.heat);

    const byKind: Partial<Record<SignalKind, number>> = {};
    for (const s of scored) {
      for (const sig of s.signals) byKind[sig.kind] = (byKind[sig.kind] ?? 0) + 1;
    }
    if (arbs.length > 0) byKind.ARB = arbs.length;

    const payload: SignalsPayload = {
      arbs,
      markets: scored.slice(0, 60),
      stats: {
        scanned: markets.length,
        flagged: scored.length,
        bullish: scored.filter((s) => s.bias > 10).length,
        bearish: scored.filter((s) => s.bias < -10).length,
        byKind,
        blockNotional: trades.reduce((sum, t) => sum + t.size * t.price, 0),
      },
    };

    return ok(payload, 15);
  } catch (err) {
    return fail(err);
  }
}
