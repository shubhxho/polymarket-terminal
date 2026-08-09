import { fetchBooks, fetchEvents, fetchHistory, fetchMarkets, fetchTrades } from "@/lib/polymarket";
import {
  aggregateFlow,
  blendedScore,
  buildCrossSection,
  findArbitrage,
  findBasketDrift,
  modelAgreement,
  scoreMarket,
  WHALE_NOTIONAL,
  type ArbOpportunity,
  type BasketDrift,
  type MarketSignals,
  type SignalKind,
} from "@/lib/signals";
import { fail, ok } from "@/lib/api-util";
import { logError, logInfo } from "@/lib/logger";
import { recordSignalSnapshot } from "@/db/snapshot";
import type { PricePoint } from "@/lib/types";
import { after } from "next/server";

export type SignalsPayload = {
  arbs: ArbOpportunity[];
  drifts: BasketDrift[];
  markets: MarketSignals[];
  stats: {
    scanned: number;
    flagged: number;
    bullish: number;
    bearish: number;
    /** Markets that got the full treatment including price history. */
    deepScanned: number;
    byKind: Partial<Record<SignalKind, number>>;
    /** Notional of block prints in the scanned window. */
    blockNotional: number;
    /** Markets that had enough history for the trained model to weigh in. */
    modeled: number;
    /** Of those, how many the model confirms versus fights the rule engine on. */
    modelConfirms: number;
    modelConflicts: number;
  };
};

/**
 * Depth and history are the slow inputs — one upstream call each, per market —
 * so only the busiest names get the full treatment. Everything else is still
 * scored on the detectors that need no extra fetch.
 */
const DEEP_SCAN_LIMIT = 48;

/**
 * The scanner behind the SIG screen.
 *
 * Runs the whole engine server-side: one client request replaces four upstream
 * calls plus ~100 per-market fetches from every connected browser.
 */
export async function GET() {
  try {
    const [markets, events, trades] = await Promise.all([
      fetchMarkets({ limit: 200, order: "volume24hr" }),
      fetchEvents({ limit: 80, order: "volume24hr" }),
      // The tape is the only source of block flow; `min` filters upstream so we
      // aren't paging thousands of $5 retail prints to find the size.
      fetchTrades({ limit: 500, minSize: WHALE_NOTIONAL }),
    ]);

    const flow = aggregateFlow(trades);
    const arbs = findArbitrage(events);
    const drifts = findBasketDrift(events);
    // The cross-section is built from the whole universe, not just the deep
    // scan — a z-score against 48 markets is barely a z-score.
    const cross = buildCrossSection(markets);

    const deep = markets.slice(0, DEEP_SCAN_LIMIT);
    const deepTokens = deep
      .map((m) => m.outcomes[0]?.tokenId)
      .filter((t): t is string => Boolean(t));

    // Books come back in one batched call; history is one call per token, so it
    // is fanned out and settled — a slow leg degrades that market's detectors
    // to silence rather than failing the whole scan.
    const [books, histories] = await Promise.all([
      fetchBooks(deepTokens).catch(() => []),
      Promise.allSettled(deepTokens.map((t) => fetchHistory(t, "1d"))),
    ]);

    const bookByToken = new Map(books.map((b) => [b.tokenId, b]));
    const historyByToken = new Map<string, PricePoint[]>();
    histories.forEach((r, i) => {
      if (r.status === "fulfilled" && r.value.length > 0) {
        historyByToken.set(deepTokens[i], r.value);
      }
    });

    const scored = markets
      .map((m) => {
        const token = m.outcomes[0]?.tokenId ?? "";
        return scoreMarket(m, {
          flow: flow.get(m.conditionId),
          book: bookByToken.get(token),
          history: historyByToken.get(token),
          cross,
        });
      })
      .filter((s): s is MarketSignals => s !== null)
      // Blend, not heat: the model reorders neighbours it has a view on while
      // markets it never saw keep their raw-heat position.
      .sort((a, b) => blendedScore(b) - blendedScore(a));

    const byKind: Partial<Record<SignalKind, number>> = {};
    for (const s of scored) {
      for (const sig of s.signals) byKind[sig.kind] = (byKind[sig.kind] ?? 0) + 1;
    }
    if (arbs.length > 0) byKind.ARB = arbs.length;
    if (drifts.length > 0) byKind.DRIFT = drifts.length;

    const payload: SignalsPayload = {
      arbs,
      drifts: drifts.slice(0, 12),
      markets: scored.slice(0, 60),
      stats: {
        scanned: markets.length,
        flagged: scored.length,
        bullish: scored.filter((s) => s.bias > 10).length,
        bearish: scored.filter((s) => s.bias < -10).length,
        deepScanned: historyByToken.size,
        byKind,
        blockNotional: trades.reduce((sum, t) => sum + t.size * t.price, 0),
        modeled: scored.filter((s) => s.model).length,
        modelConfirms: scored.filter((s) => modelAgreement(s) === "confirms").length,
        modelConflicts: scored.filter((s) => modelAgreement(s) === "conflicts").length,
      },
    };

    // Persist the scan and log it AFTER the response is sent, so the database
    // round-trip never sits on the request's hot path. No-op without DATABASE_URL.
    // `after` throws outside a request scope (e.g. a unit test); that must not
    // fail the scan, so it is guarded.
    try {
      after(() => {
        logInfo("signals.scan", {
          scanned: payload.stats.scanned,
          flagged: payload.stats.flagged,
          modeled: payload.stats.modeled,
          confirms: payload.stats.modelConfirms,
        });
        void recordSignalSnapshot(payload);
      });
    } catch {
      // no request scope — skip post-response work
    }

    return ok(payload, 15);
  } catch (err) {
    logError("signals.scan_failed", { message: err instanceof Error ? err.message : String(err) });
    return fail(err);
  }
}
