import { describe, expect, mock, test } from "bun:test";
import { blendedScore } from "@/lib/signals";
import type { Market, PricePoint } from "@/lib/types";

/**
 * End-to-end coverage of the /api/signals scan.
 *
 * Everything upstream of the route — the four Polymarket fetchers — is mocked so
 * the whole engine (detectors → model → blend → payload) runs deterministically
 * against fixed tape and history. This is the test that would catch the model
 * silently dropping out of the pipeline, the blend ordering breaking, or the new
 * `modeled/confirms/conflicts` stats drifting from the markets they summarise.
 */

const ramp = (from: number, step: number, n = 24): PricePoint[] =>
  Array.from({ length: n }, (_, i) => ({ t: i, p: from + step * i }));

const HISTORY: Record<string, PricePoint[]> = {
  tokUp: ramp(0.3, 0.01), // steady climb — momentum bullish, model leans mean-reversion
  tokDown: ramp(0.72, -0.012), // steady fall
  tokFlat: [], // no history → no model read, stays on raw heat
};

const market = (over: Partial<Market> & { conditionId: string }): Market =>
  ({
    id: over.conditionId,
    slug: over.conditionId,
    question: `Will ${over.conditionId}?`,
    outcomes: [{ label: "Yes", price: 0.5, tokenId: `tok${over.conditionId}` }],
    last: 0.5,
    volume: 0,
    volume24h: 80_000,
    volume1w: 210_000,
    liquidity: 20_000,
    active: true,
    closed: false,
    acceptingOrders: true,
    negRisk: false,
    tickSize: 0.01,
    tags: [],
    chg24h: 0,
    ...over,
  }) as Market;

const MARKETS: Market[] = [
  market({
    conditionId: "Up",
    chg24h: 8,
    outcomes: [{ label: "Yes", price: 0.5, tokenId: "tokUp" }],
  }),
  market({
    conditionId: "Down",
    chg24h: -8,
    outcomes: [{ label: "Yes", price: 0.5, tokenId: "tokDown" }],
  }),
  market({
    conditionId: "Flat",
    last: 0.04, // longshot still taking money → TAIL fires without history
    outcomes: [{ label: "Yes", price: 0.04, tokenId: "tokFlat" }],
  }),
];

mock.module("@/lib/polymarket", () => ({
  fetchMarkets: async () => MARKETS,
  fetchEvents: async () => [],
  fetchTrades: async () => [],
  fetchBooks: async () => [],
  fetchHistory: async (token: string) => HISTORY[token] ?? [],
}));

const { GET } = await import("@/app/api/signals/route");

describe("/api/signals pipeline", () => {
  test("returns a well-formed payload with the new model stats", async () => {
    const res = await GET();
    const env = await res.json();
    expect(env.ok).toBe(true);
    const body = env.data;

    expect(Array.isArray(body.markets)).toBe(true);
    expect(body.markets.length).toBeGreaterThan(0);
    expect(body.stats.scanned).toBe(MARKETS.length);
    expect(typeof body.stats.modeled).toBe("number");
    expect(typeof body.stats.modelConfirms).toBe("number");
    expect(typeof body.stats.modelConflicts).toBe("number");
  });

  test("the model runs on markets that had history and is absent where none was fetched", async () => {
    const res = await GET();
    const env = await res.json();
    expect(env.ok).toBe(true);
    const body = env.data;
    const byId = new Map<string, any>(body.markets.map((m: any) => [m.market.conditionId, m]));

    expect(byId.get("Up")?.model).toBeTruthy();
    expect(byId.get("Down")?.model).toBeTruthy();
    // The longshot fired TAIL but carried no history, so it must not invent a read.
    expect(byId.get("Flat")?.model).toBeUndefined();

    const up = byId.get("Up").model;
    expect(up.prob).toBeGreaterThan(0);
    expect(up.prob).toBeLessThan(1);
    expect(up.conviction).toBeGreaterThanOrEqual(0);
    expect(up.conviction).toBeLessThanOrEqual(1);
  });

  test("stats.modeled matches the markets that actually carry a model read", async () => {
    const res = await GET();
    const env = await res.json();
    expect(env.ok).toBe(true);
    const body = env.data;
    const withModel = body.markets.filter((m: any) => m.model).length;
    expect(body.stats.modeled).toBe(withModel);
    expect(body.stats.modelConfirms + body.stats.modelConflicts).toBeLessThanOrEqual(
      body.stats.modeled
    );
  });

  test("markets come back sorted by the blended score, not raw heat", async () => {
    const res = await GET();
    const env = await res.json();
    expect(env.ok).toBe(true);
    const body = env.data;
    const scores = body.markets.map((m: any) => blendedScore(m));
    const sorted = [...scores].sort((a, b) => b - a);
    expect(scores).toEqual(sorted);
  });
});
