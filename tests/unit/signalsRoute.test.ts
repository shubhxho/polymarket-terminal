import { describe, expect, test, vi } from "vitest";
import { blendedScore } from "@/lib/signals";
import type { Market, PricePoint } from "@/lib/types";

/**
 * End-to-end coverage of the /api/signals scan.
 *
 * Everything upstream of the route — the four Polymarket fetchers — is mocked so
 * the whole engine (detectors, model, blend, payload) runs deterministically
 * against fixed tape and history. This is the test that would catch the model
 * silently dropping out of the pipeline, the blend ordering breaking, or the new
 * `modeled/confirms/conflicts` stats drifting from the markets they summarise.
 *
 * The fixtures live in `vi.hoisted` because `vi.mock` is lifted above the
 * imports; the mock factory closes over the hoisted values so the stub and the
 * assertions share one source of truth.
 */
const { MARKETS, HISTORY } = vi.hoisted(() => {
  const ramp = (from: number, step: number, n = 24) =>
    Array.from({ length: n }, (_, i) => ({ t: i, p: from + step * i }));

  const history: Record<string, { t: number; p: number }[]> = {
    tokUp: ramp(0.3, 0.01), // steady climb — momentum bullish, model leans mean-reversion
    tokDown: ramp(0.72, -0.012), // steady fall
    tokFlat: [], // no history → no model read, stays on raw heat
  };

  const market = (over: Record<string, unknown> & { conditionId: string }) => ({
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
  });

  const markets = [
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
  ] as unknown as Market[];

  return { MARKETS: markets, HISTORY: history };
});

vi.mock("@/lib/polymarket", () => ({
  fetchMarkets: async () => MARKETS,
  fetchEvents: async () => [],
  fetchTrades: async () => [],
  fetchBooks: async () => [],
  fetchHistory: async (token: string): Promise<PricePoint[]> => HISTORY[token] ?? [],
}));

const { GET } = await import("@/app/api/signals/route");

describe("/api/signals pipeline", () => {
  test("returns a well-formed payload with the new model stats", async () => {
    const env = await (await GET()).json();
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
    const env = await (await GET()).json();
    expect(env.ok).toBe(true);
    const byId = new Map<string, any>(env.data.markets.map((m: any) => [m.market.conditionId, m]));

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

  test("markets that carry a model read also ship their live re-scoring window", async () => {
    const env = await (await GET()).json();
    const up = env.data.markets.find((m: any) => m.market.conditionId === "Up");
    expect(Array.isArray(up.recent)).toBe(true);
    expect(up.recent.length).toBeGreaterThan(0);
    expect(up.recent.length).toBeLessThanOrEqual(16);
  });

  test("stats.modeled matches the markets that actually carry a model read", async () => {
    const env = await (await GET()).json();
    const withModel = env.data.markets.filter((m: any) => m.model).length;
    expect(env.data.stats.modeled).toBe(withModel);
    expect(env.data.stats.modelConfirms + env.data.stats.modelConflicts).toBeLessThanOrEqual(
      env.data.stats.modeled
    );
  });

  test("markets come back sorted by the blended score, not raw heat", async () => {
    const env = await (await GET()).json();
    const scores = env.data.markets.map((m: any) => blendedScore(m));
    const sorted = [...scores].sort((a, b) => b - a);
    expect(scores).toEqual(sorted);
  });
});
