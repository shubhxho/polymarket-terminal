import { expect, test } from "@playwright/test";
import type { GammaEvent, GammaMarket } from "../../src/lib/polymarket";
import { scanSignals } from "../../src/lib/signals";
import { scanEdges } from "../../src/lib/signals-plus";

/**
 * Tests for the edge scanners.
 *
 * The failure mode these guard against is not a crash — it is a confident
 * signal that is an artifact of the data rather than a property of the market.
 * Every case below is one of those.
 */

function market(over: Partial<GammaMarket> & { id: string; price: number }): GammaMarket {
  return {
    id: over.id,
    question: over.question ?? `Market ${over.id}`,
    slug: `m-${over.id}`,
    groupItemTitle: over.groupItemTitle ?? `Outcome ${over.id}`,
    outcomePrices: JSON.stringify([String(over.price), String(1 - over.price)]),
    outcomes: JSON.stringify(["Yes", "No"]),
    clobTokenIds: JSON.stringify([`tok-${over.id}-y`, `tok-${over.id}-n`]),
    active: over.active ?? true,
    closed: over.closed ?? false,
    oneDayPriceChange: over.oneDayPriceChange ?? 0,
    oneWeekPriceChange: over.oneWeekPriceChange ?? 0,
    volume24hr: over.volume24hr ?? 100_000,
    bestBid: over.bestBid,
    bestAsk: over.bestAsk,
    spread: over.spread,
  };
}

function event(markets: GammaMarket[], over: Partial<GammaEvent> = {}): GammaEvent {
  return {
    id: "e1",
    slug: "who-wins",
    title: "Who wins the thing?",
    negRisk: true,
    liquidity: 500_000,
    volume24hr: 250_000,
    endDate: new Date(Date.now() + 30 * 86_400_000).toISOString(),
    markets,
    ...over,
  };
}

test.describe("ARB requires a complete outcome set", () => {
  // Three live outcomes summing to 0.85 — a genuine 15pt underround.
  const live = [
    market({ id: "a", price: 0.3 }),
    market({ id: "b", price: 0.3 }),
    market({ id: "c", price: 0.25 }),
  ];

  test("fires on a whole, genuinely underround set", () => {
    const out = scanSignals([event(live)], { kinds: ["ARB"] });
    expect(out).toHaveLength(1);
    expect(out[0].kind).toBe("ARB");
    expect(out[0].edgeBps).toBeGreaterThan(0);
  });

  test("does NOT fire when a market was dropped as closed", () => {
    // The dropped leg carried the missing 15 points. Summing the survivors
    // shows a phantom "buy-all edge" that cannot be executed, because the leg
    // you would need is not tradeable.
    const partial = [...live, market({ id: "d", price: 0.15, closed: true })];
    expect(scanSignals([event(partial)], { kinds: ["ARB"] })).toHaveLength(0);
    expect(scanEdges([event(partial)], { kinds: ["ARB"] })).toHaveLength(0);
  });

  test("does NOT fire when a market is inactive", () => {
    const partial = [...live, market({ id: "d", price: 0.15, active: false })];
    expect(scanSignals([event(partial)], { kinds: ["ARB"] })).toHaveLength(0);
    expect(scanEdges([event(partial)], { kinds: ["ARB"] })).toHaveLength(0);
  });

  test("never fires on a 2-outcome binary, whose sides sum to 1 by construction", () => {
    const binary = [market({ id: "a", price: 0.62 })];
    expect(scanSignals([event(binary)], { kinds: ["ARB"] })).toHaveLength(0);
    expect(scanEdges([event(binary)], { kinds: ["ARB"] })).toHaveLength(0);
  });

  test("never fires on an event that is not neg-risk", () => {
    expect(scanSignals([event(live, { negRisk: false })], { kinds: ["ARB"] })).toHaveLength(0);
  });
});

test.describe("scanEdges kind filtering", () => {
  /** Many moderate MOMENTUM signals plus a few strong LIQUIDITY ones. */
  function board(): GammaEvent[] {
    const movers = Array.from({ length: 40 }, (_, i) =>
      event(
        [
          market({
            id: `m${i}`,
            price: 0.5,
            oneDayPriceChange: 0.2,
            oneWeekPriceChange: 0.05,
          }),
        ],
        { id: `mv${i}`, slug: `mover-${i}`, negRisk: false },
      ),
    );
    const makers = Array.from({ length: 5 }, (_, i) =>
      event([market({ id: `l${i}`, price: 0.5, spread: 0.05 })], {
        id: `lq${i}`,
        slug: `maker-${i}`,
        negRisk: false,
        liquidity: 900_000,
      }),
    );
    return [...movers, ...makers];
  }

  test("filtering by kind returns the best of that kind, not the survivors of a global cut", () => {
    const events = board();
    // A small limit guarantees the global list is dominated by one kind.
    const all = scanEdges(events, { limit: 10 });
    const liquidityInAll = all.filter((s) => s.kind === "LIQUIDITY").length;
    const liquidityDirect = scanEdges(events, {
      kinds: ["LIQUIDITY"],
      limit: 10,
    });

    // Every returned signal is the requested kind...
    expect(liquidityDirect.every((s) => s.kind === "LIQUIDITY")).toBe(true);
    // ...and asking for the kind surfaces at least as many as the global cut
    // happened to leave behind. This is the regression: post-filtering the
    // truncated list could only ever return `liquidityInAll`.
    expect(liquidityDirect.length).toBeGreaterThanOrEqual(liquidityInAll);
    expect(liquidityDirect.length).toBe(5);
  });

  test("an unrequested kind is absent entirely", () => {
    const only = scanEdges(board(), { kinds: ["MOMENTUM"] });
    expect(only.length).toBeGreaterThan(0);
    expect(only.some((s) => s.kind !== "MOMENTUM")).toBe(false);
  });
});

test.describe("liquidity floor", () => {
  test("thin books are skipped by both scanners", () => {
    const thin = event(
      [
        market({ id: "a", price: 0.3 }),
        market({ id: "b", price: 0.3 }),
        market({ id: "c", price: 0.25 }),
      ],
      { liquidity: 100 },
    );
    expect(scanSignals([thin])).toHaveLength(0);
    expect(scanEdges([thin])).toHaveLength(0);
  });
});
