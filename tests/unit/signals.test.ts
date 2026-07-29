import { describe, expect, it } from "vitest";
import { aggregateFlow, buildCrossSection, findArbitrage, WHALE_NOTIONAL } from "@/lib/signals";
import type { EventSummary, Market, Trade } from "@/lib/types";

/**
 * The detectors read only a handful of fields off each domain object, so the
 * fixtures set exactly those and cast — a full Market has ~40 fields that would
 * only be noise here.
 */
function trade(over: Partial<Trade>): Trade {
  return {
    id: "t",
    wallet: "0x",
    side: "BUY",
    outcome: "Yes",
    outcomeIndex: 0,
    size: 0,
    price: 0,
    timestamp: 0,
    title: "",
    conditionId: "c",
    asset: "a",
    ...over,
  } as Trade;
}

function market(over: Partial<Market>): Market {
  return {
    negRisk: true,
    acceptingOrders: true,
    liquidity: 1000,
    volume24h: 0,
    volume1w: 0,
    chg24h: 0,
    ...over,
  } as Market;
}

function event(markets: Market[]): EventSummary {
  return { markets } as EventSummary;
}

describe("aggregateFlow", () => {
  it("ignores prints below the whale notional", () => {
    // 100 * 1 = 100, well under the 10k threshold.
    const flow = aggregateFlow([trade({ conditionId: "small", size: 100, price: 1 })]);
    expect(flow.has("small")).toBe(false);
  });

  it("treats buying YES and selling NO as the same bullish flow", () => {
    const big = { size: 20_000, price: 0.5 }; // 10k notional, exactly at WHALE
    expect(big.size * big.price).toBe(WHALE_NOTIONAL);
    const flow = aggregateFlow([
      trade({ conditionId: "A", side: "BUY", outcomeIndex: 0, ...big }),
      trade({ conditionId: "A", side: "SELL", outcomeIndex: 1, ...big }),
    ]);
    expect(flow.get("A")).toEqual({ net: 20_000, count: 2, gross: 20_000 });
  });

  it("nets buying NO as bearish", () => {
    const flow = aggregateFlow([
      trade({ conditionId: "B", side: "BUY", outcomeIndex: 1, size: 20_000, price: 0.5 }),
    ]);
    expect(flow.get("B")?.net).toBe(-10_000);
  });
});

describe("buildCrossSection", () => {
  it("gates volume ratios on a real weekly baseline and moves on turnover", () => {
    const cs = buildCrossSection([
      // Qualifies for both: baseline 1000 > 500, volume24h ≥ 5k.
      market({ volume24h: 10_000, volume1w: 7_000, chg24h: 3 }),
      // Turnover qualifies (move counted) but baseline 100 is too thin for a ratio.
      market({ volume24h: 6_000, volume1w: 700, chg24h: -5 }),
      // Below MIN_VOLUME_24H — excluded from both entirely.
      market({ volume24h: 1_000, volume1w: 7_000, chg24h: 99 }),
    ]);
    expect(cs.volumeRatios).toHaveLength(1);
    expect(cs.volumeRatios[0]).toBeCloseTo(Math.log(10), 4);
    expect(cs.absMoves).toEqual([3, 5]);
  });
});

describe("findArbitrage", () => {
  it("flags a basket whose bids sum above par as sell-basket", () => {
    const arb = findArbitrage([
      event([
        market({ bestBid: 0.6, bestAsk: 0.65, liquidity: 500 }),
        market({ bestBid: 0.5, bestAsk: 0.55, liquidity: 900 }),
      ]),
    ]);
    expect(arb).toHaveLength(1);
    expect(arb[0].side).toBe("sell-basket");
    expect(arb[0].edgePoints).toBeCloseTo(10, 6);
    expect(arb[0].tightestLegLiquidity).toBe(500);
  });

  it("flags a basket whose asks sum below par as buy-basket", () => {
    const arb = findArbitrage([
      event([
        market({ bestBid: 0.39, bestAsk: 0.4, liquidity: 300 }),
        market({ bestBid: 0.39, bestAsk: 0.4, liquidity: 300 }),
      ]),
    ]);
    expect(arb).toHaveLength(1);
    expect(arb[0].side).toBe("buy-basket");
    expect(arb[0].edgePoints).toBeCloseTo(20, 6);
  });

  it("finds nothing when the basket is coherent", () => {
    const arb = findArbitrage([
      event([
        market({ bestBid: 0.5, bestAsk: 0.51, liquidity: 500 }),
        market({ bestBid: 0.5, bestAsk: 0.51, liquidity: 500 }),
      ]),
    ]);
    expect(arb).toHaveLength(0);
  });

  it("skips events that are not negative-risk", () => {
    const arb = findArbitrage([
      event([
        market({ negRisk: false, bestBid: 0.6, bestAsk: 0.65, liquidity: 500 }),
        market({ negRisk: false, bestBid: 0.5, bestAsk: 0.55, liquidity: 500 }),
      ]),
    ]);
    expect(arb).toHaveLength(0);
  });

  it("refuses a basket with an unexecutable (zero-liquidity) leg", () => {
    const arb = findArbitrage([
      event([
        market({ bestBid: 0.6, bestAsk: 0.65, liquidity: 0 }),
        market({ bestBid: 0.5, bestAsk: 0.55, liquidity: 900 }),
      ]),
    ]);
    expect(arb).toHaveLength(0);
  });
});
