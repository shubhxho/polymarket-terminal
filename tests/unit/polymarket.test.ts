import { describe, expect, it } from "bun:test";
import { normalizeMarket } from "@/lib/polymarket";

describe("normalizeMarket", () => {
  it("parses JSON-encoded string arrays and zips outcomes", () => {
    const m = normalizeMarket({
      outcomes: '["Yes","No"]',
      outcomePrices: '["0.62","0.38"]',
      clobTokenIds: '["tok-yes","tok-no"]',
    });
    expect(m.outcomes).toEqual([
      { label: "Yes", price: 0.62, tokenId: "tok-yes" },
      { label: "No", price: 0.38, tokenId: "tok-no" },
    ]);
  });

  it("accepts arrays as well as encoded strings", () => {
    const m = normalizeMarket({
      outcomes: ["Yes", "No"],
      outcomePrices: [0.5, 0.5],
      clobTokenIds: ["a", "b"],
    });
    expect(m.outcomes.map((o) => o.label)).toEqual(["Yes", "No"]);
    expect(m.outcomes[0].price).toBe(0.5);
  });

  it("falls back to empty price/token when a label has no counterpart", () => {
    const m = normalizeMarket({ outcomes: '["Yes","No","Maybe"]', outcomePrices: '["0.6"]' });
    expect(m.outcomes).toEqual([
      { label: "Yes", price: 0.6, tokenId: "" },
      { label: "No", price: 0, tokenId: "" },
      { label: "Maybe", price: 0, tokenId: "" },
    ]);
  });

  it("ignores malformed JSON in the encoded arrays", () => {
    const m = normalizeMarket({ outcomes: "{not json" });
    expect(m.outcomes).toEqual([]);
  });

  it("uses lastTradePrice as the mark when present", () => {
    const m = normalizeMarket({
      outcomes: '["Yes","No"]',
      outcomePrices: '["0.6","0.4"]',
      lastTradePrice: "0.55",
    });
    expect(m.last).toBe(0.55);
  });

  it("falls back to the first outcome price when lastTradePrice is absent", () => {
    const m = normalizeMarket({ outcomes: '["Yes","No"]', outcomePrices: '["0.6","0.4"]' });
    expect(m.last).toBe(0.6);
  });

  it("converts probability changes to points", () => {
    const m = normalizeMarket({
      oneHourPriceChange: 0.012,
      oneDayPriceChange: -0.05,
      oneWeekPriceChange: 0.1,
    });
    expect(m.chg1h).toBeCloseTo(1.2, 10);
    expect(m.chg24h).toBeCloseTo(-5, 10);
    expect(m.chg1w).toBeCloseTo(10, 10);
  });

  it("prefers *Num volume/liquidity fields over the string variants", () => {
    const m = normalizeMarket({ volumeNum: 1000, volume: 999, liquidityNum: 50, liquidity: 49 });
    expect(m.volume).toBe(1000);
    expect(m.liquidity).toBe(50);
  });

  it("falls back to the non-Num volume/liquidity when *Num is missing", () => {
    const m = normalizeMarket({ volume: "777", liquidity: "12.5" });
    expect(m.volume).toBe(777);
    expect(m.liquidity).toBe(12.5);
  });

  it("treats booleans strictly and defaults acceptingOrders to true", () => {
    const open = normalizeMarket({ active: true, closed: false, negRisk: true });
    expect(open.active).toBe(true);
    expect(open.closed).toBe(false);
    expect(open.negRisk).toBe(true);
    expect(open.acceptingOrders).toBe(true);

    const truthyNotBool = normalizeMarket({ active: "yes", acceptingOrders: false });
    expect(truthyNotBool.active).toBe(false);
    expect(truthyNotBool.acceptingOrders).toBe(false);
  });

  it("defaults tickSize to 0.001 and honors the upstream value", () => {
    expect(normalizeMarket({}).tickSize).toBe(0.001);
    expect(normalizeMarket({ orderPriceMinTickSize: "0.01" }).tickSize).toBe(0.01);
  });

  it("lifts event fields and tag labels from the first event", () => {
    const m = normalizeMarket({
      events: [
        {
          id: 7,
          title: "Election",
          slug: "election",
          ticker: "ELEC",
          tags: [{ label: "Politics" }, { label: "" }, { label: "US" }],
        },
      ],
    });
    expect(m.eventId).toBe("7");
    expect(m.eventTitle).toBe("Election");
    expect(m.eventSlug).toBe("election");
    expect(m.eventTicker).toBe("ELEC");
    expect(m.tags).toEqual(["Politics", "US"]);
  });

  it("fills sane defaults for an empty raw market", () => {
    const m = normalizeMarket({});
    expect(m.id).toBe("");
    expect(m.outcomes).toEqual([]);
    expect(m.last).toBe(0);
    expect(m.volume).toBe(0);
    expect(m.active).toBe(false);
    expect(m.eventId).toBeUndefined();
    expect(m.tags).toEqual([]);
    expect(m.groupItemTitle).toBeUndefined();
  });

  it("leaves optional book fields undefined when absent, numeric when present", () => {
    expect(normalizeMarket({}).bestBid).toBeUndefined();
    const m = normalizeMarket({ bestBid: "0.6", bestAsk: "0.62", spread: "0.02" });
    expect(m.bestBid).toBe(0.6);
    expect(m.bestAsk).toBe(0.62);
    expect(m.spread).toBe(0.02);
  });
});
