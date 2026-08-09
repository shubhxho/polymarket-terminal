import { expect, test } from "@playwright/test";
import {
  bookLiquidity,
  hedgeNotionalFor,
  matchCoin,
  parseClaim,
  parseStrike,
  priceClaim,
} from "../../src/lib/derivatives";
import type { OrderBook } from "../../src/lib/execution";
import type { PerpContext } from "../../src/lib/hyperliquid";
import { intervalForHorizon } from "../../src/lib/hyperliquid";
import type { Candle } from "../../src/lib/quant";

/**
 * Tests for the Polymarket ⇄ Hyperliquid bridge.
 *
 * Parsing is the highest-risk part of the desk: a mis-read strike or, worse, a
 * touch market priced as a terminal digital produces a confident number that is
 * simply wrong. So the bar here is that anything ambiguous must return `null`
 * rather than a guess, and that is asserted as hard as the happy paths.
 */

test.describe("strike parsing", () => {
  test("reads the common dollar formats", () => {
    expect(parseStrike("Bitcoin above $120,000 on Dec 31")).toBe(120_000);
    expect(parseStrike("Will BTC hit $150k in 2025?")).toBe(150_000);
    expect(parseStrike("ETH above $4,000")).toBe(4_000);
    expect(parseStrike("Will BTC reach $1.5M?")).toBe(1_500_000);
    expect(parseStrike("SOL above $250.50 on Friday")).toBe(250.5);
  });

  test("a date's comma is not thousands grouping", () => {
    // REGRESSION: "January 31, 2026" was read as the grouped number "31," and
    // returned 31 — a $31 Bitcoin strike, which prices to ~100% and prints a
    // gigantic fabricated edge. Grouping must be `,ddd` with no space.
    expect(parseStrike("Will Bitcoin close above its high on January 31, 2026?")).toBeNull();
    expect(parseStrike("Resolves on March 5, 2027")).toBeNull();
    // A genuinely grouped number still parses.
    expect(parseStrike("Ethereum above 3,000 on Dec 1, 2025")).toBe(3_000);
    expect(parseStrike("Bitcoin above $120,000 on January 31, 2026")).toBe(120_000);
    expect(parseStrike("Above 1,250,000 total")).toBe(1_250_000);
  });

  test("a two-sided range is refused rather than halved", () => {
    // A corridor is not a threshold; taking its first number would price a
    // one-sided claim the market never offered.
    expect(parseStrike("Will Bitcoin be between $100,000 and $120,000?")).toBeNull();
    expect(parseStrike("Bitcoin closes between $90k and $110k on Friday")).toBeNull();
    expect(parseClaim("Bitcoin closes between $90k and $110k on Friday")).toBeNull();
  });

  test("does not mistake a bare year or day for a strike", () => {
    // "2025" has no $, no magnitude suffix and no comma grouping.
    expect(parseStrike("Will Bitcoin go up in 2025?")).toBeNull();
    expect(parseStrike("Bitcoin price on July 21")).toBeNull();
    expect(parseStrike("No numbers here at all")).toBeNull();
  });

  test("takes the threshold, not a trailing date", () => {
    expect(parseStrike("Bitcoin above $120,000 on December 31 2025")).toBe(120_000);
  });
});

test.describe("coin matching", () => {
  test("maps names and tickers to the Hyperliquid symbol", () => {
    expect(matchCoin("Will Bitcoin hit 150k")).toBe("BTC");
    expect(matchCoin("BTC above 120k")).toBe("BTC");
    expect(matchCoin("Ethereum above $4,000")).toBe("ETH");
    expect(matchCoin("Solana above $250")).toBe("SOL");
  });

  test("refuses coins that merely share a prefix", () => {
    // "Bitcoin Cash" is a different asset; matching it as BTC would price the
    // claim against the wrong underlying entirely.
    expect(matchCoin("Will Bitcoin Cash hit $1,000")).toBeNull();
    expect(matchCoin("Ethereum Classic above $50")).toBeNull();
  });

  test("does not fire on a substring inside another word", () => {
    expect(matchCoin("The dotted line will be signed")).toBeNull();
    expect(matchCoin("Nothing crypto about this market")).toBeNull();
  });
});

test.describe("claim classification", () => {
  test("'above ... on DATE' is a terminal digital call", () => {
    const claim = parseClaim("Bitcoin above $120,000 on December 31?");
    expect(claim).not.toBeNull();
    expect(claim?.coin).toBe("BTC");
    expect(claim?.strike).toBe(120_000);
    expect(claim?.style).toBe("TERMINAL");
    expect(claim?.direction).toBe("UP");
  });

  test("'hits ... by DATE' is a one-touch", () => {
    // This is the distinction that matters most: pricing this as a terminal
    // digital would under-price it by roughly half near the barrier.
    const claim = parseClaim("Will Bitcoin hit $150k by June 30?");
    expect(claim?.style).toBe("TOUCH");
    expect(claim?.direction).toBe("UP");
  });

  test("downside phrasing flips the direction", () => {
    expect(parseClaim("Ethereum below $2,000 on Friday?")?.direction).toBe("DOWN");
    expect(parseClaim("Will Bitcoin dip to $80,000 this month?")?.direction).toBe("DOWN");
    expect(parseClaim("Will Bitcoin dip to $80,000 this month?")?.style).toBe("TOUCH");
  });

  test("'hits' outranks a stray 'above'", () => {
    expect(parseClaim("Will BTC hit $150k, trading above its old high?")?.style).toBe("TOUCH");
  });

  test("returns null rather than guessing", () => {
    // No coin.
    expect(parseClaim("Will the Fed cut rates above $5,000?")).toBeNull();
    // No strike.
    expect(parseClaim("Will Bitcoin go up this year?")).toBeNull();
    // Coin and number, but no comparison word at all.
    expect(parseClaim("Bitcoin $120,000")).toBeNull();
  });
});

test.describe("horizon to bar length", () => {
  test("scales the bar with the life of the claim", () => {
    expect(intervalForHorizon(2)).toBe("1m");
    expect(intervalForHorizon(12)).toBe("5m");
    expect(intervalForHorizon(72)).toBe("15m");
    expect(intervalForHorizon(24 * 10)).toBe("1h");
    expect(intervalForHorizon(24 * 60)).toBe("4h");
    expect(intervalForHorizon(24 * 365)).toBe("1d");
  });

  test("degenerate horizons fall back rather than throwing", () => {
    expect(intervalForHorizon(0)).toBe("1h");
    expect(intervalForHorizon(-5)).toBe("1h");
    expect(intervalForHorizon(Number.NaN)).toBe("1h");
  });
});

/* ─────────────────────── pricing a parsed claim ─────────────────────── */

const PERP: PerpContext = {
  coin: "BTC",
  oraclePx: 100_000,
  markPx: 100_050,
  midPx: 100_040,
  premium: 0.0005,
  fundingHourly: 0.00001,
  openInterest: 5_000,
  dayNotionalVolume: 1e9,
  prevDayPx: 99_000,
  impactBid: 99_990,
  impactAsk: 100_010,
  maxLeverage: 50,
  szDecimals: 5,
};

/** Flat-ish candles with enough movement for the vol suite to be non-zero. */
function candles(n = 200): Candle[] {
  let price = 100_000;
  return Array.from({ length: n }, (_, i) => {
    const open = price;
    // Deterministic oscillation — no Math.random in tests.
    price *= 1 + 0.004 * Math.sin(i * 1.7);
    return {
      t: i * 3_600_000,
      o: open,
      h: Math.max(open, price) * 1.001,
      l: Math.min(open, price) * 0.999,
      c: price,
      v: 1,
    };
  });
}

const NOW = Date.UTC(2026, 0, 1);
const IN_30_DAYS = new Date(NOW + 30 * 86_400_000).toISOString();

function quoteWith(overrides: Partial<Parameters<typeof priceClaim>[0]> = {}) {
  const claim = parseClaim("Bitcoin above $120,000 on January 31?");
  if (!claim) throw new Error("fixture claim failed to parse");
  return priceClaim({
    claim,
    slug: "btc-120k",
    title: "Bitcoin above $120,000 on January 31?",
    outcomeLabel: "Yes",
    marketProbability: 0.25,
    perp: PERP,
    candles: candles(),
    barMinutes: 60,
    fundingHourly: 0.00001,
    endDate: IN_30_DAYS,
    now: NOW,
    ...overrides,
  });
}

/** A symmetric ladder: `levels` rungs each side, `step` apart, `size` each. */
function ladder(mid = 100_000, step = 5, size = 0.5, levels = 40): OrderBook {
  return {
    tokenId: "HL:BTC",
    asks: Array.from({ length: levels }, (_, i) => ({ price: mid + step * (i + 1), size })),
    bids: Array.from({ length: levels }, (_, i) => ({ price: mid - step * (i + 1), size })),
  };
}

test.describe("bookLiquidity", () => {
  test("measures the true top-of-book spread", () => {
    const liq = bookLiquidity(ladder(100_000, 5));
    expect(liq).not.toBeNull();
    // Best ask 100_005, best bid 99_995 ⇒ 10 wide on a 100k mid = 1bp.
    expect((liq as NonNullable<typeof liq>).spreadBps).toBeCloseTo(1, 6);
  });

  test("counts only depth inside the ±25bps band", () => {
    // 25bps of $100k is $250. With $50 rungs, offsets 50..250 are inside the
    // band and 300+ are outside — so the ladder deliberately straddles it.
    const liq = bookLiquidity(ladder(100_000, 50, 0.5, 10));
    const expected = [50, 100, 150, 200, 250].reduce(
      (sum, d) => sum + (100_000 + d) * 0.5 + (100_000 - d) * 0.5,
      0,
    );
    expect((liq as NonNullable<typeof liq>).depthUsd).toBeCloseTo(expected, 6);
    // The five rungs beyond the band must NOT be counted.
    const everything = ladder(100_000, 50, 0.5, 10)
      .asks.concat(ladder(100_000, 50, 0.5, 10).bids)
      .reduce((sum, l) => sum + l.price * l.size, 0);
    expect((liq as NonNullable<typeof liq>).depthUsd).toBeLessThan(everything);
  });

  test("reports hedge slippage against the touch, and flags a dry book", () => {
    const deep = bookLiquidity(ladder(100_000, 5, 5, 40));
    expect((deep as NonNullable<typeof deep>).hedgeFilled).toBe(true);
    // Walking a deep ladder for $10k barely leaves the touch.
    expect((deep as NonNullable<typeof deep>).hedgeSlippageBps).toBeGreaterThanOrEqual(0);
    expect((deep as NonNullable<typeof deep>).hedgeSlippageBps).toBeLessThan(5);

    // A ladder holding well under $10k cannot absorb the clip.
    const thin: OrderBook = {
      tokenId: "HL:BTC",
      asks: [{ price: 100_005, size: 0.01 }],
      bids: [{ price: 99_995, size: 0.01 }],
    };
    expect((bookLiquidity(thin) as NonNullable<typeof deep>).hedgeFilled).toBe(false);
  });

  test("depth coverage is depth measured against the clip actually needed", () => {
    const liq = bookLiquidity(ladder(100_000, 5, 5, 40), 50_000);
    const l = liq as NonNullable<typeof liq>;
    expect(l.hedgeNotionalUsd).toBe(50_000);
    expect(l.depthCoverage).toBeCloseTo(l.depthUsd / 50_000, 9);
  });

  test("an empty or one-sided book yields nothing rather than NaN", () => {
    expect(bookLiquidity({ tokenId: "HL:BTC", asks: [], bids: [] })).toBeNull();
    expect(
      bookLiquidity({
        tokenId: "HL:BTC",
        asks: [{ price: 100_005, size: 1 }],
        bids: [],
      }),
    ).toBeNull();
  });
});

test.describe("hedgeNotionalFor", () => {
  test("scales with delta and with how many contracts the position buys", () => {
    // Same dollars, same delta, cheaper contract ⇒ more contracts ⇒ bigger hedge.
    const cheap = hedgeNotionalFor(0.5, 100_000, 0.05);
    const dear = hedgeNotionalFor(0.5, 100_000, 0.5);
    expect(cheap).toBeGreaterThan(dear);
    expect(cheap / dear).toBeCloseTo(10, 6);
    // Zero delta needs no hedge at all.
    expect(hedgeNotionalFor(0, 100_000, 0.25)).toBe(0);
    // Sign of delta is irrelevant — a hedge has a size, not a direction.
    expect(hedgeNotionalFor(-0.3, 100_000, 0.25)).toBeCloseTo(
      hedgeNotionalFor(0.3, 100_000, 0.25),
      9,
    );
  });

  test("degenerate inputs return zero rather than NaN", () => {
    expect(hedgeNotionalFor(0.5, 0, 0.25)).toBe(0);
    expect(hedgeNotionalFor(Number.NaN, 100_000, 0.25)).toBe(0);
  });
});

test.describe("priceClaim", () => {
  test("produces a coherent quote from live-shaped inputs", () => {
    const q = quoteWith();
    expect(q).not.toBeNull();
    const quote = q as NonNullable<typeof q>;

    expect(quote.modelProbability).toBeGreaterThanOrEqual(0);
    expect(quote.modelProbability).toBeLessThanOrEqual(1);
    // Edge is defined as model − market, exactly.
    expect(quote.edge).toBeCloseTo(quote.modelProbability - quote.marketProbability, 12);
    // Positive funding ⇒ forward above spot.
    expect(quote.forward).toBeGreaterThan(quote.spot);
    expect(quote.vol.blended).toBeGreaterThan(0);
    expect(quote.band.lo).toBeLessThanOrEqual(quote.band.hi);
    expect(Number.isFinite(quote.z)).toBe(true);
    expect(quote.greeks).not.toBeNull();
    // No ladder was supplied, so there is no book read — not a fabricated one.
    expect(quote.book).toBeNull();
    // Venue context comes straight off the perp context.
    expect(quote.basisBps).toBeCloseTo(((PERP.markPx - PERP.oraclePx) / PERP.oraclePx) * 10_000, 9);
    expect(quote.perpChange24h).toBeCloseTo(PERP.markPx / PERP.prevDayPx - 1, 12);
  });

  test("Kelly and edge always agree in sign", () => {
    // Anchor on the model's OWN probability rather than absolute numbers: the
    // fixture's σ is whatever the synthetic candles imply, so a hardcoded
    // "0.01 is cheap" would be asserting the fixture, not the invariant.
    const model = (quoteWith() as NonNullable<ReturnType<typeof quoteWith>>).modelProbability;
    const cheap = quoteWith({ marketProbability: Math.max(model / 2, 1e-4) });
    const rich = quoteWith({ marketProbability: Math.min(model * 2 + 0.05, 0.999) });

    expect((cheap as NonNullable<typeof cheap>).edge).toBeGreaterThan(0);
    expect((cheap as NonNullable<typeof cheap>).kelly).toBeGreaterThan(0);
    expect((rich as NonNullable<typeof rich>).edge).toBeLessThan(0);
    expect((rich as NonNullable<typeof rich>).kelly).toBeLessThan(0);
    // Fair value ⇒ no stake.
    const fair = quoteWith({ marketProbability: model });
    expect((fair as NonNullable<typeof fair>).kelly).toBeCloseTo(0, 9);
  });

  test("an expired or undated market is refused, not extrapolated", () => {
    expect(quoteWith({ endDate: new Date(NOW - 86_400_000).toISOString() })).toBeNull();
    expect(quoteWith({ endDate: undefined })).toBeNull();
    expect(quoteWith({ endDate: "not-a-date" })).toBeNull();
  });

  test("a touch claim uses the barrier model, not the terminal one", () => {
    const claim = parseClaim("Will Bitcoin hit $120,000 by January 31?");
    expect(claim?.style).toBe("TOUCH");
    const touch = quoteWith({ claim: claim ?? undefined });
    const terminal = quoteWith();
    expect(touch).not.toBeNull();
    // Touching is strictly likelier than finishing above the same level, so the
    // two models must not agree — that difference IS the reason both exist.
    expect((touch as NonNullable<typeof touch>).modelProbability).toBeGreaterThan(
      (terminal as NonNullable<typeof terminal>).modelProbability,
    );
    // No closed-form greeks for a barrier.
    expect((touch as NonNullable<typeof touch>).greeks).toBeNull();
  });

  test("a barrier on the wrong side of spot is refused", () => {
    // "hits $80,000" with spot at $100,000 is a DOWN touch; asking for it as an
    // UP touch is incoherent and must not be priced.
    const claim = parseClaim("Will Bitcoin hit $80,000 by January 31?");
    expect(claim?.style).toBe("TOUCH");
    expect(claim?.direction).toBe("UP");
    expect(quoteWith({ claim: claim ?? undefined })).toBeNull();
  });

  test("net expected profit is EV on the stake minus the hedge cost", () => {
    const withBook = quoteWith({ book: ladder(100_000, 5, 5, 40) });
    const q = withBook as NonNullable<typeof withBook>;
    expect(q.book).not.toBeNull();
    expect(q.netExpectedUsd).not.toBeNull();

    const book = q.book as NonNullable<typeof q.book>;
    const hedgeCost = (book.hedgeNotionalUsd * book.hedgeSlippageBps) / 10_000;
    expect(q.netExpectedUsd as number).toBeCloseTo(10_000 * q.ev - hedgeCost, 6);
    // Carrying the hedge can only reduce the expected result.
    expect(q.netExpectedUsd as number).toBeLessThanOrEqual(10_000 * q.ev + 1e-9);
  });

  test("no ladder means no profit claim, rather than a hedge-free one", () => {
    expect((quoteWith() as NonNullable<ReturnType<typeof quoteWith>>).netExpectedUsd).toBeNull();
  });

  test("a strike orders of magnitude from spot is refused", () => {
    // Defence in depth behind the parser: even if a bad strike gets through,
    // it must not reach the model, where it would price to a hard 0 or 1.
    const absurd = parseClaim("Bitcoin above $31 on January 31?");
    expect(absurd?.strike).toBe(31);
    expect(quoteWith({ claim: absurd ?? undefined })).toBeNull();
  });

  test("a flat tape with no volatility is refused", () => {
    const flat: Candle[] = Array.from({ length: 200 }, (_, i) => ({
      t: i * 3_600_000,
      o: 100_000,
      h: 100_000,
      l: 100_000,
      c: 100_000,
      v: 1,
    }));
    expect(quoteWith({ candles: flat })).toBeNull();
  });
});
