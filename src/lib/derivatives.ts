import { asksAscending, bidsDescending, buyDollars } from "./fills";
import {
  averageFunding,
  type CandleInterval,
  type FundingComparison,
  getCandles,
  getFundingComparison,
  getFundingHistory,
  getPerpBook,
  getPerpContexts,
  INTERVAL_MINUTES,
  indexPerps,
  intervalForHorizon,
  type PerpContext,
} from "./hyperliquid";
import type { Market, OrderBook, Outcome } from "./types";
import {
  basisBps,
  type Candle,
  carryDrift,
  clamp,
  type DigitalQuote,
  digitalCall,
  digitalProbabilityExtremum,
  digitalPut,
  edgeZ,
  expectedValue,
  forwardFromFunding,
  fundingApr,
  impliedVolFromDigitalCall,
  impliedVolFromTouch,
  kellyBinary,
  msToYears,
  pickImpliedVol,
  priceDispersionBps,
  probabilityBand,
  touchProbability,
  type VolSuite,
  varianceRatio,
  volSuite,
} from "./options";

/**
 * The bridge: Polymarket claim ⇄ Hyperliquid continuous market.
 *
 * `quant.ts` knows how to price a digital and `hyperliquid.ts` knows how to
 * fetch a forward and a vol — but neither knows what a Polymarket market MEANS.
 * That translation is this file's whole job, and it is the part that is easy to
 * get quietly wrong:
 *
 *   1. WHICH UNDERLYING. "Bitcoin" / "BTC" / "₿" all mean the HL coin `BTC`.
 *   2. WHICH STRIKE. "$120,000", "$120k", "120K" are the same number.
 *   3. WHICH PAYOFF. This is the one that matters. "BTC ABOVE $120k ON Dec 31"
 *      is a TERMINAL digital — only the settlement price counts. "BTC HITS
 *      $120k BY Dec 31" is a ONE-TOUCH — any path that crosses pays. Pricing a
 *      touch market with a terminal digital under-prices it by roughly half
 *      near the barrier. We parse the phrasing and pick the right model.
 *   4. WHICH DIRECTION. above/hits ⇒ call/up-touch; below/dips ⇒ put/down-touch.
 *
 * A market we cannot confidently classify returns `null` rather than a guess —
 * a wrong model is worse than no model, and the desk prints "UNPARSED" instead.
 *
 * Not financial advice. Descriptive models of public market data.
 */

/* ────────────────────────────── parsing ────────────────────────────── */

/**
 * Aliases → Hyperliquid coin symbol. Longest-match-first when scanning a title,
 * so "Bitcoin Cash" can never be swallowed by "Bitcoin" (see `matchCoin`).
 */
const COIN_ALIASES: Record<string, string> = {
  bitcoin: "BTC",
  btc: "BTC",
  ethereum: "ETH",
  ether: "ETH",
  eth: "ETH",
  solana: "SOL",
  sol: "SOL",
  ripple: "XRP",
  xrp: "XRP",
  dogecoin: "DOGE",
  doge: "DOGE",
  cardano: "ADA",
  ada: "ADA",
  avalanche: "AVAX",
  avax: "AVAX",
  chainlink: "LINK",
  link: "LINK",
  litecoin: "LTC",
  ltc: "LTC",
  polkadot: "DOT",
  dot: "DOT",
  sui: "SUI",
  aptos: "APT",
  apt: "APT",
  arbitrum: "ARB",
  arb: "ARB",
  optimism: "OP",
  hyperliquid: "HYPE",
  hype: "HYPE",
  toncoin: "TON",
  ton: "TON",
  tron: "TRX",
  trx: "TRX",
  shiba: "kSHIB",
  pepe: "kPEPE",
  bonk: "kBONK",
};

/** Aliases we must never match as a substring of a longer, different coin. */
const COIN_BLOCKERS = ["bitcoin cash", "bch", "ethereum classic", "etc"];

/** Whether the claim settles on the terminal price or on ever touching. */
export type PayoffStyle = "TERMINAL" | "TOUCH";
export type Direction = "UP" | "DOWN";

export interface ParsedClaim {
  coin: string;
  strike: number;
  style: PayoffStyle;
  direction: Direction;
  /** The phrase that produced the classification — shown so it can be audited. */
  evidence: string;
}

/**
 * Words that make a claim path-dependent. "Hits", "reaches", "touches" and
 * "by <date>" all pay on ANY crossing; "above ... on <date>" does not.
 *
 * The downside forms matter as much as the upside ones: "will BTC DIP TO $80k"
 * is every bit as path-dependent as "will BTC HIT $150k", and leaving them out
 * meant those markets fell through to the terminal model — or, because they
 * carry no terminal keyword either, were dropped entirely.
 */
const TOUCH_WORDS =
  /\b(hit|hits|reach|reaches|reached|touch|touches|trade at|trades at|cross|crosses|get to|gets to|dips? to|drops? to|falls? to|sinks? to|crash(?:es)? to|return to|returns to)\b/i;
const TERMINAL_WORDS =
  /\b(above|below|over|under|greater than|less than|at or above|at or below|close|closes|end|ends)\b/i;
const DOWN_WORDS =
  /\b(below|under|less than|dip|dips|drop|drops|fall|falls|down to|crash|crashes)\b/i;

/**
 * A number that could be a price threshold.
 *
 * Two accepted shapes, and the distinction is load-bearing:
 *   - properly comma-grouped: `120,000` / `3,000` — groups of exactly three.
 *   - plain digits carrying a `$` or a k/m/b magnitude: `$250.50`, `150k`.
 *
 * A bare integer is NEVER a strike. That is what keeps years and day-of-month
 * out. The grouping alternative demands `,\d{3}` with no space, which is what
 * separates a real `3,000` from the `31, 2026` inside a date — the earlier
 * "contains a comma" test accepted the latter and returned 31 as a Bitcoin
 * strike, producing a claim that prices to ~100% and a fabricated edge.
 */
const STRIKE_RE = /(\$\s?)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?([kmb])?\b/gi;

/** Properly grouped thousands, e.g. `1,234,567`. */
const GROUPED_RE = /^\d{1,3}(?:,\d{3})+(?:\.\d+)?$/;

/**
 * Words that join two thresholds into a corridor rather than a single level.
 */
const RANGE_JOINER = /\b(and|to)\b|[-–—]/i;

/**
 * Every number in the text that qualifies as a price threshold, with where it
 * was found. Qualifying means a `$` prefix, a k/m/b magnitude, or proper
 * thousands grouping — a bare integer is never a strike, which is what keeps
 * years and days out.
 */
function collectStrikes(text: string): { value: number; start: number; end: number }[] {
  const out: { value: number; start: number; end: number }[] = [];
  for (const m of text.matchAll(STRIKE_RE)) {
    const digits = m[2];
    const suffix = m[3]?.toLowerCase();
    if (!m[1] && !suffix && !GROUPED_RE.test(digits)) continue;
    let value = Number.parseFloat(digits.replace(/,/g, ""));
    if (!Number.isFinite(value) || value <= 0) continue;
    if (suffix === "k") value *= 1e3;
    else if (suffix === "m") value *= 1e6;
    else if (suffix === "b") value *= 1e9;
    out.push({ value, start: m.index ?? 0, end: (m.index ?? 0) + m[0].length });
  }
  return out;
}

/**
 * Pull a price threshold out of a title, or null when there isn't exactly one.
 *
 * A corridor ("between $100k and $120k") is refused: taking its first number
 * would price a one-sided claim the market never offered. But the test has to
 * be made against QUALIFYING strikes, not against any hyphenated digits — a
 * date range like "August 3-9" is not a price range, and treating it as one
 * rejected four out of every five real crypto markets on the live board.
 */
export function parseStrike(text: string): number | null {
  const found = collectStrikes(text);
  if (found.length === 0) return null;
  if (found.length >= 2) {
    // Two thresholds joined by "and"/"to"/a dash is a corridor.
    const between = text.slice(found[0].end, found[1].start);
    if (RANGE_JOINER.test(between)) return null;
  }
  return found[0].value;
}

/**
 * Aliases pre-sorted longest-first with their word-boundary matchers already
 * compiled. `matchCoin` runs once per event on every desk build, so sorting the
 * key list and constructing ~30 `RegExp` objects inside it meant thousands of
 * throwaway allocations per request. Both are constant — hoist them.
 */
const COIN_MATCHERS: { pattern: RegExp; coin: string }[] = Object.keys(COIN_ALIASES)
  .toSorted((a, b) => b.length - a.length)
  .map((alias) => ({
    // Word-boundary match so "dot" doesn't fire inside "dotted".
    pattern: new RegExp(`\\b${alias}\\b`, "i"),
    coin: COIN_ALIASES[alias],
  }));

/** Find the coin mentioned in a title, longest alias first. */
export function matchCoin(text: string): string | null {
  const lower = text.toLowerCase();
  for (const blocked of COIN_BLOCKERS) {
    if (lower.includes(blocked)) return null;
  }
  for (const { pattern, coin } of COIN_MATCHERS) {
    if (pattern.test(lower)) {
      return coin;
    }
  }
  return null;
}

/**
 * Classify a Polymarket title into a priceable claim, or null when the phrasing
 * is ambiguous. Deliberately conservative: a market we half-understand is
 * excluded from the desk rather than priced with the wrong payoff.
 */
export function parseClaim(title: string): ParsedClaim | null {
  const coin = matchCoin(title);
  if (!coin) return null;
  const strike = parseStrike(title);
  if (strike === null) return null;

  const touchMatch = title.match(TOUCH_WORDS);
  const terminalMatch = title.match(TERMINAL_WORDS);
  // "Hits" wins over "above" — "will BTC hit $120k" is path-dependent even
  // though a stray "above" may appear elsewhere in the sentence.
  const style: PayoffStyle = touchMatch ? "TOUCH" : terminalMatch ? "TERMINAL" : "TERMINAL";
  if (!touchMatch && !terminalMatch) return null; // no comparison word at all

  const downMatch = title.match(DOWN_WORDS);
  const direction: Direction = downMatch ? "DOWN" : "UP";

  return {
    coin,
    strike,
    style,
    direction,
    evidence: (touchMatch?.[0] ?? terminalMatch?.[0] ?? "").toUpperCase(),
  };
}

/* ──────────────────────────── pricing ──────────────────────────── */

export interface DerivativeQuote {
  claim: ParsedClaim;
  /** Polymarket slug + title this quote belongs to. */
  slug: string;
  title: string;
  /** The Polymarket outcome we priced against (the YES side of the claim). */
  outcomeLabel: string;
  /** Market's implied probability, 0..1. */
  marketProbability: number;
  /** Model's probability under GBM with HL's forward and vol. */
  modelProbability: number;
  /** model − market, in probability points (0.06 = 6pp of edge). */
  edge: number;
  /** Edge as a multiple of the model's own uncertainty. |z| < 1 is noise. */
  z: number;
  /** Model uncertainty band from the vol-estimator spread. */
  band: { lo: number; hi: number; width: number };
  /** Vol the market price implies, inverted through the same model. */
  impliedVol: number | null;
  /** True when no σ reproduces the market price — GBM can't express it. */
  unattainable: boolean;
  /** Realized-vol suite from HL candles. */
  vol: VolSuite;
  /** Signed Kelly fraction at the desk's quarter-Kelly default. */
  kelly: number;
  /** EV per $1 staked buying the YES side at the market price. */
  ev: number;
  /**
   * Expected profit on a REFERENCE_POSITION_USD stake, in dollars, AFTER the
   * cost of crossing Hyperliquid's ladder to put the delta hedge on. EV alone
   * flatters every position: a 6% edge that costs 80bps of hedge slippage on a
   * hedge twice the size of the position is not a 6% trade. Null when no ladder
   * was available to cost the hedge against.
   */
  netExpectedUsd: number | null;
  /** HL validator-median oracle price — the settlement reference. */
  spot: number;
  forward: number;
  /** Continuously-compounded carry implied by the forward. */
  drift: number;
  /** Time-averaged realized hourly funding over the last week. */
  fundingHourly: number;
  fundingAprPct: number;
  /** HL funding vs the mean of Binance/Bybit, in bps/hour. Null if unlisted. */
  crossVenueBpsPerHour: number | null;
  hoursToExpiry: number;
  years: number;
  /** Lo–MacKinlay VR(4). ≈1 random walk, >1 trending, <1 mean-reverting. */
  varianceRatio: number;
  /** Cross-venue price dispersion (oracle vs mark vs mid), in bps. */
  dispersionBps: number;
  /** Perp mark vs oracle, in bps. Positive = perp rich to index. */
  basisBps: number;
  /** Cost of crossing HL's own book at $20k notional, in bps. */
  impactSpreadBps: number | null;
  /** HL open interest in base units, and 24h notional turnover in USD. */
  openInterest: number;
  perpDayVolume: number;
  /** Perp 24h return, decimal. The underlying's own momentum. */
  perpChange24h: number;
  /**
   * Live Hyperliquid ladder read — what the delta hedge actually costs. Null
   * when the book could not be fetched; the row still prices without it.
   */
  book: BookLiquidity | null;
  /** Terminal digital greeks. Null for touch claims (no closed form). */
  greeks: DigitalQuote | null;
}

/**
 * How far from spot a strike may sit before we treat it as a parsing failure.
 * Generous — genuine long-dated crypto claims do reach several multiples — but
 * decisive about the orders of magnitude that only a bad parse produces.
 */
const STRIKE_SANITY_MULTIPLE = 50;

/** Quarter-Kelly: ~94% of the growth at ~44% of the drawdown. */
export const KELLY_FRACTION = 0.25;

/**
 * Reference position the hedge is sized from: $10k of the digital itself.
 *
 * The hedge clip is NOT this number — it is derived from it. A $10k position in
 * a 5-delta contract needs a very different hedge from $10k in a 60-delta one,
 * so costing every claim at the same flat notional would say nothing about the
 * claim in front of you. See `hedgeNotionalFor`.
 */
export const REFERENCE_POSITION_USD = 10_000;

/** How far from mid still counts as "depth you could actually lift". */
const DEPTH_BAND_BPS = 25;

/**
 * Below this, the hedge is dust and there is nothing to cost.
 *
 * A deep in-the-money digital has delta ~ 0 — it already behaves like cash, so
 * it needs no perp against it. Dividing depth by a sub-dollar hedge produced a
 * coverage of 4.6e9x against live data, which reads as a number rather than as
 * "no hedge required". Infinity says the true thing.
 */
const MIN_HEDGE_USD = 1;

export interface BookLiquidity {
  /** True top-of-book spread, in bps of mid. */
  spreadBps: number;
  /** Resting notional within ±25bps of mid, both sides, in USD. */
  depthUsd: number;
  /** The delta-derived clip this book was walked for, in USD. */
  hedgeNotionalUsd: number;
  /** Slippage vs the touch for that clip, in bps. */
  hedgeSlippageBps: number;
  /** False when the ladder could not absorb the clip at all. */
  hedgeFilled: boolean;
  /**
   * `depthUsd / hedgeNotionalUsd`. Under 1 means the near book cannot cover
   * the hedge — the actionable read, since a raw depth figure means nothing
   * until you compare it to the size you actually need.
   */
  depthCoverage: number;
}

/**
 * Hedge clip implied by a claim's delta.
 *
 * A digital paying $1 has delta = ∂price/∂S. Holding N contracts leaves you
 * long N·delta of the underlying, so the offsetting perp position is
 * |N·delta|·S of notional. N itself is the reference position divided by what
 * a contract costs. That chain is why a cheap far-out-of-the-money claim can
 * need a LARGER hedge than an expensive near-the-money one: you own far more
 * contracts for the same dollars.
 */
export function hedgeNotionalFor(
  delta: number,
  spot: number,
  contractPrice: number,
  positionUsd = REFERENCE_POSITION_USD
): number {
  if (!(spot > 0) || !Number.isFinite(delta)) return 0;
  const price = clamp(contractPrice, 0.01, 0.99);
  const contracts = positionUsd / price;
  return Math.abs(contracts * delta) * spot;
}

/**
 * Read Hyperliquid's actual ladder rather than its summary statistics.
 *
 * `impactPxs` already gives HL's own $20k impact quote, but it is a single
 * number with no depth behind it. Walking the real book answers the question a
 * desk actually has — "if the model says this claim is mispriced, what does it
 * cost me to put on the delta hedge right now" — and it does so with the SAME
 * `buyDollars` walker used on the Polymarket leg, so a cross-venue trade is
 * costed with one slippage model on both sides instead of two.
 */
export function bookLiquidity(
  book: OrderBook,
  notionalUsd = REFERENCE_POSITION_USD
): BookLiquidity | null {
  const asks = asksAscending(book);
  const bids = bidsDescending(book);
  const bestAsk = asks[0]?.price;
  const bestBid = bids[0]?.price;
  if (!(bestAsk > 0) || !(bestBid > 0)) return null;

  const mid = (bestAsk + bestBid) / 2;
  if (!(mid > 0)) return null;

  const band = mid * (DEPTH_BAND_BPS / 10_000);
  let depthUsd = 0;
  for (const level of asks) {
    if (level.price > mid + band) break;
    depthUsd += level.price * level.size;
  }
  for (const level of bids) {
    if (level.price < mid - band) break;
    depthUsd += level.price * level.size;
  }

  const negligible = notionalUsd < MIN_HEDGE_USD;
  const fill = buyDollars(book, notionalUsd);
  return {
    spreadBps: ((bestAsk - bestBid) / mid) * 10_000,
    depthUsd,
    hedgeNotionalUsd: notionalUsd,
    // A hedge too small to place costs nothing and always "fills".
    hedgeSlippageBps: negligible ? 0 : fill.slippageBps,
    hedgeFilled: negligible ? true : fill.filled,
    depthCoverage: negligible ? Number.POSITIVE_INFINITY : depthUsd / notionalUsd,
  };
}

/**
 * Price one parsed claim against a live Hyperliquid context + candle history.
 *
 * Pure given its inputs — all I/O happens in `buildDesk` — so this is the
 * function to unit-test and the one to read when auditing a printed number.
 */
export function priceClaim(args: {
  claim: ParsedClaim;
  slug: string;
  title: string;
  outcomeLabel: string;
  marketProbability: number;
  perp: PerpContext;
  candles: Candle[];
  barMinutes: number;
  fundingHourly: number;
  /** Cross-venue funding for this coin, when HL publishes peers for it. */
  funding?: FundingComparison | null;
  /** Live HL ladder for this coin, when it could be fetched. */
  book?: OrderBook | null;
  endDate: string | undefined;
  now?: number;
}): DerivativeQuote | null {
  const {
    claim,
    slug,
    title,
    outcomeLabel,
    marketProbability,
    perp,
    candles,
    barMinutes,
    fundingHourly,
    endDate,
  } = args;
  const now = args.now ?? Date.now();

  const spot = perp.oraclePx;
  if (!(spot > 0)) return null;

  // Last line of defence against a mis-parsed strike. A real crypto claim sits
  // within a couple of multiples of spot; a strike 50x away is a parsing
  // artifact (a year, a day-of-month, a volume figure), and it would price to a
  // hard 0 or 1 and print an enormous fake edge rather than looking wrong.
  if (
    claim.strike > spot * STRIKE_SANITY_MULTIPLE ||
    claim.strike < spot / STRIKE_SANITY_MULTIPLE
  ) {
    return null;
  }

  const expiryMs = endDate ? new Date(endDate).getTime() : Number.NaN;
  if (!Number.isFinite(expiryMs)) return null;
  const hoursToExpiry = (expiryMs - now) / 3_600_000;
  if (!(hoursToExpiry > 0)) return null;
  const years = msToYears(expiryMs - now);

  const vol = volSuite(candles, barMinutes);
  const sigma = vol.blended;
  if (!(sigma > 0)) return null;

  const forward = forwardFromFunding(spot, fundingHourly, hoursToExpiry);
  const drift = carryDrift(spot, forward, years);

  let modelProbability: number;
  let greeks: DigitalQuote | null = null;
  let impliedVol: number | null = null;
  let unattainable = false;
  let vega = 0;

  if (claim.style === "TOUCH") {
    modelProbability = touchProbability(spot, claim.strike, years, sigma, drift);
    impliedVol = impliedVolFromTouch(marketProbability, spot, claim.strike, years, drift);
    unattainable = impliedVol === null;
    // No closed-form vega for a one-touch: bump σ by 1% and difference it.
    const bump = Math.max(sigma * 0.01, 1e-4);
    vega =
      (touchProbability(spot, claim.strike, years, sigma + bump, drift) - modelProbability) / bump;
  } else {
    const inputs = {
      spot,
      strike: claim.strike,
      years,
      sigma,
      forward,
    };
    const quote = claim.direction === "UP" ? digitalCall(inputs) : digitalPut(inputs);
    greeks = quote;
    modelProbability = quote.probability;
    vega = quote.vega;

    // Invert the market price through the SAME payoff we priced with. The put
    // is the call's complement, so invert 1 − p for a down claim.
    const callProbability = claim.direction === "UP" ? marketProbability : 1 - marketProbability;
    const roots = impliedVolFromDigitalCall(callProbability, forward, claim.strike, years);
    impliedVol = pickImpliedVol(roots, sigma);
    if (roots === null) {
      // The extremum is a CEILING: out of the money the price rises to
      // Φ(−√(−2m)) at σ* and falls away again, so a quote is unreachable when
      // it sits ABOVE that cap, never below. Comparing the wrong way round
      // labels ordinary cheap quotes "unattainable" and silently hides the
      // genuinely inexpressible ones.
      const cap = digitalProbabilityExtremum(forward, claim.strike, years);
      unattainable = cap === null || callProbability > cap.probability;
    }
  }

  if (claim.style === "TOUCH" && claim.direction === "DOWN") {
    // `touchProbability` reads the barrier's side off spot, so a down-touch is
    // already correct — but a barrier ABOVE spot on a DOWN claim is nonsense.
    if (claim.strike > spot) return null;
  }
  if (claim.style === "TOUCH" && claim.direction === "UP" && claim.strike < spot) {
    return null;
  }

  // Delta for the hedge clip. The terminal digital has a closed form; the
  // one-touch does not, so it is bumped numerically the same way its vega is.
  let delta = greeks?.delta ?? 0;
  if (claim.style === "TOUCH") {
    const bump = Math.max(spot * 1e-4, 1e-8);
    delta =
      (touchProbability(spot + bump, claim.strike, years, sigma, drift) - modelProbability) / bump;
  }
  const hedgeNotionalUsd = hedgeNotionalFor(delta, spot, marketProbability);

  const book = args.book ? bookLiquidity(args.book, hedgeNotionalUsd) : null;

  const band = probabilityBand(modelProbability, vega, vol.spread);
  const edge = modelProbability - marketProbability;
  const z = edgeZ(edge, band.width / 2);
  const kelly = kellyBinary(modelProbability, marketProbability, KELLY_FRACTION);
  const ev = expectedValue(modelProbability, marketProbability);

  return {
    claim,
    slug,
    title,
    outcomeLabel,
    marketProbability,
    modelProbability,
    edge,
    z,
    band,
    impliedVol,
    unattainable,
    vol,
    kelly,
    ev,
    spot,
    forward,
    drift,
    fundingHourly,
    fundingAprPct: fundingApr(fundingHourly) * 100,
    crossVenueBpsPerHour: args.funding?.dislocationBpsPerHour ?? null,
    hoursToExpiry,
    years,
    varianceRatio: varianceRatio(candles, 4),
    dispersionBps: priceDispersionBps([perp.oraclePx, perp.markPx, perp.midPx]),
    basisBps: basisBps(perp.markPx, perp.oraclePx),
    impactSpreadBps:
      perp.impactBid > 0 && perp.impactAsk > 0
        ? ((perp.impactAsk - perp.impactBid) / ((perp.impactAsk + perp.impactBid) / 2)) * 10_000
        : null,
    openInterest: perp.openInterest,
    perpDayVolume: perp.dayNotionalVolume,
    perpChange24h: perp.prevDayPx > 0 ? perp.markPx / perp.prevDayPx - 1 : 0,
    book,
    netExpectedUsd:
      book === null
        ? null
        : REFERENCE_POSITION_USD * ev - (book.hedgeNotionalUsd * book.hedgeSlippageBps) / 10_000,
    greeks,
  };
}

/* ──────────────────────────── the desk ──────────────────────────── */

/** How many bars of history each horizon gets. More bars, steadier σ. */
const LOOKBACK_BARS = 240;

export interface Desk {
  rows: DerivativeQuote[];
  /** Markets that mentioned a coin but could not be classified. */
  unparsed: { slug: string; title: string; reason: string }[];
  /** Perp contexts for the coins we actually priced — the venue strip. */
  perps: PerpContext[];
  /** Cross-venue funding for those coins. */
  funding: Map<string, FundingComparison>;
}

/**
 * Find the YES outcome of a claim within an event.
 *
 * A binary crypto market expands to Yes/No; a multi-market event ("What price
 * will BTC hit in July?") expands to one row per strike bucket, and each row's
 * own title carries its own strike. We handle both by preferring an outcome
 * literally labelled "Yes" and falling back to the leading outcome.
 */
function yesOutcome(outcomes: Outcome[]): Outcome | null {
  const yes = outcomes.find((o) => /^yes$/i.test(o.label.trim()));
  return yes ?? outcomes[0] ?? null;
}

/**
 * Build the whole derivatives desk for a set of Polymarket events.
 *
 * Fetches are deduped by coin — a board with twelve BTC markets pulls BTC's
 * candles and funding once, not twelve times. Any single coin failing to load
 * drops only its own rows.
 */
export async function buildDesk(
  markets: Market[],
  opts: { now?: number; maxRows?: number } = {}
): Promise<Desk> {
  const now = opts.now ?? Date.now();
  const maxRows = opts.maxRows ?? 40;

  const unparsed: Desk["unparsed"] = [];
  const candidates: {
    claim: ParsedClaim;
    slug: string;
    title: string;
    outcome: Outcome;
    endDate: string | undefined;
  }[] = [];

  for (const m of markets) {
    if (!m.active || m.closed) continue;
    // A market in a multi-outcome event carries its strike in its own question;
    // a standalone binary carries it in the event title. Try the most specific
    // text first and fall back, so "Bitcoin price on July 31 — $120,000" and
    // "Bitcoin above $120,000 on July 31" both resolve.
    const specific = m.groupItemTitle ? `${m.eventTitle ?? ""} ${m.groupItemTitle}` : m.question;
    const claim = parseClaim(specific) ?? parseClaim(m.question) ?? parseClaim(m.eventTitle ?? "");
    const title = m.groupItemTitle
      ? `${m.eventTitle ?? m.question} — ${m.groupItemTitle}`
      : m.question;

    if (!matchCoin(specific) && !matchCoin(m.question) && !matchCoin(m.eventTitle ?? "")) {
      continue; // not a crypto market at all — silent, not "unparsed"
    }
    const outcome = yesOutcome(m.outcomes);
    if (!claim || !outcome) {
      unparsed.push({
        slug: m.eventSlug ?? m.slug,
        title,
        reason: claim ? "NO OUTCOME" : "UNCLASSIFIED PHRASING",
      });
      continue;
    }
    candidates.push({
      claim,
      slug: m.eventSlug ?? m.slug,
      title,
      outcome,
      endDate: m.endDate,
    });
  }

  const empty: Desk = { rows: [], unparsed, perps: [], funding: new Map() };
  if (candidates.length === 0) return empty;

  // The perp universe and the cross-venue funding table are board-wide, so they
  // are pulled once regardless of how many claims we are pricing.
  const [perps, funding] = await Promise.all([
    getPerpContexts().catch(() => null),
    getFundingComparison().catch(() => new Map<string, FundingComparison>()),
  ]);
  if (!perps) return empty;
  const byCoin = indexPerps(perps);

  /** Horizon in hours, or NaN when the market has no usable end date. */
  const horizonOf = (endDate: string | undefined): number =>
    endDate ? (new Date(endDate).getTime() - now) / 3_600_000 : Number.NaN;

  // Candles are keyed by (coin, bar length): a board with twelve BTC markets
  // that all sit in the same horizon bucket pulls BTC's history exactly once.
  const wanted = new Map<string, { coin: string; interval: CandleInterval }>();
  const coinsNeeded = new Set<string>();
  for (const c of candidates) {
    if (!byCoin.has(c.claim.coin)) continue;
    const hours = horizonOf(c.endDate);
    if (!(hours > 0)) continue;
    const interval = intervalForHorizon(hours);
    wanted.set(`${c.claim.coin}|${interval}`, { coin: c.claim.coin, interval });
    coinsNeeded.add(c.claim.coin);
  }

  const candlesByKey = new Map<string, Candle[]>();
  const fundingByCoin = new Map<string, number>();
  const bookByCoin = new Map<string, OrderBook>();

  await Promise.all([
    // One L2 ladder per coin. This is what turns "the model disagrees" into
    // "and here is what acting on it costs" — see `bookLiquidity`.
    ...[...coinsNeeded].map(async (coin) => {
      try {
        bookByCoin.set(coin, await getPerpBook(coin));
      } catch {
        // Depth is supplementary; the claim still prices without it.
      }
    }),
    // Realized funding: one week of prints, averaged. Far steadier than the
    // single next-hour rate, which is noisy enough that pricing a month-out
    // forward off it would be indefensible.
    ...[...coinsNeeded].map(async (coin) => {
      try {
        fundingByCoin.set(coin, averageFunding(await getFundingHistory(coin, 168, now)));
      } catch {
        fundingByCoin.set(coin, byCoin.get(coin)?.fundingHourly ?? 0);
      }
    }),
    ...[...wanted.entries()].map(async ([key, { coin, interval }]) => {
      try {
        candlesByKey.set(key, await getCandles(coin, interval, LOOKBACK_BARS, now));
      } catch {
        // One coin/interval failing loses only its own rows.
      }
    }),
  ]);

  const rows: DerivativeQuote[] = [];
  for (const c of candidates) {
    const perp = byCoin.get(c.claim.coin);
    if (!perp) {
      unparsed.push({
        slug: c.slug,
        title: c.title,
        reason: `NO HL PERP FOR ${c.claim.coin}`,
      });
      continue;
    }
    const hours = horizonOf(c.endDate);
    if (!(hours > 0)) continue;
    const interval = intervalForHorizon(hours);
    const candles = candlesByKey.get(`${c.claim.coin}|${interval}`);
    // Under ~10 bars every vol estimator is noise, not an estimate.
    if (!candles || candles.length < 10) continue;

    const quote = priceClaim({
      claim: c.claim,
      slug: c.slug,
      title: c.title,
      outcomeLabel: c.outcome.label,
      marketProbability: c.outcome.price,
      perp,
      candles,
      barMinutes: INTERVAL_MINUTES[interval],
      fundingHourly: fundingByCoin.get(c.claim.coin) ?? perp.fundingHourly,
      funding: funding.get(c.claim.coin) ?? null,
      book: bookByCoin.get(c.claim.coin) ?? null,
      endDate: c.endDate,
      now,
    });
    if (quote) rows.push(quote);
  }

  // Rank by conviction, not raw edge: |z| is the edge measured in units of the
  // model's OWN uncertainty, so a 4pp edge the model is sure about outranks a
  // 9pp edge sitting inside a ±12pp band.
  const priced = rows.toSorted((a, b) => Math.abs(b.z) - Math.abs(a.z)).slice(0, maxRows);
  const coins = new Set(priced.map((r) => r.claim.coin));

  return {
    rows: priced,
    unparsed: unparsed.slice(0, 12),
    perps: perps.filter((p) => coins.has(p.coin)),
    funding,
  };
}
