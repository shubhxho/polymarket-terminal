/**
 * Signal engine.
 *
 * Pure functions over already-normalised data — no I/O, no React — so the
 * scoring can be reasoned about and tested in isolation from the feeds. It
 * runs server-side for the scanner and client-side on the detail screen.
 *
 * Three rules run through everything here:
 *
 * 1. **A signal must be actionable.** "Price went up" is not a signal; "price
 *    went up on volume six standard deviations above the cross-section, with
 *    the book still bid" is. Anything that merely restates a column already on
 *    screen is left out.
 * 2. **Unusual is measured against a population, not a constant.** Where a
 *    threshold used to be a magic number, it is now a robust z-score against
 *    either the market's own history or the day's cross-section of markets. A
 *    quiet Tuesday and a election night should not need different code.
 * 3. **Silence beats noise.** Every detector returns `null` unless its
 *    preconditions are met, and thin markets are excluded before scoring
 *    rather than being allowed to produce enormous meaningless percentages.
 */

import {
  autocorrelation,
  bandZ,
  bookStats,
  clamp,
  driftPerDay,
  percentileOf,
  realisedVol,
  robustZ,
  saturate,
  volCompression,
  type BookStats,
} from "./quant";
import type { EventSummary, Market, OrderBook, PricePoint, Trade } from "./types";

export type SignalKind =
  | "ARB"
  | "DRIFT"
  | "SURGE"
  | "MOMENTUM"
  | "REVERSAL"
  | "BREAKOUT"
  | "COIL"
  | "WHALE"
  | "IMBALANCE"
  | "LEAN"
  | "TAIL"
  | "EXPIRY"
  | "THIN";

export type Direction = "bullish" | "bearish" | "neutral";

export type Signal = {
  readonly kind: SignalKind;
  readonly direction: Direction;
  /** 0..100. How notable this reading is, comparable across kinds. */
  readonly strength: number;
  /**
   * 0..1. How much the inputs justify the reading — short history, a thin
   * book or a handful of prints all reduce it. Strength says how loud, this
   * says how much to believe it.
   */
  readonly confidence: number;
  /** The standardised statistic behind the call, for transparency. */
  readonly z?: number;
  /** Terse value, for a badge: "6.2σ vol". */
  readonly headline: string;
  /** One sentence a trader can act on. */
  readonly detail: string;
};

export type MarketSignals = {
  readonly market: Market;
  readonly signals: readonly Signal[];
  /** -100..100. Net directional read; sign is bullish-on-YES. */
  readonly bias: number;
  /** 0..100. How much this market deserves attention, regardless of side. */
  readonly heat: number;
  /**
   * 0..100. How much the directional signals agree with each other. Four
   * signals all pointing up is a different proposition from four that cancel,
   * and a bare composite score cannot tell those apart.
   */
  readonly conviction: number;
  readonly stats: MarketStats;
};

/** Derived measures surfaced alongside the signals for the detail rail. */
export type MarketStats = {
  realisedVol: number;
  driftPerDay: number;
  /** Drift divided by volatility — trend quality, not trend size. */
  trendQuality: number;
  autocorrelation: number;
  bandZ: number;
  volCompression: number;
  book: BookStats;
  volumeZ: number;
  moveZ: number;
};

export type ArbOpportunity = {
  readonly event: EventSummary;
  readonly side: "sell-basket" | "buy-basket";
  /** Sum of the leg prices that form the basket. */
  readonly basket: number;
  /** Guaranteed edge in probability points, before fees and slippage. */
  readonly edgePoints: number;
  readonly legs: number;
  /** Thinnest leg's resting liquidity — the real cap on executable size. */
  readonly tightestLegLiquidity: number;
};

// ── Tunables ───────────────────────────────────────────────────────────────
/** Below this turnover, ratios explode off tiny denominators. */
const MIN_VOLUME_24H = 5_000;
/** A print at or above this notional is treated as informed size. */
export const WHALE_NOTIONAL = 10_000;
/** Prices this close to a bound are settled in all but name. */
const SETTLED_BAND = 0.015;

/** Relative weight of each kind when compounding into `heat`. */
const HEAT_WEIGHT: Record<SignalKind, number> = {
  ARB: 1.0,
  DRIFT: 0.8,
  SURGE: 0.9,
  WHALE: 0.85,
  MOMENTUM: 0.75,
  BREAKOUT: 0.75,
  REVERSAL: 0.7,
  IMBALANCE: 0.6,
  LEAN: 0.55,
  COIL: 0.5,
  EXPIRY: 0.5,
  TAIL: 0.4,
  // Execution quality is information about cost, not about value.
  THIN: 0.2,
};

/**
 * Cross-sectional context: the distribution of the whole scanned universe.
 *
 * Passing this in is what lets a detector say "this is a 4σ move *today*"
 * instead of "this is above 3 points", which means something different on a
 * quiet weekend and on an FOMC afternoon.
 */
export type CrossSection = {
  /** log(24h volume / own weekly baseline) for every qualifying market. */
  volumeRatios: number[];
  /** |24h change in points| for every qualifying market. */
  absMoves: number[];
};

export function buildCrossSection(markets: readonly Market[]): CrossSection {
  const volumeRatios: number[] = [];
  const absMoves: number[] = [];
  for (const m of markets) {
    if (m.volume24h < MIN_VOLUME_24H) continue;
    const baseline = m.volume1w / 7;
    if (baseline > 500) volumeRatios.push(Math.log(m.volume24h / baseline));
    absMoves.push(Math.abs(m.chg24h ?? 0));
  }
  return { volumeRatios, absMoves };
}

export type ScoreContext = {
  flow?: FlowStat;
  book?: OrderBook;
  history?: readonly PricePoint[];
  cross?: CrossSection;
};

// ── Detectors ──────────────────────────────────────────────────────────────

/** Turnover unusual against the day's cross-section of markets. */
function surge(m: Market, cross: CrossSection | undefined): Signal | null {
  const baseline = m.volume1w / 7;
  // A market younger than a week has no meaningful baseline.
  if (baseline < 1_000) return null;
  const ratio = m.volume24h / baseline;
  if (ratio < 1.5) return null;

  const z = cross ? robustZ(Math.log(ratio), cross.volumeRatios) : 0;
  // Without a population to compare against, fall back to the raw ratio.
  const strength = cross
    ? Math.round(100 * saturate(Math.max(0, z), 2.5))
    : Math.round(100 * saturate(ratio - 1, 3));
  if (strength < 12) return null;

  const pct = cross ? percentileOf(Math.log(ratio), cross.volumeRatios) : 0.5;
  return {
    kind: "SURGE",
    // Volume says attention, not direction; the 24h move supplies the sign.
    direction: (m.chg24h ?? 0) > 1 ? "bullish" : (m.chg24h ?? 0) < -1 ? "bearish" : "neutral",
    strength,
    confidence: clamp(saturate(m.volume24h, 50_000), 0.3, 1),
    z,
    headline: `${ratio.toFixed(1)}× vol`,
    detail: `Turning over ${ratio.toFixed(1)}× its own weekly average${
      cross ? `, ${(pct * 100).toFixed(0)}th percentile across the board today` : ""
    }.`,
  };
}

/**
 * A trend worth trading, measured as drift per unit of volatility.
 *
 * Raw point movement rewards whichever market is noisiest. Dividing by
 * realised volatility asks the better question — is this repricing large
 * *relative to how much this market normally jumps around* — and lag-1
 * autocorrelation confirms the moves are persisting rather than oscillating.
 */
function momentum(m: Market, history: readonly PricePoint[] | undefined): Signal | null {
  if (!history || history.length < 12) return null;
  const drift = driftPerDay(history);
  const vol = realisedVol(history);
  if (vol < 1e-6) return null;
  const quality = drift / vol;
  if (Math.abs(quality) < 0.35) return null;

  const ac = autocorrelation(history);
  // Persistent increments corroborate a trend; reverting ones undercut it.
  const persistence = clamp(0.5 + ac, 0.2, 1);

  return {
    kind: "MOMENTUM",
    direction: drift > 0 ? "bullish" : "bearish",
    strength: Math.round(100 * saturate(Math.abs(quality), 0.9) * persistence),
    confidence: clamp(saturate(history.length, 60), 0.3, 1),
    z: quality,
    headline: `${drift > 0 ? "+" : ""}${drift.toFixed(1)}pt/d`,
    detail: `Drifting ${drift > 0 ? "up" : "down"} ${Math.abs(drift).toFixed(
      1
    )} points a day against ${vol.toFixed(1)} points of daily noise${
      ac > 0.1 ? ", and the moves are persisting" : ""
    }.`,
  };
}

/** Price stretched beyond its own band while increments start reverting. */
function reversal(m: Market, history: readonly PricePoint[] | undefined): Signal | null {
  if (!history || history.length < 20) return null;
  const z = bandZ(history);
  if (Math.abs(z) < 1.6) return null;
  const ac = autocorrelation(history);
  // A stretched price that is still trending is a breakout, not a fade.
  if (ac > 0.05) return null;

  return {
    kind: "REVERSAL",
    // Stretched high means the fade is downward.
    direction: z > 0 ? "bearish" : "bullish",
    strength: Math.round(100 * saturate(Math.abs(z) - 1.4, 1.4)),
    confidence: clamp(saturate(-ac + 0.15, 0.25), 0.25, 0.9),
    z,
    headline: `${z > 0 ? "+" : ""}${z.toFixed(1)}σ stretched`,
    detail: `Sitting ${Math.abs(z).toFixed(1)} standard deviations ${
      z > 0 ? "above" : "below"
    } its recent range while moves are mean-reverting — the stretch is more likely to unwind than extend.`,
  };
}

/** Stretched *and* still trending, with volatility expanding: a breakout. */
function breakout(history: readonly PricePoint[] | undefined): Signal | null {
  if (!history || history.length < 24) return null;
  const z = bandZ(history);
  if (Math.abs(z) < 1.8) return null;
  const ac = autocorrelation(history);
  if (ac <= 0.05) return null;
  const expansion = volCompression(history);
  if (expansion < 1.2) return null;

  return {
    kind: "BREAKOUT",
    direction: z > 0 ? "bullish" : "bearish",
    strength: Math.round(100 * saturate(Math.abs(z) - 1.5, 1.2) * clamp(expansion / 2, 0.4, 1)),
    confidence: clamp(saturate(ac, 0.2), 0.3, 0.95),
    z,
    headline: `${z > 0 ? "break up" : "break down"}`,
    detail: `Left its range to the ${z > 0 ? "upside" : "downside"} with volatility ${expansion.toFixed(
      1
    )}× its earlier level and moves still persisting.`,
  };
}

/** Volatility contracted hard — a coiled range that tends to resolve. */
function coil(history: readonly PricePoint[] | undefined): Signal | null {
  if (!history || history.length < 30) return null;
  const c = volCompression(history);
  if (c > 0.55) return null;
  const z = bandZ(history);
  // Only interesting while the price is still inside the range.
  if (Math.abs(z) > 1.5) return null;

  return {
    kind: "COIL",
    direction: "neutral",
    strength: Math.round(100 * saturate(0.55 - c, 0.3)),
    confidence: 0.55,
    z: c,
    headline: `${(c * 100).toFixed(0)}% vol`,
    detail: `Volatility has fallen to ${(c * 100).toFixed(
      0
    )}% of its earlier level with price still mid-range — compressed ranges resolve, and this one has not yet.`,
  };
}

/**
 * Net direction of large prints, scaled by the market's own turnover.
 *
 * Absolute whale flow favours whatever is busiest today, so this measures
 * conviction as a share of the market's 24h volume instead.
 */
function whale(m: Market, flow: FlowStat | undefined): Signal | null {
  if (!flow || flow.count === 0) return null;
  if (Math.abs(flow.net) < WHALE_NOTIONAL) return null;
  const share = Math.abs(flow.net) / Math.max(m.volume24h, 1);
  // One-sidedness: net over gross. A whale that bought and sold equally is not
  // expressing a view, however large the tickets were.
  const oneSided = flow.gross > 0 ? Math.abs(flow.net) / flow.gross : 0;

  return {
    kind: "WHALE",
    direction: flow.net > 0 ? "bullish" : "bearish",
    strength: Math.round(100 * (saturate(share, 0.08) * 0.6 + oneSided * 0.4)),
    confidence: clamp(saturate(flow.count, 3) * oneSided + 0.2, 0.2, 1),
    headline: `${flow.net > 0 ? "+" : "−"}$${abbreviate(Math.abs(flow.net))}`,
    detail: `${flow.count} block${flow.count === 1 ? "" : "s"} ≥ $${abbreviate(
      WHALE_NOTIONAL
    )} netting ${flow.net > 0 ? "buy" : "sell"} $${abbreviate(Math.abs(flow.net))} — ${(
      share * 100
    ).toFixed(0)}% of 24h volume, ${(oneSided * 100).toFixed(0)}% one-sided.`,
  };
}

/** Resting capital skewed to one side, weighted by what each side risks. */
function imbalance(stats: BookStats): Signal | null {
  const total = stats.bidNotional + stats.askNotional;
  if (total < 5_000) return null;
  if (Math.abs(stats.imbalance) < 0.35) return null;

  return {
    kind: "IMBALANCE",
    direction: stats.imbalance > 0 ? "bullish" : "bearish",
    strength: Math.round(100 * saturate(Math.abs(stats.imbalance) - 0.3, 0.35)),
    confidence: clamp(saturate(total, 40_000), 0.25, 1),
    z: stats.imbalance,
    headline: `${stats.imbalance > 0 ? "+" : ""}${(stats.imbalance * 100).toFixed(0)}% book`,
    detail: `Resting capital is ${(Math.abs(stats.imbalance) * 100).toFixed(0)}% skewed to the ${
      stats.imbalance > 0 ? "bid" : "ask"
    } — $${abbreviate(stats.bidNotional)} bid against $${abbreviate(stats.askNotional)} offered.`,
  };
}

/**
 * Micro-price lean: where the book says the next trade actually goes.
 *
 * Weighting each side by the *opposite* side's size puts fair value nearer the
 * thin side, because that is the side that gets taken. It is the shortest-horizon
 * signal here and the only one that speaks to the very next tick.
 */
function lean(stats: BookStats): Signal | null {
  if (stats.microPrice === undefined || stats.mid === undefined) return null;
  if (stats.spread === undefined || stats.spread <= 0) return null;
  // Only meaningful relative to the spread it sits inside.
  const fraction = stats.microLean / (stats.spread * 100);
  if (Math.abs(fraction) < 0.18) return null;

  return {
    kind: "LEAN",
    direction: stats.microLean > 0 ? "bullish" : "bearish",
    strength: Math.round(100 * saturate(Math.abs(fraction) - 0.15, 0.3)),
    confidence: clamp(saturate(stats.bidNotional + stats.askNotional, 25_000), 0.2, 0.85),
    z: fraction,
    headline: `${stats.microLean > 0 ? "+" : ""}${stats.microLean.toFixed(2)}¢ lean`,
    detail: `Size-weighted fair value sits ${Math.abs(stats.microLean).toFixed(2)}¢ ${
      stats.microLean > 0 ? "above" : "below"
    } mid — ${(Math.abs(fraction) * 100).toFixed(0)}% of the way across the spread toward the ${
      stats.microLean > 0 ? "offer" : "bid"
    }.`,
  };
}

/** Extreme price still attracting real money — the favourite-longshot zone. */
function tail(m: Market): Signal | null {
  const p = m.last;
  if (p > 0.06 && p < 0.94) return null;
  if (p <= SETTLED_BAND || p >= 1 - SETTLED_BAND) return null;
  if (m.volume24h < 25_000) return null;
  const isLongshot = p <= 0.06;

  return {
    kind: "TAIL",
    direction: "neutral",
    strength: Math.round(100 * saturate(m.volume24h / 25_000, 3)),
    confidence: 0.5,
    headline: isLongshot ? "longshot" : "favourite",
    detail: `Trading at ${(p * 100).toFixed(1)}¢ on $${abbreviate(m.volume24h)} of 24h volume — ${
      isLongshot
        ? "longshots price rich, so the offer is usually the better side"
        : "heavy favourites price cheap relative to true odds"
    }.`,
  };
}

/** Close to resolution and still genuinely uncertain: maximum sensitivity. */
function expiry(m: Market): Signal | null {
  if (!m.endDate) return null;
  const hours = (new Date(m.endDate).getTime() - Date.now()) / 3_600_000;
  if (!Number.isFinite(hours) || hours <= 0 || hours > 72) return null;
  if (m.last < 0.12 || m.last > 0.88) return null;

  return {
    kind: "EXPIRY",
    direction: "neutral",
    strength: Math.round(100 * saturate(72 - hours, 36)),
    confidence: 0.8,
    headline: hours < 24 ? `${hours.toFixed(0)}h left` : `${(hours / 24).toFixed(0)}d left`,
    detail: `Resolves in ${
      hours < 24 ? `${hours.toFixed(0)} hours` : `${(hours / 24).toFixed(1)} days`
    } while still priced at ${(m.last * 100).toFixed(0)}¢ — small news moves this hard from here.`,
  };
}

/**
 * Execution warning: crossing costs more than most edges are worth.
 *
 * A book is expensive to trade when it is *either* wide *or* hollow, and the
 * two are distinct hazards — a one-tick spread with nothing resting behind it
 * fills once and gaps, which a spread-only check misses entirely. So this fires
 * on a wide relative spread or a low two-way liquidity score, and names the
 * thinner side, since that is the direction that will actually hurt.
 */
function thin(m: Market, stats: BookStats): Signal | null {
  const spread = stats.spread ?? m.spread;
  const mid = stats.mid ?? m.last;
  if (spread === undefined || mid <= 0 || spread < 0.01) return null;
  const relative = spread / mid;

  // With a real book, hollowness counts too; on a bare quote, only the spread.
  const haveBook = stats.bestBid !== undefined;
  const hollow = haveBook ? clamp((0.2 - stats.liquidityScore) / 0.2, 0, 1) : 0;
  const wide = saturate(Math.max(0, relative - 0.05), 0.2);
  const severity = Math.max(wide, hollow);
  if (severity < 0.15) return null;

  // Which direction is the trap: the side with less resting within a cent.
  const thinnerSide = stats.costUpOneCent <= stats.costDownOneCent ? "offer" : "bid";
  const thinnerDepth = Math.min(stats.costUpOneCent, stats.costDownOneCent);

  return {
    kind: "THIN",
    direction: "neutral",
    strength: Math.round(100 * severity),
    confidence: 0.7,
    z: relative,
    headline: `${(spread * 100).toFixed(1)}¢ wide`,
    detail: `Spread is ${(relative * 100).toFixed(0)}% of mid${
      thinnerDepth > 0
        ? `, with only $${abbreviate(thinnerDepth)} resting within a cent on the ${thinnerSide}`
        : ""
    } — crossing costs more than most edges are worth.`,
  };
}

// ── Aggregation ────────────────────────────────────────────────────────────

export type FlowStat = { net: number; count: number; gross: number };

/**
 * Nets large prints per market.
 *
 * A BUY of the YES token and a SELL of the NO token are the same directional
 * bet, so the outcome index flips the sign — otherwise a two-sided event's
 * flow cancels itself out to nothing.
 */
export function aggregateFlow(trades: readonly Trade[]): Map<string, FlowStat> {
  const out = new Map<string, FlowStat>();
  for (const t of trades) {
    const value = t.size * t.price;
    if (value < WHALE_NOTIONAL) continue;
    const yesSide = t.outcomeIndex === 0;
    const buying = t.side === "BUY";
    // Bullish-on-YES when buying YES or selling NO.
    const signed = (yesSide === buying ? 1 : -1) * value;
    const prev = out.get(t.conditionId) ?? { net: 0, count: 0, gross: 0 };
    out.set(t.conditionId, {
      net: prev.net + signed,
      count: prev.count + 1,
      gross: prev.gross + value,
    });
  }
  return out;
}

/** Runs every detector over one market and compounds the results. */
export function scoreMarket(market: Market, ctx: ScoreContext = {}): MarketSignals | null {
  if (market.volume24h < MIN_VOLUME_24H) return null;
  if (!market.acceptingOrders) return null;
  // Settled-but-unresolved markets generate spurious extremes on every axis.
  if (market.last <= SETTLED_BAND || market.last >= 1 - SETTLED_BAND) return null;

  const book = bookStats(ctx.book);
  const history = ctx.history;

  const signals = [
    surge(market, ctx.cross),
    momentum(market, history),
    reversal(market, history),
    breakout(history),
    coil(history),
    whale(market, ctx.flow),
    imbalance(book),
    lean(book),
    tail(market),
    expiry(market),
    thin(market, book),
  ].filter((s): s is Signal => s !== null);

  if (signals.length === 0) return null;

  // Every contribution is scaled by both its kind's weight and its own
  // confidence, so a loud reading built on twelve data points cannot outrank a
  // quieter one built on six hundred.
  let heatAcc = 0;
  let biasAcc = 0;
  let directionalWeight = 0;
  let directionalSigned = 0;

  for (const s of signals) {
    const w = HEAT_WEIGHT[s.kind] * s.confidence;
    heatAcc += s.strength * w;
    if (s.direction === "bullish" || s.direction === "bearish") {
      const sign = s.direction === "bullish" ? 1 : -1;
      biasAcc += sign * s.strength * w;
      directionalWeight += s.strength * w;
      directionalSigned += sign * s.strength * w;
    }
  }

  // Compound rather than average: three independent signals agreeing is a much
  // stronger read than one strong signal alone, but returns still diminish.
  const heat = Math.round(100 * saturate(heatAcc, 90));
  const conviction =
    directionalWeight > 0 ? Math.round(100 * Math.abs(directionalSigned / directionalWeight)) : 0;

  const vol = history ? realisedVol(history) : 0;
  const drift = history ? driftPerDay(history) : 0;

  return {
    market,
    signals: [...signals].sort(
      (a, b) =>
        b.strength * HEAT_WEIGHT[b.kind] * b.confidence -
        a.strength * HEAT_WEIGHT[a.kind] * a.confidence
    ),
    bias: clamp(Math.round(biasAcc / 1.5), -100, 100),
    heat: clamp(heat, 0, 100),
    conviction,
    stats: {
      realisedVol: vol,
      driftPerDay: drift,
      trendQuality: vol > 1e-6 ? drift / vol : 0,
      autocorrelation: history ? autocorrelation(history) : 0,
      bandZ: history ? bandZ(history) : 0,
      volCompression: history ? volCompression(history) : 1,
      book,
      volumeZ: ctx.cross
        ? robustZ(
            Math.log(Math.max(market.volume24h / Math.max(market.volume1w / 7, 1), 1e-6)),
            ctx.cross.volumeRatios
          )
        : 0,
      moveZ: ctx.cross ? robustZ(Math.abs(market.chg24h ?? 0), ctx.cross.absMoves) : 0,
    },
  };
}

/**
 * Mutually-exclusive basket mispricing.
 *
 * On a negative-risk event exactly one leg resolves YES, so the YES legs form a
 * basket worth exactly $1 at resolution. If every leg can be sold for more than
 * $1 in total, or bought for less, that difference is locked in.
 *
 * Only `negRisk` events qualify. A sports event carries 20+ *unrelated* legs
 * (moneyline, spreads, totals) whose prices sum to nothing meaningful, and
 * summing those would manufacture a fake 15x "arbitrage" on every game.
 */
export function findArbitrage(events: readonly EventSummary[]): ArbOpportunity[] {
  const out: ArbOpportunity[] = [];

  for (const event of events) {
    if (!event.markets.every((m) => m.negRisk)) continue;
    const legs = event.markets.filter((m) => m.acceptingOrders);
    if (legs.length < 2 || legs.length !== event.markets.length) continue;

    const bids = legs.map((m) => m.bestBid);
    const asks = legs.map((m) => m.bestAsk);
    if (bids.some((b) => b === undefined) || asks.some((a) => a === undefined)) continue;

    const sumBid = (bids as number[]).reduce((a, b) => a + b, 0);
    const sumAsk = (asks as number[]).reduce((a, b) => a + b, 0);

    // The basket can only be lifted as fast as its thinnest leg. A leg with no
    // resting liquidity makes the whole thing unexecutable, and printing a
    // "2.5 point edge" nobody can capture is worse than printing nothing.
    const tightestLegLiquidity = Math.min(...legs.map((m) => m.liquidity));
    if (tightestLegLiquidity <= 0) continue;

    // Sell every leg for more than the $1 the basket can ever pay out.
    if (sumBid > 1.005) {
      out.push({
        event,
        side: "sell-basket",
        basket: sumBid,
        edgePoints: (sumBid - 1) * 100,
        legs: legs.length,
        tightestLegLiquidity,
      });
      continue;
    }

    // Buy every leg for less than the $1 exactly one of them will pay.
    // A leg quoted at 99¢+ usually has no real offer resting behind it, which
    // would inflate `sumAsk` and hide a fake edge, so require genuine two-way
    // quotes on every leg before trusting the buy side.
    const quotesAreReal = (asks as number[]).every((a) => a < 0.99);
    if (quotesAreReal && sumAsk < 0.995) {
      out.push({
        event,
        side: "buy-basket",
        basket: sumAsk,
        edgePoints: (1 - sumAsk) * 100,
        legs: legs.length,
        tightestLegLiquidity,
      });
    }
  }

  return out.sort((a, b) => b.edgePoints - a.edgePoints);
}

export type BasketDrift = {
  readonly event: EventSummary;
  /** Sum of leg mid prices. Should sit at 1.00 on a coherent book. */
  readonly basket: number;
  /** Signed deviation in probability points. */
  readonly driftPoints: number;
  /**
   * The basket's own quoting uncertainty, in points: the sum of every leg's
   * half-spread. Drift smaller than this is indistinguishable from where the
   * mids happen to sit inside their spreads.
   */
  readonly noisePoints: number;
  /** Drift as a multiple of that noise. Above 1 it is real. */
  readonly ratio: number;
  readonly legs: number;
};

/**
 * Coherence drift on mutually-exclusive events.
 *
 * Distinct from `findArbitrage`: that one needs a *crossable* edge on real
 * quotes. This measures how far the mid-price basket has wandered from the 100¢
 * it must settle at — pressure that has built without yet opening a spread you
 * can trade. It is an early warning that one leg is repricing and its siblings
 * have not caught up.
 */
export function findBasketDrift(events: readonly EventSummary[]): BasketDrift[] {
  const out: BasketDrift[] = [];

  for (const event of events) {
    if (!event.markets.every((m) => m.negRisk)) continue;
    const legs = event.markets.filter(
      (m) => m.acceptingOrders && m.bestBid !== undefined && m.bestAsk !== undefined
    );
    if (legs.length < 2 || legs.length !== event.markets.length) continue;

    // Every leg must carry a genuine two-way market before its mid means
    // anything. An unquoted leg reports bestBid 0 against bestAsk 0.999, whose
    // "mid" is 50¢ of pure fiction — on a 128-leg field like a presidential
    // nomination that summed to a basket of 44.46 and a 4,345-point "drift".
    const properlyQuoted = legs.every((m) => m.bestAsk! < 0.99 && m.bestAsk! - m.bestBid! <= 0.1);
    if (!properlyQuoted) continue;

    const basket = legs.reduce((sum, m) => sum + (m.bestBid! + m.bestAsk!) / 2, 0);
    const driftPoints = (basket - 1) * 100;
    // A coherent field cannot drift this far. Anything beyond it is a quoting
    // artefact that slipped the guard above, not a tradeable dislocation.
    if (Math.abs(driftPoints) > 25) continue;

    // Compare the drift against the basket's own quoting noise rather than a
    // constant. Each leg's mid could legitimately sit anywhere within half a
    // spread of true, and those uncertainties accumulate with leg count — so a
    // 1-point drift is loud on a 3-leg field and silent on a 30-leg one.
    const noisePoints = legs.reduce((sum, m) => sum + (m.bestAsk! - m.bestBid!), 0) * 100 * 0.5;
    const ratio = noisePoints > 1e-6 ? Math.abs(driftPoints) / noisePoints : 0;
    if (ratio < 1) continue;

    out.push({ event, basket, driftPoints, noisePoints, ratio, legs: legs.length });
  }

  return out.sort((a, b) => b.ratio - a.ratio);
}

function abbreviate(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return n.toFixed(0);
}

/** Display metadata. Keyed by the union so a new kind cannot be forgotten. */
export const SIGNAL_META: Record<
  SignalKind,
  { label: string; tone: "direction" | "warn" | "info"; blurb: string }
> = {
  ARB: { label: "ARB", tone: "warn", blurb: "Mutually-exclusive basket is mispriced against $1" },
  DRIFT: {
    label: "DRIFT",
    tone: "warn",
    blurb: "Basket mid has wandered from the 100¢ it settles at",
  },
  SURGE: { label: "SURGE", tone: "info", blurb: "Turnover unusual against today's cross-section" },
  MOMENTUM: {
    label: "MOM",
    tone: "direction",
    blurb: "Drift per unit of volatility — trend quality",
  },
  REVERSAL: {
    label: "REV",
    tone: "direction",
    blurb: "Stretched beyond its band while moves revert",
  },
  BREAKOUT: { label: "BRK", tone: "direction", blurb: "Left its range with volatility expanding" },
  COIL: { label: "COIL", tone: "info", blurb: "Volatility compressed — a range yet to resolve" },
  WHALE: { label: "WHALE", tone: "direction", blurb: "Net direction of block prints ≥ $10k" },
  IMBALANCE: { label: "BOOK", tone: "direction", blurb: "Resting capital skewed to one side" },
  LEAN: { label: "LEAN", tone: "direction", blurb: "Micro-price lean — pressure on the next tick" },
  TAIL: { label: "TAIL", tone: "info", blurb: "Extreme price still drawing real turnover" },
  EXPIRY: { label: "EXPY", tone: "info", blurb: "Near resolution and still genuinely uncertain" },
  THIN: {
    label: "THIN",
    tone: "warn",
    blurb: "Spread is a large fraction of mid — costly to cross",
  },
};
