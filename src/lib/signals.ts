/**
 * Signal engine.
 *
 * Pure functions over already-normalised data — no I/O, no React — so the
 * scoring can be reasoned about and tested in isolation from the feeds.
 *
 * Two rules run through everything here:
 *
 * 1. **A signal must be actionable.** "Price went up" is not a signal; "price
 *    went up on 6x its normal volume while the book stayed bid" is. Anything
 *    that merely restates a column already on screen is left out.
 * 2. **Silence beats noise.** Every detector returns `null` unless its
 *    preconditions are met, and thin markets are excluded before scoring
 *    rather than being allowed to produce enormous meaningless percentages.
 */

import type { EventSummary, Market, OrderBook, Trade } from "./types";

export type SignalKind =
  | "ARB"
  | "SURGE"
  | "MOMENTUM"
  | "REVERSAL"
  | "WHALE"
  | "IMBALANCE"
  | "LONGSHOT"
  | "EXPIRY"
  | "WIDE";

export type Direction = "bullish" | "bearish" | "neutral";

export type Signal = {
  readonly kind: SignalKind;
  readonly direction: Direction;
  /** 0..100. How notable this reading is, comparable across kinds. */
  readonly strength: number;
  /** Terse value, for a badge: "6.2x vol". */
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
};

export type ArbOpportunity = {
  readonly event: EventSummary;
  readonly side: "sell-basket" | "buy-basket";
  /** Sum of the leg prices that form the basket. */
  readonly basket: number;
  /** Guaranteed edge in probability points, before fees and slippage. */
  readonly edgePoints: number;
  readonly legs: number;
  /** Thinnest leg's resting notional — the real cap on executable size. */
  readonly tightestLegLiquidity: number;
};

// ── Tunables ───────────────────────────────────────────────────────────────
// Markets below this turnover produce wild ratios off tiny denominators, so
// they are excluded from scoring entirely rather than filtered afterwards.
const MIN_VOLUME_24H = 5_000;
/** A print at or above this notional is treated as informed size. */
export const WHALE_NOTIONAL = 10_000;
/** Prices this close to a bound are settled in all but name. */
const SETTLED_BAND = 0.015;

/** Relative weight of each kind when compounding into `heat`. */
const HEAT_WEIGHT: Record<SignalKind, number> = {
  ARB: 1.0,
  SURGE: 0.9,
  WHALE: 0.85,
  MOMENTUM: 0.7,
  REVERSAL: 0.7,
  IMBALANCE: 0.6,
  EXPIRY: 0.5,
  LONGSHOT: 0.4,
  // A wide spread is information about execution quality, not about value.
  WIDE: 0.2,
};

/** Maps an unbounded ratio into 0..100 with diminishing returns. */
function saturate(value: number, halfPoint: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0;
  return Math.round((100 * value) / (value + halfPoint));
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

// ── Detectors ──────────────────────────────────────────────────────────────

/**
 * Turnover well above the market's own recent baseline.
 *
 * Compared against its own weekly average rather than against other markets,
 * so a quiet market waking up scores as highly as a permanently busy one.
 */
function surge(m: Market): Signal | null {
  const baseline = m.volume1w / 7;
  // A market younger than a week has no meaningful baseline.
  if (baseline < 1_000) return null;
  const ratio = m.volume24h / baseline;
  if (ratio < 2) return null;
  return {
    kind: "SURGE",
    // Volume alone says attention, not direction; the 24h move supplies sign.
    direction: (m.chg24h ?? 0) > 1 ? "bullish" : (m.chg24h ?? 0) < -1 ? "bearish" : "neutral",
    strength: saturate(ratio - 1, 3),
    headline: `${ratio.toFixed(1)}x vol`,
    detail: `24h turnover is ${ratio.toFixed(1)}x this market's weekly average — ${
      m.volume24h >= baseline * 5 ? "a step change in attention" : "unusual interest"
    }.`,
  };
}

/**
 * A repricing that is still underway: the last hour is pushing the same way as
 * the last day, so the move has not yet mean-reverted.
 */
function momentum(m: Market): Signal | null {
  const d = m.chg24h ?? 0;
  const h = m.chg1h ?? 0;
  if (Math.abs(d) < 3) return null;
  if (h === 0 || Math.sign(h) !== Math.sign(d)) return null;
  // Hourly rate running above the day's average pace means it is accelerating.
  const pace = Math.abs(h) / (Math.abs(d) / 24);
  return {
    kind: "MOMENTUM",
    direction: d > 0 ? "bullish" : "bearish",
    strength: clamp(saturate(Math.abs(d), 12) * (pace > 1 ? 1 : 0.7), 0, 100),
    headline: `${d > 0 ? "+" : ""}${d.toFixed(1)}pt trend`,
    detail: `Moved ${Math.abs(d).toFixed(1)} points over 24h and still going ${
      d > 0 ? "up" : "down"
    }${pace > 1 ? ", accelerating into the last hour" : ""}.`,
  };
}

/** The last hour fighting the day — an exhausted or freshly-repriced move. */
function reversal(m: Market): Signal | null {
  const d = m.chg24h ?? 0;
  const h = m.chg1h ?? 0;
  if (Math.abs(d) < 3 || Math.abs(h) < 1.5) return null;
  if (Math.sign(h) === Math.sign(d)) return null;
  // How much of the day's move the last hour has already given back.
  const retrace = Math.abs(h) / Math.abs(d);
  return {
    kind: "REVERSAL",
    direction: h > 0 ? "bullish" : "bearish",
    strength: saturate(retrace, 0.35),
    headline: `${(retrace * 100).toFixed(0)}% retrace`,
    detail: `Last hour (${h > 0 ? "+" : ""}${h.toFixed(1)}pt) is fighting the 24h move (${
      d > 0 ? "+" : ""
    }${d.toFixed(1)}pt) — ${retrace > 0.5 ? "the move is unwinding" : "early signs of a fade"}.`,
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
  return {
    kind: "WHALE",
    direction: flow.net > 0 ? "bullish" : "bearish",
    strength: clamp(saturate(share, 0.08) * 0.7 + saturate(flow.count, 4) * 0.3, 0, 100),
    headline: `${flow.net > 0 ? "+" : "-"}$${abbreviate(Math.abs(flow.net))} blocks`,
    detail: `${flow.count} block${flow.count === 1 ? "" : "s"} ≥ $${abbreviate(
      WHALE_NOTIONAL
    )} netting ${flow.net > 0 ? "buy" : "sell"} $${abbreviate(Math.abs(flow.net))} — ${(
      share * 100
    ).toFixed(0)}% of 24h volume.`,
  };
}

/**
 * Resting size skewed to one side of the book.
 *
 * Weighted by each side's *cost* (bids pay `price`, asks risk `1 - price`), so
 * a wall of 2¢ asks is not mistaken for real conviction.
 */
function imbalance(book: OrderBook | undefined): Signal | null {
  if (!book || book.bids.length === 0 || book.asks.length === 0) return null;
  const bidNotional = book.bids.slice(0, 10).reduce((s, l) => s + l.price * l.size, 0);
  const askNotional = book.asks.slice(0, 10).reduce((s, l) => s + (1 - l.price) * l.size, 0);
  const total = bidNotional + askNotional;
  if (total < 5_000) return null;
  const skew = (bidNotional - askNotional) / total;
  if (Math.abs(skew) < 0.35) return null;
  return {
    kind: "IMBALANCE",
    direction: skew > 0 ? "bullish" : "bearish",
    strength: saturate(Math.abs(skew) - 0.3, 0.35),
    headline: `${skew > 0 ? "+" : ""}${(skew * 100).toFixed(0)}% book`,
    detail: `Top-of-book capital is ${(Math.abs(skew) * 100).toFixed(0)}% skewed to the ${
      skew > 0 ? "bid" : "ask"
    } — $${abbreviate(bidNotional)} bid vs $${abbreviate(askNotional)} offered.`,
  };
}

/** Extreme price still attracting real money — the favourite-longshot zone. */
function longshot(m: Market): Signal | null {
  const p = m.last;
  if (p > 0.06 && p < 0.94) return null;
  if (p <= SETTLED_BAND || p >= 1 - SETTLED_BAND) return null;
  if (m.volume24h < 25_000) return null;
  const isLongshot = p <= 0.06;
  return {
    kind: "LONGSHOT",
    direction: "neutral",
    strength: saturate(m.volume24h / 25_000, 3),
    headline: isLongshot ? "longshot bid" : "favourite",
    detail: `Trading at ${(p * 100).toFixed(1)}¢ on $${abbreviate(
      m.volume24h
    )} of 24h volume — ${
      isLongshot
        ? "longshots are structurally overpriced, so the ask side is usually the edge"
        : "heavy favourites are structurally underpriced relative to true odds"
    }.`,
  };
}

/** Close to resolution and still genuinely uncertain: maximum price sensitivity. */
function expiry(m: Market): Signal | null {
  if (!m.endDate) return null;
  const hours = (new Date(m.endDate).getTime() - Date.now()) / 3_600_000;
  if (!Number.isFinite(hours) || hours <= 0 || hours > 72) return null;
  if (m.last < 0.12 || m.last > 0.88) return null;
  return {
    kind: "EXPIRY",
    direction: "neutral",
    strength: saturate(72 - hours, 36),
    headline: `${hours < 24 ? `${hours.toFixed(0)}h` : `${(hours / 24).toFixed(0)}d`} to res`,
    detail: `Resolves in ${
      hours < 24 ? `${hours.toFixed(0)} hours` : `${(hours / 24).toFixed(1)} days`
    } while still priced at ${(m.last * 100).toFixed(
      0
    )}¢ — small news moves the price hard from here.`,
  };
}

/** Execution warning: the spread is a large fraction of what you'd pay. */
function wide(m: Market): Signal | null {
  const bid = m.bestBid;
  const ask = m.bestAsk;
  if (bid === undefined || ask === undefined || ask <= bid) return null;
  const spread = ask - bid;
  const mid = (ask + bid) / 2;
  if (mid <= 0) return null;
  const relative = spread / mid;
  if (relative < 0.08 || spread < 0.01) return null;
  return {
    kind: "WIDE",
    direction: "neutral",
    strength: saturate(relative, 0.25),
    headline: `${(spread * 100).toFixed(1)}¢ wide`,
    detail: `Spread is ${(relative * 100).toFixed(
      0
    )}% of mid — crossing it costs more than most edges are worth.`,
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
    const notional = t.size * t.price;
    if (notional < WHALE_NOTIONAL) continue;
    const yesSide = t.outcomeIndex === 0;
    const buying = t.side === "BUY";
    // Bullish-on-YES when buying YES or selling NO.
    const signed = (yesSide === buying ? 1 : -1) * notional;
    const prev = out.get(t.conditionId) ?? { net: 0, count: 0, gross: 0 };
    out.set(t.conditionId, {
      net: prev.net + signed,
      count: prev.count + 1,
      gross: prev.gross + notional,
    });
  }
  return out;
}

/** Runs every detector over one market and compounds the results. */
export function scoreMarket(
  market: Market,
  ctx: { flow?: FlowStat; book?: OrderBook } = {}
): MarketSignals | null {
  if (market.volume24h < MIN_VOLUME_24H) return null;
  if (!market.acceptingOrders) return null;
  // Settled-but-unresolved markets generate spurious extremes on every axis.
  if (market.last <= SETTLED_BAND || market.last >= 1 - SETTLED_BAND) return null;

  const signals = [
    surge(market),
    momentum(market),
    reversal(market),
    whale(market, ctx.flow),
    imbalance(ctx.book),
    longshot(market),
    expiry(market),
    wide(market),
  ].filter((s): s is Signal => s !== null);

  if (signals.length === 0) return null;

  let bias = 0;
  let heatAcc = 0;
  for (const s of signals) {
    const weight = HEAT_WEIGHT[s.kind];
    heatAcc += s.strength * weight;
    if (s.direction === "bullish") bias += s.strength * weight;
    if (s.direction === "bearish") bias -= s.strength * weight;
  }

  // Compound rather than average: three independent signals agreeing is a much
  // stronger read than one strong signal alone, but returns still diminish.
  const heat = clamp(Math.round(saturate(heatAcc, 90)), 0, 100);

  return {
    market,
    signals: signals.sort((a, b) => b.strength * HEAT_WEIGHT[b.kind] - a.strength * HEAT_WEIGHT[a.kind]),
    bias: clamp(Math.round(bias / 1.5), -100, 100),
    heat,
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
  ARB: { label: "ARB", tone: "warn", blurb: "Basket of mutually-exclusive legs is mispriced against $1" },
  SURGE: { label: "SURGE", tone: "info", blurb: "24h volume far above this market's own weekly baseline" },
  MOMENTUM: { label: "MOM", tone: "direction", blurb: "Sustained repricing, last hour agreeing with the day" },
  REVERSAL: { label: "REV", tone: "direction", blurb: "Last hour fighting the 24h move" },
  WHALE: { label: "WHALE", tone: "direction", blurb: "Net direction of block prints ≥ $10k" },
  IMBALANCE: { label: "BOOK", tone: "direction", blurb: "Resting capital skewed to one side" },
  LONGSHOT: { label: "TAIL", tone: "info", blurb: "Extreme price still drawing real turnover" },
  EXPIRY: { label: "EXPY", tone: "info", blurb: "Near resolution and still genuinely uncertain" },
  WIDE: { label: "WIDE", tone: "warn", blurb: "Spread is a large fraction of mid — costly to cross" },
};
