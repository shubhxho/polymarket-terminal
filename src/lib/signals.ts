import { eventOutcomes, type GammaEvent, type Outcome } from "./polymarket";

/**
 * Client-side "edge" scanner for prediction markets.
 *
 * This is deliberately NOT a high-frequency execution engine — Polymarket is a
 * slow, on-chain-settled prediction market, and this app is a read-only viewer.
 * What it does instead is compute the genuinely useful, honestly-defined signals
 * a trader would scan a board for, cheaply enough to run over a few hundred
 * markets in a Web Worker without blocking the UI:
 *
 *   ARB       cross-outcome mispricing on mutually-exclusive (negRisk) events.
 *             For a market where exactly one outcome resolves YES, the YES prices
 *             should sum to 1. sum < 1 ⇒ buying every YES locks in a profit
 *             (underround); sum > 1 ⇒ the book carries an overround / vig.
 *   MOMENTUM  directional 24h move in a market that is STILL LIVE — i.e. the
 *             price is inside a tradeable band and the event is not already
 *             settling in-play. The 1w change classifies the move as an
 *             acceleration, a continuation, or a genuine reversal.
 *
 * Scoring is the part that makes the board readable. Each signal gets a 0–100
 * score built from a *non-saturating* magnitude curve (so a 60pt in-game blowout
 * doesn't tie a 12pt swing at 100) multiplied by real book quality (depth ×
 * turnover) and, for momentum, a time weight that discounts markets about to
 * settle. Net effect: a moderate move on a deep, liquid, still-open book
 * outranks a huge move on a thin book that's already decided. Not financial
 * advice.
 */

export type SignalKind = "ARB" | "MOMENTUM";

/** A tradeable leg of a signal — the CLOB token the fill simulator hits. */
export interface SignalLeg {
  tokenId: string;
  label: string;
  price: number;
}

export interface Signal {
  slug: string;
  title: string;
  kind: SignalKind;
  /** Signed edge in basis points (1bp = 0.01%). Sign encodes direction. */
  edgeBps: number;
  /** 0–100 composite rank: edge magnitude discounted by book quality/time. */
  score: number;
  /** Bid/ask spread on the reference outcome, in basis points (execution cost). */
  spreadBps: number | null;
  liquidity: number;
  volume24h: number;
  /** Short human-readable rationale, terminal-styled uppercase. */
  detail: string;
  /**
   * Edge after crossing the spread on every leg, in bps. Positive = a real
   * buy-all (or sell-all) profit at the touch. Null when a leg is unquoted.
   * This is the number that decides whether `edgeBps` is a trade or a fact.
   */
  executableBps: number | null;
  /** Outcomes to fill: all legs for ARB, just the mover for MOMENTUM. */
  legs: SignalLeg[];
}

function legsOf(outcomes: { tokenId?: string; label: string; price: number }[]): SignalLeg[] {
  const legs: SignalLeg[] = [];
  for (const o of outcomes) {
    if (o.tokenId) legs.push({ tokenId: o.tokenId, label: o.label, price: o.price });
  }
  return legs;
}

/**
 * The arb you could actually put on, priced off the touch rather than the mid.
 *
 * The mid-price sum is a fair-value statement; it is not a trade. Buying every
 * YES costs the ASKS, so the executable underround is `1 − Σ(bestAsk)`.
 * Selling the complete set collects the BIDS, so the executable overround is
 * `Σ(bestBid) − 1`. Those two numbers are routinely several hundred bps worse
 * than the mid-based edge, which is exactly why a board full of "2% arbs"
 * contains almost nothing executable.
 *
 * Returns null when any leg is missing a quote — a basket is only as real as
 * its worst-quoted leg, and guessing one is how a phantom arb gets printed.
 */
function executableArb(outs: Outcome[]): {
  askSum: number;
  bidSum: number;
  buyBps: number;
  sellBps: number;
} | null {
  let askSum = 0;
  let bidSum = 0;
  for (const o of outs) {
    if (typeof o.bestAsk !== "number" || typeof o.bestBid !== "number") return null;
    if (!(o.bestAsk > 0) || !(o.bestBid > 0)) return null;
    askSum += o.bestAsk;
    bidSum += o.bestBid;
  }
  return {
    askSum,
    bidSum,
    buyBps: (1 - askSum) * 10000,
    sellBps: (bidSum - 1) * 10000,
  };
}

export interface ScanOptions {
  /** Minimum |arb edge| to report, in bps. Default 40bps (0.40%). */
  arbMinBps?: number;
  /** Minimum |24h move| to report as momentum, in price points. Default 5pts. */
  momentumMinPts?: number;
  /** Floor on book depth ($) for a market to be scannable. Default 5000. */
  minLiquidity?: number;
  /**
   * Momentum only fires when the reference outcome's price sits inside this
   * band — a market that has already run to ~0/1 is decided, not a signal.
   * Default [0.15, 0.85].
   */
  tradeBand?: [number, number];
  /** Momentum on a book wider than this spread (bps) is untradeable. Default 2500 (25%). */
  maxSpreadBps?: number;
  /** Restrict output to these signal kinds. Default: all. */
  kinds?: SignalKind[];
  /** Max signals returned. Omit for no cap — truncation is a display concern. */
  limit?: number;
  /** Reference "now" in ms, for time-to-expiry. Default Date.now(). */
  now?: number;
}

const DEFAULTS = {
  arbMinBps: 40,
  momentumMinPts: 5,
  minLiquidity: 5000,
  tradeBand: [0.15, 0.85] as [number, number],
  maxSpreadBps: 2500,
  limit: Number.POSITIVE_INFINITY,
};

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/** Depth-of-book weight: ~$1 → 0, ~$1M → 1. */
function liquidityWeight(liquidity: number): number {
  return clamp(Math.log10(Math.max(1, liquidity)) / 6, 0, 1);
}

function volumeWeight(volume: number): number {
  return clamp(Math.log10(Math.max(1, volume)) / 6, 0, 1);
}

/**
 * Combined book quality — a market needs BOTH depth and turnover to score well,
 * so `sqrt(depth × turnover)`, with a small floor so a decent book isn't zeroed.
 */
function bookQuality(liquidity: number, volume: number): number {
  const q = Math.sqrt(liquidityWeight(liquidity) * volumeWeight(volume));
  return clamp(0.15 + 0.85 * q, 0, 1);
}

/**
 * Saturating-but-not-flat magnitude curve: `1 - e^(-x/k)`. Unlike a hard
 * `min(x/k, 1)` this keeps rising past the knee, so bigger edges still rank
 * above smaller ones instead of all pinning at 100.
 */
function magnitude(x: number, k: number): number {
  return 1 - Math.exp(-Math.abs(x) / k);
}

/** Discount markets that are about to settle (in-play sports, expiring bins). */
function timeWeight(hoursToExpiry: number): number {
  if (!Number.isFinite(hoursToExpiry)) return 0.9; // no end date → treat as open
  if (hoursToExpiry <= 0) return 0.1;
  // Ramp from 0.3 (settling within hours) to 1.0 (≥3 days out).
  return clamp(0.3 + 0.7 * clamp(hoursToExpiry / 72, 0, 1), 0.3, 1);
}

function hoursUntil(iso: string | undefined, now: number): number {
  if (!iso) return Number.POSITIVE_INFINITY;
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return Number.POSITIVE_INFINITY;
  return (t - now) / 3_600_000;
}

function spreadBpsOf(o: { spread?: number; bestBid?: number; bestAsk?: number }): number | null {
  if (typeof o.spread === "number" && o.spread > 0) return o.spread * 10000;
  if (typeof o.bestBid === "number" && typeof o.bestAsk === "number") {
    const s = o.bestAsk - o.bestBid;
    return s > 0 ? s * 10000 : null;
  }
  return null;
}

export function scanSignals(events: GammaEvent[], opts: ScanOptions = {}): Signal[] {
  const arbMinBps = opts.arbMinBps ?? DEFAULTS.arbMinBps;
  const momentumMinPts = opts.momentumMinPts ?? DEFAULTS.momentumMinPts;
  const minLiquidity = opts.minLiquidity ?? DEFAULTS.minLiquidity;
  const [bandLo, bandHi] = opts.tradeBand ?? DEFAULTS.tradeBand;
  const maxSpreadBps = opts.maxSpreadBps ?? DEFAULTS.maxSpreadBps;
  const limit = opts.limit ?? DEFAULTS.limit;
  const now = opts.now ?? Date.now();
  const wantArb = !opts.kinds || opts.kinds.includes("ARB");
  const wantMom = !opts.kinds || opts.kinds.includes("MOMENTUM");

  const signals: Signal[] = [];

  for (const event of events) {
    if (!event.markets || event.markets.length === 0) continue;
    const outs = eventOutcomes(event);
    if (outs.length === 0) continue;

    const liquidity = event.liquidity ?? 0;
    const volume24h = event.volume24hr ?? 0;
    if (liquidity < minLiquidity) continue; // skip thin books entirely
    const quality = bookQuality(liquidity, volume24h);
    const liqW = liquidityWeight(liquidity);
    const hours = hoursUntil(event.endDate, now);
    const lead = outs[0];
    const spreadBps = spreadBpsOf(lead);
    // True only when every market in the event survived the active/closed
    // filter, i.e. the outcome set we are summing is the whole event.
    const completeSet = outs.length > 0 && event.markets.every((m) => m.active && !m.closed);

    // ── ARB: mutually-exclusive outcomes should price to a sum of 1 ──
    // That identity holds ONLY over a COMPLETE outcome set. `eventOutcomes`
    // drops markets that are inactive or closed, so if any were dropped we are
    // summing a subset — which always undershoots 1 and manufactures a
    // "buy-all underround" that does not exist. Polymarket's own neg-risk docs
    // make the same point from the other side: augmented neg-risk events carry
    // unnamed placeholder outcomes that the UI hides and warns against trading,
    // so a set that isn't whole cannot be arbitraged.
    if (wantArb && event.negRisk && outs.length >= 3 && completeSet) {
      const sum = outs.reduce((s, o) => s + o.price, 0);
      const edgeBps = (1 - sum) * 10000; // >0 underround (buyable), <0 overround
      if (Math.abs(edgeBps) >= arbMinBps) {
        // What survives crossing the spread on every leg.
        const exec = executableArb(outs);
        const executableBps = exec == null ? null : edgeBps > 0 ? exec.buyBps : exec.sellBps;
        // Depth matters more than turnover for capturing a book edge, and an
        // edge that dies at the touch is not an edge: an unexecutable basket is
        // scored down hard rather than dropped, so the board still shows the
        // mispricing while making clear it cannot be lifted.
        const executablePenalty = executableBps == null ? 0.6 : executableBps > 0 ? 1 : 0.25;
        const score = clamp(
          100 * magnitude(edgeBps, 250) * (0.5 + 0.5 * liqW) * executablePenalty,
          0,
          100,
        );
        // Underround: buying every YES for $sum returns $1 → (1/sum − 1) profit.
        // Overround: the book's built-in vig you pay is (sum − 1).
        const ret = edgeBps > 0 ? (1 / sum - 1) * 100 : (sum - 1) * 100;
        const dir =
          edgeBps > 0
            ? `UNDERROUND · BUY-ALL +${ret.toFixed(2)}%`
            : `OVERROUND · VIG ${ret.toFixed(2)}%`;
        signals.push({
          slug: event.slug,
          title: event.title,
          kind: "ARB",
          edgeBps,
          score,
          spreadBps,
          liquidity,
          volume24h,
          executableBps,
          detail: `${outs.length} OUTCOMES SUM ${(sum * 100).toFixed(1)}% · ${dir}${
            executableBps == null
              ? " · NO TOUCH QUOTE"
              : ` · AT TOUCH ${executableBps > 0 ? "+" : ""}${(executableBps / 100).toFixed(2)}%`
          }`,
          legs: legsOf(outs),
        });
      }
    }

    // ── MOMENTUM: strongest 24h mover that is still a live, tradeable book ──
    if (wantMom) {
      const mover = outs.reduce((a, b) => (Math.abs(b.change24h) > Math.abs(a.change24h) ? b : a));
      const movePts = mover.change24h * 100;
      const moverSpread = spreadBpsOf(mover);
      const inBand = mover.price >= bandLo && mover.price <= bandHi;
      const tradeable = moverSpread == null || moverSpread <= maxSpreadBps;
      if (Math.abs(movePts) >= momentumMinPts && inBand && tradeable) {
        // Compare against the weekly change of the SAME market that moved — not
        // markets[0], which in a multi-outcome event is a different contract.
        const moverMkt = event.markets.find((m) => m.id === mover.marketId);
        const wkPts = (moverMkt?.oneWeekPriceChange ?? 0) * 100;
        const wkAbs = Math.abs(wkPts);
        // Classify the move against the established weekly trend:
        //   NEW      no trend yet (week barely moved) — this is fresh flow
        //   REVERSAL today's move opposes the week's direction
        //   ACCEL    same direction, and today alone ≥ the whole week's net move
        //   EXTEND   same direction, continuing a larger multi-day trend
        const trend =
          wkAbs < 2
            ? "NEW"
            : Math.sign(movePts) !== Math.sign(wkPts)
              ? "REVERSAL"
              : Math.abs(movePts) >= wkAbs
                ? "ACCEL"
                : "EXTEND";
        const wkTag = wkAbs < 2 ? "" : ` 1W ${wkPts > 0 ? "+" : ""}${wkPts.toFixed(1)}`;
        const arrow = movePts > 0 ? "▲" : "▼";
        // Wider book ⇒ more of the move is unrealizable slippage.
        const spreadQuality = moverSpread == null ? 0.75 : clamp(1 - moverSpread / 2000, 0.1, 1);
        const score = clamp(
          100 * magnitude(movePts, 12) * quality * timeWeight(hours) * spreadQuality,
          0,
          100,
        );
        signals.push({
          slug: event.slug,
          title: event.title,
          kind: "MOMENTUM",
          edgeBps: movePts * 100,
          score,
          spreadBps: moverSpread,
          liquidity,
          volume24h,
          // A directional entry pays half the spread to get in.
          executableBps: moverSpread == null ? null : movePts * 100 - moverSpread / 2,
          detail: `${arrow} ${Math.abs(movePts).toFixed(1)}PTS/24H → ${(mover.price * 100).toFixed(0)}% · ${mover.label.toUpperCase()} · ${trend}${wkTag}`,
          legs: legsOf([mover]),
        });
      }
    }
  }

  const ranked = signals.toSorted((a, b) => b.score - a.score);
  return Number.isFinite(limit) ? ranked.slice(0, limit) : ranked;
}
