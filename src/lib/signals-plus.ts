import { daysUntil, eventOutcomes, fmtUsd, type GammaEvent } from "./polymarket";

/**
 * Enhanced edge scanner — a superset of the base ARB/MOMENTUM kernel that adds
 * a genuine LIQUIDITY (maker / spread-capture) read and scores every signal on
 * a comparable, liquidity-aware 0–100 scale.
 *
 * Lives beside `signals.ts` (rather than replacing it) so the board's radar and
 * this fuller /signals view can evolve independently. ARB and MOMENTUM keep the
 * exact, proven definitions from the base kernel; LIQUIDITY is new:
 *
 *   ARB        mutually-exclusive (negRisk) YES prices should sum to 1.
 *              sum<1 ⇒ buy-all underround (real edge); sum>1 ⇒ overround / vig.
 *   MOMENTUM   strongest 24h move, read against the 1w trend (accel/extend/reversal).
 *   LIQUIDITY  a deep book carrying a wide bid/ask — a spread a maker can capture.
 *              Rewards depth × spread width; a 1% spread on a $1M book beats a
 *              4% spread on a $2k book because only the deep one is workable.
 *
 * Not financial advice — these are descriptive reads of public market data.
 */

export type EdgeKind = "ARB" | "MOMENTUM" | "LIQUIDITY" | "RESOLUTION";

export interface EdgeSignal {
  slug: string;
  title: string;
  kind: EdgeKind;
  /** Signed edge in basis points (1bp = 0.01%). Sign encodes direction. */
  edgeBps: number;
  /** 0–100 composite: edge magnitude discounted by how tradeable the book is. */
  score: number;
  /** Bid/ask spread on the reference outcome, in bps (execution cost). */
  spreadBps: number | null;
  liquidity: number;
  volume24h: number;
  /** Short, terminal-styled uppercase rationale. */
  detail: string;
}

export interface ScanEdgeOptions {
  /** Minimum |arb edge| to report, in bps. Default 40bps (0.40%). */
  arbMinBps?: number;
  /** Minimum |24h move| to report as momentum, in price points. Default 4pts. */
  momentumMinPts?: number;
  /** Minimum liquidity for a LIQUIDITY signal, in USD. Default $25k. */
  liqMinUsd?: number;
  /** Minimum spread for a LIQUIDITY signal, in bps. Default 120bps (1.20%). */
  liqMinSpreadBps?: number;
  /** Max days-to-resolution for a RESOLUTION signal. Default 3 days. */
  resolutionMaxDays?: number;
  /** Minimum liquidity for a RESOLUTION signal, in USD. Default $5k. */
  resolutionMinUsd?: number;
  /**
   * Floor on book depth ($) for an event to be scannable at all. Default $5k.
   * Without it the board fills with double-digit "edges" on markets holding a
   * few hundred dollars of depth, which are untradeable at any size and crowd
   * out the signals that aren't.
   */
  minLiquidity?: number;
  /** Max signals returned. Default 60. */
  limit?: number;
}

const DEFAULTS: Required<ScanEdgeOptions> = {
  arbMinBps: 40,
  momentumMinPts: 4,
  liqMinUsd: 25_000,
  liqMinSpreadBps: 120,
  resolutionMaxDays: 3,
  resolutionMinUsd: 5_000,
  minLiquidity: 5_000,
  limit: 60,
};

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/** Confidence from depth of book: ~$1 → 0, ~$1M → 1. */
function liquidityWeight(liquidity: number): number {
  return clamp(Math.log10(Math.max(1, liquidity)) / 6, 0, 1);
}

function volumeWeight(volume: number): number {
  return clamp(Math.log10(Math.max(1, volume)) / 6, 0, 1);
}

function spreadBpsOf(o: { spread?: number; bestBid?: number; bestAsk?: number }): number | null {
  if (typeof o.spread === "number" && o.spread > 0) return o.spread * 10000;
  if (typeof o.bestBid === "number" && typeof o.bestAsk === "number") {
    const s = o.bestAsk - o.bestBid;
    return s > 0 ? s * 10000 : null;
  }
  return null;
}

export function scanEdges(events: GammaEvent[], opts: ScanEdgeOptions = {}): EdgeSignal[] {
  const {
    arbMinBps,
    momentumMinPts,
    liqMinUsd,
    liqMinSpreadBps,
    resolutionMaxDays,
    resolutionMinUsd,
    minLiquidity,
    limit,
  } = { ...DEFAULTS, ...opts };
  const signals: EdgeSignal[] = [];

  for (const event of events) {
    if (!event.markets || event.markets.length === 0) continue;
    const outs = eventOutcomes(event);
    if (outs.length === 0) continue;

    const liquidity = event.liquidity ?? 0;
    const volume24h = event.volume24hr ?? 0;
    if (liquidity < minLiquidity) continue; // untradeable at any size
    const liqW = liquidityWeight(liquidity);
    const volW = volumeWeight(volume24h);
    const lead = outs[0];
    const spreadBps = spreadBpsOf(lead);

    // Fields every signal for this event shares — spread once, reuse per kind.
    const base = {
      slug: event.slug,
      title: event.title,
      spreadBps,
      liquidity,
      volume24h,
    };

    // ── ARB: mutually-exclusive outcomes should price to a sum of 1 ──
    // Requires ≥3 outcomes. A 2-outcome negRisk market is a plain binary whose
    // Yes/No prices are constructed to sum to ~1, so any "edge" there is the
    // rounding on the quote, not a mispricing — including them floods the board
    // with phantom ARB signals.
    if (event.negRisk && outs.length >= 3) {
      const sum = outs.reduce((s, o) => s + o.price, 0);
      const edgeBps = (1 - sum) * 10000; // >0 underround (buyable), <0 overround
      if (Math.abs(edgeBps) >= arbMinBps) {
        const score = clamp((Math.abs(edgeBps) / 300) * 100 * (0.4 + 0.6 * liqW), 0, 100);
        const dir = edgeBps > 0 ? "UNDERROUND · BUY-ALL EDGE" : "OVERROUND · VIG";
        signals.push({
          ...base,
          kind: "ARB",
          edgeBps,
          score,
          detail: `${outs.length} OUTCOMES SUM ${(sum * 100).toFixed(1)}% · ${dir}`,
        });
      }
    }

    // ── MOMENTUM: strongest directional 24h mover on the board ──
    const mover = outs.reduce((a, b) => (Math.abs(b.change24h) > Math.abs(a.change24h) ? b : a));
    const movePts = mover.change24h * 100;
    if (Math.abs(movePts) >= momentumMinPts) {
      // Compare against the weekly change of the SAME market that moved. Using
      // markets[0] classifies the mover against a different contract entirely in
      // any multi-outcome event, which mislabels most ACCEL/REVERSAL calls.
      const moverMkt = event.markets.find((m) => m.id === mover.marketId);
      const wk = (moverMkt?.oneWeekPriceChange ?? 0) * 100;
      const accel =
        wk !== 0 && Math.sign(movePts) === Math.sign(wk)
          ? Math.abs(movePts) > Math.abs(wk) / 7
            ? "ACCEL"
            : "EXTEND"
          : "REVERSAL";
      const arrow = movePts > 0 ? "▲" : "▼";
      const score = clamp((Math.abs(movePts) / 15) * 100 * (0.5 + 0.5 * volW), 0, 100);
      signals.push({
        ...base,
        kind: "MOMENTUM",
        edgeBps: movePts * 100,
        score,
        detail: `${arrow} ${Math.abs(movePts).toFixed(1)}PTS/24H · ${mover.label.toUpperCase()} · ${accel}`,
      });
    }

    // ── LIQUIDITY: a deep book carrying a capturable spread ──
    // Only meaningful where the book is deep enough to actually work: a wide
    // spread on a dead market is noise, not an opportunity.
    if (spreadBps != null && spreadBps >= liqMinSpreadBps && liquidity >= liqMinUsd) {
      // Reward spread width, but gate hard on depth so shallow books can't rank.
      const score = clamp((spreadBps / 400) * 100 * (0.25 + 0.75 * liqW), 0, 100);
      signals.push({
        ...base,
        kind: "LIQUIDITY",
        edgeBps: spreadBps, // capturable spread, always positive
        score,
        detail: `DEEP BOOK ${fmtUsd(liquidity)} · CAPTURE ${(spreadBps / 100).toFixed(1)}% SPREAD · ${lead.label.toUpperCase()}`,
      });
    }

    // ── RESOLUTION: imminent settlement while the outcome is still contested ──
    // A market resolving within a few days whose leader is nowhere near 0/100
    // still has live probability mass — the closer to settlement, the sharper
    // the repricing tends to be.
    const days = daysUntil(event.endDate);
    if (
      Number.isFinite(days) &&
      days >= 0 &&
      days <= resolutionMaxDays &&
      liquidity >= resolutionMinUsd &&
      lead.price >= 0.15 &&
      lead.price <= 0.85
    ) {
      // Uncertainty peaks at a 50/50 leader, →0 as the leader nears certainty.
      const uncertainty = 1 - Math.abs(2 * lead.price - 1);
      const urgency = clamp((resolutionMaxDays - days) / resolutionMaxDays, 0, 1);
      const edgeBps = uncertainty * 10000;
      const score = clamp(uncertainty * (0.5 + 0.5 * urgency) * 100 * (0.5 + 0.5 * liqW), 0, 100);
      const when = days < 1 ? "<1D" : `${Math.ceil(days)}D`;
      signals.push({
        ...base,
        kind: "RESOLUTION",
        edgeBps,
        score,
        detail: `RESOLVES IN ${when} · ${(lead.price * 100).toFixed(0)}% LEADER STILL LIVE · ${lead.label.toUpperCase()}`,
      });
    }
  }

  return signals.toSorted((a, b) => b.score - a.score).slice(0, limit);
}
