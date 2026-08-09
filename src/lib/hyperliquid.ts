import type { OrderBook } from "./types";
import type { Candle } from "./options";

/**
 * Hyperliquid read-only client (HyperCore `info` API).
 *
 * WHY HYPERLIQUID IS THE RIGHT SECOND VENUE
 * -----------------------------------------
 * Polymarket tells you the probability of a discrete claim. To decide whether
 * that probability is RIGHT you need the continuous market for the same
 * underlying, and you need three things from it that a spot price alone can't
 * give:
 *
 *   1. AN ORACLE PRICE. HL publishes `oraclePx`, a validator-median of external
 *      spot venues. That is the number the market actually settles funding
 *      against — cleaner than any single exchange's last trade.
 *   2. A CARRY CURVE. Perps have no expiry, so funding IS the term structure.
 *      An hourly funding rate compounds into a forward price, and a forward is
 *      exactly what a digital option needs to be priced honestly.
 *   3. REAL DEPTH. `l2Book` returns the full ladder, so a claimed edge can be
 *      walked against actual size rather than a top-of-book mid.
 *
 * Everything here is public, unauthenticated, read-only. No keys, no signing,
 * no orders — this module cannot place a trade.
 *
 * Endpoint: POST https://api.hyperliquid.xyz/info with a `{ type }` body.
 * Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
 */

const HL_INFO = "https://api.hyperliquid.xyz/info";

async function info<T>(body: Record<string, unknown>, revalidate: number): Promise<T> {
  const res = await fetch(HL_INFO, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    next: { revalidate },
  });
  if (!res.ok) throw new Error(`hyperliquid ${res.status} for ${body.type}`);
  return res.json() as Promise<T>;
}

const num = (v: string | number | undefined | null): number => {
  const n = typeof v === "string" ? Number.parseFloat(v) : (v ?? Number.NaN);
  return Number.isFinite(n) ? n : Number.NaN;
};

/* ─────────────────────────── perp contexts ─────────────────────────── */

interface RawPerpMeta {
  universe: {
    name: string;
    szDecimals: number;
    maxLeverage: number;
    isDelisted?: boolean;
    onlyIsolated?: boolean;
  }[];
}

interface RawPerpCtx {
  funding: string;
  openInterest: string;
  prevDayPx: string;
  dayNtlVlm: string;
  premium: string | null;
  oraclePx: string;
  markPx: string;
  midPx: string | null;
  impactPxs: [string, string] | null;
  /** HIP-3 dexes only — absent on the main perp dex. We don't read it. */
  dayBaseVlm?: string;
}

export interface PerpContext {
  /** HL coin symbol, e.g. "BTC". */
  coin: string;
  /** Validator-median external spot price — the settlement reference. */
  oraclePx: number;
  /** HL's mark price (impact-price EMA blended with oracle). */
  markPx: number;
  /** Book mid, or NaN on an empty book. */
  midPx: number;
  /** (mark − oracle)/oracle as reported by HL. */
  premium: number;
  /** Funding rate per HOUR, decimal. Positive = longs pay shorts. */
  fundingHourly: number;
  /** Open interest, in base units. */
  openInterest: number;
  /** 24h notional volume, USD. */
  dayNotionalVolume: number;
  /** Price 24h ago, for the daily change. */
  prevDayPx: number;
  /** Impact bid/ask — the price a $20k-notional order would move to. */
  impactBid: number;
  impactAsk: number;
  maxLeverage: number;
  szDecimals: number;
}

/**
 * Perp universe + live state in one round trip.
 *
 * `metaAndAssetCtxs` returns `[meta, ctxs]` as positional arrays — the i-th ctx
 * belongs to the i-th universe entry — so the zip below is load-bearing, not
 * cosmetic. Delisted assets are dropped: they still appear in the universe with
 * a stale mark and would silently poison any consensus price built from them.
 */
export async function getPerpContexts(revalidate = 15): Promise<PerpContext[]> {
  const [meta, ctxs] = await info<[RawPerpMeta, RawPerpCtx[]]>(
    { type: "metaAndAssetCtxs" },
    revalidate
  );
  const out: PerpContext[] = [];
  for (let i = 0; i < meta.universe.length; i++) {
    const u = meta.universe[i];
    const c = ctxs[i];
    if (!u || !c || u.isDelisted) continue;
    const oraclePx = num(c.oraclePx);
    const markPx = num(c.markPx);
    if (!(oraclePx > 0) || !(markPx > 0)) continue;
    out.push({
      coin: u.name,
      oraclePx,
      markPx,
      midPx: num(c.midPx),
      premium: num(c.premium) || 0,
      fundingHourly: num(c.funding) || 0,
      openInterest: num(c.openInterest) || 0,
      dayNotionalVolume: num(c.dayNtlVlm) || 0,
      prevDayPx: num(c.prevDayPx) || markPx,
      impactBid: num(c.impactPxs?.[0]),
      impactAsk: num(c.impactPxs?.[1]),
      maxLeverage: u.maxLeverage,
      szDecimals: u.szDecimals,
    });
  }
  return out;
}

export type PerpContextMap = Map<string, PerpContext>;

export const indexPerps = (ctxs: PerpContext[]): PerpContextMap =>
  new Map(ctxs.map((c) => [c.coin, c]));

/* ───────────────────────── cross-venue funding ───────────────────────── */

type RawPredictedFundings = [
  string,
  [
    string,
    {
      fundingRate: string;
      nextFundingTime: number;
      fundingIntervalHours?: number;
    } | null,
  ][],
][];

/**
 * Funding clock per venue, in hours.
 *
 * `predictedFundings` does NOT return an interval — each venue reports the raw
 * rate for its OWN settlement period. Hyperliquid settles hourly while the CEXs
 * settle every 8 hours, so the printed rates are not comparable: differencing
 * them as-is overstates HL by 8×. Defaulting a missing interval to 8 is just as
 * wrong the other way — it divides HL's already-hourly rate by 8 and shrinks
 * every dislocation to an eighth of its true size. So the clock is looked up
 * per venue, and an unrecognised venue is skipped rather than guessed at.
 */
const VENUE_INTERVAL_HOURS: Record<string, number> = {
  HlPerp: 1,
  BinPerp: 8,
  BybitPerp: 8,
};

export interface VenueFunding {
  venue: string;
  /** Normalized to an HOURLY rate so venues on 1h/4h/8h clocks compare. */
  hourly: number;
  intervalHours: number;
  nextFundingTime: number;
}

export interface FundingComparison {
  coin: string;
  venues: VenueFunding[];
  /** Hyperliquid's own hourly rate, if listed. */
  hlHourly: number | null;
  /** Mean hourly across the other venues — the "outside" market. */
  peerHourly: number | null;
  /** hl − peer, in bps per hour. The cross-venue carry dislocation. */
  dislocationBpsPerHour: number | null;
}

/**
 * `predictedFundings` gives the next funding rate on HL *and* on Binance/Bybit
 * for the same coin. Venues run different funding clocks, so the raw rates are
 * not comparable — each is divided by its own settlement interval to reach a
 * common hourly basis before anything is differenced.
 *
 * A persistent HL-vs-peer gap is a real, mechanical cross-venue carry trade,
 * and it also tells you whether HL's forward (which we price digitals off) is
 * idiosyncratic or corroborated by the wider market.
 */
export async function getFundingComparison(
  revalidate = 60
): Promise<Map<string, FundingComparison>> {
  const raw = await info<RawPredictedFundings>({ type: "predictedFundings" }, revalidate);
  const out = new Map<string, FundingComparison>();
  for (const [coin, entries] of raw) {
    const venues: VenueFunding[] = [];
    for (const [venue, data] of entries) {
      if (!data) continue;
      const rate = num(data.fundingRate);
      if (!Number.isFinite(rate)) continue;
      // Prefer an interval the API actually sent; otherwise the known clock for
      // this venue. A venue we don't know the clock for is dropped — a wrong
      // normalization is worse than a missing peer.
      const intervalHours = data.fundingIntervalHours ?? VENUE_INTERVAL_HOURS[venue];
      if (!(intervalHours > 0)) continue;
      venues.push({
        venue,
        hourly: rate / intervalHours,
        intervalHours,
        nextFundingTime: data.nextFundingTime,
      });
    }
    if (venues.length === 0) continue;
    const hl = venues.find((v) => v.venue === "HlPerp") ?? null;
    const peers = venues.filter((v) => v.venue !== "HlPerp");
    const peerHourly =
      peers.length > 0 ? peers.reduce((s, v) => s + v.hourly, 0) / peers.length : null;
    out.set(coin, {
      coin,
      venues,
      hlHourly: hl?.hourly ?? null,
      peerHourly,
      dislocationBpsPerHour: hl && peerHourly !== null ? (hl.hourly - peerHourly) * 10_000 : null,
    });
  }
  return out;
}

/* ──────────────────────────── candles ──────────────────────────── */

interface RawCandle {
  t: number;
  T: number;
  s: string;
  i: string;
  o: string;
  c: string;
  h: string;
  l: string;
  v: string;
  n: number;
}

export type CandleInterval = "1m" | "5m" | "15m" | "1h" | "4h" | "1d";

export const INTERVAL_MINUTES: Record<CandleInterval, number> = {
  "1m": 1,
  "5m": 5,
  "15m": 15,
  "1h": 60,
  "4h": 240,
  "1d": 1440,
};

/**
 * Pick a bar length for a given horizon.
 *
 * Vol estimated on bars far coarser than the horizon is stale; estimated on
 * bars far finer, it is microstructure noise (bid-ask bounce inflates every
 * range estimator). The rule of thumb that survives both failures is ~100–300
 * bars covering roughly the life of the claim, so the bar scales with it.
 */
export function intervalForHorizon(hours: number): CandleInterval {
  if (!Number.isFinite(hours) || hours <= 0) return "1h";
  if (hours <= 6) return "1m";
  if (hours <= 24) return "5m";
  if (hours <= 96) return "15m";
  if (hours <= 24 * 21) return "1h";
  if (hours <= 24 * 90) return "4h";
  return "1d";
}

/**
 * OHLCV history — the input to every realized-vol estimator.
 *
 * `lookbackBars` is converted to a wall-clock window because the API takes a
 * time range, not a count. We ask for the window and take the last N bars, so a
 * gap in HL's history shortens the sample rather than silently shifting it.
 */
export async function getCandles(
  coin: string,
  interval: CandleInterval,
  lookbackBars: number,
  now = Date.now(),
  revalidate = 60
): Promise<Candle[]> {
  const minutes = INTERVAL_MINUTES[interval];
  const startTime = now - lookbackBars * minutes * 60_000;
  const raw = await info<RawCandle[]>(
    {
      type: "candleSnapshot",
      req: { coin, interval, startTime, endTime: now },
    },
    revalidate
  );
  return raw
    .map((k) => ({
      t: k.t,
      o: num(k.o),
      h: num(k.h),
      l: num(k.l),
      c: num(k.c),
      v: num(k.v),
    }))
    .filter((k) => k.o > 0 && k.h > 0 && k.l > 0 && k.c > 0)
    .slice(-lookbackBars);
}

/* ──────────────────────────── order book ──────────────────────────── */

interface RawL2 {
  coin: string;
  time: number;
  levels: [{ px: string; sz: string; n: number }[], { px: string; sz: string; n: number }[]];
}

/**
 * Live ladder, adapted straight into the SAME `OrderBook` shape the Polymarket
 * fill simulator in `execution.ts` already walks. That reuse is the point: one
 * slippage model, two venues, so a cross-venue trade can be costed end to end
 * with the same arithmetic on both legs.
 *
 * HL returns `levels: [bids, asks]` positionally, best-first.
 */
export async function getPerpBook(coin: string, revalidate = 5): Promise<OrderBook> {
  const raw = await info<RawL2>({ type: "l2Book", coin }, revalidate);
  const parse = (side: { px: string; sz: string }[] | undefined) =>
    (side ?? [])
      .map((l) => ({ price: num(l.px), size: num(l.sz) }))
      .filter((l) => Number.isFinite(l.price) && Number.isFinite(l.size));
  return {
    tokenId: `HL:${coin}`,
    bids: parse(raw.levels?.[0]),
    asks: parse(raw.levels?.[1]),
    timestamp: raw.time ?? Date.now(),
  };
}

/* ──────────────────────── historical funding ──────────────────────── */

export interface FundingPoint {
  time: number;
  /** Hourly rate, decimal. */
  rate: number;
  premium: number;
}

/**
 * Realized funding history. The mean over the window is a far better carry
 * estimate than the single next-hour print, which is noisy enough that pricing
 * a month-out forward off it would be indefensible.
 */
export async function getFundingHistory(
  coin: string,
  hours = 168,
  now = Date.now(),
  revalidate = 300
): Promise<FundingPoint[]> {
  const raw = await info<{ coin: string; fundingRate: string; premium: string; time: number }[]>(
    { type: "fundingHistory", coin, startTime: now - hours * 3_600_000 },
    revalidate
  );
  return raw
    .map((f) => ({
      time: f.time,
      rate: num(f.fundingRate),
      premium: num(f.premium),
    }))
    .filter((f) => Number.isFinite(f.rate));
}

/** Time-average of a funding history window — the carry we actually price off. */
export function averageFunding(history: FundingPoint[]): number {
  if (history.length === 0) return 0;
  return history.reduce((s, f) => s + f.rate, 0) / history.length;
}
