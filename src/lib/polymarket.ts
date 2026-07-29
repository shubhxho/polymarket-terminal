import type {
  EventSummary,
  Holder,
  HistoryInterval,
  Market,
  OrderBook,
  Position,
  PricePoint,
  Trade,
} from "./types";

const GAMMA = "https://gamma-api.polymarket.com";
const CLOB = "https://clob.polymarket.com";
const DATA = "https://data-api.polymarket.com";

/** Upstream sometimes 5xxs under load; one retry turns a blank panel into a
 *  half-second stall. `revalidate` is seconds of Next data-cache reuse. */
async function get<T>(url: string, revalidate: number, retries = 1): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, {
        next: { revalidate },
        headers: { accept: "application/json" },
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText} :: ${url}`);
      return (await res.json()) as T;
    } catch (err) {
      lastErr = err;
      if (attempt < retries) await new Promise((r) => setTimeout(r, 250));
    }
  }
  throw lastErr;
}

function n(v: unknown, fallback = 0): number {
  const x = typeof v === "string" ? parseFloat(v) : typeof v === "number" ? v : NaN;
  return Number.isFinite(x) ? x : fallback;
}

/** Gamma returns `outcomes`/`outcomePrices`/`clobTokenIds` as JSON-encoded
 *  strings rather than arrays, so every read has to go through this. */
function jsonArray(v: unknown): string[] {
  if (Array.isArray(v)) return v.map(String);
  if (typeof v === "string") {
    try {
      const parsed = JSON.parse(v);
      return Array.isArray(parsed) ? parsed.map(String) : [];
    } catch {
      return [];
    }
  }
  return [];
}

type RawMarket = Record<string, unknown>;

export function normalizeMarket(raw: RawMarket): Market {
  const labels = jsonArray(raw.outcomes);
  const prices = jsonArray(raw.outcomePrices).map((p) => n(p));
  const tokens = jsonArray(raw.clobTokenIds);

  const outcomes = labels.map((label, i) => ({
    label,
    price: prices[i] ?? 0,
    tokenId: tokens[i] ?? "",
  }));

  const events = Array.isArray(raw.events) ? (raw.events as RawMarket[]) : [];
  const ev = events[0];
  const evTags = ev && Array.isArray(ev.tags) ? (ev.tags as RawMarket[]) : [];

  // `lastTradePrice` is the truest mark, but it is absent on thin markets —
  // fall back to the quoted outcome price.
  const last = n(raw.lastTradePrice, outcomes[0]?.price ?? 0);

  return {
    id: String(raw.id ?? ""),
    slug: String(raw.slug ?? ""),
    question: String(raw.question ?? ""),
    groupItemTitle: raw.groupItemTitle ? String(raw.groupItemTitle) : undefined,
    conditionId: String(raw.conditionId ?? ""),
    icon: raw.icon ? String(raw.icon) : undefined,
    description: raw.description ? String(raw.description) : undefined,
    outcomes,
    last,
    bestBid: raw.bestBid !== undefined ? n(raw.bestBid) : undefined,
    bestAsk: raw.bestAsk !== undefined ? n(raw.bestAsk) : undefined,
    spread: raw.spread !== undefined ? n(raw.spread) : undefined,
    // Upstream expresses changes in probability (0..1); screens want points.
    chg1h: n(raw.oneHourPriceChange) * 100,
    chg24h: n(raw.oneDayPriceChange) * 100,
    chg1w: n(raw.oneWeekPriceChange) * 100,
    volume: n(raw.volumeNum, n(raw.volume)),
    volume24h: n(raw.volume24hr),
    volume1w: n(raw.volume1wk),
    liquidity: n(raw.liquidityNum, n(raw.liquidity)),
    endDate: raw.endDate ? String(raw.endDate) : undefined,
    startDate: raw.startDate ? String(raw.startDate) : undefined,
    active: raw.active === true,
    closed: raw.closed === true,
    acceptingOrders: raw.acceptingOrders !== false,
    negRisk: raw.negRisk === true,
    tickSize: n(raw.orderPriceMinTickSize, 0.001),
    eventId: ev ? String(ev.id) : undefined,
    eventTitle: ev ? String(ev.title ?? "") : undefined,
    eventSlug: ev ? String(ev.slug ?? "") : undefined,
    eventTicker: ev ? String(ev.ticker ?? "") : undefined,
    tags: evTags.map((t) => String(t.label ?? "")).filter(Boolean),
  };
}

function normalizeEvent(raw: RawMarket): EventSummary {
  const markets = Array.isArray(raw.markets) ? (raw.markets as RawMarket[]) : [];
  const tags = Array.isArray(raw.tags) ? (raw.tags as RawMarket[]) : [];
  return {
    id: String(raw.id ?? ""),
    ticker: String(raw.ticker ?? ""),
    slug: String(raw.slug ?? ""),
    title: String(raw.title ?? ""),
    icon: raw.icon ? String(raw.icon) : undefined,
    volume: n(raw.volume),
    volume24h: n(raw.volume24hr),
    liquidity: n(raw.liquidity),
    openInterest: n(raw.openInterest),
    endDate: raw.endDate ? String(raw.endDate) : undefined,
    competitive: raw.competitive !== undefined ? n(raw.competitive) : undefined,
    markets: markets
      .map((m) => normalizeMarket({ ...m, events: [{ ...raw, tags }] }))
      .filter((m) => m.outcomes.length > 0),
    tags: tags.map((t) => String(t.label ?? "")).filter(Boolean),
  };
}

export type MarketQuery = {
  limit?: number;
  offset?: number;
  order?: string;
  ascending?: boolean;
  tagId?: string;
  closed?: boolean;
  ids?: string[];
  slug?: string;
  tokenIds?: string[];
};

export async function fetchMarkets(q: MarketQuery = {}): Promise<Market[]> {
  const p = new URLSearchParams();
  p.set("limit", String(q.limit ?? 60));
  if (q.offset) p.set("offset", String(q.offset));
  p.set("order", q.order ?? "volume24hr");
  p.set("ascending", String(q.ascending ?? false));
  // A slug (or explicit id list) is a unique key — never constrain it by
  // status, or a market that has since resolved (e.g. an expired
  // "…-end-of-july" question) returns nothing and can't be opened at all.
  const byKey = Boolean(q.slug) || (q.ids?.length ?? 0) > 0;
  if (byKey) {
    // no active/closed filter — match the exact market whatever its status
  } else if (q.closed === undefined) {
    p.set("active", "true");
    p.set("closed", "false");
  } else {
    p.set("closed", String(q.closed));
  }
  if (q.tagId) p.set("tag_id", q.tagId);
  if (q.slug) p.set("slug", q.slug);
  q.ids?.forEach((id) => p.append("id", id));
  q.tokenIds?.forEach((t) => p.append("clob_token_ids", t));

  const raw = await get<RawMarket[]>(`${GAMMA}/markets?${p}`, 10);
  return (Array.isArray(raw) ? raw : []).map(normalizeMarket);
}

export async function fetchEvents(q: MarketQuery = {}): Promise<EventSummary[]> {
  const p = new URLSearchParams();
  p.set("limit", String(q.limit ?? 30));
  if (q.offset) p.set("offset", String(q.offset));
  p.set("order", q.order ?? "volume24hr");
  p.set("ascending", String(q.ascending ?? false));
  // Same rule as fetchMarkets: a slug is exact, so don't hide resolved events.
  if (!q.slug) {
    p.set("active", "true");
    p.set("closed", "false");
  }
  if (q.tagId) p.set("tag_id", q.tagId);
  if (q.slug) p.set("slug", q.slug);

  const raw = await get<RawMarket[]>(`${GAMMA}/events?${p}`, 10);
  return (Array.isArray(raw) ? raw : []).map(normalizeEvent);
}

export async function searchPolymarket(
  query: string,
  limit = 12
): Promise<{ events: EventSummary[]; markets: Market[] }> {
  const p = new URLSearchParams({
    q: query,
    limit_per_type: String(limit),
    events_status: "active",
  });
  const raw = await get<{ events?: RawMarket[]; markets?: RawMarket[] }>(
    `${GAMMA}/public-search?${p}`,
    15
  );

  const events = (raw.events ?? []).map(normalizeEvent);
  const direct = (raw.markets ?? []).map(normalizeMarket);

  // Upstream almost always returns an empty top-level `markets` array and puts
  // the real matches inside each event, so a naive read shows "no markets" for
  // a query that in fact matched dozens. Fall back to the event legs, ranked
  // by turnover, and de-duplicate against anything upstream did return.
  const seen = new Set(direct.map((m) => m.id));
  const nested = events
    .flatMap((e) => e.markets)
    .filter((m) => m.id && !seen.has(m.id) && seen.add(m.id));

  return {
    events,
    markets: [...direct, ...nested].sort((a, b) => b.volume24h - a.volume24h),
  };
}

function normalizeBook(raw: RawMarket): OrderBook {
  const level = (v: unknown): { price: number; size: number }[] =>
    (Array.isArray(v) ? v : []).map((l) => {
      const o = l as RawMarket;
      return { price: n(o.price), size: n(o.size) };
    });
  return {
    tokenId: String(raw.asset_id ?? ""),
    // CLOB returns bids ascending and asks descending; screens render both
    // best-first, so sort here rather than in every consumer.
    bids: level(raw.bids).sort((a, b) => b.price - a.price),
    asks: level(raw.asks).sort((a, b) => a.price - b.price),
    timestamp: n(raw.timestamp, Date.now()),
  };
}

export async function fetchBooks(tokenIds: string[]): Promise<OrderBook[]> {
  const ids = tokenIds.filter(Boolean);
  if (ids.length === 0) return [];
  const res = await fetch(`${CLOB}/books`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(ids.map((token_id) => ({ token_id }))),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`books ${res.status}`);
  const raw = (await res.json()) as RawMarket[];
  return (Array.isArray(raw) ? raw : []).map(normalizeBook);
}

export async function fetchHistory(
  tokenId: string,
  interval: HistoryInterval = "1d",
  fidelity?: number
): Promise<PricePoint[]> {
  // Point spacing in minutes. Too fine over a long window returns thousands of
  // points the chart can't resolve; these keep each series near 200-600 points.
  const defaults: Record<HistoryInterval, number> = {
    "1h": 1,
    "6h": 5,
    "1d": 10,
    "1w": 60,
    "1m": 240,
    max: 720,
  };
  const p = new URLSearchParams({
    market: tokenId,
    interval,
    fidelity: String(fidelity ?? defaults[interval]),
  });
  const raw = await get<{ history?: { t: number; p: number }[] }>(
    `${CLOB}/prices-history?${p}`,
    20
  );
  return (raw.history ?? []).map((h) => ({ t: n(h.t), p: n(h.p) }));
}

function normalizeTrade(raw: RawMarket, i: number): Trade {
  return {
    id: `${String(raw.transactionHash ?? i)}-${String(raw.asset ?? "")}-${i}`,
    wallet: String(raw.proxyWallet ?? ""),
    name: raw.name ? String(raw.name) : raw.pseudonym ? String(raw.pseudonym) : undefined,
    side: raw.side === "SELL" ? "SELL" : "BUY",
    outcome: String(raw.outcome ?? ""),
    outcomeIndex: n(raw.outcomeIndex),
    size: n(raw.size),
    price: n(raw.price),
    timestamp: n(raw.timestamp),
    title: String(raw.title ?? ""),
    slug: raw.slug ? String(raw.slug) : undefined,
    conditionId: String(raw.conditionId ?? ""),
    asset: String(raw.asset ?? ""),
  };
}

export async function fetchTrades(opts: {
  conditionId?: string;
  user?: string;
  limit?: number;
  /** Filter to trades at or above this notional (size x price) in USDC. */
  minSize?: number;
}): Promise<Trade[]> {
  const p = new URLSearchParams();
  p.set("limit", String(opts.limit ?? 60));
  p.set("takerOnly", "true");
  if (opts.conditionId) p.set("market", opts.conditionId);
  if (opts.user) p.set("user", opts.user);
  if (opts.minSize) p.set("filterAmount", String(opts.minSize));
  const raw = await get<RawMarket[]>(`${DATA}/trades?${p}`, 3);
  return (Array.isArray(raw) ? raw : []).map(normalizeTrade);
}

export async function fetchHolders(conditionId: string, limit = 12): Promise<Holder[][]> {
  const p = new URLSearchParams({ market: conditionId, limit: String(limit) });
  const raw = await get<{ token: string; holders: RawMarket[] }[]>(`${DATA}/holders?${p}`, 30);
  return (Array.isArray(raw) ? raw : []).map((group) =>
    (group.holders ?? []).map((h) => ({
      wallet: String(h.proxyWallet ?? ""),
      name: h.name ? String(h.name) : h.pseudonym ? String(h.pseudonym) : undefined,
      amount: n(h.amount),
      outcomeIndex: n(h.outcomeIndex),
    }))
  );
}

export async function fetchPositions(user: string, limit = 50): Promise<Position[]> {
  const p = new URLSearchParams({
    user,
    limit: String(limit),
    sortBy: "CURRENT",
    sortDirection: "DESC",
    sizeThreshold: "1",
  });
  const raw = await get<RawMarket[]>(`${DATA}/positions?${p}`, 10);
  return (Array.isArray(raw) ? raw : []).map((r) => ({
    conditionId: String(r.conditionId ?? ""),
    asset: String(r.asset ?? ""),
    title: String(r.title ?? ""),
    slug: r.slug ? String(r.slug) : undefined,
    outcome: String(r.outcome ?? ""),
    size: n(r.size),
    avgPrice: n(r.avgPrice),
    curPrice: n(r.curPrice),
    value: n(r.currentValue),
    cashPnl: n(r.cashPnl),
    percentPnl: n(r.percentPnl),
    realizedPnl: n(r.realizedPnl),
    redeemable: r.redeemable === true,
    endDate: r.endDate ? String(r.endDate) : undefined,
  }));
}

export async function fetchTags(): Promise<{ id: string; label: string; slug: string }[]> {
  const raw = await get<RawMarket[]>(`${GAMMA}/tags?limit=200&order=id&ascending=true`, 3600);
  return (Array.isArray(raw) ? raw : []).map((t) => ({
    id: String(t.id ?? ""),
    label: String(t.label ?? ""),
    slug: String(t.slug ?? ""),
  }));
}
