const GAMMA = "https://gamma-api.polymarket.com";
const CLOB = "https://clob.polymarket.com";

export interface GammaMarket {
  id: string;
  question: string;
  slug: string;
  groupItemTitle?: string;
  outcomes?: string; // JSON-encoded string[]
  outcomePrices?: string; // JSON-encoded string[]
  clobTokenIds?: string; // JSON-encoded string[]
  lastTradePrice?: number;
  bestBid?: number;
  bestAsk?: number;
  spread?: number;
  oneDayPriceChange?: number;
  oneWeekPriceChange?: number;
  volume24hr?: number;
  volumeNum?: number;
  liquidityNum?: number;
  endDate?: string;
  active?: boolean;
  closed?: boolean;
  icon?: string;
}

export interface GammaEvent {
  id: string;
  slug: string;
  title: string;
  description?: string;
  icon?: string;
  image?: string;
  endDate?: string;
  liquidity?: number;
  volume?: number;
  volume24hr?: number;
  volume1wk?: number;
  openInterest?: number;
  commentCount?: number;
  negRisk?: boolean;
  markets: GammaMarket[];
  tags?: { id: string; label: string; slug: string }[];
}

export interface PricePoint {
  t: number; // unix seconds
  p: number; // price 0..1
}

export interface Outcome {
  marketId: string;
  label: string;
  price: number;
  change24h: number;
  volume24h: number;
  bestBid?: number;
  bestAsk?: number;
  spread?: number;
  tokenId?: string;
}

async function gamma<T>(path: string, revalidate: number): Promise<T> {
  const res = await fetch(`${GAMMA}${path}`, { next: { revalidate } });
  if (!res.ok) {
    throw new Error(`Gamma API ${res.status} for ${path}`);
  }
  return res.json();
}

/**
 * Same call, but explicitly opted OUT of Next's data cache.
 *
 * Used only for the oversized list endpoints. Asking Next to cache a 3MB
 * response is not a no-op that quietly degrades: it serializes the body,
 * discovers it exceeds the 2MB ceiling, throws it away and logs an error —
 * every request, for a cache entry that can never exist. Declaring `no-store`
 * says the true thing, skips that work, and leaves `cachedList` as the single
 * caching layer for these paths.
 */
async function gammaUncached<T>(path: string): Promise<T> {
  const res = await fetch(`${GAMMA}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Gamma API ${res.status} for ${path}`);
  }
  return res.json();
}

/* ─────────────────────── the list-payload problem ─────────────────────── */

/**
 * Gamma's `/events` list endpoint returns 2.5–4MB for a single page.
 *
 * Next's data cache refuses to store any entry over 2MB, so `next: { revalidate }`
 * on these calls did not merely underperform — it failed outright on every
 * request ("items over 2MB can not be cached"), meaning the board refetched
 * multiple megabytes on every single render and logged an error each time.
 *
 * Almost all of that weight is prose we never render on a list: per-market
 * `description` (the full resolution rules), images, and dozens of unused
 * fields. So instead of trying to cache the raw response, we cache the TRIMMED
 * projection — the ~30 fields the board, the scanners and the desk actually
 * read. That is two orders of magnitude smaller, fits the cache comfortably,
 * and is also less to hand to the Web Worker.
 */
const LIST_TTL_MS = 60_000;
/** Bounded so a long-lived server can't accumulate pages indefinitely. */
const LIST_CACHE_MAX = 64;

const listCache = new Map<string, { at: number; events: GammaEvent[] }>();

/** Keep only what a list view reads. Drops the multi-KB prose fields. */
function trimForList(event: GammaEvent): GammaEvent {
  return {
    id: event.id,
    slug: event.slug,
    title: event.title,
    icon: event.icon,
    endDate: event.endDate,
    liquidity: event.liquidity,
    volume: event.volume,
    volume24hr: event.volume24hr,
    openInterest: event.openInterest,
    negRisk: event.negRisk,
    tags: event.tags,
    markets: (event.markets ?? []).map((m) => ({
      id: m.id,
      question: m.question,
      slug: m.slug,
      groupItemTitle: m.groupItemTitle,
      outcomes: m.outcomes,
      outcomePrices: m.outcomePrices,
      clobTokenIds: m.clobTokenIds,
      lastTradePrice: m.lastTradePrice,
      bestBid: m.bestBid,
      bestAsk: m.bestAsk,
      spread: m.spread,
      oneDayPriceChange: m.oneDayPriceChange,
      oneWeekPriceChange: m.oneWeekPriceChange,
      volume24hr: m.volume24hr,
      endDate: m.endDate,
      active: m.active,
      closed: m.closed,
    })),
  };
}

/**
 * Fetch a list endpoint, trim it, and memoize the trimmed result in-process.
 *
 * This REPLACES Next's data cache for these paths rather than layering on it,
 * because that cache provably cannot hold them. The TTL matches the 60s
 * revalidate the fetches used to ask for, so refresh behaviour is unchanged;
 * what changes is that the cache now actually holds something.
 */
async function cachedList(
  path: string,
  extract: (raw: unknown) => GammaEvent[],
): Promise<GammaEvent[]> {
  const hit = listCache.get(path);
  const now = Date.now();
  if (hit && now - hit.at < LIST_TTL_MS) return hit.events;

  const raw = await gammaUncached<unknown>(path);
  const events = extract(raw).map(trimForList);

  if (listCache.size >= LIST_CACHE_MAX) {
    // Map preserves insertion order, so the first key is the oldest entry.
    const oldest = listCache.keys().next().value;
    if (oldest !== undefined) listCache.delete(oldest);
  }
  listCache.set(path, { at: now, events });
  return events;
}

export async function getTopEvents(
  tagSlug?: string,
  limit = 25,
  offset = 0,
): Promise<GammaEvent[]> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
    active: "true",
    closed: "false",
    order: "volume24hr",
    ascending: "false",
  });
  if (tagSlug) params.set("tag_slug", tagSlug);
  return cachedList(`/events?${params}`, (raw) =>
    Array.isArray(raw) ? (raw as GammaEvent[]) : [],
  );
}

export async function getRelatedEvents(
  slug: string,
  tagSlugs: string[],
  limit = 6,
): Promise<GammaEvent[]> {
  const tag = tagSlugs[0];
  if (!tag) return [];
  const events = await getTopEvents(tag, limit + 1, 0);
  return events.filter((e) => e.slug !== slug).slice(0, limit);
}

export async function searchEvents(query: string): Promise<GammaEvent[]> {
  const params = new URLSearchParams({
    q: query,
    limit_per_type: "40",
    events_status: "active",
  });
  // `trimForList` already normalizes `markets` to an array, which matters here:
  // search results sometimes omit the nested markets entirely.
  return cachedList(`/public-search?${params}`, (raw) => {
    const data = raw as { events?: GammaEvent[] };
    return data.events ?? [];
  });
}

export async function getEventBySlug(slug: string): Promise<GammaEvent | null> {
  const events = await gamma<GammaEvent[]>(`/events?slug=${encodeURIComponent(slug)}`, 30);
  return events[0] ?? null;
}

export async function getPriceHistory(tokenId: string, interval: string): Promise<PricePoint[]> {
  const fidelity =
    interval === "1d"
      ? "5"
      : interval === "1w"
        ? "30"
        : interval === "1m"
          ? "60"
          : interval === "3m"
            ? "240"
            : "1440";
  const params = new URLSearchParams({
    market: tokenId,
    interval,
    fidelity,
  });
  const res = await fetch(`${CLOB}/prices-history?${params}`, {
    next: { revalidate: 120 },
  });
  if (!res.ok) return [];
  const data: { history?: PricePoint[] } = await res.json();
  return data.history ?? [];
}

function parseJsonArray(value: string | undefined): string[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * Memoized per event object.
 *
 * `eventOutcomes` is called five or six times for the same event in a single
 * board render — the sort comparator, the row, the breadth bar, the movers
 * strip, the ticker — and each call `JSON.parse`s three encoded arrays per
 * market. On a 25-row board of multi-outcome events that is thousands of
 * redundant parses per request. A WeakMap keyed on the event object collapses
 * them to one, and because every request deserializes fresh objects there is no
 * staleness risk: a new fetch is a new key.
 */
const outcomeCache = new WeakMap<GammaEvent, Outcome[]>();

/**
 * Flatten an event's markets into displayable outcomes. Multi-market events
 * (e.g. "World Cup Winner") use each market's Yes side labeled by
 * groupItemTitle; single binary markets expand to their Yes/No outcomes.
 */
export function eventOutcomes(event: GammaEvent): Outcome[] {
  const cached = outcomeCache.get(event);
  if (cached) return cached;
  const computed = computeOutcomes(event);
  outcomeCache.set(event, computed);
  return computed;
}

function computeOutcomes(event: GammaEvent): Outcome[] {
  const open = event.markets.filter((m) => m.active && !m.closed);
  if (open.length === 0) return [];

  if (open.length === 1) {
    const m = open[0];
    const labels = parseJsonArray(m.outcomes);
    const prices = parseJsonArray(m.outcomePrices).map(Number);
    const tokens = parseJsonArray(m.clobTokenIds);
    return labels.map((label, i) => ({
      marketId: m.id,
      label,
      price: prices[i] ?? 0,
      change24h: i === 0 ? (m.oneDayPriceChange ?? 0) : -(m.oneDayPriceChange ?? 0),
      volume24h: m.volume24hr ?? 0,
      bestBid: m.bestBid,
      bestAsk: m.bestAsk,
      spread: m.spread,
      tokenId: tokens[i],
    }));
  }

  return open
    .map((m) => {
      const prices = parseJsonArray(m.outcomePrices).map(Number);
      const tokens = parseJsonArray(m.clobTokenIds);
      return {
        marketId: m.id,
        label: m.groupItemTitle || m.question,
        price: prices[0] ?? m.lastTradePrice ?? 0,
        change24h: m.oneDayPriceChange ?? 0,
        volume24h: m.volume24hr ?? 0,
        bestBid: m.bestBid,
        bestAsk: m.bestAsk,
        spread: m.spread,
        tokenId: tokens[0],
      };
    })
    .toSorted((a, b) => b.price - a.price);
}

/** Top outcome of an event, for the dashboard table. */
export function leadingOutcome(event: GammaEvent): Outcome | null {
  return eventOutcomes(event)[0] ?? null;
}

export function fmtUsd(value: number | undefined): string {
  const v = value ?? 0;
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
  return `$${v.toFixed(0)}`;
}

export function fmtCents(price: number): string {
  return `${(price * 100).toFixed(1)}¢`;
}

export function fmtPct(price: number): string {
  return `${(price * 100).toFixed(price >= 0.995 || price < 0.01 ? 1 : 0)}%`;
}

export function fmtChange(change: number): string {
  const pts = change * 100;
  const sign = pts > 0 ? "+" : "";
  return `${sign}${pts.toFixed(1)}`;
}

export function daysUntil(iso: string | undefined): number {
  if (!iso) return Infinity;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return Infinity;
  return (d.getTime() - Date.now()) / 86400000;
}

export function fmtDate(iso: string | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d
    .toLocaleDateString("en-US", {
      month: "short",
      day: "2-digit",
      year: "numeric",
      timeZone: "UTC",
    })
    .toUpperCase();
}
