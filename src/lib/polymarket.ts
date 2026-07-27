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
  return gamma<GammaEvent[]>(`/events?${params}`, 60);
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
  const data = await gamma<{ events?: GammaEvent[] }>(
    `/public-search?${params}`,
    60,
  );
  // Search results omit nested markets sometimes; keep shape consistent.
  return (data.events ?? []).map((e) => ({ ...e, markets: e.markets ?? [] }));
}

export async function getEventBySlug(slug: string): Promise<GammaEvent | null> {
  const events = await gamma<GammaEvent[]>(
    `/events?slug=${encodeURIComponent(slug)}`,
    30,
  );
  return events[0] ?? null;
}

export async function getPriceHistory(
  tokenId: string,
  interval: string,
): Promise<PricePoint[]> {
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
 * Flatten an event's markets into displayable outcomes. Multi-market events
 * (e.g. "World Cup Winner") use each market's Yes side labeled by
 * groupItemTitle; single binary markets expand to their Yes/No outcomes.
 */
export function eventOutcomes(event: GammaEvent): Outcome[] {
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
      change24h:
        i === 0 ? (m.oneDayPriceChange ?? 0) : -(m.oneDayPriceChange ?? 0),
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
    .sort((a, b) => b.price - a.price);
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
