import type { NextRequest } from "next/server";
import type { BookLevel, OrderBook } from "@/lib/execution";

const CLOB = "https://clob.polymarket.com";
const MAX_TOKENS = 60;

interface RawLevel {
  price: string;
  size: string;
}
interface RawBook {
  asset_id?: string;
  bids?: RawLevel[];
  asks?: RawLevel[];
}

function parseLevels(levels: RawLevel[] | undefined): BookLevel[] {
  if (!levels) return [];
  return levels
    .map((l) => ({ price: Number(l.price), size: Number(l.size) }))
    .filter((l) => Number.isFinite(l.price) && Number.isFinite(l.size));
}

/**
 * Proxies Polymarket's CLOB batch order-book endpoint so the client fill
 * simulator can read live depth without CORS. GET /api/book?tokens=id1,id2,…
 */
export async function GET(request: NextRequest) {
  const raw = request.nextUrl.searchParams.get("tokens") ?? "";
  const tokens = raw
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean)
    .slice(0, MAX_TOKENS);

  if (tokens.length === 0) {
    return Response.json({ error: "no tokens" }, { status: 400 });
  }

  let books: RawBook[];
  try {
    const res = await fetch(`${CLOB}/books`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(tokens.map((token_id) => ({ token_id }))),
      next: { revalidate: 5 },
    });
    if (!res.ok) throw new Error(`clob ${res.status}`);
    books = await res.json();
  } catch {
    return Response.json({ error: "book fetch failed" }, { status: 502 });
  }

  const out: OrderBook[] = (Array.isArray(books) ? books : [])
    .filter((b) => b.asset_id)
    .map((b) => ({
      tokenId: b.asset_id as string,
      bids: parseLevels(b.bids),
      asks: parseLevels(b.asks),
    }));

  return Response.json({ books: out });
}
