import type { NextRequest } from "next/server";
import { type GammaEvent, getTopEvents, leadingOutcome, searchEvents } from "@/lib/polymarket";

/** CSV cell: quote and escape, so a comma in a title can't shift the columns. */
const cell = (v: string): string => `"${v.replace(/"/g, '""')}"`;

/**
 * A CSV is read by a spreadsheet, not by a person, so every numeric column is
 * emitted RAW. Writing the display formats ("$1.2M", "63%", "JAN 04 2026") into
 * the file turns each of those columns into text that will not sum, sort or
 * chart on the other end — the export exists precisely so the numbers can be
 * worked on elsewhere.
 */
export async function GET(request: NextRequest) {
  const tag = request.nextUrl.searchParams.get("tag") ?? "";
  const q = request.nextUrl.searchParams.get("q") ?? "";

  let events: GammaEvent[];
  try {
    events = q ? await searchEvents(q) : await getTopEvents(tag || undefined, 200, 0);
  } catch {
    return new Response("Failed to fetch market data", { status: 502 });
  }

  const rows = events.filter((e) => e.markets.length > 0);

  const header = [
    "Slug",
    "Market",
    "Leading Outcome",
    "Implied Probability",
    "24H Change (pts)",
    "24H Volume (USD)",
    "Liquidity (USD)",
    "Open Interest (USD)",
    "End Date (ISO)",
  ].join(",");

  const lines = rows.map((event) => {
    const lead = leadingOutcome(event);
    const end = event.endDate ? new Date(event.endDate) : null;
    return [
      cell(event.slug),
      cell(event.title),
      cell(lead?.label ?? ""),
      lead ? lead.price.toFixed(4) : "",
      ((lead?.change24h ?? 0) * 100).toFixed(2),
      String(event.volume24hr ?? 0),
      String(event.liquidity ?? 0),
      String(event.openInterest ?? 0),
      end && !Number.isNaN(end.getTime()) ? end.toISOString() : "",
    ].join(",");
  });

  const csv = [header, ...lines].join("\r\n");
  const slug = q ? `search-${q.slice(0, 20)}` : tag || "trending";
  const filename = `polymarket-${slug}.csv`;

  return new Response(csv, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="${filename}"`,
    },
  });
}
