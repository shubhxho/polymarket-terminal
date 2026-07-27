import type { NextRequest } from "next/server";
import {
  fmtDate,
  fmtPct,
  fmtUsd,
  getTopEvents,
  leadingOutcome,
  searchEvents,
} from "@/lib/polymarket";

export async function GET(request: NextRequest) {
  const tag = request.nextUrl.searchParams.get("tag") ?? "";
  const q = request.nextUrl.searchParams.get("q") ?? "";

  let events;
  try {
    events = q
      ? await searchEvents(q)
      : await getTopEvents(tag || undefined, 200, 0);
  } catch {
    return new Response("Failed to fetch market data", { status: 502 });
  }

  const rows = events.filter((e) => e.markets.length > 0);

  const header = [
    "Market",
    "Leading Outcome",
    "Odds",
    "24H Change (pts)",
    "24H Volume",
    "Liquidity",
    "Open Interest",
    "End Date",
  ].join(",");

  const lines = rows.map((event) => {
    const lead = leadingOutcome(event);
    const change = lead?.change24h ?? 0;
    const pts = (change * 100).toFixed(1);
    const changeFmt =
      change > 0.001 ? `+${pts}` : change < -0.001 ? pts : `±0.0`;
    return [
      `"${event.title.replace(/"/g, '""')}"`,
      `"${(lead?.label ?? "—").replace(/"/g, '""')}"`,
      lead ? fmtPct(lead.price) : "—",
      changeFmt,
      fmtUsd(event.volume24hr),
      fmtUsd(event.liquidity),
      fmtUsd(event.openInterest),
      fmtDate(event.endDate),
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
