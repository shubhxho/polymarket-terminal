import Link from "next/link";
import {
  fmtChange,
  fmtPct,
  getTopEvents,
  leadingOutcome,
} from "@/lib/polymarket";

export async function Ticker() {
  let events;
  try {
    events = await getTopEvents(undefined, 20, 0);
  } catch {
    return null;
  }

  if (events.length === 0) return null;

  const items = events.flatMap((e) => {
    const lead = leadingOutcome(e);
    if (!lead) return [];
    return [
      {
        slug: e.slug,
        title: e.title,
        price: lead.price,
        change: lead.change24h,
        label: lead.label,
      },
    ];
  });

  if (items.length === 0) return null;

  // Duplicate 3× for a seamless infinite scroll
  const tripled = [...items, ...items, ...items];

  return (
    <div className="relative flex items-stretch border-b border-edge bg-panel-raised">
      {/* Pinned live-feed label */}
      <div className="z-10 flex shrink-0 items-center gap-1.5 border-r border-edge bg-panel px-3 text-[10px] font-bold tracking-widest text-accent shadow-[6px_0_10px_-4px_rgba(0,0,0,0.7)]">
        <span className="live-dot inline-block h-1.5 w-1.5 rounded-full bg-accent" />
        <span className="hidden glow-soft sm:inline">LIVE FEED</span>
      </div>

      {/* Scrolling belt with soft edge fades */}
      <div
        className="relative flex-1 overflow-hidden"
        style={{
          maskImage:
            "linear-gradient(90deg, transparent, #000 3%, #000 97%, transparent)",
          WebkitMaskImage:
            "linear-gradient(90deg, transparent, #000 3%, #000 97%, transparent)",
        }}
      >
        <div className="animate-ticker flex min-w-full gap-0 py-1.5">
          {tripled.map((item, i) => {
            const changeColor =
              item.change > 0.001
                ? "text-accent"
                : item.change < -0.001
                  ? "text-red"
                  : "text-muted/60";
            const arrow =
              item.change > 0.001 ? "▲" : item.change < -0.001 ? "▼" : "·";
            return (
              <Link
                key={i}
                href={`/event/${item.slug}`}
                className="group flex shrink-0 items-center gap-2 px-4 text-[11px] hover:bg-panel"
              >
                <span className="text-muted/30 group-hover:text-accent/60">
                  ·
                </span>
                <span className="max-w-[160px] truncate text-muted/70 group-hover:text-foreground">
                  {item.title}
                </span>
                <span className="shrink-0 font-bold tabular-nums text-amber">
                  {fmtPct(item.price)}
                </span>
                <span className={`shrink-0 tabular-nums ${changeColor}`}>
                  {arrow} {fmtChange(item.change)}
                </span>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
