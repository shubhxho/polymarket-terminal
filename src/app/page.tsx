import Link from "next/link";
import { Suspense } from "react";
import { DownloadButton } from "@/components/download-button";
import { Sparkline } from "@/components/sparkline";
import {
  daysUntil,
  fmtChange,
  fmtDate,
  fmtPct,
  fmtUsd,
  type GammaEvent,
  getPriceHistory,
  getTopEvents,
  leadingOutcome,
  type PricePoint,
  searchEvents,
} from "@/lib/polymarket";
import { type Signal, scanSignals } from "@/lib/signals";

const TAGS = [
  { slug: "", label: "TRENDING" },
  { slug: "politics", label: "POLITICS" },
  { slug: "sports", label: "SPORTS" },
  { slug: "crypto", label: "CRYPTO" },
  { slug: "economy", label: "ECONOMY" },
  { slug: "geopolitics", label: "GEOPOLITICS" },
  { slug: "tech", label: "TECH" },
  { slug: "pop-culture", label: "CULTURE" },
];

const SORT_KEYS = ["vol", "liq", "oi", "change", "ends", "odds"] as const;
type SortKey = (typeof SORT_KEYS)[number];

const PAGE_SIZE = 25;

interface SearchParams {
  tag?: string;
  q?: string;
  sort?: string;
  dir?: string;
  page?: string;
}

export default async function Home({ searchParams }: { searchParams: Promise<SearchParams> }) {
  const { tag = "", q = "", sort = "vol", dir = "desc", page = "1" } = await searchParams;

  return (
    <main className="flex flex-1 flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <nav className="flex flex-wrap gap-px border border-edge bg-edge">
          {TAGS.map((t) => {
            const active = t.slug === tag && !q;
            return (
              <Link
                key={t.slug}
                href={t.slug ? `/?tag=${t.slug}` : "/"}
                className={`px-3 py-1.5 text-xs tracking-wider transition-colors ${
                  active
                    ? "bg-accent font-bold text-black shadow-[0_0_16px_var(--accent-dim)]"
                    : "bg-panel text-muted hover:bg-panel-raised hover:text-accent"
                }`}
              >
                {t.label}
              </Link>
            );
          })}
        </nav>
        <div className="flex items-center gap-2">
          <form
            action="/"
            className="flex items-center gap-0 border border-edge bg-panel transition-shadow focus-within:border-accent/60 focus-within:shadow-[0_0_16px_-2px_var(--accent-dim)]"
          >
            <span className="cursor-blink pl-2 text-accent">&gt;</span>
            <input
              type="text"
              name="q"
              defaultValue={q}
              placeholder="SEARCH MARKETS_"
              autoComplete="off"
              data-search
              className="w-48 bg-transparent px-2 py-1.5 text-xs uppercase tracking-wider text-foreground placeholder:text-muted focus:outline-none sm:w-64"
            />
            <span className="hidden px-2 text-[10px] text-muted/40 sm:block">/</span>
            <button
              type="submit"
              className="border-l border-edge px-3 py-1.5 text-xs text-muted hover:bg-panel-raised hover:text-accent"
            >
              GO
            </button>
          </form>
          <DownloadButton tag={tag} query={q} />
        </div>
      </div>

      <Suspense key={`${tag}|${q}|${sort}|${dir}|${page}`} fallback={<BoardSkeleton />}>
        <Board tag={tag} query={q} sort={sort} dir={dir} page={Number(page) || 1} />
      </Suspense>
    </main>
  );
}

function sortHref(
  tag: string,
  query: string,
  sortKey: string,
  currentSort: string,
  currentDir: string,
): string {
  const p = new URLSearchParams();
  if (tag) p.set("tag", tag);
  if (query) p.set("q", query);
  p.set("sort", sortKey);
  p.set("dir", sortKey === currentSort && currentDir === "desc" ? "asc" : "desc");
  return `/?${p}`;
}

function pageHref(tag: string, query: string, sort: string, dir: string, page: number): string {
  const p = new URLSearchParams();
  if (tag) p.set("tag", tag);
  if (query) p.set("q", query);
  if (sort !== "vol") p.set("sort", sort);
  if (dir !== "desc") p.set("dir", dir);
  p.set("page", String(page));
  return `/?${p}`;
}

function applySortRows(rows: GammaEvent[], sort: string, dir: string): GammaEvent[] {
  if (!sort || sort === "vol") return rows;
  const asc = dir === "asc";
  return rows.toSorted((a, b) => {
    let va = 0;
    let vb = 0;
    if (sort === "liq") {
      va = a.liquidity ?? 0;
      vb = b.liquidity ?? 0;
    } else if (sort === "oi") {
      va = a.openInterest ?? 0;
      vb = b.openInterest ?? 0;
    } else if (sort === "change") {
      va = leadingOutcome(a)?.change24h ?? 0;
      vb = leadingOutcome(b)?.change24h ?? 0;
    } else if (sort === "ends") {
      va = a.endDate ? new Date(a.endDate).getTime() : Infinity;
      vb = b.endDate ? new Date(b.endDate).getTime() : Infinity;
    } else if (sort === "odds") {
      va = leadingOutcome(a)?.price ?? 0;
      vb = leadingOutcome(b)?.price ?? 0;
    }
    return asc ? va - vb : vb - va;
  });
}

async function Board({
  tag,
  query,
  sort,
  dir,
  page,
}: {
  tag: string;
  query: string;
  sort: string;
  dir: string;
  page: number;
}) {
  // True server-side pagination: fetch only one page at a time.
  // Search results come back as a flat list and are paginated client-side.
  const offset = (page - 1) * PAGE_SIZE;
  let events: GammaEvent[];
  let isLastPage = false;

  try {
    if (query) {
      const all = await searchEvents(query);
      const filtered = all.filter((e) => e.markets.length > 0);
      events = applySortRows(filtered, sort, dir).slice(offset, offset + PAGE_SIZE);
      isLastPage = offset + PAGE_SIZE >= filtered.length;
    } else {
      // Fetch PAGE_SIZE + 1 to detect if there's a next page
      const fetched = await getTopEvents(tag || undefined, PAGE_SIZE + 1, offset);
      isLastPage = fetched.length <= PAGE_SIZE;
      const trimmed = fetched.slice(0, PAGE_SIZE).filter((e) => e.markets.length > 0);
      events = applySortRows(trimmed, sort, dir);
    }
  } catch {
    return (
      <div className="border border-red/40 bg-panel p-5 panel-lit">
        <div className="flex items-center gap-2 border-b border-red/20 pb-2 text-[11px] tracking-widest text-red">
          <span className="glow-red">▲</span> DATA FEED ERROR · CODE 0x1A
        </div>
        <div className="mt-3 space-y-1 text-xs leading-5">
          <p className="text-muted">
            <span className="text-red">$</span> connect gamma-api.polymarket.com
          </p>
          <p className="text-red/80">→ CONNECTION REFUSED — upstream unreachable</p>
          <p className="text-muted/70">
            → retry: reload the page · the feed auto-recovers when the API responds
            <span className="cursor-blink ml-1 text-red">▊</span>
          </p>
        </div>
      </div>
    );
  }

  if (events.length === 0) {
    return (
      <div className="border border-edge bg-panel p-5 panel-lit">
        <div className="flex items-center gap-2 border-b border-edge pb-2 text-[11px] tracking-widest text-muted">
          <span className="text-amber">∅</span> NO RESULTS
        </div>
        <div className="mt-3 space-y-1 text-xs leading-5">
          <p className="text-muted">
            <span className="text-accent">$</span> query {query ? `"${query}"` : "market feed"}
          </p>
          <p className="text-muted/70">
            → 0 MARKETS MATCHED
            {query ? ` "${query.toUpperCase()}"` : ""} — try a different term or tag
            <span className="cursor-blink ml-1 text-accent">▊</span>
          </p>
        </div>
      </div>
    );
  }

  const totalVol24 = events.reduce((s, e) => s + (e.volume24hr ?? 0), 0);
  const totalLiq = events.reduce((s, e) => s + (e.liquidity ?? 0), 0);
  const totalOI = events.reduce((s, e) => s + (e.openInterest ?? 0), 0);
  const pageLabel = page > 1 ? ` · PAGE ${page}` : "";

  // Fetch a 7-day price trace for each row's leading outcome, in parallel.
  // Cached upstream (revalidate) so repeat renders are cheap; failures degrade
  // to an empty trace rather than breaking the board.
  const sparks: PricePoint[][] = await Promise.all(
    events.map(async (e) => {
      const token = leadingOutcome(e)?.tokenId;
      if (!token) return [];
      try {
        return await getPriceHistory(token, "1w");
      } catch {
        return [];
      }
    }),
  );

  return (
    <>
      <div className="grid grid-cols-2 gap-px border border-edge bg-edge sm:grid-cols-4">
        <Stat label={`MARKETS${pageLabel}`} value={String(events.length)} />
        <Stat label="24H VOLUME" value={fmtUsd(totalVol24)} accent />
        <Stat label="LIQUIDITY" value={fmtUsd(totalLiq)} />
        <Stat label="OPEN INTEREST" value={fmtUsd(totalOI)} />
      </div>

      {events.length >= 4 && <BreadthBar rows={events} />}

      {(() => {
        // Run the same edge kernel the terminal uses, but server-side over the
        // current board, and surface it as a visible panel.
        const signals = scanSignals(events, { limit: 6 });
        return signals.length > 0 ? <EdgeRadar signals={signals} /> : null;
      })()}

      {!query && page === 1 && events.length >= 6 && <MoversBar rows={events} />}

      <div className="overflow-x-auto border border-edge panel-lit">
        <table className="w-full min-w-full border-collapse text-left lg:min-w-[880px]">
          <thead>
            <tr className="border-b border-edge bg-panel-raised text-[11px] tracking-widest text-muted [&>th]:whitespace-nowrap">
              <th className="w-8 px-2 py-2 font-normal">#</th>
              <th className="px-2 py-2 font-normal">MARKET</th>
              <th className="hidden px-2 py-2 text-right font-normal md:table-cell">LEADER</th>
              <SortTh
                sortKey="odds"
                label="ODDS"
                currentSort={sort}
                currentDir={dir}
                tag={tag}
                query={query}
              />
              <SortTh
                sortKey="change"
                label="24H Δ"
                currentSort={sort}
                currentDir={dir}
                tag={tag}
                query={query}
              />
              <th className="hidden px-2 py-2 text-right font-normal lg:table-cell">7D TREND</th>
              <SortTh
                sortKey="vol"
                label="24H VOL"
                currentSort={sort}
                currentDir={dir}
                tag={tag}
                query={query}
                thClass="hidden sm:table-cell"
              />
              <SortTh
                sortKey="liq"
                label="LIQUIDITY"
                currentSort={sort}
                currentDir={dir}
                tag={tag}
                query={query}
                thClass="hidden lg:table-cell"
              />
              <SortTh
                sortKey="oi"
                label="OPEN INT"
                currentSort={sort}
                currentDir={dir}
                tag={tag}
                query={query}
                thClass="hidden lg:table-cell"
              />
              <SortTh
                sortKey="ends"
                label="ENDS"
                currentSort={sort}
                currentDir={dir}
                tag={tag}
                query={query}
                thClass="hidden md:table-cell"
              />
            </tr>
          </thead>
          <tbody>
            {(() => {
              const maxVol = events.reduce((m, e) => Math.max(m, e.volume24hr ?? 0), 1);
              return events.map((event, i) => (
                <Row
                  key={event.id}
                  event={event}
                  index={offset + i + 1}
                  maxVol={maxVol}
                  spark={sparks[i]}
                />
              ));
            })()}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between border border-edge bg-panel px-2 py-2 text-xs panel-lit">
        <span className="flex items-center gap-1.5 text-muted">
          <span className="text-accent/60">▪</span>
          ROWS {offset + 1}–{offset + events.length} · PAGE {page}
          {isLastPage ? <span className="text-accent/70">· END OF FEED</span> : ""}
        </span>
        <div className="flex items-center gap-px border border-edge bg-edge">
          {page > 1 ? (
            <Link
              href={pageHref(tag, query, sort, dir, page - 1)}
              className="bg-panel px-3 py-1.5 text-muted hover:bg-panel-raised hover:text-accent"
            >
              ← PREV
            </Link>
          ) : (
            <span className="cursor-not-allowed bg-panel px-3 py-1.5 text-muted/30">← PREV</span>
          )}
          <span className="bg-accent px-3 py-1.5 font-bold text-black">{page}</span>
          {!isLastPage ? (
            <Link
              href={pageHref(tag, query, sort, dir, page + 1)}
              className="bg-panel px-3 py-1.5 text-muted hover:bg-panel-raised hover:text-accent"
            >
              NEXT →
            </Link>
          ) : (
            <span className="cursor-not-allowed bg-panel px-3 py-1.5 text-muted/30">NEXT →</span>
          )}
        </div>
      </div>
    </>
  );
}

function SortTh({
  label,
  sortKey,
  currentSort,
  currentDir,
  tag,
  query,
  thClass = "",
}: {
  label: string;
  sortKey: SortKey;
  currentSort: string;
  currentDir: string;
  tag: string;
  query: string;
  thClass?: string;
}) {
  const active = currentSort === sortKey;
  const href = sortHref(tag, query, sortKey, currentSort, currentDir);
  return (
    <th className={`px-2 py-2 font-normal ${thClass}`}>
      <Link
        href={href}
        className={`flex items-center justify-end gap-1 hover:text-foreground ${active ? "text-foreground" : ""}`}
      >
        {label}
        <span className={`text-[9px] ${active ? "text-accent" : "opacity-25"}`}>
          {active ? (currentDir === "asc" ? "▲" : "▼") : "↕"}
        </span>
      </Link>
    </th>
  );
}

function Row({
  event,
  index,
  maxVol,
  spark,
}: {
  event: GammaEvent;
  index: number;
  maxVol: number;
  spark: PricePoint[];
}) {
  const lead = leadingOutcome(event);
  const change = lead?.change24h ?? 0;
  const changeColor = change > 0.001 ? "text-accent" : change < -0.001 ? "text-red" : "text-muted";
  const arrow = change > 0.001 ? "▲" : change < -0.001 ? "▼" : "·";

  const days = daysUntil(event.endDate);
  const endsColor = days < 1 ? "text-red" : days < 7 ? "text-amber" : "text-muted";
  const endsBadge = days < 1 ? "EXPIRING" : days < 7 ? `${Math.ceil(days)}D` : null;

  const volPct = maxVol > 0 ? ((event.volume24hr ?? 0) / maxVol) * 100 : 0;
  const odds = lead?.price ?? 0;

  const oddsColor = odds > 0.66 ? "text-accent" : odds < 0.34 ? "text-red" : "text-amber";
  const oddsBar = odds > 0.66 ? "bg-accent/40" : odds < 0.34 ? "bg-red/40" : "bg-amber/40";

  return (
    <tr
      data-term-index={index}
      data-term-row={event.slug}
      style={{ animationDelay: `${((index - 1) % PAGE_SIZE) * 14}ms` }}
      className={`group rise border-b border-edge transition-colors last:border-b-0 hover:bg-panel-raised ${
        index % 2 === 0 ? "bg-panel-raised/25" : "bg-panel"
      }`}
    >
      <td className="border-l-2 border-l-transparent px-2 py-2.5 text-[11px] text-muted transition-colors group-hover:border-l-accent group-hover:text-accent/70">
        {String(index).padStart(2, "0")}
      </td>
      <td className="max-w-[46vw] px-2 py-2.5 sm:max-w-sm">
        <Link href={`/event/${event.slug}`} className="flex items-center gap-2.5">
          {event.icon ? (
            // Icon hosts come from the Gamma feed at runtime. next/image
            // hard-errors on any host not in `images.remotePatterns`, so one
            // new CDN upstream would blank the whole board; a plain lazy <img>
            // degrades to a missing icon instead.
            // oxlint-disable-next-line no-img-element
            <img
              src={event.icon}
              alt=""
              width={20}
              height={20}
              loading="lazy"
              decoding="async"
              className="h-5 w-5 shrink-0 rounded-sm object-cover opacity-90"
            />
          ) : (
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-sm bg-panel-raised text-[11px] text-muted">
              ·
            </span>
          )}
          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-2">
              <span className="truncate text-foreground group-hover:text-accent">
                {event.title}
              </span>
              {event.negRisk && (
                <span className="shrink-0 rounded-sm bg-amber/10 px-1 text-[9px] tracking-wider text-amber/70">
                  NR
                </span>
              )}
            </span>
            {lead && (
              <span className="mt-0.5 block truncate text-[10px] text-cyan/80 md:hidden">
                ▸ {lead.label}
              </span>
            )}
          </span>
        </Link>
      </td>
      <td className="hidden max-w-36 truncate px-2 py-2.5 text-right text-[11px] text-cyan md:table-cell">
        {lead?.label ?? "—"}
      </td>
      <td
        className={`relative px-2 py-2.5 text-right font-bold tabular-nums ${oddsColor} group-hover:glow-soft`}
      >
        {lead ? fmtPct(lead.price) : "—"}
        {lead && (
          <div
            className={`absolute bottom-0 left-0 h-[2px] ${oddsBar} transition-all`}
            style={{ width: `${Math.min(odds * 100, 100)}%` }}
          />
        )}
      </td>
      <td className={`whitespace-nowrap px-2 py-2.5 text-right tabular-nums ${changeColor}`}>
        {arrow} {fmtChange(change)}
      </td>
      <td className="hidden px-2 py-2 text-right lg:table-cell">
        <Sparkline points={spark} />
      </td>
      <td className="relative hidden px-2 py-2.5 text-right tabular-nums text-foreground sm:table-cell">
        {fmtUsd(event.volume24hr)}
        <div
          className="absolute bottom-0 left-0 h-[2px] bg-accent/20"
          style={{ width: `${volPct}%` }}
        />
      </td>
      <td className="hidden px-2 py-2.5 text-right tabular-nums text-muted lg:table-cell">
        {fmtUsd(event.liquidity)}
      </td>
      <td className="hidden px-2 py-2.5 text-right tabular-nums text-muted lg:table-cell">
        {fmtUsd(event.openInterest)}
      </td>
      <td
        className={`hidden whitespace-nowrap px-2 py-2.5 text-right text-[11px] md:table-cell ${endsColor}`}
      >
        {endsBadge && (
          <span className="mr-1.5 rounded-sm bg-current/10 px-1 py-0.5 text-[9px] tracking-wider">
            {endsBadge}
          </span>
        )}
        {fmtDate(event.endDate)}
      </td>
    </tr>
  );
}

function fmtEdge(bps: number): string {
  const pct = bps / 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

function EdgeRadar({ signals }: { signals: Signal[] }) {
  const arbs = signals.filter((s) => s.kind === "ARB").length;
  const mom = signals.length - arbs;
  return (
    <div className="border border-edge bg-panel panel-lit">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge px-3 py-2 text-[10px] tracking-widest">
        <span className="flex items-center gap-1.5 text-muted">
          <span className="glow-soft text-accent">◈</span>
          EDGE RADAR
          <span className="text-muted/50">
            · {signals.length} SIGNAL{signals.length === 1 ? "" : "S"}
          </span>
        </span>
        <span className="flex items-center gap-3 text-[10px] text-muted/60">
          {arbs > 0 && <span className="text-cyan">{arbs} ARB</span>}
          {mom > 0 && <span className="text-amber">{mom} MOMENTUM</span>}
          <span className="hidden sm:inline">SCORE = EDGE × BOOK QUALITY</span>
        </span>
      </div>
      <div className="grid gap-px bg-edge sm:grid-cols-2 lg:grid-cols-3">
        {signals.map((s) => (
          <SignalCard key={`${s.kind}-${s.slug}`} signal={s} />
        ))}
      </div>
    </div>
  );
}

function SignalCard({ signal: s }: { signal: Signal }) {
  const isArb = s.kind === "ARB";
  const buyable = s.edgeBps > 0;
  // ARB: green when there's a buyable underround edge, amber when it's just vig.
  // MOMENTUM: green up, red down.
  const kindColor = isArb ? "text-cyan" : buyable ? "text-accent" : "text-red";
  const edgeColor = isArb
    ? buyable
      ? "text-accent"
      : "text-amber"
    : buyable
      ? "text-accent"
      : "text-red";
  const barColor = isArb ? "bg-cyan/60" : buyable ? "bg-accent/60" : "bg-red/60";

  return (
    <Link
      href={`/event/${s.slug}`}
      className="group flex flex-col gap-1.5 bg-panel px-3 py-2.5 transition-colors hover:bg-panel-raised"
    >
      <div className="flex items-center justify-between gap-2 text-[10px] tracking-widest">
        <span className={`rounded-sm border border-current/30 px-1 py-0.5 ${kindColor}`}>
          {s.kind}
        </span>
        <span className="flex items-center gap-1.5 text-muted/60">
          SCORE
          <span className="font-bold tabular-nums text-foreground">{Math.round(s.score)}</span>
        </span>
      </div>
      <p className="truncate text-xs text-foreground group-hover:text-accent">{s.title}</p>
      {/* Score bar */}
      <div className="h-1 w-full overflow-hidden rounded-sm bg-panel-raised ring-1 ring-inset ring-edge">
        <div className={`h-full ${barColor}`} style={{ width: `${Math.min(s.score, 100)}%` }} />
      </div>
      <div className="flex items-center justify-between gap-2 text-[10px] tabular-nums">
        <span className={`font-bold ${edgeColor}`}>EDGE {fmtEdge(s.edgeBps)}</span>
        <span className="text-muted/60">
          {s.spreadBps != null ? `${(s.spreadBps / 100).toFixed(1)}% SPR` : "NO BOOK"}
        </span>
      </div>
      <p className="truncate text-[10px] text-muted/70">{s.detail}</p>
    </Link>
  );
}

function BreadthBar({ rows }: { rows: GammaEvent[] }) {
  let up = 0;
  let down = 0;
  let flat = 0;
  let sumChange = 0;
  for (const e of rows) {
    const c = leadingOutcome(e)?.change24h ?? 0;
    sumChange += c;
    if (c > 0.001) up++;
    else if (c < -0.001) down++;
    else flat++;
  }
  const total = rows.length || 1;
  const upPct = (up / total) * 100;
  const flatPct = (flat / total) * 100;
  const downPct = (down / total) * 100;
  const net = (sumChange / total) * 100; // avg 24h change in points
  const bullish = net >= 0;
  const netColor = net > 0.05 ? "text-accent" : net < -0.05 ? "text-red" : "text-muted";
  const label =
    upPct >= 60
      ? "RISK-ON"
      : downPct >= 60
        ? "RISK-OFF"
        : up > down
          ? "LEANING BULLISH"
          : down > up
            ? "LEANING BEARISH"
            : "MIXED";

  return (
    <div className="border border-edge bg-panel px-2 py-2.5 panel-lit">
      <div className="mb-2 flex items-center justify-between text-[10px] tracking-widest">
        <span className="flex items-center gap-1.5 text-muted">
          <span className={bullish ? "text-accent/60" : "text-red/60"}>{bullish ? "▲" : "▼"}</span>
          MARKET BREADTH
          <span className={`glow-soft ${netColor}`}>· {label}</span>
        </span>
        <span className={`tabular-nums ${netColor}`}>
          AVG {net >= 0 ? "+" : ""}
          {net.toFixed(1)}pp
        </span>
      </div>
      {/* Segmented breadth bar */}
      <div className="flex h-2 w-full overflow-hidden rounded-sm bg-panel-raised ring-1 ring-inset ring-edge">
        <div
          className="h-full bg-accent/70 shadow-[0_0_8px_-1px_var(--accent)] transition-all"
          style={{ width: `${upPct}%` }}
        />
        <div className="h-full bg-muted/25 transition-all" style={{ width: `${flatPct}%` }} />
        <div className="h-full bg-red/70 transition-all" style={{ width: `${downPct}%` }} />
      </div>
      <div className="mt-1.5 flex items-center gap-4 text-[10px] tabular-nums text-muted">
        <span className="flex items-center gap-1 text-accent">
          <span className="inline-block h-1.5 w-1.5 rounded-sm bg-accent/70" />
          {up} UP
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-1.5 w-1.5 rounded-sm bg-muted/40" />
          {flat} FLAT
        </span>
        <span className="flex items-center gap-1 text-red">
          <span className="inline-block h-1.5 w-1.5 rounded-sm bg-red/70" />
          {down} DOWN
        </span>
        <span className="ml-auto text-muted/50">{upPct.toFixed(0)}% ADVANCING</span>
      </div>
    </div>
  );
}

function MoversBar({ rows }: { rows: GammaEvent[] }) {
  const byChange = rows.toSorted(
    (a, b) => (leadingOutcome(b)?.change24h ?? 0) - (leadingOutcome(a)?.change24h ?? 0),
  );
  const gainers = byChange.slice(0, 5);
  const losers = byChange.toReversed().slice(0, 5);

  return (
    <div className="grid grid-cols-1 gap-px border border-edge bg-edge text-[11px] sm:grid-cols-2">
      <MoversColumn title="TOP GAINERS" rows={gainers} dir="up" />
      <MoversColumn title="TOP LOSERS" rows={losers} dir="down" />
    </div>
  );
}

function MoversColumn({
  title,
  rows,
  dir,
}: {
  title: string;
  rows: GammaEvent[];
  dir: "up" | "down";
}) {
  const up = dir === "up";
  const tone = up ? "text-accent" : "text-red";
  const bar = up ? "bg-accent/50" : "bg-red/50";
  const arrow = up ? "▲" : "▼";
  const maxAbs = rows.reduce(
    (m, e) => Math.max(m, Math.abs(leadingOutcome(e)?.change24h ?? 0)),
    0.0001,
  );
  return (
    <div className="bg-panel px-2 py-2.5 panel-lit">
      <div className={`mb-2 flex items-center gap-1.5 text-[10px] tracking-widest ${tone}`}>
        <span className="glow-soft">{arrow}</span>
        {title}
      </div>
      <div className="flex flex-col gap-0.5">
        {rows.map((e, i) => {
          const lead = leadingOutcome(e);
          const change = lead?.change24h ?? 0;
          const mag = (Math.abs(change) / maxAbs) * 100;
          return (
            <Link
              key={e.id}
              href={`/event/${e.slug}`}
              className="group relative -mx-1 flex items-center justify-between gap-2 overflow-hidden rounded-sm px-1 py-0.5 hover:bg-panel-raised"
            >
              <span
                className={`absolute inset-y-0 left-0 ${bar} opacity-15 transition-all group-hover:opacity-25`}
                style={{ width: `${mag}%` }}
              />
              <div className="relative flex min-w-0 items-center gap-2">
                <span className="w-3 shrink-0 text-right text-[9px] tabular-nums text-muted/40">
                  {i + 1}
                </span>
                {e.icon && (
                  // Feed-supplied host — see Row.
                  // oxlint-disable-next-line no-img-element
                  <img
                    src={e.icon}
                    alt=""
                    width={14}
                    height={14}
                    loading="lazy"
                    decoding="async"
                    className="h-3.5 w-3.5 shrink-0 rounded-sm object-cover opacity-75 group-hover:opacity-100"
                  />
                )}
                <span className="truncate text-muted group-hover:text-foreground">{e.title}</span>
              </div>
              <div className="relative flex shrink-0 items-center gap-2">
                <span className="text-[10px] text-muted/60">{fmtPct(lead?.price ?? 0)}</span>
                <span className={`tabular-nums ${tone}`}>
                  {arrow} {fmtChange(change)}
                </span>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="group relative bg-panel px-2 py-2.5 panel-lit">
      {accent && (
        <span className="absolute inset-x-0 top-0 h-px bg-accent/50 shadow-[0_0_8px_var(--accent-dim)]" />
      )}
      <div className="flex items-center gap-1.5 text-[10px] tracking-widest text-muted">
        <span className={accent ? "text-accent/60" : "text-muted/50"}>{accent ? "◆" : "◇"}</span>
        {label}
      </div>
      <div
        className={`text-lg font-bold tabular-nums ${accent ? "text-accent glow-soft" : "text-foreground"}`}
      >
        {value}
      </div>
    </div>
  );
}

/**
 * Skeleton geometry, declared once. The ragged widths are what make a loading
 * state read as "text arriving" rather than "grey boxes", and hoisting them out
 * gives every placeholder a stable identity instead of a list index.
 */
const SKELETON_MOVER_WIDTHS = [72, 65, 58, 51, 44];
const SKELETON_FEED_ROWS = Array.from({ length: 8 }, (_, i) => ({
  id: `feed-${i}`,
  width: 40 - (i % 3) * 6,
}));

function BoardSkeleton() {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-px border border-edge bg-edge sm:grid-cols-4">
        {["MARKETS", "24H VOLUME", "LIQUIDITY", "OPEN INTEREST"].map((l, i) => (
          <div key={l} className="bg-panel px-2 py-2.5 panel-lit">
            <div className="flex items-center gap-1.5 text-[10px] tracking-widest text-muted/50">
              <span className={i === 1 ? "text-accent/60" : "text-muted/40"}>
                {i === 1 ? "◆" : "◇"}
              </span>
              {l}
            </div>
            <div className="shimmer mt-1 h-5 w-20 rounded-sm bg-panel-raised" />
          </div>
        ))}
      </div>
      <div className="border border-edge bg-panel px-2 py-2.5 panel-lit">
        <div className="mb-2 text-[10px] tracking-widest text-muted/50">
          MARKET BREADTH · SCANNING
        </div>
        <div className="shimmer h-2 w-full rounded-sm bg-panel-raised" />
      </div>
      <div className="grid grid-cols-2 gap-px border border-edge bg-edge">
        {["▲ TOP GAINERS", "▼ TOP LOSERS"].map((l) => (
          <div key={l} className="bg-panel px-2 py-2.5 panel-lit">
            <div className="mb-2 text-[10px] tracking-widest text-muted/40">{l}</div>
            {SKELETON_MOVER_WIDTHS.map((w) => (
              <div
                key={w}
                className="shimmer my-1 h-3 rounded-sm bg-panel-raised"
                style={{ width: `${w}%` }}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="scan-sweep overflow-hidden border border-edge bg-panel panel-lit">
        <div className="border-b border-edge bg-panel-raised px-2 py-2 text-[11px] tracking-widest text-muted/50">
          <span className="text-accent">&gt;</span> STREAMING MARKET FEED
          <span className="cursor-blink text-accent">▊</span>
        </div>
        {SKELETON_FEED_ROWS.map((row) => (
          <div
            key={row.id}
            className="flex items-center gap-3 border-b border-edge/60 px-2 py-2.5 last:border-b-0"
          >
            <div className="h-5 w-5 shrink-0 rounded-sm bg-panel-raised" />
            <div
              className="shimmer h-3 rounded-sm bg-panel-raised"
              style={{ width: `${row.width}%` }}
            />
            <div className="shimmer ml-auto h-3 w-16 rounded-sm bg-panel-raised" />
            <div className="shimmer h-3 w-12 rounded-sm bg-panel-raised" />
          </div>
        ))}
      </div>
    </div>
  );
}
