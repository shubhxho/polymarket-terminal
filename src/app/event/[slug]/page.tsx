import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Suspense } from "react";
import { type ChartSeries, PriceChart } from "@/components/price-chart";
import {
  daysUntil,
  eventOutcomes,
  fmtChange,
  fmtDate,
  fmtPct,
  fmtUsd,
  type GammaEvent,
  getEventBySlug,
  getPriceHistory,
  getRelatedEvents,
  leadingOutcome,
  type Outcome,
} from "@/lib/polymarket";

const RANGES = [
  { key: "1d", label: "1D" },
  { key: "1w", label: "1W" },
  { key: "1m", label: "1M" },
  { key: "3m", label: "3M" },
  { key: "max", label: "MAX" },
];

interface Params {
  slug: string;
}

export async function generateMetadata({
  params,
}: {
  params: Promise<Params>;
}): Promise<Metadata> {
  const { slug } = await params;
  const event = await getEventBySlug(slug);
  if (!event) return { title: "MARKET NOT FOUND" };
  const lead = leadingOutcome(event);
  return {
    title: event.title,
    description: lead
      ? `${lead.label} at ${fmtPct(lead.price)} — ${fmtUsd(event.volume24hr)} 24h volume`
      : `${fmtUsd(event.volume24hr)} 24h volume on Polymarket`,
  };
}

export default async function EventPage({
  params,
  searchParams,
}: {
  params: Promise<Params>;
  searchParams: Promise<{ r?: string }>;
}) {
  const [{ slug }, { r }] = await Promise.all([params, searchParams]);
  const range = RANGES.some((x) => x.key === r) ? (r as string) : "1w";

  const event = await getEventBySlug(slug);
  if (!event) notFound();

  const outcomes = eventOutcomes(event);
  const multiMarket =
    event.markets.filter((m) => m.active && !m.closed).length > 1;

  const days = daysUntil(event.endDate);
  const endsColor =
    days < 1 ? "text-red" : days < 7 ? "text-amber" : "text-muted";
  const urgencyLabel =
    days < 1 ? "EXPIRING TODAY" : days < 7 ? `${Math.ceil(days)}D LEFT` : null;

  return (
    <main className="flex flex-1 flex-col gap-3">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-1.5 text-xs text-muted">
        <Link href="/" className="hover:text-accent">
          ~/MARKETS
        </Link>
        <span>/</span>
        <span className="text-foreground/60">{event.slug.toUpperCase()}</span>
      </nav>

      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4 border border-edge bg-panel p-4 panel-lit">
        <div className="flex items-start gap-3">
          {event.icon && (
            <img
              src={event.icon}
              alt=""
              width={44}
              height={44}
              className="h-11 w-11 shrink-0 rounded-sm object-cover"
            />
          )}
          <div className="min-w-0">
            <h1 className="text-base font-bold leading-snug text-foreground">
              {event.title}
            </h1>
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px]">
              <span className={endsColor}>
                ENDS {fmtDate(event.endDate)}
                {urgencyLabel && (
                  <span className="ml-1.5 rounded-sm bg-current/15 px-1 py-0.5 text-[9px] tracking-wider">
                    {urgencyLabel}
                  </span>
                )}
              </span>
              {event.negRisk && (
                <span className="rounded-sm bg-amber/10 px-1.5 text-[10px] tracking-wider text-amber">
                  NEG-RISK
                </span>
              )}
              {(event.tags ?? []).slice(0, 4).map((t) => (
                <Link
                  key={t.id}
                  href={`/?tag=${t.slug}`}
                  className="text-muted/70 hover:text-accent"
                >
                  #{t.slug.toUpperCase()}
                </Link>
              ))}
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-px bg-edge sm:grid-cols-4">
          <HeaderStat label="24H VOL" value={fmtUsd(event.volume24hr)} accent />
          <HeaderStat label="TOTAL VOL" value={fmtUsd(event.volume)} />
          <HeaderStat label="LIQUIDITY" value={fmtUsd(event.liquidity)} />
          <HeaderStat label="OPEN INT" value={fmtUsd(event.openInterest)} />
        </div>
      </div>

      {/* Chart section */}
      <div className="flex items-center justify-between">
        <h2 className="text-xs tracking-widest text-muted">
          PRICE HISTORY{multiMarket ? " · TOP 5 OUTCOMES" : ""}
        </h2>
        <div className="flex gap-px border border-edge bg-edge">
          {RANGES.map((x) => (
            <Link
              key={x.key}
              href={`/event/${event.slug}?r=${x.key}`}
              className={`px-3 py-1 text-[11px] tracking-wider transition-colors ${
                x.key === range
                  ? "bg-accent font-bold text-black shadow-[0_0_14px_var(--accent-dim)]"
                  : "bg-panel text-muted hover:bg-panel-raised hover:text-accent"
              }`}
            >
              {x.label}
            </Link>
          ))}
        </div>
      </div>

      <Suspense key={range} fallback={<ChartSkeleton />}>
        <Chart outcomes={outcomes} range={range} multiMarket={multiMarket} />
      </Suspense>

      {/* Order book */}
      <h2 className="mt-1 text-xs tracking-widest text-muted">ORDER BOOK</h2>
      <div className="overflow-x-auto border border-edge">
        <table className="w-full min-w-[640px] border-collapse text-left">
          <thead>
            <tr className="border-b border-edge bg-panel-raised text-[11px] tracking-widest text-muted">
              <th className="px-3 py-2 font-normal">OUTCOME</th>
              <th className="px-3 py-2 text-right font-normal">PRICE</th>
              <th className="px-3 py-2 text-right font-normal">24H Δ</th>
              <th className="px-3 py-2 text-right font-normal">BID</th>
              <th className="px-3 py-2 text-right font-normal">ASK</th>
              <th className="px-3 py-2 text-right font-normal">SPREAD</th>
              <th className="px-3 py-2 text-right font-normal">24H VOL</th>
              <th className="w-44 px-3 py-2 font-normal">IMPLIED</th>
            </tr>
          </thead>
          <tbody>
            {outcomes.map((o) => (
              <OutcomeRow key={`${o.marketId}-${o.label}`} outcome={o} />
            ))}
          </tbody>
          {outcomes.length > 1 &&
            (() => {
              const total = outcomes.reduce((s, o) => s + o.price, 0);
              const overround = total - 1;
              return (
                <tfoot>
                  <tr className="border-t border-edge/60 bg-panel-raised">
                    <td
                      className="px-3 py-1.5 text-[10px] tracking-wider text-muted"
                      colSpan={4}
                    >
                      TOTAL IMPLIED PROBABILITY
                    </td>
                    <td
                      className={`px-3 py-1.5 text-right text-[11px] tabular-nums ${overround > 0.02 ? "text-amber" : "text-muted"}`}
                      colSpan={5}
                    >
                      {(total * 100).toFixed(1)}%
                      {overround > 0.02 &&
                        ` · +${(overround * 100).toFixed(1)}% HOUSE EDGE`}
                    </td>
                  </tr>
                </tfoot>
              );
            })()}
        </table>
      </div>

      {/* Resolution rules */}
      {event.description && (
        <details className="border border-edge bg-panel">
          <summary className="cursor-pointer px-3 py-2 text-xs tracking-widest text-muted hover:text-accent">
            RESOLUTION RULES
          </summary>
          <p className="whitespace-pre-wrap border-t border-edge px-3 py-3 text-xs leading-relaxed text-foreground/80">
            {event.description}
          </p>
        </details>
      )}

      {/* Actions row */}
      <div className="flex items-center gap-3">
        <a
          href={`https://polymarket.com/event/${event.slug}`}
          target="_blank"
          rel="noopener noreferrer"
          className="border border-edge bg-panel px-3 py-1.5 text-xs text-muted hover:border-edge-bright hover:text-accent"
        >
          TRADE ON POLYMARKET ↗
        </a>
        <Link href="/" className="text-xs text-muted hover:text-accent">
          ← BACK TO MARKETS
        </Link>
      </div>

      {/* Related events */}
      {(event.tags ?? []).length > 0 && (
        <Suspense fallback={null}>
          <RelatedSection slug={event.slug} tags={event.tags ?? []} />
        </Suspense>
      )}
    </main>
  );
}

async function RelatedSection({
  slug,
  tags,
}: {
  slug: string;
  tags: { id: string; label: string; slug: string }[];
}) {
  const related = await getRelatedEvents(
    slug,
    tags.map((t) => t.slug),
  );
  if (related.length === 0) return null;

  return (
    <>
      <h2 className="mt-1 text-xs tracking-widest text-muted">
        RELATED MARKETS · #{tags[0]?.slug.toUpperCase()}
      </h2>
      <div className="grid gap-px border border-edge bg-edge sm:grid-cols-2 lg:grid-cols-3">
        {related.map((e) => {
          const lead = leadingOutcome(e);
          const change = lead?.change24h ?? 0;
          const changeColor =
            change > 0.001
              ? "text-accent"
              : change < -0.001
                ? "text-red"
                : "text-muted";
          return (
            <Link
              key={e.id}
              href={`/event/${e.slug}`}
              className="group flex items-start gap-3 bg-panel p-3 hover:bg-panel-raised"
            >
              {e.icon ? (
                <img
                  src={e.icon}
                  alt=""
                  width={32}
                  height={32}
                  className="h-8 w-8 shrink-0 rounded-sm object-cover opacity-80"
                />
              ) : (
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-sm bg-panel-raised text-muted">
                  ·
                </span>
              )}
              <div className="min-w-0 flex-1">
                <p className="truncate text-[12px] font-medium text-foreground group-hover:text-accent">
                  {e.title}
                </p>
                <div className="mt-0.5 flex items-center gap-3 text-[11px]">
                  {lead && (
                    <>
                      <span className="font-bold tabular-nums text-amber">
                        {fmtPct(lead.price)}
                      </span>
                      <span className={`tabular-nums ${changeColor}`}>
                        {change > 0.001 ? "▲" : change < -0.001 ? "▼" : "·"}{" "}
                        {fmtChange(change)}
                      </span>
                    </>
                  )}
                  <span className="ml-auto text-muted">
                    {fmtUsd(e.volume24hr)}
                  </span>
                </div>
              </div>
            </Link>
          );
        })}
      </div>
    </>
  );
}

async function Chart({
  outcomes,
  range,
  multiMarket,
}: {
  outcomes: Outcome[];
  range: string;
  multiMarket: boolean;
}) {
  const charted = (multiMarket ? outcomes : outcomes.slice(0, 1))
    .filter((o) => o.tokenId)
    .slice(0, 5);

  const series: ChartSeries[] = await Promise.all(
    charted.map(async (o) => ({
      label: o.label,
      points: await getPriceHistory(o.tokenId as string, range),
    })),
  );

  return <PriceChart series={series} />;
}

function OutcomeRow({ outcome: o }: { outcome: Outcome }) {
  const changeColor =
    o.change24h > 0.001
      ? "text-accent"
      : o.change24h < -0.001
        ? "text-red"
        : "text-muted";
  const changeArrow =
    o.change24h > 0.001 ? "▲" : o.change24h < -0.001 ? "▼" : "·";
  const barColor =
    o.price > 0.7
      ? "bg-accent shadow-[0_0_8px_-1px_var(--accent)]"
      : o.price > 0.3
        ? "bg-accent-dim"
        : "bg-muted/40";

  return (
    <tr className="group border-b border-edge bg-panel last:border-b-0 hover:bg-panel-raised">
      <td className="max-w-xs truncate px-3 py-2.5 font-bold text-foreground group-hover:text-accent">
        {o.label}
      </td>
      <td className="px-3 py-2.5 text-right font-bold tabular-nums text-amber group-hover:glow-soft">
        {fmtPct(o.price)}
      </td>
      <td className={`px-3 py-2.5 text-right tabular-nums ${changeColor}`}>
        {changeArrow} {fmtChange(o.change24h)}
      </td>
      <td className="px-3 py-2.5 text-right tabular-nums text-accent">
        {o.bestBid != null ? fmtPct(o.bestBid) : "—"}
      </td>
      <td className="px-3 py-2.5 text-right tabular-nums text-red">
        {o.bestAsk != null ? fmtPct(o.bestAsk) : "—"}
      </td>
      <td className="px-3 py-2.5 text-right tabular-nums text-muted">
        {o.spread != null ? (o.spread * 100).toFixed(1) : "—"}
      </td>
      <td className="px-3 py-2.5 text-right tabular-nums text-foreground">
        {fmtUsd(o.volume24h)}
      </td>
      <td className="px-3 py-2.5">
        <div className="flex items-center gap-2">
          <div className="h-2.5 min-w-[60px] flex-1 overflow-hidden bg-panel-raised ring-1 ring-inset ring-edge">
            <div
              className={`h-full transition-all ${barColor}`}
              style={{ width: `${Math.min(o.price * 100, 100)}%` }}
            />
          </div>
          <span className="w-10 shrink-0 text-right text-[10px] tabular-nums text-muted">
            {(o.price * 100).toFixed(1)}%
          </span>
        </div>
      </td>
    </tr>
  );
}

function HeaderStat({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="relative bg-panel px-3 py-2 panel-lit">
      {accent && (
        <span className="absolute inset-x-0 top-0 h-px bg-accent/50 shadow-[0_0_8px_var(--accent-dim)]" />
      )}
      <div className="text-[10px] tracking-widest text-muted">{label}</div>
      <div
        className={`text-sm font-bold tabular-nums ${accent ? "text-accent glow-soft" : "text-foreground"}`}
      >
        {value}
      </div>
    </div>
  );
}

function ChartSkeleton() {
  return (
    <div className="flex h-56 items-center justify-center border border-edge bg-panel text-muted">
      <span className="text-accent">&gt;</span>&nbsp;LOADING CHART
      <span className="cursor-blink">▊</span>
    </div>
  );
}
