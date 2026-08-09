"use client";

import { motion } from "motion/react";
import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { useTerminal } from "@/components/TerminalProvider";
import { PriceChart, SERIES_COLORS, type Series } from "@/components/ui/PriceChart";
import { Empty, ErrorBox, Field, Loading, Panel } from "@/components/ui/Panel";
import { useMarketSocket } from "@/hooks/useMarketSocket";
import { usePoll } from "@/hooks/usePoll";
import {
  aggregateFlow,
  modelAgreement,
  scoreMarket,
  SIGNAL_META,
  type MarketSignals,
} from "@/lib/signals";
import {
  cents,
  compact,
  dateShort,
  dirClass,
  shortAddr,
  signed,
  timeToExpiry,
  truncate,
  usd,
} from "@/lib/format";
import { panelVariants, staggerContainer, tapScale } from "@/lib/motion";
import type {
  EventSummary,
  Holder,
  HistoryInterval,
  Market,
  OrderBook,
  PricePoint,
  Trade,
} from "@/lib/types";

/**
 * Heavy, below-the-fold children are split out of the initial chunk and pulled
 * in only when this screen actually mounts. The order-book ladder and the
 * time-&-sales tape are both client-only and each its own module, so deferring
 * them trims the entry bundle without changing what renders — a brief `Loading`
 * placeholder stands in until the chunk arrives, matching the states these
 * panels already show while their feeds resolve.
 *
 * `PriceChart` is deliberately left statically imported: `SERIES_COLORS` (a
 * runtime value from the same module) is used in this file's `series` memo, so
 * the module is in the bundle regardless and code-splitting the component would
 * add a loading flicker while removing nothing.
 */
const OrderBookLadder = dynamic(
  () => import("@/components/OrderBook").then((m) => m.OrderBookLadder),
  { ssr: false, loading: () => <Loading /> }
);
const TradeTape = dynamic(() => import("@/components/TradeTape").then((m) => m.TradeTape), {
  ssr: false,
  loading: () => <Loading />,
});
// The ticket pulls in the wallet + signing path, none of which the read-only
// screens need — keep it out of the entry bundle and load it with the rest of
// the below-the-fold right rail.
const OrderTicket = dynamic(() => import("@/components/OrderTicket").then((m) => m.OrderTicket), {
  ssr: false,
  loading: () => <Loading />,
});

const INTERVALS: { key: HistoryInterval; label: string }[] = [
  { key: "1h", label: "1H" },
  { key: "6h", label: "6H" },
  { key: "1d", label: "1D" },
  { key: "1w", label: "1W" },
  { key: "1m", label: "1M" },
  { key: "max", label: "MAX" },
];

/** The history endpoint accepts at most 8 tokens, and more than 8 lines on one
 *  chart is unreadable anyway — deeper legs stay in the table below. */
const MAX_SERIES = 8;

/**
 * The analytics launchpad for a single event or market.
 *
 * Resolves an event slug first (the common case — most Polymarket URLs are
 * events, which carry every leg), falling back to a standalone market slug.
 * Chart, book, tape and holders all key off one selected outcome token so the
 * whole screen moves together.
 */
export default function DetailScreen({ slug, kind }: { slug: string; kind: "event" | "market" }) {
  const { toggleWatch, isWatched, addAlert, toast } = useTerminal();
  const [interval, setInterval] = useState<HistoryInterval>("1d");
  const [selectedToken, setSelectedToken] = useState<string | null>(null);

  const eventQ = usePoll<EventSummary[]>(
    kind === "event" ? `/api/events?slug=${encodeURIComponent(slug)}&limit=1` : null,
    20000
  );
  // A market slug (or an event slug that resolved to nothing) falls through to
  // the markets endpoint so a direct market link still opens.
  const marketFallback = kind === "market" || (!eventQ.loading && (eventQ.data?.length ?? 0) === 0);
  const marketQ = usePoll<Market[]>(
    marketFallback ? `/api/markets?slug=${encodeURIComponent(slug)}&limit=1` : null,
    20000
  );

  // Last resort: if the exact slug resolves to nothing on both endpoints (a
  // renamed or truly-gone market), search by the de-slugified name and open the
  // closest hit — so the feed never dead-ends on "no market found". This is the
  // "try SRCH" hint, done automatically. The trailing `-<n>` disambiguator that
  // Polymarket appends to slugs is dropped from the query.
  const directEmpty =
    !eventQ.loading &&
    !marketQ.loading &&
    (eventQ.data?.length ?? 0) === 0 &&
    (marketQ.data?.length ?? 0) === 0;
  const searchTerm = slug.replace(/-\d+$/, "").replace(/-/g, " ").trim();
  const searchQ = usePoll<{ events: EventSummary[]; markets: Market[] }>(
    directEmpty && searchTerm ? `/api/search?q=${encodeURIComponent(searchTerm)}&limit=6` : null,
    30000
  );
  const searchEvent =
    searchQ.data?.events?.find((e) => e.slug === slug) ?? searchQ.data?.events?.[0] ?? null;
  const searchMarket =
    searchQ.data?.markets?.find((m) => m.slug === slug) ?? searchQ.data?.markets?.[0] ?? null;

  const event = eventQ.data?.[0] ?? searchEvent ?? null;
  const markets = useMemo<Market[]>(() => {
    if (event) {
      // Rank legs by turnover so the chart shows the contested ones.
      return [...event.markets].sort((a, b) => b.volume24h - a.volume24h);
    }
    if (marketQ.data?.length) return marketQ.data;
    return searchMarket ? [searchMarket] : [];
  }, [event, marketQ.data, searchMarket]);

  const title = event?.title ?? markets[0]?.question ?? slug;
  const primary = markets[0];

  // Every outcome across every leg, flattened. Binary markets contribute both
  // Yes and No; multi-outcome events contribute one "Yes" per leg.
  const outcomes = useMemo(() => {
    if (markets.length === 1) {
      return markets[0].outcomes.map((o) => ({
        tokenId: o.tokenId,
        label: o.label,
        price: o.price,
        market: markets[0],
      }));
    }
    return markets
      .filter((m) => m.outcomes[0]?.tokenId)
      .map((m) => ({
        tokenId: m.outcomes[0].tokenId,
        label: m.groupItemTitle || m.question,
        price: m.outcomes[0].price,
        market: m,
      }));
  }, [markets]);

  // Selection defaults to the leading outcome by falling back here rather than
  // by seeding state in an effect — which also means a slug whose outcomes
  // arrive late never renders an empty book panel first. The screen is
  // remounted per slug by the shell, so no cross-slug reset is needed either.
  const selected = outcomes.find((o) => o.tokenId === selectedToken) ?? outcomes[0];
  const selectedId = selected?.tokenId ?? null;
  const selectedMarket = selected?.market ?? primary;

  const chartTokens = useMemo(
    () => outcomes.slice(0, MAX_SERIES).map((o) => o.tokenId),
    [outcomes]
  );

  const historyQ = usePoll<{ tokenId: string; points: PricePoint[] }[]>(
    chartTokens.length > 0
      ? `/api/history?tokens=${chartTokens.join(",")}&interval=${interval}`
      : null,
    30000
  );
  const tradesQ = usePoll<Trade[]>(
    selectedMarket ? `/api/trades?condition=${selectedMarket.conditionId}&limit=60` : null,
    5000
  );
  const holdersQ = usePoll<Holder[][]>(
    selectedMarket ? `/api/holders?condition=${selectedMarket.conditionId}&limit=10` : null,
    60000
  );
  const bookSnapshot = usePoll<OrderBook[]>(
    selectedId ? `/api/books?tokens=${selectedId}` : null,
    15000
  );

  // Live feed for every charted outcome, so the ladder and the outcome list
  // tick together without a second subscription.
  const feed = useMarketSocket(chartTokens, chartTokens.length > 0);
  const liveBook = selectedId ? feed.books.get(selectedId) : undefined;
  const book = liveBook ?? bookSnapshot.data?.[0];

  // The signal engine is pure, so the detail screen runs it in the browser on
  // data it has already fetched — the market, the live book and the tape it is
  // showing anyway. No extra request, and it stays in step with the socket.
  const signals = useMemo(() => {
    if (!selectedMarket) return null;
    const flow = aggregateFlow(tradesQ.data ?? []).get(selectedMarket.conditionId);
    // Feed the token's own history in too, so the same trained model that ranks
    // the scan also weighs in here — the detail screen runs the whole engine,
    // not a cheaper subset of it.
    const token = selectedMarket.outcomes[0]?.tokenId;
    const history = token ? historyQ.data?.find((h) => h.tokenId === token)?.points : undefined;
    return scoreMarket(selectedMarket, { flow, book, history });
  }, [selectedMarket, tradesQ.data, book, historyQ.data]);

  const series = useMemo<Series[]>(() => {
    if (!historyQ.data) return [];
    return historyQ.data
      .map((h, i) => {
        const o = outcomes.find((x) => x.tokenId === h.tokenId);
        return {
          tokenId: h.tokenId,
          label: o ? truncate(o.label, 22) : `#${i + 1}`,
          color: SERIES_COLORS[i % SERIES_COLORS.length],
          points: h.points,
        };
      })
      .filter((s) => s.points.length > 1);
  }, [historyQ.data, outcomes]);

  const loading =
    eventQ.loading ||
    (marketFallback && marketQ.loading) ||
    (directEmpty && searchQ.loading && markets.length === 0);
  const notFound = !loading && markets.length === 0;

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loading text={`resolving ${slug}`} />
      </div>
    );
  }

  if (notFound) {
    return (
      <div className="p-3">
        <ErrorBox message={`no market found for slug "${slug}" — try SRCH to locate it by name`} />
      </div>
    );
  }

  const watched = selected ? isWatched(selected.tokenId) : false;

  return (
    <motion.div
      className="flex h-full min-h-0 flex-col gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      {/* ── Instrument header ─────────────────────────────────────────── */}
      <motion.div
        variants={panelVariants}
        className="shrink-0 border-b border-edge bg-surface px-2 py-1.5"
      >
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm2 font-bold text-accent" title={title}>
              {title}
            </h1>
            <div className="mt-[2px] flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] text-muted">
              <span className="text-faint">{slug}</span>
              {(event?.tags ?? primary?.tags ?? []).slice(0, 5).map((t) => (
                <span key={t} className="border border-edge px-1 text-info-weak uppercase">
                  {t}
                </span>
              ))}
              {primary?.negRisk ? (
                <span className="border border-accent-2/40 px-1 text-accent-2">NEG-RISK</span>
              ) : null}
              {primary && !primary.acceptingOrders ? (
                <span className="border border-down-weak px-1 text-down">CLOSED TO ORDERS</span>
              ) : null}
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-1">
            <motion.button
              whileTap={tapScale}
              onClick={() => {
                if (!selected) return;
                toggleWatch({
                  slug: event?.slug ?? selectedMarket?.slug ?? slug,
                  label: selected.label,
                  tokenId: selected.tokenId,
                  marketId: selectedMarket?.id ?? "",
                  addedAt: Date.now(),
                });
              }}
              className={`border px-1.5 py-[1px] text-[10px] tracking-wide ${
                watched
                  ? "border-accent text-accent"
                  : "border-edge text-muted hover:border-accent-weak hover:text-accent"
              }`}
            >
              {watched ? "★ WATCHING" : "☆ WATCH"}
            </motion.button>
            <a
              href={`https://polymarket.com/event/${event?.slug ?? slug}`}
              target="_blank"
              rel="noreferrer"
              className="border border-edge px-1.5 py-[1px] text-[10px] tracking-wide text-muted hover:border-info-weak hover:text-info"
            >
              OPEN ON POLYMARKET ↗
            </a>
          </div>
        </div>

        <div className="mt-1.5 flex flex-wrap gap-x-5 gap-y-1">
          <HeaderStat label="Volume" value={usd(event?.volume ?? primary?.volume)} />
          <HeaderStat label="24h Vol" value={usd(event?.volume24h ?? primary?.volume24h)} />
          <HeaderStat label="Liquidity" value={usd(event?.liquidity ?? primary?.liquidity)} />
          {event?.openInterest ? (
            <HeaderStat label="Open Int" value={usd(event.openInterest)} />
          ) : null}
          <HeaderStat label="Legs" value={String(markets.length)} />
          <HeaderStat
            label="Resolves"
            value={dateShort(event?.endDate ?? primary?.endDate)}
            sub={timeToExpiry(event?.endDate ?? primary?.endDate)}
          />
          <HeaderStat
            label="Feed"
            value={feed.status.toUpperCase()}
            tone={feed.status === "live" ? "text-up" : "text-accent-weak"}
          />
        </div>
      </motion.div>

      {/* ── Workspace ─────────────────────────────────────────────────── */}
      <motion.div
        variants={staggerContainer}
        className="grid min-h-0 flex-1 grid-cols-1 gap-2 overflow-y-auto xl:grid-cols-[minmax(0,1fr)_300px] xl:overflow-visible"
      >
        <motion.div variants={panelVariants} className="flex min-h-0 min-w-0 flex-col gap-2">
          <Panel
            title="PRICE HISTORY"
            right={
              <span className="flex items-center gap-1">
                {INTERVALS.map((iv) => (
                  <motion.button
                    key={iv.key}
                    whileTap={tapScale}
                    onClick={() => setInterval(iv.key)}
                    className={`px-1 ${
                      interval === iv.key ? "font-medium text-accent" : "text-faint hover:text-ink"
                    }`}
                  >
                    {iv.label}
                  </motion.button>
                ))}
              </span>
            }
            className="shrink-0"
          >
            {historyQ.loading ? (
              <Loading text="loading history" />
            ) : (
              <>
                <PriceChart series={series} height={214} />
                <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 border-t border-edge pt-1">
                  {series.map((s) => {
                    const first = s.points[0]?.p ?? 0;
                    const last = s.points[s.points.length - 1]?.p ?? 0;
                    const chg = (last - first) * 100;
                    return (
                      <button
                        key={s.tokenId}
                        onClick={() => setSelectedToken(s.tokenId)}
                        className={`flex items-baseline gap-1 text-[10px] ${
                          selectedId === s.tokenId ? "text-ink" : "text-muted"
                        }`}
                      >
                        <span
                          className="inline-block h-[7px] w-[7px]"
                          style={{ background: s.color }}
                        />
                        <span className="truncate">{s.label}</span>
                        <span className="text-ink">{cents(last)}¢</span>
                        <span className={dirClass(chg)}>{signed(chg)}</span>
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </Panel>

          <Panel
            title="SIGNALS"
            right={
              signals
                ? `heat ${signals.heat} · bias ${signed(signals.bias, 0)}${
                    signals.model ? ` · model ${Math.round(signals.model.prob * 100)}%` : ""
                  }`
                : "none"
            }
            className="shrink-0"
          >
            <SignalStrip signals={signals} />
          </Panel>

          <Panel
            title={`TIME & SALES · ${truncate(selected?.label ?? "", 28)}`}
            right={tradesQ.refreshing ? "sync…" : undefined}
            className="min-h-0 flex-1 max-xl:min-h-[220px]"
            flush
          >
            {tradesQ.loading ? (
              <Loading />
            ) : tradesQ.error ? (
              <ErrorBox message={tradesQ.error} />
            ) : (
              <TradeTape trades={tradesQ.data ?? []} />
            )}
          </Panel>
        </motion.div>

        <motion.div
          variants={panelVariants}
          className="flex min-h-0 min-w-0 flex-col gap-2 overflow-y-auto"
        >
          <Panel title="OUTCOMES" className="max-h-[190px] shrink-0" flush>
            <OutcomeList
              outcomes={outcomes}
              selectedToken={selectedId}
              onSelect={setSelectedToken}
              quotes={feed.quotes}
              feedVersion={feed.version}
            />
          </Panel>

          <Panel
            title="ORDER BOOK"
            right={liveBook ? "live" : "rest"}
            className="min-h-[326px] shrink-0"
          >
            <OrderBookLadder book={book} depth={9} outcomeLabel={selected?.label} />
          </Panel>

          <Panel title="TRADE" right={truncate(selected?.label ?? "", 18)} className="shrink-0">
            <OrderTicket
              tokenId={selectedId ?? undefined}
              outcomeLabel={selected?.label}
              markPrice={selected?.price}
              tickSize={selectedMarket?.tickSize ?? 0.001}
              negRisk={selectedMarket?.negRisk ?? false}
              acceptingOrders={selectedMarket?.acceptingOrders ?? false}
              book={book}
            />
          </Panel>

          <Panel title="MARKET DATA" className="shrink-0">
            <Field label="Condition" value={shortAddr(selectedMarket?.conditionId)} />
            <Field label="Token" value={shortAddr(selected?.tokenId)} />
            <Field label="Tick size" value={selectedMarket?.tickSize ?? "--"} />
            <Field
              label="24h change"
              value={signed(selectedMarket?.chg24h)}
              tone={dirClass(selectedMarket?.chg24h)}
            />
            <Field
              label="1w change"
              value={signed(selectedMarket?.chg1w)}
              tone={dirClass(selectedMarket?.chg1w)}
            />
            <Field label="Leg volume" value={usd(selectedMarket?.volume)} />
            <Field label="Leg liquidity" value={usd(selectedMarket?.liquidity)} />
            <div className="mt-1.5 border-t border-edge pt-1.5">
              <AlertForm
                disabled={!selected}
                onArm={(op, target) => {
                  if (!selected || !selectedMarket) return;
                  addAlert({
                    tokenId: selected.tokenId,
                    marketId: selectedMarket.id,
                    label: selected.label,
                    op,
                    target,
                  });
                }}
                onError={(msg) => toast(msg, "error")}
              />
            </div>
          </Panel>

          <Panel title="TOP HOLDERS" className="max-h-[190px] shrink-0" flush>
            <HolderList groups={holdersQ.data} loading={holdersQ.loading} />
          </Panel>
        </motion.div>
      </motion.div>

      {(primary?.description || event?.title) && (
        <details className="shrink-0 border-t border-edge bg-surface px-2 py-1">
          <summary className="cursor-pointer text-[10px] tracking-wide text-accent-weak uppercase">
            Resolution criteria
          </summary>
          <p className="mt-1 max-h-[110px] overflow-auto pr-2 text-tiny leading-relaxed whitespace-pre-wrap text-muted">
            {primary?.description ?? "No description provided."}
          </p>
        </details>
      )}
    </motion.div>
  );
}

/**
 * Compact read-out of everything the engine flagged on this market.
 *
 * Each signal shows its own strength bar rather than only the composite: two
 * weak signals and one overwhelming one can produce the same heat, and which
 * of those it is changes what you'd do about it.
 */
function SignalStrip({ signals }: { signals: MarketSignals | null }) {
  if (!signals || signals.signals.length === 0) {
    return (
      <div className="py-1 text-[10px] tracking-wide text-faint uppercase">
        nothing flagged — market is quiet on every detector
      </div>
    );
  }

  const model = signals.model;
  const agree = model ? modelAgreement(signals) : "neutral";
  const modelGlyph =
    agree === "confirms"
      ? { mark: "✓", tone: "text-accent-2", word: "confirms bias" }
      : agree === "conflicts"
        ? { mark: "✕", tone: "text-warn", word: "conflicts with bias" }
        : { mark: "·", tone: "text-faint", word: "no clear lean" };

  return (
    <div className="flex flex-col gap-[3px]">
      {/* The trained model's own read, kept visually apart from the detector
          rows above it — a different kind of evidence, not one more detector. */}
      {model ? (
        <div
          className="mb-[3px] flex items-baseline gap-2 border-b border-edge/60 pb-[4px]"
          title={`Trained model puts ${Math.round(
            model.prob * 100
          )}% on YES rising from here (conviction ${(model.conviction * 100).toFixed(
            0
          )}%), and ${modelGlyph.word}. ${(model.auc * 100).toFixed(0)}% AUC out of sample — it tilts the read, it does not drive it.`}
        >
          <span
            className={`w-[46px] shrink-0 border border-edge-strong px-1 text-center text-[10px] font-bold ${
              model.direction === "bullish" ? "text-up" : "text-down"
            }`}
          >
            MODEL
          </span>
          <span className="h-[6px] w-[40px] shrink-0 self-center bg-edge" aria-hidden>
            <span
              className={`block h-full ${model.direction === "bullish" ? "bg-up" : "bg-down"}`}
              style={{ width: `${model.conviction * 100}%` }}
            />
          </span>
          <span
            className={`w-[84px] shrink-0 text-tiny ${
              model.direction === "bullish" ? "text-up" : "text-down"
            }`}
          >
            {Math.round(model.prob * 100)}% up
          </span>
          <span className={`min-w-0 flex-1 truncate text-[10px] ${modelGlyph.tone}`}>
            {modelGlyph.mark} {modelGlyph.word}
          </span>
        </div>
      ) : null}
      {signals.signals.map((s) => {
        const meta = SIGNAL_META[s.kind];
        const tone =
          meta.tone === "warn"
            ? "border-accent-weak text-accent"
            : s.direction === "bullish"
              ? "border-up-weak text-up"
              : s.direction === "bearish"
                ? "border-down-weak text-down"
                : "border-edge-strong text-muted";
        return (
          <div key={s.kind} className="flex items-baseline gap-2">
            <span
              className={`w-[46px] shrink-0 border px-1 text-center text-[10px] font-bold ${tone}`}
              title={meta.blurb}
            >
              {meta.label}
            </span>
            <span className="h-[6px] w-[40px] shrink-0 self-center bg-edge" aria-hidden>
              <span
                className={`block h-full ${
                  s.direction === "bullish"
                    ? "bg-up"
                    : s.direction === "bearish"
                      ? "bg-down"
                      : "bg-accent-weak"
                }`}
                style={{ width: `${s.strength}%` }}
              />
            </span>
            <span className="w-[84px] shrink-0 text-tiny text-accent">{s.headline}</span>
            <span className="min-w-0 flex-1 truncate text-[10px] text-muted" title={s.detail}>
              {s.detail}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function HeaderStat({
  label,
  value,
  sub,
  tone = "text-ink",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: string;
}) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-[10px] tracking-wide text-muted uppercase">{label}</span>
      <span className={`text-tiny ${tone}`}>{value}</span>
      {sub ? <span className="text-[10px] text-faint">({sub})</span> : null}
    </span>
  );
}

function OutcomeList({
  outcomes,
  selectedToken,
  onSelect,
  quotes,
  feedVersion,
}: {
  outcomes: { tokenId: string; label: string; price: number }[];
  selectedToken: string | null;
  onSelect: (t: string) => void;
  quotes: Map<string, { bid?: number; ask?: number; last?: number }>;
  feedVersion: number;
}) {
  // feedVersion changes whenever the quote map mutates in place.
  void feedVersion;

  if (outcomes.length === 0) return <Empty />;

  return (
    <div className="text-tiny">
      {outcomes.map((o) => {
        const q = quotes.get(o.tokenId);
        const bid = q?.bid;
        const ask = q?.ask;
        const mark =
          q?.last ?? (bid !== undefined && ask !== undefined ? (bid + ask) / 2 : o.price);
        const active = o.tokenId === selectedToken;
        return (
          <button
            key={o.tokenId}
            onClick={() => onSelect(o.tokenId)}
            className={`flex w-full items-center gap-1.5 border-b border-edge/40 px-1.5 py-[3px] text-left hover:bg-surface-2 ${
              active ? "row-sel" : ""
            }`}
          >
            <span className="min-w-0 flex-1 truncate text-ink" title={o.label}>
              {o.label}
            </span>
            <span className="w-[34px] shrink-0 text-right text-up/80">{cents(bid, 0)}</span>
            <span className="w-[34px] shrink-0 text-right text-down/80">{cents(ask, 0)}</span>
            <span className="w-[40px] shrink-0 text-right font-bold text-accent">
              {cents(mark)}¢
            </span>
            {/* Probability bar doubles as an at-a-glance ranking. */}
            <span className="hidden h-[6px] w-[38px] shrink-0 bg-edge sm:block">
              <span
                className="block h-full bg-info-weak"
                style={{ width: `${Math.max(0, Math.min(100, mark * 100))}%` }}
              />
            </span>
          </button>
        );
      })}
    </div>
  );
}

function HolderList({ groups, loading }: { groups: Holder[][] | null; loading: boolean }) {
  if (loading) return <Loading />;
  const rows = (groups ?? []).flat();
  if (rows.length === 0) return <Empty text="no holder data" />;
  const max = Math.max(...rows.map((r) => r.amount), 1);

  return (
    <div className="text-tiny">
      {rows
        .sort((a, b) => b.amount - a.amount)
        .slice(0, 20)
        .map((h, i) => (
          <div
            key={`${h.wallet}-${h.outcomeIndex}-${i}`}
            className="relative flex items-center gap-1.5 border-b border-edge/40 px-1.5 py-[2px]"
          >
            <span
              className="absolute inset-y-0 right-0 bg-info/8"
              style={{ width: `${(h.amount / max) * 100}%` }}
              aria-hidden
            />
            <span className="relative w-[10px] shrink-0 text-right text-faint">{i + 1}</span>
            <span className="relative min-w-0 flex-1 truncate text-ink/85">
              {h.name ? truncate(h.name, 16) : shortAddr(h.wallet)}
            </span>
            <span className="relative w-[22px] shrink-0 text-center text-[10px] text-info-weak">
              {h.outcomeIndex === 0 ? "YES" : "NO"}
            </span>
            <span className="relative w-[52px] shrink-0 text-right text-ink">
              {compact(h.amount)}
            </span>
          </div>
        ))}
    </div>
  );
}

function AlertForm({
  disabled,
  onArm,
  onError,
}: {
  disabled: boolean;
  onArm: (op: "gte" | "lte", target: number) => void;
  onError: (msg: string) => void;
}) {
  const [op, setOp] = useState<"gte" | "lte">("gte");
  const [target, setTarget] = useState("");

  const submit = () => {
    const v = parseFloat(target);
    if (!Number.isFinite(v) || v <= 0 || v >= 100) {
      onError("alert target must be between 0 and 100 cents");
      return;
    }
    onArm(op, v / 100);
    setTarget("");
  };

  return (
    <div className="flex items-center gap-1">
      <span className="text-[10px] tracking-wide text-muted uppercase">Alert</span>
      <motion.button
        whileTap={tapScale}
        onClick={() => setOp((o) => (o === "gte" ? "lte" : "gte"))}
        className="border border-edge px-1 text-tiny text-accent hover:border-accent-weak"
        title="Toggle direction"
      >
        {op === "gte" ? "≥" : "≤"}
      </motion.button>
      <input
        value={target}
        onChange={(e) => setTarget(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") submit();
          e.stopPropagation();
        }}
        inputMode="decimal"
        placeholder="cents"
        className="w-[52px] border border-edge px-1 text-tiny text-ink placeholder:text-faint"
      />
      <motion.button
        whileTap={tapScale}
        onClick={submit}
        disabled={disabled}
        className="border border-accent-weak px-1.5 text-[10px] font-bold tracking-wide text-accent hover:bg-accent hover:text-canvas disabled:opacity-40"
      >
        ARM
      </motion.button>
    </div>
  );
}
