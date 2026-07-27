"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { SignalsPayload } from "@/app/api/signals/route";
import { useTerminal } from "@/components/TerminalProvider";
import { Empty, ErrorBox, Field, Loading, Panel } from "@/components/ui/Panel";
import { usePoll } from "@/hooks/usePoll";
import { cents, compact, dirClass, signed, timeToExpiry, truncate, usd } from "@/lib/format";
import {
  SIGNAL_META,
  type ArbOpportunity,
  type MarketSignals,
  type Signal,
  type SignalKind,
} from "@/lib/signals";

/** Declared order of the union, so the chip row and legend never reshuffle. */
const KINDS = Object.keys(SIGNAL_META) as SignalKind[];

/**
 * The scanner screen.
 *
 * Everything on the page is the output of one server-side pass over the tape,
 * the books and the event tree, so the ranking, the filters and the detail rail
 * are always reading the same snapshot rather than three drifting fetches.
 *
 * Arbitrage sits above the ranking because it is the only thing here that is
 * risk-free if it is real, and it decays in seconds — the ranked table is a
 * research queue, the arb strip is an alarm.
 */
export default function SignalsScreen() {
  const { go } = useTerminal();
  const { data, error, loading, refreshing } = usePoll<SignalsPayload>("/api/signals", 20000);

  const [active, setActive] = useState<readonly SignalKind[]>([]);
  const [sel, setSel] = useState(0);
  const bodyRef = useRef<HTMLDivElement | null>(null);

  const toggleKind = useCallback((k: SignalKind) => {
    setActive((prev) => (prev.includes(k) ? prev.filter((x) => x !== k) : [...prev, k]));
  }, []);

  const rows = useMemo<MarketSignals[]>(() => {
    const all = data?.markets ?? [];
    if (active.length === 0) return all;
    // A market qualifies on *any* selected kind: the chips read as "show me
    // these", not as a conjunction nobody could satisfy with four filters on.
    return all.filter((m) => m.signals.some((s) => active.includes(s.kind)));
  }, [data, active]);

  // Changing the filter is a new question, so the cursor goes back to the top
  // rather than landing on whatever happens to occupy that index now.
  const filterKey = active.join(",");
  const [prevFilter, setPrevFilter] = useState(filterKey);
  if (prevFilter !== filterKey) {
    setPrevFilter(filterKey);
    setSel(0);
  }

  // A shrinking result set must not leave the cursor past the last row.
  // Clamped during render so no frame highlights a row that no longer exists.
  const [prevCount, setPrevCount] = useState(rows.length);
  if (prevCount !== rows.length) {
    setPrevCount(rows.length);
    if (sel > rows.length - 1) setSel(Math.max(0, rows.length - 1));
  }

  const openMarket = useCallback(
    (m: MarketSignals) => {
      const slug = m.market.eventSlug || m.market.slug;
      go({ fn: "DES", slug, kind: m.market.eventSlug ? "event" : "market" }, `DES ${slug}`);
    },
    [go]
  );

  // Bound to the window rather than the table so the command line keeps focus.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA";
      const isNav = e.key === "ArrowDown" || e.key === "ArrowUp";
      if (typing && !isNav) return;
      if (rows.length === 0) return;

      if (isNav) {
        e.preventDefault();
        setSel((s) => {
          const next = e.key === "ArrowDown" ? s + 1 : s - 1;
          return Math.max(0, Math.min(rows.length - 1, next));
        });
        return;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        openMarket(rows[sel]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows, sel, openMarket]);

  // Keep the highlighted row inside the scroll viewport.
  useEffect(() => {
    const el = bodyRef.current?.querySelector<HTMLElement>(`[data-row="${sel}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  const stats = data?.stats;
  const arbs = data?.arbs ?? [];
  const selected = rows[sel];
  // A failed refresh keeps the last good scan on screen; the strip says so.
  const stale = !!error && !!data;

  const directional = (stats?.bullish ?? 0) + (stats?.bearish ?? 0);
  const bullPct = directional === 0 ? 0 : ((stats?.bullish ?? 0) / directional) * 100;

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      {/* ── Scan strip ─────────────────────────────────────────────────── */}
      <div className="shrink-0 border border-edge bg-surface">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-[2px] px-1.5 py-[3px] text-[10px] tracking-wide uppercase">
          <span className="shrink-0 text-accent-weak">Signals</span>
          <span className="text-muted">
            <span className="text-info">SCANNED</span> {stats?.scanned ?? "--"}
          </span>
          <span className="text-muted">
            <span className="text-info">FLAGGED</span>{" "}
            <span className="text-ink">{stats?.flagged ?? "--"}</span>
          </span>

          <span className="flex items-center gap-2 text-muted">
            <span>
              <span className="text-info">BULL</span>{" "}
              <span className="text-up">{stats?.bullish ?? 0}</span>
            </span>
            <span
              className="flex h-[6px] w-[120px] shrink-0 border border-edge"
              title="Net-bullish vs net-bearish markets"
            >
              <span className="bg-up/70" style={{ width: `${bullPct}%` }} />
              <span className="flex-1 bg-down/70" />
            </span>
            <span>
              <span className="text-info">BEAR</span>{" "}
              <span className="text-down">{stats?.bearish ?? 0}</span>
            </span>
          </span>

          <span className="text-muted" title="Notional of block prints in the scanned window">
            <span className="text-info">BLOCK NTNL</span>{" "}
            <span className="text-ink">{stats ? usd(stats.blockNotional) : "--"}</span>
          </span>

          <span className="ml-auto flex shrink-0 items-center gap-2 text-muted">
            {active.length > 0 ? (
              <button
                onClick={() => setActive([])}
                className="border border-edge px-1 text-[10px] tracking-wide text-muted uppercase hover:border-edge-strong hover:text-accent-weak"
              >
                clear
              </button>
            ) : null}
            {stale ? <span className="text-down">stale</span> : null}
            {refreshing ? <span className="text-accent-weak">···</span> : null}
          </span>
        </div>

        {/* Filter chips — one per kind the scan actually produced. */}
        <div className="flex flex-wrap items-center gap-2 border-t border-edge px-1.5 py-[3px]">
          {KINDS.filter((k) => (stats?.byKind[k] ?? 0) > 0).map((k) => {
            const on = active.includes(k);
            return (
              <button
                key={k}
                onClick={() => toggleKind(k)}
                title={SIGNAL_META[k].blurb}
                className={`border px-1.5 py-[1px] text-[10px] tracking-wide uppercase ${
                  on
                    ? "border-accent bg-accent/8 text-accent"
                    : "border-edge text-muted hover:border-edge-strong hover:text-accent-weak"
                }`}
              >
                {SIGNAL_META[k].label}{" "}
                <span className={on ? "text-accent" : "text-faint"}>{stats?.byKind[k]}</span>
              </button>
            );
          })}
          {stats && KINDS.every((k) => (stats.byKind[k] ?? 0) === 0) ? (
            <span className="text-[10px] tracking-widest text-faint uppercase">no kinds</span>
          ) : null}
        </div>
      </div>

      {/* ── Arbitrage ──────────────────────────────────────────────────── */}
      {arbs.length > 0 ? (
        <Panel
          title="Arbitrage"
          right={`${arbs.length} live`}
          className="max-h-[140px] shrink-0 border-accent-weak"
          flush
        >
          <div className="border-b border-edge bg-surface-2 px-1.5 py-[2px] text-[10px] text-muted">
            exactly one leg of a negative-risk event resolves YES, so the basket settles at 100¢ —
            an edge is the gap to that, before fees and slippage
          </div>
          {arbs.map((a) => (
            <ArbRow
              key={`${a.event.id}-${a.side}`}
              arb={a}
              onOpen={() =>
                go(
                  { fn: "DES", slug: a.event.slug, kind: "event" },
                  `DES ${a.event.slug}`
                )
              }
            />
          ))}
        </Panel>
      ) : null}

      {/* ── Ranking + detail ───────────────────────────────────────────── */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-2 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Panel
          title="Ranked Signals"
          right={`${rows.length}${active.length > 0 ? ` / ${data?.markets.length ?? 0}` : ""}`}
          flush
          className="min-h-0"
        >
          {loading ? (
            <Loading text="scanning" />
          ) : error && !data ? (
            <div className="p-1.5">
              <ErrorBox message={error} />
            </div>
          ) : rows.length === 0 ? (
            <Empty text={active.length > 0 ? "no markets match filter" : "no signals"} />
          ) : (
            <div ref={bodyRef} className="min-w-[880px] text-tiny">
              <div className="sticky top-0 z-10 flex items-center gap-1 border-b border-edge-strong bg-surface-2 px-1 py-[3px] text-[10px] tracking-wide text-accent-weak uppercase">
                <span className="w-[22px] shrink-0 text-right">#</span>
                <span
                  className="w-[62px] shrink-0 text-right"
                  title="Composite attention score, 0-100, regardless of side"
                >
                  Heat
                </span>
                <span className="min-w-0 flex-1">Market</span>
                <span className="w-[52px] shrink-0 text-right" title="Last traded probability, in cents">
                  Last
                </span>
                <span className="w-[52px] shrink-0 text-right" title="Change over 24 hours, in points">
                  24H
                </span>
                <span className="w-[64px] shrink-0 text-right" title="24-hour notional volume">
                  Vol 24H
                </span>
                <span
                  className="w-[74px] shrink-0 text-right"
                  title="Net directional read, -100..100, positive is bullish on YES"
                >
                  Bias
                </span>
                <span className="w-[212px] shrink-0">Signals</span>
              </div>

              {rows.map((m, i) => (
                <SignalRow
                  key={m.market.id || m.market.slug || i}
                  index={i}
                  row={m}
                  selected={i === sel}
                  onSelect={() => setSel(i)}
                  onOpen={() => openMarket(m)}
                />
              ))}
            </div>
          )}
        </Panel>

        <div className="hidden min-h-0 flex-col gap-2 xl:flex">
          <Panel title="Signal Detail" className="min-h-0 flex-1">
            {selected ? <Detail row={selected} /> : <Empty text="select a market" />}
          </Panel>
          <Panel title="Legend" className="max-h-[190px] shrink-0">
            {KINDS.map((k) => (
              <div key={k} className="border-b border-edge/60 py-[2px] last:border-0">
                <span className="text-info">{SIGNAL_META[k].label}</span>{" "}
                <span className="text-[10px] text-muted">{SIGNAL_META[k].blurb}</span>
              </div>
            ))}
          </Panel>
        </div>
      </div>
    </div>
  );
}

/* ── Arbitrage row ────────────────────────────────────────────────────── */

function ArbRow({ arb, onOpen }: { arb: ArbOpportunity; onOpen: () => void }) {
  return (
    <div className="flex items-center gap-2 border-b border-edge/40 px-1.5 py-[2px] text-tiny hover:bg-surface-2">
      <button
        onClick={onOpen}
        title={arb.event.title}
        className="min-w-0 flex-1 truncate text-left text-ink hover:text-accent"
      >
        {truncate(arb.event.title, 58)}
      </button>
      <span
        className={`w-[92px] shrink-0 text-right text-[10px] tracking-wide uppercase ${
          arb.side === "sell-basket" ? "text-down" : "text-up"
        }`}
        title={
          arb.side === "sell-basket"
            ? "Sell every leg for more than the $1 the basket can ever pay"
            : "Buy every leg for less than the $1 exactly one of them will pay"
        }
      >
        {arb.side === "sell-basket" ? "Sell basket" : "Buy basket"}
      </span>
      <span className="w-[44px] shrink-0 text-right text-muted" title="Legs in the basket">
        {arb.legs} leg
      </span>
      <span className="w-[60px] shrink-0 text-right text-ink" title="Sum of the leg prices">
        {cents(arb.basket)}¢
      </span>
      <span
        className="w-[72px] shrink-0 text-right text-sm2 font-bold text-accent"
        title="Guaranteed edge in probability points, before fees and slippage"
      >
        {signed(arb.edgePoints, 2)}
      </span>
      <span
        className="w-[64px] shrink-0 text-right text-info-weak"
        title="Thinnest leg's resting notional — the real cap on executable size"
      >
        {usd(arb.tightestLegLiquidity)}
      </span>
    </div>
  );
}

/* ── Ranked row ───────────────────────────────────────────────────────── */

function SignalRow({
  index,
  row,
  selected,
  onSelect,
  onOpen,
}: {
  index: number;
  row: MarketSignals;
  selected: boolean;
  onSelect: () => void;
  onOpen: () => void;
}) {
  const m = row.market;
  const label = m.groupItemTitle || m.question;
  const context = m.eventTitle;

  return (
    <div
      data-row={index}
      onClick={onSelect}
      onDoubleClick={onOpen}
      role="button"
      tabIndex={-1}
      className={`flex cursor-pointer items-center gap-1 border-b border-edge/40 px-1 py-[2px] hover:bg-surface-2 ${
        selected ? "row-sel" : ""
      }`}
    >
      <span className="w-[22px] shrink-0 text-right text-faint">{index + 1}</span>
      <Heat value={row.heat} />
      <span className="flex min-w-0 flex-1 items-baseline gap-1.5">
        <span className="truncate text-ink" title={m.question}>
          {truncate(label, 56)}
        </span>
        {context && context !== label ? (
          <span className="hidden truncate text-[10px] text-faint lg:inline">
            {truncate(context, 34)}
          </span>
        ) : null}
      </span>
      <span className="w-[52px] shrink-0 text-right font-bold text-ink">{cents(m.last)}</span>
      <span className={`w-[52px] shrink-0 text-right ${dirClass(m.chg24h)}`}>
        {signed(m.chg24h)}
      </span>
      <span className="w-[64px] shrink-0 text-right text-ink/80">{compact(m.volume24h)}</span>
      <Bias value={row.bias} />
      <span className="flex w-[212px] shrink-0 items-center gap-2 overflow-hidden">
        {row.signals.map((s) => (
          <Badge key={s.kind} signal={s} />
        ))}
      </span>
    </div>
  );
}

/** Compact 0..100 meter. Amber because heat is chrome, not direction. */
function Heat({ value }: { value: number }) {
  return (
    <span className="flex w-[62px] shrink-0 items-center justify-end gap-1">
      <span className="h-[6px] w-[28px] shrink-0 border border-edge bg-surface-2">
        <span className="block h-full bg-accent" style={{ width: `${value}%` }} />
      </span>
      <span className="w-[22px] text-right text-accent">{value}</span>
    </span>
  );
}

/** Centre-origin diverging bar: right in green for bullish, left in red. */
function Bias({ value }: { value: number }) {
  const half = Math.abs(value) / 2;
  return (
    <span className="flex w-[74px] shrink-0 items-center justify-end gap-1">
      <span className="relative h-[6px] w-[40px] shrink-0 border border-edge bg-surface-2">
        {value >= 0 ? (
          <span className="absolute inset-y-0 left-1/2 bg-up" style={{ width: `${half}%` }} />
        ) : (
          <span className="absolute inset-y-0 right-1/2 bg-down" style={{ width: `${half}%` }} />
        )}
        <span className="absolute inset-y-0 left-1/2 w-px bg-edge-strong" />
      </span>
      <span className={`w-[26px] text-right ${dirClass(value)}`}>{signed(value, 0)}</span>
    </span>
  );
}

/**
 * Colour carries the read: green/red only when the signal has a side, amber for
 * the execution warnings, dim for the ones that are context rather than a view.
 */
function badgeTone(s: Signal): string {
  if (SIGNAL_META[s.kind].tone === "warn") return "border-accent-weak text-accent";
  if (s.direction === "bullish") return "border-up-weak text-up";
  if (s.direction === "bearish") return "border-down-weak text-down";
  return "border-edge-strong text-muted";
}

function Badge({ signal }: { signal: Signal }) {
  return (
    <span
      title={signal.detail}
      className={`shrink-0 border px-[3px] text-[10px] leading-[13px] tracking-wide uppercase ${badgeTone(
        signal
      )}`}
    >
      {SIGNAL_META[signal.kind].label}
    </span>
  );
}

/* ── Detail rail ──────────────────────────────────────────────────────── */

function Detail({ row }: { row: MarketSignals }) {
  const m = row.market;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-tiny text-ink">{m.question}</div>
      {m.eventTitle && m.eventTitle !== m.question ? (
        <div className="text-[10px] text-faint">{m.eventTitle}</div>
      ) : null}

      <div>
        <Field label="Last" value={cents(m.last)} tone="font-bold text-ink" />
        <Field label="1H" value={signed(m.chg1h)} tone={dirClass(m.chg1h)} />
        <Field label="24H" value={signed(m.chg24h)} tone={dirClass(m.chg24h)} />
        <Field label="1W" value={signed(m.chg1w)} tone={dirClass(m.chg1w)} />
        <Field label="Vol 24H" value={usd(m.volume24h)} />
        <Field label="Liquidity" value={usd(m.liquidity)} tone="text-info-weak" />
        <Field label="Spread" value={m.spread === undefined ? "--" : `${cents(m.spread)}¢`} />
        <Field label="Expiry" value={timeToExpiry(m.endDate)} tone="text-muted" />
      </div>

      <div className="flex flex-col gap-1.5">
        {row.signals.map((s) => (
          <div key={s.kind} className="border-t border-edge pt-1">
            <div className="flex items-center gap-1.5">
              <span className="w-[38px] shrink-0 text-[10px] tracking-wide text-info uppercase">
                {SIGNAL_META[s.kind].label}
              </span>
              <span className="h-[5px] w-[40px] shrink-0 border border-edge bg-surface-2">
                <span
                  className="block h-full bg-accent"
                  style={{ width: `${Math.max(0, Math.min(100, s.strength))}%` }}
                />
              </span>
              <span className="min-w-0 flex-1 truncate text-right font-bold text-accent">
                {s.headline}
              </span>
            </div>
            <div className="pt-[2px] text-[10px] leading-[14px] text-muted">{s.detail}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
