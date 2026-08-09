"use client";

import { motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { SignalsPayload } from "@/app/api/signals/route";
import { useTerminal } from "@/components/TerminalProvider";
import { Chip, Empty, ErrorBox, Field, Loading, Panel } from "@/components/ui/Panel";
import { usePoll } from "@/hooks/usePoll";
import { useMarketSocket } from "@/hooks/useMarketSocket";
import { useLiveModel } from "@/hooks/useLiveModel";
import { useMesh } from "@/components/MeshProvider";
import type { LiveRead } from "@/lib/liveModel";
import type { Consensus } from "@/lib/signalMesh";
import { cents, compact, dirClass, signed, timeToExpiry, truncate, usd } from "@/lib/format";
import { panelVariants, staggerContainer } from "@/lib/motion";
import { clamp } from "@/lib/quant";
import {
  modelAgreement,
  SIGNAL_META,
  type Agreement,
  type ArbOpportunity,
  type BasketDrift,
  type MarketSignals,
  type Signal,
  type SignalKind,
} from "@/lib/signals";
import {
  MODEL_AUC,
  MODEL_CALIBRATION,
  type Calibration,
  type ReliabilityBin,
} from "@/lib/mlSignal";

/** Declared order of the union, so the chip row and the legend never reshuffle. */
const KINDS = Object.keys(SIGNAL_META) as SignalKind[];

/** Badges that fit a ranked row before the column starts clipping. */
const MAX_ROW_BADGES = 4;

const DEEP_SCAN_TITLE =
  "Deep scan: markets that also got price history and a full order book pulled, so the time-series and microstructure detectors could run on them. Everything else is scored on the cheap detectors alone.";

const DISLOCATION_NOTE =
  "Exactly one leg of a negative-risk event resolves YES, so the basket settles at 100¢. Arbitrage is a crossable edge on real quotes; drift is mid-price pressure that has not yet opened one.";

const MODEL_NOTE = `A trained network scores each deep-scanned market's own price path and reports the probability it drifts up over the next few hours. It ran at ${(
  MODEL_AUC * 100
).toFixed(
  0
)}% AUC out of sample — a real but weak edge — so it only nudges the ranking toward markets it independently agrees with, and never overrules the book. Confirms means it points the same way as the rule-engine bias; conflicts means it fights it.`;

/**
 * The scanner screen.
 *
 * Everything on the page is the output of one server-side pass over the tape,
 * the books and the event tree, so the ranking, the filters and the detail rail
 * are always reading the same snapshot rather than three drifting fetches.
 *
 * Dislocations sit above the ranking because a basket that does not sum to 100¢
 * is the only thing here that is true regardless of anyone's opinion, and it
 * decays in seconds — the ranked table is a research queue, that panel is an
 * alarm. Everything below it is a probabilistic read, which is why strength and
 * confidence are drawn as two separate marks rather than multiplied into one
 * number nobody can take apart again.
 */
export default function SignalsScreen() {
  const { go } = useTerminal();
  const { data, error, loading, refreshing } = usePoll<SignalsPayload>("/api/signals", 20000);

  // Live model overlay: subscribe to the ranked markets' tokens and re-score the
  // model between polls off the socket. The poll stays the source of truth for
  // heat, bias, arbs and the cross-section; only the model read and its blended
  // value tick live. Capped so a huge scan doesn't open hundreds of subscriptions.
  const liveTokens = useMemo(
    () =>
      (data?.markets ?? [])
        .map((m) => m.market.outcomes[0]?.tokenId)
        .filter((t): t is string => Boolean(t))
        .slice(0, 60),
    [data]
  );
  const feed = useMarketSocket(liveTokens, liveTokens.length > 0);
  const liveModel = useLiveModel(data?.markets ?? [], feed);

  // Desk consensus from the P2P mesh, when peers are connected — the scanner
  // marks a market other terminals independently agree on.
  const mesh = useMesh();

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

  const openEvent = useCallback(
    (slug: string) => go({ fn: "DES", slug, kind: "event" }, `DES ${slug}`),
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
  const drifts = data?.drifts ?? [];
  const selected = rows[sel];
  // A failed refresh keeps the last good scan on screen; the strip says so.
  const stale = !!error && !!data;

  const bullish = stats?.bullish ?? 0;
  const bearish = stats?.bearish ?? 0;
  const directional = bullish + bearish;
  const bullPct = directional === 0 ? 50 : (bullish / directional) * 100;

  // Confirm/conflict split recomputed from the live overlay, so the header tracks
  // the model as price moves rather than freezing at the last poll. Falls back to
  // the poll's own numbers exactly when no live tick has landed yet.
  const { confirms, conflicts } = useMemo(() => {
    let c = 0;
    let x = 0;
    for (const m of data?.markets ?? []) {
      const lv = liveModel.get(m.market.id);
      const a = modelAgreement(lv ? { ...m, model: lv.model } : m);
      if (a === "confirms") c++;
      else if (a === "conflicts") x++;
    }
    return { confirms: c, conflicts: x };
  }, [data, liveModel]);
  const modelCalls = confirms + conflicts;
  const confirmPct = modelCalls === 0 ? 50 : (confirms / modelCalls) * 100;

  const liveKinds = KINDS.filter((k) => (stats?.byKind[k] ?? 0) > 0);

  return (
    <motion.div
      className="flex h-full min-h-0 flex-col gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      {/* ── Scan header ────────────────────────────────────────────────── */}
      <Panel
        title="Scan"
        right={
          <span className="flex items-center gap-2">
            {stale ? <span className="text-warn">stale</span> : null}
            {refreshing ? <span className="text-accent">···</span> : null}
            {feed.status === "live" ? (
              <span
                className="flex items-center gap-1 text-up"
                title="Model is re-scoring live off the market socket between the 20-second polls."
              >
                <span className="dot" /> live
              </span>
            ) : null}
            {mesh.peers.size > 0 ? (
              <span
                className="text-accent-2"
                title={`Signal mesh: ${mesh.peers.size} peer(s) over ${mesh.links} link(s) sharing reads. Ranked rows the desk agrees on are marked.`}
              >
                desk {mesh.peers.size}
              </span>
            ) : null}
            <span>every 20s</span>
          </span>
        }
        className="shrink-0"
        flush
        animate
      >
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3 px-2.5 py-2">
          <Stat
            label="Scanned"
            value={stats ? String(stats.scanned) : "—"}
            title="Markets pulled from the tape and put through the engine."
          />
          <Stat
            label="Flagged"
            value={stats ? String(stats.flagged) : "—"}
            title="Markets where at least one detector fired. Everything else stayed silent."
          />
          <Stat
            label="Deep"
            value={stats ? String(stats.deepScanned) : "—"}
            title={DEEP_SCAN_TITLE}
          />

          <div className="flex flex-col gap-[3px]">
            <span className="eyebrow">Bull / Bear</span>
            <span
              className="flex items-center gap-1.5"
              title="Markets whose net directional read is clearly bullish versus clearly bearish."
            >
              <span className="w-[18px] text-right text-tiny font-medium text-up">{bullish}</span>
              <span className="flex h-[6px] w-[120px] shrink-0 overflow-hidden rounded-sm border border-edge bg-surface-2">
                <span className="bg-up" style={{ width: `${bullPct}%` }} />
                <span className="flex-1 bg-down" />
              </span>
              <span className="w-[18px] text-tiny font-medium text-down">{bearish}</span>
            </span>
          </div>

          <Stat
            label="Block notional"
            value={stats ? usd(stats.blockNotional) : "—"}
            title="Total notional of the block prints in the scanned window — the size that moved, not the count."
          />

          <div className="flex flex-col gap-[3px]">
            <span className="eyebrow" title={MODEL_NOTE}>
              Model {stats ? stats.modeled : "—"}
            </span>
            <span
              className="flex items-center gap-1.5"
              title={`The trained model took a directional view on ${modelCalls} of them: ${confirms} confirm the rule engine, ${conflicts} conflict. ${MODEL_NOTE}`}
            >
              <span className="w-[18px] text-right text-tiny font-medium text-accent-2">
                {confirms}
              </span>
              <span className="flex h-[6px] w-[120px] shrink-0 overflow-hidden rounded-sm border border-edge bg-surface-2">
                <span className="bg-accent-2" style={{ width: `${confirmPct}%` }} />
                <span className="flex-1 bg-warn" />
              </span>
              <span className="w-[18px] text-tiny font-medium text-warn">{conflicts}</span>
            </span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5 border-t border-edge px-2.5 py-1.5">
          <span className="eyebrow mr-1">Kinds</span>
          {liveKinds.map((k) => (
            <Chip
              key={k}
              active={active.includes(k)}
              onClick={() => toggleKind(k)}
              title={SIGNAL_META[k].blurb}
            >
              {SIGNAL_META[k].label}
              <span className="text-faint">{stats?.byKind[k]}</span>
            </Chip>
          ))}
          {liveKinds.length === 0 ? (
            <span className="text-[11px] text-faint">nothing fired this pass</span>
          ) : null}
          {active.length > 0 ? (
            <Chip onClick={() => setActive([])} title="Clear every filter">
              clear
            </Chip>
          ) : null}
        </div>
      </Panel>

      {/* ── Dislocations ───────────────────────────────────────────────── */}
      {arbs.length > 0 || drifts.length > 0 ? (
        <Panel
          title="Dislocations"
          right={`${arbs.length} arb · ${drifts.length} drift`}
          className="max-h-[220px] shrink-0"
          flush
          animate
        >
          <p className="border-b border-edge px-2.5 py-1.5 text-[11px] leading-[15px] text-faint">
            {DISLOCATION_NOTE}
          </p>

          {arbs.length > 0 ? (
            <>
              <div className="flex items-center justify-between border-b border-edge bg-surface-2 px-2.5 py-[3px]">
                <span className="eyebrow">Arbitrage</span>
                <span className="eyebrow">Edge, points</span>
              </div>
              {arbs.map((a) => (
                <ArbRow
                  key={`${a.event.id}-${a.side}`}
                  arb={a}
                  onOpen={() => openEvent(a.event.slug)}
                />
              ))}
            </>
          ) : null}

          {drifts.length > 0 ? (
            <>
              <div className="flex items-center justify-between border-b border-edge bg-surface-2 px-2.5 py-[3px]">
                <span className="eyebrow">Basket drift</span>
                <span className="eyebrow">Drift ÷ quoting noise</span>
              </div>
              {drifts.map((d) => (
                <DriftRow key={d.event.id} drift={d} onOpen={() => openEvent(d.event.slug)} />
              ))}
            </>
          ) : null}
        </Panel>
      ) : null}

      {/* ── Ranking + detail rail ──────────────────────────────────────── */}
      <motion.div
        variants={panelVariants}
        className="grid min-h-0 flex-1 grid-cols-1 gap-2 xl:grid-cols-[minmax(0,1fr)_330px]"
      >
        <Panel
          title="Ranked signals"
          right={`${rows.length}${active.length > 0 ? ` / ${data?.markets.length ?? 0}` : ""}`}
          flush
          className="min-h-0 flex-1"
        >
          {loading ? (
            <Loading text="scanning" />
          ) : error && !data ? (
            <div className="p-2.5">
              <ErrorBox message={error} />
            </div>
          ) : rows.length === 0 ? (
            <Empty text={active.length > 0 ? "no markets match that filter" : "no signals"} />
          ) : (
            <div ref={bodyRef} className="min-w-[1040px] text-tiny">
              <div className="sticky top-0 z-10 flex items-center gap-2 border-b border-edge bg-surface-2 px-2.5 py-[4px]">
                <span className="eyebrow w-[22px] shrink-0 text-right">#</span>
                <span
                  className="eyebrow w-[58px] shrink-0 text-right"
                  title="Composite attention score, 0-100 — how much this market deserves a look, regardless of which way."
                >
                  Heat
                </span>
                <span className="eyebrow min-w-0 flex-1">Market</span>
                <span
                  className="eyebrow w-[48px] shrink-0 text-right"
                  title="Last traded probability, in cents."
                >
                  Last
                </span>
                <span
                  className="eyebrow w-[48px] shrink-0 text-right"
                  title="Change over 24 hours, in probability points."
                >
                  24H
                </span>
                <span
                  className="eyebrow w-[58px] shrink-0 text-right"
                  title="24-hour notional volume."
                >
                  Vol 24H
                </span>
                <span
                  className="eyebrow w-[76px] shrink-0 text-right"
                  title="Net directional read, -100 to +100. Positive is bullish on YES."
                >
                  Bias
                </span>
                <span
                  className="eyebrow w-[54px] shrink-0 text-right"
                  title="Conviction, 0-100: how much the directional signals agree with each other. Four signals all pointing the same way is a different proposition from four that cancel."
                >
                  Conv
                </span>
                <span className="eyebrow w-[92px] shrink-0 text-right" title={MODEL_NOTE}>
                  Model
                </span>
                <span className="eyebrow w-[216px] shrink-0">Signals</span>
              </div>

              {rows.map((m, i) => (
                <SignalRow
                  key={m.market.id || m.market.slug || i}
                  index={i}
                  row={m}
                  live={liveModel.get(m.market.id)}
                  desk={mesh.consensusMap.get(m.market.id || m.market.slug)}
                  selected={i === sel}
                  onSelect={() => setSel(i)}
                  onOpen={() => openMarket(m)}
                />
              ))}
            </div>
          )}
        </Panel>

        <div className="hidden min-h-0 flex-col gap-2 overflow-auto xl:flex">
          {selected ? (
            <DetailRail row={selected} live={liveModel.get(selected.market.id)} />
          ) : (
            <Panel title="Detail" className="shrink-0">
              <Empty text="select a market" />
            </Panel>
          )}

          <Panel title="Legend" className="shrink-0">
            {KINDS.map((k) => (
              <div
                key={k}
                className="flex items-baseline gap-2 border-b border-edge/60 py-[4px] last:border-0"
              >
                <span className="eyebrow w-[46px] shrink-0">{SIGNAL_META[k].label}</span>
                <span className="min-w-0 text-[11px] leading-[15px] text-muted">
                  {SIGNAL_META[k].blurb}
                </span>
              </div>
            ))}
          </Panel>
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ── Scan header parts ────────────────────────────────────────────────── */

function Stat({ label, value, title }: { label: string; value: string; title: string }) {
  return (
    <div className="flex flex-col gap-[1px]" title={title}>
      <span className="eyebrow">{label}</span>
      <span className="text-sm2 font-medium text-ink">{value}</span>
    </div>
  );
}

/* ── Dislocation rows ─────────────────────────────────────────────────── */

function ArbRow({ arb, onOpen }: { arb: ArbOpportunity; onOpen: () => void }) {
  return (
    <div className="flex items-center gap-2 border-b border-edge/60 px-2.5 py-[4px] text-tiny last:border-0 hover:bg-surface-2">
      <button
        onClick={onOpen}
        title={arb.event.title}
        className="min-w-0 flex-1 truncate text-left text-ink hover:text-accent"
      >
        {truncate(arb.event.title, 52)}
      </button>
      <Chip
        tone="accent"
        title={
          arb.side === "sell-basket"
            ? "Sell every leg for more than the $1 the basket can ever pay out."
            : "Buy every leg for less than the $1 exactly one of them will pay."
        }
      >
        {arb.side === "sell-basket" ? "SELL BASKET" : "BUY BASKET"}
      </Chip>
      <span className="w-[46px] shrink-0 text-right text-muted" title="Legs in the basket.">
        {arb.legs} leg
      </span>
      <span
        className="w-[56px] shrink-0 text-right text-ink"
        title="Sum of the leg prices that form the basket."
      >
        {cents(arb.basket)}¢
      </span>
      <span
        className="w-[66px] shrink-0 text-right text-sm2 font-semibold text-accent"
        title="Guaranteed edge in probability points, before fees and slippage."
      >
        {signed(arb.edgePoints, 2)}
      </span>
      <span
        className="w-[62px] shrink-0 text-right text-muted"
        title="Thinnest leg's resting liquidity — the real cap on executable size."
      >
        {usd(arb.tightestLegLiquidity)}
      </span>
    </div>
  );
}

function DriftRow({ drift, onOpen }: { drift: BasketDrift; onOpen: () => void }) {
  return (
    <div className="flex items-center gap-2 border-b border-edge/60 px-2.5 py-[4px] text-tiny last:border-0 hover:bg-surface-2">
      <button
        onClick={onOpen}
        title={drift.event.title}
        className="min-w-0 flex-1 truncate text-left text-ink hover:text-accent"
      >
        {truncate(drift.event.title, 52)}
      </button>
      <span className="w-[46px] shrink-0 text-right text-muted" title="Legs in the basket.">
        {drift.legs} leg
      </span>
      <span
        className="w-[56px] shrink-0 text-right text-ink"
        title="Sum of the leg mid prices. A coherent book sits at 100¢."
      >
        {cents(drift.basket)}¢
      </span>
      <span
        className="w-[56px] shrink-0 text-right text-muted"
        title="Signed deviation of the mid basket from 100¢, in probability points."
      >
        {signed(drift.driftPoints, 2)}
      </span>
      <span
        className="w-[78px] shrink-0 text-right text-sm2 font-semibold text-accent"
        title={`Drift as a multiple of the basket's own quoting noise (${drift.noisePoints.toFixed(
          2
        )} points, the sum of every leg's half-spread). Below 1× the mids could sit there by accident.`}
      >
        {drift.ratio.toFixed(1)}× noise
      </span>
    </div>
  );
}

/* ── Ranked row ───────────────────────────────────────────────────────── */

function SignalRow({
  index,
  row,
  live,
  desk,
  selected,
  onSelect,
  onOpen,
}: {
  index: number;
  row: MarketSignals;
  live?: LiveRead;
  desk?: Consensus;
  selected: boolean;
  onSelect: () => void;
  onOpen: () => void;
}) {
  const m = row.market;
  // Overlay the live model read onto the poll row for the model column only —
  // heat, bias and the badges stay as the scan computed them.
  const modelRow = live ? { ...row, model: live.model } : row;
  const label = m.groupItemTitle || m.question;
  const context = m.eventTitle;

  return (
    <div
      data-row={index}
      onClick={onSelect}
      onDoubleClick={onOpen}
      role="button"
      tabIndex={-1}
      className={`flex cursor-pointer items-center gap-2 border-b border-edge/60 px-2.5 py-[4px] hover:bg-surface-2 ${
        selected ? "row-sel" : ""
      }`}
    >
      <span className="w-[22px] shrink-0 text-right text-faint">{index + 1}</span>
      <Heat value={row.heat} />
      <span className="flex min-w-0 flex-1 items-baseline gap-2">
        <DeskDot desk={desk} />
        <span className="truncate text-ink" title={m.question}>
          {truncate(label, 54)}
        </span>
        {context && context !== label ? (
          <span className="hidden truncate text-[11px] text-faint lg:inline">
            {truncate(context, 32)}
          </span>
        ) : null}
      </span>
      <span className="w-[48px] shrink-0 text-right font-medium text-ink">{cents(m.last)}</span>
      <span className={`w-[48px] shrink-0 text-right ${dirClass(m.chg24h)}`}>
        {signed(m.chg24h)}
      </span>
      <span className="w-[58px] shrink-0 text-right text-muted">{compact(m.volume24h)}</span>
      <Bias value={row.bias} />
      <Conviction value={row.conviction} />
      <ModelCell row={modelRow} />
      {/* Signals are pre-sorted by weighted strength, so the leading few are
          the ones worth the width. Overflow is counted rather than clipped —
          a badge sliced in half reads as a rendering bug, and the rail shows
          the full set for whichever row is selected anyway. */}
      <span className="flex w-[216px] shrink-0 items-center gap-1 overflow-hidden">
        {row.signals.slice(0, MAX_ROW_BADGES).map((s) => (
          <Badge key={s.kind} signal={s} />
        ))}
        {row.signals.length > MAX_ROW_BADGES ? (
          <span
            className="shrink-0 text-[11px] text-faint"
            title={row.signals
              .slice(MAX_ROW_BADGES)
              .map((s) => SIGNAL_META[s.kind].label)
              .join(", ")}
          >
            +{row.signals.length - MAX_ROW_BADGES}
          </span>
        ) : null}
      </span>
    </div>
  );
}

/**
 * A dot that lights only when the P2P mesh has a directional consensus on this
 * market — green when peers lean bullish, red bearish, brighter the more of them
 * agree. Absent entirely off the mesh, so a solo terminal's rows stay clean.
 */
function DeskDot({ desk }: { desk?: Consensus }) {
  if (!desk || desk.voters === 0) return null;
  const tone = desk.agreement > 0 ? "text-up" : desk.agreement < 0 ? "text-down" : "text-faint";
  return (
    <span
      className={`dot shrink-0 ${tone}`}
      style={{ opacity: 0.4 + 0.6 * Math.min(1, desk.voters / 3) }}
      title={`Desk consensus: ${desk.bullish} bullish / ${desk.bearish} bearish across ${desk.voters} peer(s) — ${(desk.agreement * 100).toFixed(0)}% net.`}
    />
  );
}

/** Compact 0..100 attention meter. Accent, because heat is chrome, not a side. */
function Heat({ value }: { value: number }) {
  return (
    <span className="flex w-[58px] shrink-0 items-center justify-end gap-1.5">
      <span className="h-[6px] w-[28px] shrink-0 overflow-hidden rounded-sm border border-edge bg-surface-2">
        <span className="block h-full bg-accent" style={{ width: `${clamp(value, 0, 100)}%` }} />
      </span>
      <span className="w-[20px] text-right text-accent">{value}</span>
    </span>
  );
}

/** Centre-origin diverging bar: right for bullish, left for bearish. */
function Bias({ value }: { value: number }) {
  const half = Math.abs(value) / 2;
  return (
    <span className="flex w-[76px] shrink-0 items-center justify-end gap-1.5">
      <span className="relative h-[6px] w-[44px] shrink-0 overflow-hidden rounded-sm border border-edge bg-surface-2">
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
 * Agreement among the directional signals, kept visually distinct from heat:
 * a quieter secondary hue, because a market can be loud and incoherent at once
 * and the two readings must not blur into each other.
 */
function Conviction({ value }: { value: number }) {
  return (
    <span
      className="flex w-[54px] shrink-0 items-center justify-end gap-1.5"
      title="Conviction, 0-100: how much the directional signals agree with each other. High heat with low conviction means a busy market with no consensus."
    >
      <span className="h-[6px] w-[22px] shrink-0 overflow-hidden rounded-sm border border-edge bg-surface-2">
        <span className="block h-full bg-accent-2" style={{ width: `${clamp(value, 0, 100)}%` }} />
      </span>
      <span className="w-[20px] text-right text-muted">{value}</span>
    </span>
  );
}

const AGREE_GLYPH: Record<Agreement, { mark: string; tone: string; word: string }> = {
  confirms: { mark: "✓", tone: "text-accent-2", word: "confirms" },
  conflicts: { mark: "✕", tone: "text-warn", word: "conflicts with" },
  neutral: { mark: "·", tone: "text-faint", word: "is undecided on" },
};

/**
 * The model's own directional read, kept beside the rule engine's rather than
 * blended into it.
 *
 * The number is the probability the model puts on YES rising; the bar is that
 * probability as a centre-origin lean, coloured by side exactly like `Bias` so
 * the eye can check agreement across the two columns at a glance. The leading
 * glyph is the verdict on whether the two actually line up — a market where the
 * book leans one way and the model the other is precisely the row worth a second
 * look, and it should not need arithmetic to spot.
 */
function ModelCell({ row }: { row: MarketSignals }) {
  const m = row.model;
  if (!m) {
    return (
      <span
        className="w-[92px] shrink-0 text-right text-faint"
        title="Not deep-scanned, so there was no price history to run the model on."
      >
        —
      </span>
    );
  }

  const agree = modelAgreement(row);
  const glyph = AGREE_GLYPH[agree];
  const pct = Math.round(m.prob * 100);
  const up = m.direction === "bullish";
  const half = m.conviction * 50;

  return (
    <span
      className="flex w-[92px] shrink-0 items-center justify-end gap-1.5"
      title={`Model puts ${pct}% on YES rising from here (conviction ${(m.conviction * 100).toFixed(
        0
      )}%), and ${glyph.word} the rule-engine bias. ${(m.auc * 100).toFixed(
        0
      )}% AUC out of sample.`}
    >
      <span className={`w-[10px] shrink-0 text-center ${glyph.tone}`}>{glyph.mark}</span>
      <span className="relative h-[6px] w-[40px] shrink-0 overflow-hidden rounded-sm border border-edge bg-surface-2">
        {up ? (
          <span className="absolute inset-y-0 left-1/2 bg-up" style={{ width: `${half}%` }} />
        ) : (
          <span className="absolute inset-y-0 right-1/2 bg-down" style={{ width: `${half}%` }} />
        )}
        <span className="absolute inset-y-0 left-1/2 w-px bg-edge-strong" />
      </span>
      <span className={`w-[24px] text-right ${up ? "text-up" : "text-down"}`}>{pct}%</span>
    </span>
  );
}

/**
 * Colour carries the side, opacity carries the belief.
 *
 * A 90-strength reading built on twelve data points and one built on six
 * hundred would otherwise print the identical badge, so confidence is mapped to
 * opacity — a weak badge visibly recedes rather than needing to be hovered.
 */
function badgeTone(s: Signal): "up" | "down" | "warn" | "neutral" {
  if (SIGNAL_META[s.kind].tone === "warn") return "warn";
  if (s.direction === "bullish") return "up";
  if (s.direction === "bearish") return "down";
  return "neutral";
}

function Badge({ signal }: { signal: Signal }) {
  const meta = SIGNAL_META[signal.kind];
  return (
    <span
      className="inline-flex shrink-0"
      style={{ opacity: 0.35 + 0.65 * clamp(signal.confidence, 0, 1) }}
    >
      <Chip
        tone={badgeTone(signal)}
        title={`${meta.label} · ${signal.headline} — ${signal.detail} Confidence ${(
          clamp(signal.confidence, 0, 1) * 100
        ).toFixed(0)}%.`}
      >
        {meta.label}
      </Chip>
    </span>
  );
}

/* ── Detail rail ──────────────────────────────────────────────────────── */

function DetailRail({ row, live }: { row: MarketSignals; live?: LiveRead }) {
  const m = row.market;
  const s = row.stats;
  const book = s.book;
  const modelRow = live ? { ...row, model: live.model } : row;

  return (
    <>
      <Panel title="Detail" right={`heat ${row.heat}`} className="shrink-0">
        <div className="flex flex-col gap-2">
          <div className="text-sm2 leading-[17px] text-ink">{m.question}</div>
          {m.eventTitle && m.eventTitle !== m.question ? (
            <div className="text-[11px] text-faint">{m.eventTitle}</div>
          ) : null}

          <div>
            <Field label="Last" value={`${cents(m.last)}¢`} />
            <Field label="24H" value={signed(m.chg24h)} tone={dirClass(m.chg24h)} />
            <Field label="Vol 24H" value={usd(m.volume24h)} />
            <Field label="Expiry" value={timeToExpiry(m.endDate)} tone="text-muted" />
            <Field label="Bias" value={signed(row.bias, 0)} tone={dirClass(row.bias)} />
            <Field label="Conviction" value={`${row.conviction}`} tone="text-accent-2" />
            <Field
              label="Condition"
              value={<span className="mono text-[11px]">{truncate(m.conditionId, 14)}</span>}
              tone="text-muted"
            />
          </div>
        </div>
      </Panel>

      {modelRow.model ? <ModelPanel row={modelRow} /> : null}

      <Panel title="Quant" className="shrink-0">
        <QuantField
          label="Realised vol"
          value={`${s.realisedVol.toFixed(1)} pts/d`}
          title="How much this market normally moves: the standard deviation of its price changes, rescaled to probability points per day."
        />
        <QuantField
          label="Drift"
          value={`${signed(s.driftPerDay, 2)} pts/d`}
          tone={dirClass(s.driftPerDay)}
          title="The straight-line slope of price against time, in probability points per day — the size of the repricing, ignoring the noise around it."
        />
        <QuantField
          label="Trend quality"
          value={s.trendQuality.toFixed(2)}
          title="Drift divided by volatility. It asks whether the move is large relative to how much this market normally jumps around, rather than large in absolute points."
        />
        <QuantField
          label="Autocorr, lag 1"
          value={s.autocorrelation.toFixed(2)}
          title="Correlation between one price change and the next. Positive means moves persist and the market is genuinely repricing; negative means it is oscillating inside a range."
        />
        <QuantField
          label="Band z"
          value={`${signed(s.bandZ, 2)}σ`}
          title="How far the latest price sits from the middle of its own recent range, measured in standard deviations of that range. Beyond ±2 it is stretched."
        />
        <QuantField
          label="Vol compression"
          value={`${(s.volCompression * 100).toFixed(0)}%`}
          title="Recent volatility as a percentage of its earlier baseline. Well under 100% means the market has gone quiet relative to itself — a coiled range."
        />
        <QuantField
          label="Micro lean"
          value={`${signed(book.microLean, 2)}¢`}
          tone={dirClass(book.microLean)}
          title="Where the book says the next trade goes: size-weighted fair value minus mid, in cents. Weighting each side by the opposite side's size pulls fair value toward the thin side, because that is the side that gets taken."
        />
        <QuantField
          label="Book imbalance"
          value={`${(book.imbalance * 100).toFixed(0)}%`}
          tone={dirClass(book.imbalance)}
          title="Resting capital on the bid minus the ask as a share of the total, counting what each side actually risks — a bid risks its price, an offer risks the rest of the dollar."
        />
        <QuantField
          label="Cost to move 1¢"
          value={usd(book.costToMoveOneCent)}
          title="Dollars resting within a cent of mid on the thinner side — what it costs to push the price a single tick in the direction that's harder to move. A book is only as tradable as its shallower side."
        />
        <QuantField
          label="Liquidity"
          value={`${Math.round(book.liquidityScore * 100)}`}
          title="Execution quality, 0–100. High only when the spread is tight and real capital rests on the thinner side at once — a one-sided wall behind a wide spread scores low, because you still cross the spread to trade it."
        />
      </Panel>

      <Panel title="Signals" right={`${row.signals.length}`} className="shrink-0">
        <div className="flex flex-col gap-2.5">
          {row.signals.map((sig) => (
            <SignalDetail key={sig.kind} signal={sig} />
          ))}
        </div>
      </Panel>
    </>
  );
}

/**
 * A `Field` whose label carries an explanation.
 *
 * The wrapper owns the rule so the row still separates: `Field`'s own hairline
 * is suppressed by its `last:border-0` once it is an only child.
 */
function QuantField({
  label,
  value,
  tone,
  title,
}: {
  label: string;
  value: string;
  tone?: string;
  title: string;
}) {
  return (
    <div title={title} className="border-b border-edge/60 last:border-0">
      <Field label={label} value={value} tone={tone} />
    </div>
  );
}

/**
 * The model's read, given a full panel on the detail rail because the rail is
 * where a trader has already decided a market is worth the time — so the honest
 * caveats fit here in a way they never would in a table cell. The header carries
 * the model's out-of-sample AUC so the strength of every number below it is read
 * in the same glance, and the footer states plainly that this only ever tilts
 * the ranking rather than driving it.
 */
function ModelPanel({ row }: { row: MarketSignals }) {
  const m = row.model;
  if (!m) return null;
  const agree = modelAgreement(row);
  const glyph = AGREE_GLYPH[agree];
  const pct = Math.round(m.prob * 100);
  const up = m.direction === "bullish";

  return (
    <Panel title="Model" right={`${(m.auc * 100).toFixed(0)}% AUC`} className="shrink-0">
      <div className="flex flex-col gap-2">
        {/* Probability as a 0–100 track with the coin-flip line marked, so how
            far off the fence the model is reads as distance, not just a digit. */}
        <div className="flex items-center gap-2">
          <span className="relative h-[8px] min-w-0 flex-1 overflow-hidden rounded-sm border border-edge bg-surface-2">
            {up ? (
              <span
                className="absolute inset-y-0 left-1/2 bg-up"
                style={{ width: `${m.conviction * 50}%` }}
              />
            ) : (
              <span
                className="absolute inset-y-0 right-1/2 bg-down"
                style={{ width: `${m.conviction * 50}%` }}
              />
            )}
            <span className="absolute inset-y-0 left-1/2 w-px bg-edge-strong" />
          </span>
          <span
            className={`w-[42px] text-right text-sm2 font-semibold ${up ? "text-up" : "text-down"}`}
          >
            {pct}%
          </span>
        </div>

        <div>
          <Field label="P(YES rises)" value={`${pct}%`} tone={up ? "text-up" : "text-down"} />
          <Field
            label="Direction"
            value={up ? "bullish" : "bearish"}
            tone={up ? "text-up" : "text-down"}
          />
          <Field
            label="Conviction"
            value={`${(m.conviction * 100).toFixed(0)}%`}
            tone="text-accent-2"
          />
          <Field
            label="vs. engine"
            value={
              <span className={glyph.tone}>
                {glyph.mark} {glyph.word.replace(" with", "")}
              </span>
            }
          />
        </div>

        {MODEL_CALIBRATION ? <ReliabilityDiagram cal={MODEL_CALIBRATION} /> : null}

        <p className="text-[11px] leading-[15px] text-faint">
          {(m.auc * 100).toFixed(0)}% AUC out of sample — a real but weak edge, so it only tilts the
          ranking toward markets it agrees with. It never overrides the book.
          {MODEL_CALIBRATION ? (
            <>
              {" "}
              Probability is temperature-calibrated on validation; held-out reliability error{" "}
              {(MODEL_CALIBRATION.ece_heldout * 100).toFixed(1)}% (down from{" "}
              {(MODEL_CALIBRATION.ece_before * 100).toFixed(1)}%), so the number means what it says.
            </>
          ) : null}
        </p>
      </div>
    </Panel>
  );
}

/**
 * Reliability diagram — the "does 70% mean 70%?" chart.
 *
 * Predicted probability runs along x, the empirical hit rate up y, both on the
 * same 0..1 scale, so a perfectly calibrated model traces the dotted diagonal.
 * The model's line is drawn over it and each point is sized by how many
 * validation windows fell in that bin, so a wobble on a bin of four is visibly
 * lighter than a true miss on a bin of ten thousand. This is the picture behind
 * the single ECE number — the honest bridge from what the model claims to what
 * actually happened.
 */
function ReliabilityDiagram({ cal }: { cal: Calibration }) {
  const bins = cal.reliability;
  if (bins.length < 2) return null;

  const S = 100;
  const P = 9;
  const toX = (v: number) => P + clamp(v, 0, 1) * (S - 2 * P);
  const toY = (v: number) => S - (P + clamp(v, 0, 1) * (S - 2 * P));
  const maxN = Math.max(...bins.map((b) => b.n));
  const line = bins.map(
    (b: ReliabilityBin) => `${toX(b.conf).toFixed(1)},${toY(b.acc).toFixed(1)}`
  );

  return (
    <div title="Reliability diagram: predicted probability (x) against the actual hit rate on validation (y). The dotted line is perfect calibration; the model's line hugging it is what the low ECE measures. Points are sized by how many windows landed in each bin.">
      <div className="eyebrow mb-1">Reliability</div>
      <svg viewBox={`0 0 ${S} ${S}`} className="w-full" style={{ maxHeight: 132 }} aria-hidden>
        <rect
          x={P}
          y={P}
          width={S - 2 * P}
          height={S - 2 * P}
          fill="none"
          strokeWidth={0.5}
          className="stroke-current text-faint"
        />
        <line
          x1={toX(0)}
          y1={toY(0)}
          x2={toX(1)}
          y2={toY(1)}
          strokeWidth={0.7}
          strokeDasharray="2 2"
          className="stroke-current text-faint"
        />
        <polyline
          points={line.join(" ")}
          fill="none"
          strokeWidth={1.1}
          strokeLinejoin="round"
          className="stroke-current text-accent"
        />
        {bins.map((b, i) => (
          <circle
            key={i}
            cx={toX(b.conf)}
            cy={toY(b.acc)}
            r={1 + 2.6 * Math.sqrt(b.n / maxN)}
            className="fill-current text-accent"
          />
        ))}
      </svg>
    </div>
  );
}

function SignalDetail({ signal }: { signal: Signal }) {
  const meta = SIGNAL_META[signal.kind];
  const confidence = clamp(signal.confidence, 0, 1);

  return (
    <div className="border-t border-edge pt-2 first:border-0 first:pt-0">
      <div className="flex items-center gap-2">
        <span className="eyebrow w-[42px] shrink-0">{meta.label}</span>
        <span
          className="h-[5px] w-[46px] shrink-0 overflow-hidden rounded-sm border border-edge bg-surface-2"
          title="Strength: how notable this reading is, on a 0-100 scale comparable across every kind."
        >
          <span
            className="block h-full bg-accent"
            style={{ width: `${clamp(signal.strength, 0, 100)}%` }}
          />
        </span>
        <span className="w-[20px] shrink-0 text-[11px] text-faint">{signal.strength}</span>
        <span className="min-w-0 flex-1 truncate text-right text-tiny font-medium text-accent">
          {signal.headline}
        </span>
      </div>

      <div className="flex items-center gap-2 pt-[4px]">
        <span className="eyebrow w-[42px] shrink-0">Conf</span>
        <ConfidenceDots value={confidence} />
        <span className="text-[11px] text-muted">{(confidence * 100).toFixed(0)}%</span>
        {signal.z !== undefined ? (
          <span
            className="ml-auto text-[11px] text-faint"
            title="The standardised statistic behind the call, shown so the reading can be checked rather than trusted."
          >
            z {signal.z.toFixed(2)}
          </span>
        ) : null}
      </div>

      <p className="pt-[4px] text-[11px] leading-[15px] text-muted">{signal.detail}</p>
    </div>
  );
}

/**
 * Confidence as five discrete dots rather than a second continuous bar — a
 * different mark for a different quantity, so strength and belief cannot be
 * read off each other by mistake.
 */
function ConfidenceDots({ value }: { value: number }) {
  const lit = Math.max(1, Math.round(value * 5));
  return (
    <span
      className="flex items-center gap-[3px]"
      title="Confidence: how much the inputs justify the reading. Short history, a thin book or a handful of prints all pull it down. Strength says how loud, this says how much to believe it."
    >
      {[0, 1, 2, 3, 4].map((i) => (
        <span
          key={i}
          className={`dot ${i < lit ? "text-accent-2" : "text-faint"}`}
          style={{ opacity: i < lit ? 1 : 0.3 }}
        />
      ))}
    </span>
  );
}
