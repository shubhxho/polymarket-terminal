"use client";

import { motion } from "motion/react";
import { useMemo, useState, type ReactNode } from "react";
import { TradeTape } from "@/components/TradeTape";
import { useTerminal } from "@/components/TerminalProvider";
import { Empty, ErrorBox, Field, Loading, Panel } from "@/components/ui/Panel";
import { usePoll } from "@/hooks/usePoll";
import { clock, compact, truncate, usd } from "@/lib/format";
import { panelVariants, staggerContainer, tapScale } from "@/lib/motion";
import type { Trade } from "@/lib/types";

/** Size floors map straight onto the `min` notional query param. */
const SIZE_FILTERS: { label: string; min: number }[] = [
  { label: "ALL", min: 0 },
  { label: "$1K+", min: 1_000 },
  { label: "$10K+", min: 10_000 },
  { label: "$50K+", min: 50_000 },
  { label: "$250K+", min: 250_000 },
];

const SIDE_FILTERS = ["ALL", "BUY", "SELL"] as const;
type SideFilter = (typeof SIDE_FILTERS)[number];

type ActiveRow = {
  key: string;
  title: string;
  slug?: string;
  count: number;
  notional: number;
};

/**
 * TAS — the consolidated print tape.
 *
 * Size filtering happens server-side (a $250k floor over 200 prints is a very
 * different window than the top 200 unfiltered prints), while the side filter
 * is client-side so toggling BUY/SELL never costs a round trip. The right rail
 * always aggregates the *unfiltered* window so the buy/sell proportion bar
 * stays meaningful when one side is hidden.
 */
export default function TapeScreen() {
  const { go } = useTerminal();
  const [min, setMin] = useState(0);
  const [side, setSide] = useState<SideFilter>("ALL");

  const { data, error, loading, updatedAt } = usePoll<Trade[]>(
    `/api/trades?limit=200&min=${min}`,
    4000
  );

  const trades = useMemo(() => data ?? [], [data]);

  const filtered = useMemo(
    () => (side === "ALL" ? trades : trades.filter((t) => t.side === side)),
    [trades, side]
  );

  const flow = useMemo(() => {
    let buy = 0;
    let sell = 0;
    let largest = 0;
    let largestTitle = "";
    for (const t of trades) {
      const n = t.size * t.price;
      if (t.side === "BUY") buy += n;
      else sell += n;
      if (n > largest) {
        largest = n;
        largestTitle = t.title;
      }
    }
    const total = buy + sell;
    return {
      buy,
      sell,
      total,
      count: trades.length,
      avg: trades.length ? total / trades.length : 0,
      largest,
      largestTitle,
      // Split the bar 50/50 when the window is empty rather than dividing by 0.
      buyPct: total > 0 ? (buy / total) * 100 : 50,
    };
  }, [trades]);

  const active = useMemo(() => {
    const byMarket = new Map<string, ActiveRow>();
    for (const t of trades) {
      const key = t.conditionId || t.title;
      const n = t.size * t.price;
      const row = byMarket.get(key);
      if (row) {
        row.count += 1;
        row.notional += n;
        if (!row.slug && t.slug) row.slug = t.slug;
      } else {
        byMarket.set(key, { key, title: t.title, slug: t.slug, count: 1, notional: n });
      }
    }
    return [...byMarket.values()].sort((a, b) => b.notional - a.notional).slice(0, 10);
  }, [trades]);

  const stamp = updatedAt ? clock(new Date(updatedAt)) : "--:--:--";

  return (
    <motion.div
      className="flex h-full min-h-0 flex-col gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      {/* ── Toolbar ─────────────────────────────────────────────────────── */}
      <motion.div
        variants={panelVariants}
        className="flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border border-edge bg-surface px-1.5 py-1"
      >
        <span className="text-[10px] tracking-wide text-info uppercase">Size</span>
        <div className="flex items-center gap-1">
          {SIZE_FILTERS.map((f) => (
            <Chip key={f.label} active={min === f.min} onClick={() => setMin(f.min)}>
              {f.label}
            </Chip>
          ))}
        </div>

        <span className="ml-2 text-[10px] tracking-wide text-info uppercase">Side</span>
        <div className="flex items-center gap-1">
          {SIDE_FILTERS.map((s) => (
            <Chip key={s} active={side === s} onClick={() => setSide(s)}>
              {s}
            </Chip>
          ))}
        </div>

        <span className="ml-auto text-[10px] text-muted">
          {min === 0 ? "no size floor" : `notional ≥ ${usd(min)}`}
        </span>
      </motion.div>

      {/* ── Tape + rail ─────────────────────────────────────────────────── */}
      <motion.div variants={staggerContainer} className="flex min-h-0 flex-1 gap-2">
        <Panel
          title="Consolidated Tape"
          flush
          className="min-w-0 flex-1"
          animate
          right={`${filtered.length} prints · updated ${stamp}`}
        >
          {loading ? (
            <Loading text="loading tape" />
          ) : error && trades.length === 0 ? (
            <div className="p-1.5">
              <ErrorBox message={error} />
            </div>
          ) : (
            <TradeTape trades={filtered} showMarket />
          )}
        </Panel>

        <motion.aside
          variants={panelVariants}
          className="hidden w-[260px] shrink-0 flex-col gap-2 lg:flex"
        >
          <Panel title="Flow" className="shrink-0" right={`${flow.count}`}>
            <Field label="Total notional" value={usd(flow.total)} tone="text-accent" />
            <Field label="Buy notional" value={usd(flow.buy)} tone="text-up" />
            <Field label="Sell notional" value={usd(flow.sell)} tone="text-down" />

            <div className="flex h-[6px] w-full border border-edge">
              <div className="bg-up" style={{ width: `${flow.buyPct}%` }} />
              <div className="bg-down" style={{ width: `${100 - flow.buyPct}%` }} />
            </div>
            <div className="flex items-baseline justify-between pt-[2px] pb-1 text-[10px]">
              <span className="text-up">{flow.buyPct.toFixed(1)}% BUY</span>
              <span className="text-down">{(100 - flow.buyPct).toFixed(1)}% SELL</span>
            </div>

            <Field label="Prints" value={compact(flow.count)} />
            <Field label="Avg print" value={usd(flow.avg)} />
            <Field
              label="Largest print"
              value={
                <span title={flow.largestTitle}>{flow.largest > 0 ? usd(flow.largest) : "--"}</span>
              }
              tone="text-accent"
            />
          </Panel>

          <Panel title="Most Active" flush className="min-h-0 flex-1">
            {active.length === 0 ? (
              <Empty text="no prints" />
            ) : (
              <div className="text-tiny">
                <div className="sticky top-0 z-10 flex items-center gap-1 border-b border-edge bg-surface-2 px-1 py-[2px] text-[10px] tracking-wide text-accent-weak uppercase">
                  <span className="min-w-0 flex-1">Market</span>
                  <span className="w-[26px] shrink-0 text-right">N</span>
                  <span className="w-[52px] shrink-0 text-right">Notional</span>
                </div>
                {active.map((r) => (
                  <div
                    key={r.key}
                    onClick={() => {
                      if (r.slug) go({ fn: "DES", slug: r.slug, kind: "market" }, `DES ${r.slug}`);
                    }}
                    role={r.slug ? "button" : undefined}
                    tabIndex={r.slug ? 0 : undefined}
                    onKeyDown={(e) => {
                      if ((e.key === "Enter" || e.key === " ") && r.slug) {
                        e.preventDefault();
                        go({ fn: "DES", slug: r.slug, kind: "market" }, `DES ${r.slug}`);
                      }
                    }}
                    className={`flex items-center gap-1 border-b border-edge/30 px-1 py-[2px] hover:bg-surface-2 ${
                      r.slug ? "cursor-pointer" : ""
                    }`}
                  >
                    <span className="min-w-0 flex-1 truncate text-ink/85" title={r.title}>
                      {truncate(r.title, 30)}
                    </span>
                    <span className="w-[26px] shrink-0 text-right text-muted">{r.count}</span>
                    <span className="w-[52px] shrink-0 text-right text-accent">
                      ${compact(r.notional)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </motion.aside>
      </motion.div>
    </motion.div>
  );
}

/** Toolbar toggle. Active reads as amber-on-black with an amber rule. */
function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <motion.button
      type="button"
      whileTap={tapScale}
      onClick={onClick}
      className={`border px-1.5 py-[1px] text-[10px] tracking-wide uppercase ${
        active
          ? "border-accent bg-accent/8 font-medium text-accent"
          : "border-edge bg-surface-2 text-muted hover:border-edge-strong hover:text-ink"
      }`}
    >
      {children}
    </motion.button>
  );
}
