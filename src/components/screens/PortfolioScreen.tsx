"use client";

import { motion } from "motion/react";
import { useCallback, useMemo, useState } from "react";
import { TradeTape } from "@/components/TradeTape";
import { useTerminal } from "@/components/TerminalProvider";
import { useWallet } from "@/hooks/useWallet";
import { Empty, ErrorBox, Loading, Panel } from "@/components/ui/Panel";
import { cn } from "@/lib/cn";
import { copyToClipboard } from "@/lib/clipboard";
import { usePoll } from "@/hooks/usePoll";
import { cents, clock, compact, dirClass, signed, truncate, usd } from "@/lib/format";
import { panelVariants, staggerContainer, tapScale } from "@/lib/motion";
import type { Position, Trade } from "@/lib/types";

const POLL_MS = 15_000;

/** Numeric columns sort by value; the two text columns sort lexically. */
type SortKey =
  | "title"
  | "outcome"
  | "size"
  | "avgPrice"
  | "curPrice"
  | "value"
  | "cashPnl"
  | "percentPnl";

type Sort = { key: SortKey; dir: "asc" | "desc" };

const COLUMNS: {
  key: SortKey;
  label: string;
  /** Layout classes; the market column flexes, the rest are fixed gutters. */
  width: string;
  title: string;
}[] = [
  // Headers render as <button>, which centres text by default — every column
  // has to state its alignment or it won't sit over its own values.
  { key: "title", label: "Market", width: "min-w-0 flex-1 text-left", title: "Market question" },
  { key: "outcome", label: "Outcome", width: "w-[76px] shrink-0 text-left", title: "Outcome held" },
  { key: "size", label: "Shares", width: "w-[64px] shrink-0 text-right", title: "Share count" },
  {
    key: "avgPrice",
    label: "Avg",
    width: "w-[46px] shrink-0 text-right",
    title: "Average entry, in cents",
  },
  {
    key: "curPrice",
    label: "Last",
    width: "w-[46px] shrink-0 text-right",
    title: "Current mark, in cents",
  },
  {
    key: "value",
    label: "Value",
    width: "w-[62px] shrink-0 text-right",
    title: "Mark-to-market value",
  },
  {
    key: "cashPnl",
    label: "P&L",
    width: "w-[62px] shrink-0 text-right",
    title: "Unrealised P&L in dollars",
  },
  {
    key: "percentPnl",
    label: "P&L%",
    width: "w-[56px] shrink-0 text-right",
    title: "Unrealised P&L in percent",
  },
];

/**
 * Wallet blotter: aggregates on top, the position book in the middle, that
 * wallet's own prints at the bottom. Everything is derived from the two feeds,
 * so a bad address surfaces as one error box rather than three empty panels.
 */
export default function PortfolioScreen({ user }: { user: string }) {
  const { go, toast } = useTerminal();
  const { isMe } = useWallet();
  const isOwn = isMe(user);
  const [sort, setSort] = useState<Sort>({ key: "value", dir: "desc" });

  const positions = usePoll<Position[]>(
    `/api/positions?user=${encodeURIComponent(user)}&limit=100`,
    POLL_MS
  );
  const trades = usePoll<Trade[]>(`/api/trades?user=${encodeURIComponent(user)}&limit=40`, POLL_MS);

  const rows = useMemo(() => positions.data ?? [], [positions.data]);

  const totals = useMemo(() => {
    let value = 0;
    let cashPnl = 0;
    let cost = 0;
    let realized = 0;
    let redeemable = 0;
    for (const p of rows) {
      value += p.value;
      cashPnl += p.cashPnl;
      cost += p.size * p.avgPrice;
      realized += p.realizedPnl;
      if (p.redeemable) redeemable += 1;
    }
    return {
      value,
      cashPnl,
      cost,
      realized,
      redeemable,
      // Return on capital actually deployed, not on current mark.
      returnPct: cost > 0 ? (cashPnl / cost) * 100 : 0,
    };
  }, [rows]);

  const sorted = useMemo(() => {
    const { key, dir } = sort;
    const mul = dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      switch (key) {
        case "title":
          return mul * a.title.localeCompare(b.title);
        case "outcome":
          return mul * a.outcome.localeCompare(b.outcome);
        default:
          return mul * (a[key] - b[key]);
      }
    });
  }, [rows, sort]);

  const toggleSort = useCallback((key: SortKey) => {
    setSort((s) =>
      s.key === key
        ? { key, dir: s.dir === "desc" ? "asc" : "desc" }
        : // A fresh numeric column is far more useful largest-first.
          { key, dir: key === "title" || key === "outcome" ? "asc" : "desc" }
    );
  }, []);

  const copyAddress = useCallback(() => copyToClipboard(user, toast), [user, toast]);

  return (
    <motion.div
      className="flex h-full min-h-0 flex-col gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      <motion.div
        variants={panelVariants}
        className="flex shrink-0 items-center gap-2 border border-edge bg-surface px-1.5 py-[3px]"
      >
        <span className="shrink-0 text-[10px] tracking-wide text-info uppercase">Wallet</span>
        {isOwn && (
          <span
            title="This is your connected Phantom wallet"
            className="shrink-0 border border-accent-weak px-1.5 py-[1px] text-[10px] tracking-wide text-accent uppercase"
          >
            You
          </span>
        )}
        <span className="mono min-w-0 truncate text-tiny text-ink" title={user}>
          {user}
        </span>
        <motion.button
          whileTap={tapScale}
          onClick={copyAddress}
          title="Copy address to clipboard"
          className="shrink-0 border border-edge-strong px-1.5 py-[1px] text-[10px] tracking-wide text-muted uppercase hover:border-accent-weak hover:text-accent"
        >
          Copy
        </motion.button>
        <span className="ml-auto shrink-0 text-[10px] text-faint">
          {positions.updatedAt ? `UPD ${clock(new Date(positions.updatedAt))}` : "--"}
        </span>
      </motion.div>

      <motion.div
        variants={panelVariants}
        className="grid shrink-0 grid-cols-3 gap-2 md:grid-cols-6"
      >
        <Stat label="Market Value" value={usd(totals.value)} />
        <Stat label="Unrealised P&L" value={usd(totals.cashPnl)} tone={dirClass(totals.cashPnl)} />
        <Stat
          label="Return"
          value={`${signed(totals.returnPct)}%`}
          tone={dirClass(totals.returnPct)}
        />
        <Stat label="Realised P&L" value={usd(totals.realized)} tone={dirClass(totals.realized)} />
        <Stat label="Open Pos" value={compact(rows.length)} />
        <Stat
          label="Redeemable"
          value={compact(totals.redeemable)}
          tone={totals.redeemable > 0 ? "text-accent" : "text-ink"}
        />
      </motion.div>

      <Panel
        title="Open Positions"
        right={`${sorted.length} pos`}
        className="min-h-0 flex-1"
        flush
        animate
      >
        {positions.error ? (
          <div className="p-1.5">
            <ErrorBox message={positions.error} />
          </div>
        ) : positions.loading ? (
          <Loading text="loading positions" />
        ) : sorted.length === 0 ? (
          <Empty text="no open positions" />
        ) : (
          <div className="min-w-[760px] text-tiny">
            <div className="sticky top-0 z-10 flex items-center gap-1 border-b border-edge-strong bg-surface-2 px-1 py-[3px] text-[10px] tracking-wide text-accent-weak uppercase">
              {COLUMNS.map((c) => (
                <motion.button
                  key={c.key}
                  whileTap={tapScale}
                  onClick={() => toggleSort(c.key)}
                  title={c.title}
                  className={`${c.width} truncate uppercase hover:text-accent ${
                    sort.key === c.key ? "text-accent" : ""
                  }`}
                >
                  {c.label}
                  {sort.key === c.key ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
                </motion.button>
              ))}
              <span className="w-[52px] shrink-0" />
            </div>

            {sorted.map((p) => {
              const open = p.slug
                ? () => go({ fn: "DES", slug: p.slug!, kind: "market" }, `DES ${p.slug}`)
                : undefined;
              return (
                <div
                  key={p.asset || `${p.conditionId}-${p.outcome}`}
                  onClick={open}
                  role={open ? "button" : undefined}
                  tabIndex={open ? 0 : undefined}
                  onKeyDown={
                    open
                      ? (e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            open();
                          }
                        }
                      : undefined
                  }
                  className={cn(
                    "flex items-center gap-1 border-b border-edge/40 px-1 py-[2px] hover:bg-surface-2",
                    p.slug && "cursor-pointer"
                  )}
                >
                  <span className="min-w-0 flex-1 truncate text-ink" title={p.title}>
                    {truncate(p.title, 60)}
                  </span>
                  <span className="w-[76px] shrink-0 truncate text-info" title={p.outcome}>
                    {p.outcome}
                  </span>
                  <span className="w-[64px] shrink-0 text-right text-muted">{compact(p.size)}</span>
                  <span className="w-[46px] shrink-0 text-right text-muted">
                    {cents(p.avgPrice)}
                  </span>
                  <span className="w-[46px] shrink-0 text-right font-bold text-ink">
                    {cents(p.curPrice)}
                  </span>
                  <span className="w-[62px] shrink-0 text-right text-ink/85">{usd(p.value)}</span>
                  <span className={`w-[62px] shrink-0 text-right ${dirClass(p.cashPnl)}`}>
                    {usd(p.cashPnl)}
                  </span>
                  <span className={`w-[56px] shrink-0 text-right ${dirClass(p.percentPnl)}`}>
                    {signed(p.percentPnl)}
                  </span>
                  <span className="w-[52px] shrink-0 text-right">
                    {p.redeemable ? (
                      <span
                        title="Resolved — proceeds can be redeemed"
                        className="border border-accent-weak px-1.5 py-[1px] text-[10px] tracking-wide text-accent"
                      >
                        REDEEM
                      </span>
                    ) : null}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </Panel>

      <Panel
        title="Recent Activity"
        right={trades.data ? `${trades.data.length} prints` : undefined}
        className="h-[30%] min-h-[96px] shrink-0"
        flush
        animate
      >
        {trades.error ? (
          <div className="p-1.5">
            <ErrorBox message={trades.error} />
          </div>
        ) : trades.loading ? (
          <Loading text="loading tape" />
        ) : (
          <TradeTape trades={trades.data ?? []} showMarket />
        )}
      </Panel>
    </motion.div>
  );
}

/** Bordered aggregate cell: dim label over a large, optionally tinted value. */
function Stat({
  label,
  value,
  tone = "text-ink",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="flex min-w-0 flex-col border border-edge bg-surface px-1.5 py-[3px]">
      <span className="truncate text-[10px] tracking-wide text-info uppercase">{label}</span>
      <span className={`truncate text-sm2 font-bold ${tone}`}>{value}</span>
    </div>
  );
}
