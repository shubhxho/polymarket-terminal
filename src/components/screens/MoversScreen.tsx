"use client";

import { motion } from "motion/react";
import { useMemo, useState } from "react";
import { MarketGrid, type GridColumn } from "@/components/MarketGrid";
import { AsyncBody, Panel, Refreshing } from "@/components/ui/Panel";
import { usePoll } from "@/hooks/usePoll";
import { compact } from "@/lib/format";
import { panelVariants, staggerContainer, tapScale } from "@/lib/motion";
import type { Market } from "@/lib/types";

type Timeframe = "1H" | "24H" | "1W";

const TIMEFRAMES: Timeframe[] = ["1H", "24H", "1W"];

const CHANGE_COLUMN: Record<Timeframe, GridColumn> = {
  "1H": "chg1h",
  "24H": "chg24h",
  "1W": "chg1w",
};

function changeOf(m: Market, tf: Timeframe): number | undefined {
  return tf === "1H" ? m.chg1h : tf === "24H" ? m.chg24h : m.chg1w;
}

/**
 * Liquidity floor for ranking. A market with a few hundred dollars of 24h flow
 * can print a 40-point "move" off one stale fill, which would otherwise own the
 * whole leaderboard. $5k of 24h notional is roughly where a book is quoted
 * continuously enough that a change reflects opinion rather than a single
 * clearing trade — below it the number is noise, not a mover.
 */
const MIN_VOLUME_24H = 5000;

/** Prices within 1.5 points of either bound are treated as already decided. */
const SETTLED_BAND = 0.015;

const LIST_SIZE = 25;

/**
 * Session leaders and laggards.
 *
 * One universe (the 200 most active markets) ranked two ways, so the eye can
 * read the tape's dispersion in a single glance: what is being bid up on the
 * left, what is being sold on the right, and the breadth strip above saying
 * whether the move is broad or a handful of names.
 */
export default function MoversScreen() {
  const [tf, setTf] = useState<Timeframe>("24H");
  const { data, error, loading, refreshing } = usePoll<Market[]>(
    "/api/markets?limit=200&order=volume24hr",
    10000
  );

  const { gainers, losers, advancing, declining } = useMemo(() => {
    // Splitting by sign rather than slicing head/tail of one sorted list keeps
    // the two panels disjoint when fewer than 2×LIST_SIZE markets qualify.
    const up: Market[] = [];
    const down: Market[] = [];
    for (const m of data ?? []) {
      if (m.volume24h < MIN_VOLUME_24H) continue;
      // A settled market is not a mover. Sports and event markets pin to 0 or
      // 100 the moment the result is known and then sit there for hours before
      // formal resolution, which would otherwise fill both panels with ±99
      // point "moves" nobody can trade.
      if (!m.acceptingOrders) continue;
      if (m.last <= SETTLED_BAND || m.last >= 1 - SETTLED_BAND) continue;
      const chg = changeOf(m, tf);
      if (chg === undefined || Number.isNaN(chg) || chg === 0) continue;
      (chg > 0 ? up : down).push(m);
    }
    up.sort((a, b) => (changeOf(b, tf) ?? 0) - (changeOf(a, tf) ?? 0));
    down.sort((a, b) => (changeOf(a, tf) ?? 0) - (changeOf(b, tf) ?? 0));
    return {
      gainers: up.slice(0, LIST_SIZE),
      losers: down.slice(0, LIST_SIZE),
      advancing: up.length,
      declining: down.length,
    };
  }, [data, tf]);

  const columns = useMemo<GridColumn[]>(
    () => ["last", CHANGE_COLUMN[tf], "vol24h", "liquidity", "expiry"],
    [tf]
  );

  const breadth = advancing + declining;
  const upPct = breadth === 0 ? 0 : (advancing / breadth) * 100;
  // A failed refresh keeps the last good ranking on screen; the strip says so.
  const stale = !!error && !!data;

  return (
    <motion.div
      className="flex h-full min-h-0 flex-col gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      <motion.div
        variants={panelVariants}
        className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border border-edge bg-surface px-1.5 py-[3px] text-[10px] tracking-wide uppercase"
      >
        <span className="shrink-0 text-info">Movers</span>

        <span className="flex shrink-0 items-center gap-2">
          {TIMEFRAMES.map((t) => (
            <motion.button
              key={t}
              whileTap={tapScale}
              onClick={() => setTf(t)}
              title={`Rank by change over ${t}`}
              // Toggle group: expose the selected timeframe to assistive tech,
              // not by the accent colour alone.
              aria-pressed={t === tf}
              className={`border px-1.5 py-[1px] text-[10px] tracking-wide uppercase ${
                t === tf
                  ? "border-accent bg-accent/8 text-accent"
                  : "border-edge text-muted hover:border-edge-strong hover:text-accent-weak"
              }`}
            >
              {t}
            </motion.button>
          ))}
        </span>

        <span className="flex min-w-0 items-center gap-2 text-muted">
          <span>
            <span className="text-info">ADV</span> <span className="text-up">{advancing}</span>
          </span>
          <span
            className="flex h-[6px] w-[120px] shrink-0 border border-edge"
            title="Advancing vs declining"
          >
            <span className="bg-up/70" style={{ width: `${upPct}%` }} />
            <span className="flex-1 bg-down/70" />
          </span>
          <span>
            <span className="text-info">DEC</span> <span className="text-down">{declining}</span>
          </span>
        </span>

        <span className="ml-auto flex shrink-0 items-center gap-3 text-muted">
          <span>
            <span className="text-info">UNIV</span> {breadth}
          </span>
          <span>
            <span className="text-info">MIN VOL</span> {compact(MIN_VOLUME_24H)}
          </span>
          {stale ? <span className="text-warn">stale</span> : null}
          <Refreshing show={refreshing} />
        </span>
      </motion.div>

      <motion.div
        variants={staggerContainer}
        className="grid min-h-0 flex-1 grid-cols-1 gap-2 overflow-y-auto xl:grid-cols-2 xl:overflow-visible"
      >
        <motion.div variants={panelVariants} className="flex min-h-0 min-w-0 flex-col">
          <Panel
            title={`Top Gainers · ${tf}`}
            right={`${gainers.length}`}
            flush
            className="min-h-0 flex-1"
          >
            <AsyncBody loading={loading} error={error} hasData={!!data} loadingText="ranking">
              <MarketGrid markets={gainers} columns={columns} emptyText="no advancers" />
            </AsyncBody>
          </Panel>
        </motion.div>

        <motion.div variants={panelVariants} className="flex min-h-0 min-w-0 flex-col">
          <Panel
            title={`Top Losers · ${tf}`}
            right={`${losers.length}`}
            flush
            className="min-h-0 flex-1"
          >
            <AsyncBody loading={loading} error={error} hasData={!!data} loadingText="ranking">
              <MarketGrid markets={losers} columns={columns} emptyText="no decliners" />
            </AsyncBody>
          </Panel>
        </motion.div>
      </motion.div>
    </motion.div>
  );
}
