"use client";

import { motion } from "motion/react";
import { useMemo, useState, type FormEvent } from "react";
import { useTerminal } from "@/components/TerminalProvider";
import { Empty, Panel } from "@/components/ui/Panel";
import { useMarketSocket } from "@/hooks/useMarketSocket";
import { cents, clock, signed, truncate } from "@/lib/format";
import { staggerContainer, tapScale } from "@/lib/motion";
import type { Alert } from "@/lib/types";

/** How close (in probability points) an alert has to be before it reads hot. */
const NEAR_POINTS = 2;

type AlertRow = {
  alert: Alert;
  /** Live mark, 0..1 — last print, else the mid of top-of-book. */
  live?: number;
  /** Points still to travel before the condition is satisfied. */
  gap?: number;
};

/**
 * ALRT — display and CRUD for price alerts.
 *
 * Firing lives in `AlertEngine`, not here: the engine must keep evaluating
 * while the user is on any other screen, so this component deliberately owns
 * no side effects beyond its own form state.
 */
export default function AlertsScreen() {
  const { alerts, addAlert, removeAlert, watchlist, toast } = useTerminal();

  const tokenIds = useMemo(() => alerts.map((a) => a.tokenId), [alerts]);
  const feed = useMarketSocket(tokenIds);

  const rows = useMemo<AlertRow[]>(() => {
    return alerts.map((alert) => {
      const q = feed.quotes.get(alert.tokenId);
      const live =
        q?.last ?? (q?.bid !== undefined && q?.ask !== undefined ? (q.bid + q.ask) / 2 : undefined);
      const gap =
        live === undefined
          ? undefined
          : (alert.op === "gte" ? alert.target - live : live - alert.target) * 100;
      return { alert, live, gap };
    });
    // The feed mutates its maps in place, so `quotes` never changes identity —
    // `version` is the only thing that can make this recompute.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [alerts, feed.quotes, feed.version]);

  return (
    <motion.div
      className="flex h-full min-h-0 flex-col gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      <Panel
        title="Price Alerts"
        flush
        className="min-h-0 flex-1"
        right={`${alerts.length} armed · feed ${feed.status}`}
        animate
      >
        {rows.length === 0 ? (
          <Empty text="no alerts armed" />
        ) : (
          <div className="min-w-[560px] text-tiny">
            <div className="sticky top-0 z-10 flex items-center gap-1 border-b border-edge-strong bg-surface-2 px-1 py-[3px] text-[10px] tracking-wide text-accent-weak uppercase">
              <span className="min-w-0 flex-1">Market</span>
              <span className="w-[62px] shrink-0 text-right">Cond</span>
              <span className="w-[52px] shrink-0 text-right">Live</span>
              <span className="w-[56px] shrink-0 text-right" title="Points to trigger">
                Dist
              </span>
              <span className="w-[62px] shrink-0 text-right">Armed</span>
              <span className="w-[70px] shrink-0 text-right">Status</span>
              <span className="w-[16px] shrink-0" />
            </div>

            {rows.map(({ alert, live, gap }) => {
              const triggered = alert.triggeredAt !== undefined;
              const near = !triggered && gap !== undefined && gap <= NEAR_POINTS;
              return (
                <div
                  key={alert.id}
                  className="flex items-center gap-1 border-b border-edge/40 px-1 py-[2px] hover:bg-surface-2"
                >
                  <span className="min-w-0 flex-1 truncate text-ink" title={alert.label}>
                    {truncate(alert.label, 56)}
                  </span>
                  <span className="w-[62px] shrink-0 text-right text-info">
                    {alert.op === "gte" ? "≥" : "≤"} {cents(alert.target)}
                  </span>
                  <span className="w-[52px] shrink-0 text-right font-bold text-ink">
                    {cents(live)}
                  </span>
                  <span
                    className={`w-[56px] shrink-0 text-right ${near ? "text-up" : "text-muted"}`}
                    title="Probability points remaining before this alert fires"
                  >
                    {gap === undefined ? "--" : signed(gap)}
                  </span>
                  <span className="w-[62px] shrink-0 text-right text-faint">
                    {clock(new Date(alert.createdAt))}
                  </span>
                  <span
                    className={`w-[70px] shrink-0 text-right ${
                      triggered ? "font-bold text-accent" : "text-muted"
                    }`}
                    title={
                      alert.triggeredAt !== undefined
                        ? `Triggered ${clock(new Date(alert.triggeredAt))}`
                        : "Waiting on the live feed"
                    }
                  >
                    {triggered ? "TRIGGERED" : "ARMED"}
                  </span>
                  <motion.button
                    type="button"
                    whileTap={tapScale}
                    onClick={() => removeAlert(alert.id)}
                    title="Disarm alert"
                    className="w-[16px] shrink-0 text-center text-faint hover:text-down"
                  >
                    ✕
                  </motion.button>
                </div>
              );
            })}
          </div>
        )}
      </Panel>

      <ArmForm watchlist={watchlist} onArm={addAlert} onError={(m) => toast(m, "error")} />
    </motion.div>
  );
}

function ArmForm({
  watchlist,
  onArm,
  onError,
}: {
  watchlist: ReturnType<typeof useTerminal>["watchlist"];
  onArm: ReturnType<typeof useTerminal>["addAlert"];
  onError: (message: string) => void;
}) {
  const [tokenId, setTokenId] = useState("");
  const [op, setOp] = useState<Alert["op"]>("gte");
  const [target, setTarget] = useState("50");

  // Falling back to the first row means the form is usable without touching the
  // select, and survives the watchlist loading in after first paint.
  const selected = watchlist.find((w) => w.tokenId === tokenId) ?? watchlist[0];

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!selected) {
      onError("watchlist is empty — nothing to arm");
      return;
    }
    const raw = target.trim();
    const value = Number(raw);
    if (!raw || !Number.isFinite(value)) {
      onError("target must be a number of cents, e.g. 62.5");
      return;
    }
    if (value < 0 || value > 100) {
      onError("target must be between 0 and 100 cents");
      return;
    }
    onArm({
      tokenId: selected.tokenId,
      marketId: selected.marketId,
      label: selected.label,
      op,
      target: value / 100,
    });
  };

  return (
    <Panel title="Arm New Alert" className="shrink-0" animate>
      {watchlist.length === 0 ? (
        <Empty
          text="No markets to alert on"
          hint="Add markets to your watchlist first — press W on any row"
        />
      ) : (
        <form onSubmit={submit} className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <label className="text-[10px] tracking-wide text-info uppercase" htmlFor="alrt-market">
            Market
          </label>
          <select
            id="alrt-market"
            value={selected?.tokenId ?? ""}
            onChange={(e) => setTokenId(e.target.value)}
            className="min-w-0 max-w-[320px] flex-1 border border-edge bg-surface-2 px-1 py-[1px] text-tiny text-ink focus:border-accent focus:outline-none"
          >
            {watchlist.map((w) => (
              <option key={w.tokenId} value={w.tokenId}>
                {truncate(w.label, 48)} · {w.tokenId.slice(0, 8)}…
              </option>
            ))}
          </select>

          <span className="text-[10px] tracking-wide text-info uppercase">Op</span>
          <div className="flex items-center gap-1">
            {(["gte", "lte"] as const).map((o) => (
              <motion.button
                key={o}
                type="button"
                whileTap={tapScale}
                onClick={() => setOp(o)}
                title={o === "gte" ? "Fire at or above target" : "Fire at or below target"}
                className={`border px-2 py-[1px] text-tiny ${
                  op === o
                    ? "border-accent bg-accent/8 font-medium text-accent"
                    : "border-edge bg-surface-2 text-muted hover:border-edge-strong hover:text-ink"
                }`}
              >
                {o === "gte" ? "≥" : "≤"}
              </motion.button>
            ))}
          </div>

          <label className="text-[10px] tracking-wide text-info uppercase" htmlFor="alrt-target">
            Target ¢
          </label>
          <input
            id="alrt-target"
            type="number"
            min={0}
            max={100}
            step={0.1}
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            className="w-[64px] border border-edge bg-surface-2 px-1 py-[1px] text-right text-tiny text-ink focus:border-accent"
          />

          <motion.button
            type="submit"
            whileTap={tapScale}
            className="rounded-sm bg-accent-soft px-2.5 py-[3px] text-[11px] font-medium text-accent-ink hover:brightness-95"
          >
            Arm
          </motion.button>
        </form>
      )}

      <p className="mt-1 border-t border-edge/60 pt-1 text-[10px] text-muted">
        Alerts are evaluated against the live websocket feed and only fire while the terminal is
        open — nothing is stored or watched server-side.
      </p>
    </Panel>
  );
}
