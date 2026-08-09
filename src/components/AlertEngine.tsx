"use client";

import { useEffect, useMemo, useRef } from "react";
import { useMarketSocket } from "@/hooks/useMarketSocket";
import { useTerminal } from "@/components/TerminalProvider";
import { cents } from "@/lib/format";

/**
 * Evaluates armed price alerts against the live websocket feed.
 *
 * Mounted once in the shell so alerts fire on any screen. Deliberately
 * *crossing*-based rather than level-based: an alert armed at ≥60¢ on a market
 * already trading at 70¢ would otherwise fire instantly and uselessly, so the
 * first observed price only establishes a baseline.
 */
export function AlertEngine() {
  const { alerts, markAlertTriggered, toast } = useTerminal();

  const armed = useMemo(() => alerts.filter((a) => !a.triggeredAt), [alerts]);
  const tokenIds = useMemo(() => armed.map((a) => a.tokenId), [armed]);
  const feed = useMarketSocket(tokenIds, tokenIds.length > 0);

  // Last price seen per token, so we can detect the moment of crossing.
  const seen = useRef(new Map<string, number>());

  useEffect(() => {
    if (armed.length === 0) return;

    // Drop baselines for tokens that are no longer armed (their alert fired or was
    // removed). Otherwise a re-armed alert on the same token would find a stale
    // `prev` and could fire instantly off a minutes-old price — the very
    // instant-fire this crossing design exists to prevent — and the map would grow
    // unbounded as tokens come and go.
    const armedTokens = new Set(armed.map((a) => a.tokenId));
    for (const k of seen.current.keys()) {
      if (!armedTokens.has(k)) seen.current.delete(k);
    }

    for (const alert of armed) {
      const q = feed.quotes.get(alert.tokenId);
      const price =
        q?.last ?? (q?.bid !== undefined && q?.ask !== undefined ? (q.bid + q.ask) / 2 : undefined);
      if (price === undefined) continue;

      const prev = seen.current.get(alert.tokenId);
      seen.current.set(alert.tokenId, price);
      if (prev === undefined) continue;

      const crossed =
        alert.op === "gte"
          ? prev < alert.target && price >= alert.target
          : prev > alert.target && price <= alert.target;
      if (!crossed) continue;

      const text = `ALERT · ${alert.label} ${alert.op === "gte" ? "≥" : "≤"} ${cents(
        alert.target
      )}¢ (now ${cents(price)}¢)`;
      markAlertTriggered(alert.id);
      toast(text, "warn");

      // Best-effort desktop notification; silently skipped when not granted.
      if (typeof Notification !== "undefined" && Notification.permission === "granted") {
        new Notification("Polymarket Terminal", { body: text });
      }
    }
    // `feed.version` is the render trigger — the quotes map mutates in place.
  }, [feed.version, feed.quotes, armed, markAlertTriggered, toast]);

  // Ask for notification permission the first time an alert is armed.
  useEffect(() => {
    if (armed.length === 0) return;
    if (typeof Notification === "undefined") return;
    if (Notification.permission === "default") void Notification.requestPermission();
  }, [armed.length]);

  return null;
}
