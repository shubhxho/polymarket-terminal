/**
 * Structured logging — console always, database when one is configured.
 *
 * Every call prints to the server console (so logs show up in Vercel/`bun` runs
 * regardless) and, when `DATABASE_URL` is set, also lands a row in `event_log`.
 * The database write is fire-and-forget and fully swallowed: logging must never
 * throw into, slow, or fail the request it is describing.
 */

import { db } from "@/db/client";
import { eventLog } from "@/db/schema";

export type LogLevel = "info" | "warn" | "error";

export function log(level: LogLevel, event: string, context?: unknown): void {
  const tag = `[${level}] ${event}`;
  // This module IS the logger — the console is the point here.
  /* eslint-disable no-console */
  if (level === "error") console.error(tag, context ?? "");
  else if (level === "warn") console.warn(tag, context ?? "");
  else console.log(tag, context ?? "");
  /* eslint-enable no-console */

  // Fire-and-forget: resolve the (memoised) client, then write. `log` stays
  // synchronous for its hot-path callers — the database round-trip is detached.
  void db().then((d) => {
    if (!d) return;
    void d
      .insert(eventLog)
      .values({ level, event: event.slice(0, 160), context: context ?? null })
      .catch(() => {
        // A logging failure is never allowed to surface.
      });
  });
}

export const logInfo = (event: string, context?: unknown) => log("info", event, context);
export const logWarn = (event: string, context?: unknown) => log("warn", event, context);
export const logError = (event: string, context?: unknown) => log("error", event, context);
