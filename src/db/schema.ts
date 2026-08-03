/**
 * Drizzle schema (PostgreSQL).
 *
 * The terminal itself is read-only over live Polymarket data, so the database
 * exists for the two things worth keeping across time:
 *
 *   - `signalSnapshots` — one row per scan of `/api/signals`, so the *live*
 *     signals can be reviewed and back-tested after the fact rather than
 *     evaporating on the next 20-second poll;
 *   - `eventLog` — a structured application log (see `lib/logger.ts`), so
 *     "log everything" has somewhere to land.
 */

import {
  bigserial,
  doublePrecision,
  index,
  integer,
  jsonb,
  pgTable,
  timestamp,
  uniqueIndex,
  varchar,
} from "drizzle-orm/pg-core";

export const signalSnapshots = pgTable(
  "signal_snapshots",
  {
    id: bigserial("id", { mode: "number" }).primaryKey(),
    capturedAt: timestamp("captured_at", { withTimezone: true }).notNull().defaultNow(),
    scanned: integer("scanned").notNull(),
    flagged: integer("flagged").notNull(),
    modeled: integer("modeled").notNull(),
    confirms: integer("confirms").notNull(),
    conflicts: integer("conflicts").notNull(),
    bullish: integer("bullish").notNull(),
    bearish: integer("bearish").notNull(),
    arbs: integer("arbs").notNull(),
    drifts: integer("drifts").notNull(),
    blockNotional: doublePrecision("block_notional").notNull(),
    topMarketId: varchar("top_market_id", { length: 128 }),
    topHeat: integer("top_heat"),
    /** Compact top-N rows (id, question, heat, bias, model prob) for backtesting. */
    markets: jsonb("markets"),
    /**
     * `capturedAt` floored to the minute, set at insert time. A unique index on
     * it makes the write path idempotent per minute: every `/api/signals` scan
     * attempts an insert, but at most one row per minute survives
     * (`onConflictDoNothing`). Without it each edge revalidation (~4/min at
     * s-maxage=15, more across instances) wrote a near-duplicate row, bloating
     * the table and giving the backtest a ragged, uneven series.
     *
     * A plain column, not a Postgres generated one: `date_trunc('minute', …)`
     * over a `timestamptz` is not IMMUTABLE, so it can't back a stored generated
     * column. Computing the bucket in the writer sidesteps that.
     */
    minuteBucket: timestamp("minute_bucket", { withTimezone: true }).notNull(),
  },
  (t) => [
    index("captured_at_idx").on(t.capturedAt),
    uniqueIndex("snapshot_minute_uq").on(t.minuteBucket),
  ]
);

export const eventLog = pgTable(
  "event_log",
  {
    id: bigserial("id", { mode: "number" }).primaryKey(),
    at: timestamp("at", { withTimezone: true }).notNull().defaultNow(),
    level: varchar("level", { length: 16 }).notNull(),
    event: varchar("event", { length: 160 }).notNull(),
    context: jsonb("context"),
  },
  (t) => [index("event_idx").on(t.event), index("at_idx").on(t.at)]
);

export type SignalSnapshotRow = typeof signalSnapshots.$inferInsert;
export type EventLogRow = typeof eventLog.$inferInsert;
