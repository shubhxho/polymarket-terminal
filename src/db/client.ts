/**
 * Lazily-constructed Drizzle client (PostgreSQL, node-postgres).
 *
 * The whole database layer is **optional**: with no `DATABASE_URL` set, `db()`
 * returns `null` and every consumer (the logger, the snapshot writer) becomes a
 * silent no-op. That is deliberate — the app builds, the CI gate runs and the
 * current deploy works with no database at all, and the moment the env var is
 * added persistence just switches on with no code change.
 */

import { drizzle, type NodePgDatabase } from "drizzle-orm/node-postgres";
import * as schema from "./schema";

export type Db = NodePgDatabase<typeof schema>;

let cached: Db | null | undefined;

/** The Drizzle client, or `null` when `DATABASE_URL` is unset. Memoised. */
export function db(): Db | null {
  if (cached !== undefined) return cached;
  const url = process.env.DATABASE_URL;
  cached = url ? drizzle(url, { schema }) : null;
  return cached;
}

/** Whether a database is configured — cheap check without building a client. */
export const dbEnabled = (): boolean => Boolean(process.env.DATABASE_URL);
