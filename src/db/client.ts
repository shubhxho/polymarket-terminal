/**
 * Lazily-constructed Drizzle client (PostgreSQL, node-postgres).
 *
 * The whole database layer is **optional**: with no `DATABASE_URL` set, `db()`
 * returns `null` and every consumer (the logger, the snapshot writer) becomes a
 * silent no-op. That is deliberate — the app builds, the CI gate runs and the
 * current deploy works with no database at all, and the moment the env var is
 * added persistence just switches on with no code change.
 */

import type { NodePgDatabase } from "drizzle-orm/node-postgres";
import * as schema from "./schema";

export type Db = NodePgDatabase<typeof schema>;

let cached: Promise<Db | null> | undefined;

/**
 * The Drizzle client, or `null` when `DATABASE_URL` is unset. Memoised.
 *
 * `drizzle-orm/node-postgres` (and its `pg` dependency) is loaded through a
 * dynamic `import()` *inside* the URL guard, never at module scope. Without a
 * database that graph is never touched — which also sidesteps a Turbopack dev
 * bug that crashes on the ESM-default import of the CommonJS `pg` package. The
 * cost is that `db()` is async; the result is a cached promise, so every call
 * after the first resolves instantly.
 */
export function db(): Promise<Db | null> {
  return (cached ??= (async () => {
    const url = process.env.DATABASE_URL;
    if (!url) return null;
    const { drizzle } = await import("drizzle-orm/node-postgres");
    return drizzle(url, { schema });
  })());
}

/** Whether a database is configured — cheap check without building a client. */
export const dbEnabled = (): boolean => Boolean(process.env.DATABASE_URL);
