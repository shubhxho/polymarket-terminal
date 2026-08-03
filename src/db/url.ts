/**
 * Normalise a Postgres connection string for node-postgres.
 *
 * The deployment URL carries `sslrootcert=system` — libpq's "validate against
 * the OS trust store". The pinned `pg-connection-string@2.14.0` doesn't
 * understand that value: it treats `system` as a file path and `readFileSync`s
 * it, throwing `ENOENT: open 'system'` from the `pg.Client` constructor before
 * a socket is ever opened. That surfaced as silent `snapshot.insert_failed`
 * logs at runtime and a hung `drizzle-kit migrate`.
 *
 * Dropping just that one parameter fixes both: `sslmode=verify-full` still
 * validates the server certificate (and hostname) against Node's bundled CA
 * set, which the managed Postgres cert chains to — verified connecting in
 * ~250ms with the param removed.
 */
export function pgConnectionUrl(raw: string): string {
  try {
    const u = new URL(raw);
    if (u.searchParams.get("sslrootcert") === "system") {
      u.searchParams.delete("sslrootcert");
    }
    return u.toString();
  } catch {
    // Not a parseable URL (e.g. a bare socket path) — hand it through as-is.
    return raw;
  }
}
