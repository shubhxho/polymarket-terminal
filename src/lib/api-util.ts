/** Uniform envelope so client hooks can treat every endpoint the same.
 *
 *  Built on the Web-standard `Response.json()` static method — the same WHATWG
 *  surface Bun implements natively and Next 16 documents for route handlers —
 *  rather than the `next/server` wrapper, so the API layer carries no framework
 *  coupling and would run unchanged on Bun, Node, Deno or the edge. */
export function ok<T>(data: T, sMaxAge = 5): Response {
  return Response.json(
    { ok: true, data, ts: Date.now() },
    {
      headers: {
        "cache-control": `public, s-maxage=${sMaxAge}, stale-while-revalidate=${sMaxAge * 6}`,
      },
    }
  );
}

export function fail(err: unknown, status = 502): Response {
  const message = err instanceof Error ? err.message : String(err);
  return Response.json({ ok: false, error: message, ts: Date.now() }, { status });
}

/** Clamp a user-supplied `limit` so a stray query can't ask for 10k rows. */
export function limitOf(v: string | null, fallback: number, max: number): number {
  const n = v ? parseInt(v, 10) : NaN;
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.min(n, max);
}
