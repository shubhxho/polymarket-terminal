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

export type Envelope<T> = { ok: boolean; data?: T; error?: string; ts?: number };

/**
 * Read an `ok()`-shaped JSON envelope from a fetch `Response`, or throw a
 * legible error.
 *
 * The guard before `res.json()` matters: a redirect to an HTML page — a Vercel
 * auth wall on a `*.vercel.app` preview, a 5xx error page, an offline captive
 * portal — would otherwise surface as an opaque "Unexpected token '<'" from
 * `JSON.parse`. This turns that into a message that says what actually happened.
 */
export async function readEnvelope<T>(res: Response): Promise<T | null> {
  const contentType = res.headers.get("content-type") ?? "";
  if (!res.ok || !contentType.includes("application/json")) {
    if (res.redirected || contentType.includes("text/html")) {
      throw new Error(
        "feed blocked — this deployment is behind Vercel authentication; open the app's real domain or sign in"
      );
    }
    throw new Error(`HTTP ${res.status} — non-JSON response`);
  }
  const json = (await res.json()) as Envelope<T>;
  if (!json.ok) throw new Error(json.error ?? `HTTP ${res.status}`);
  return (json.data ?? null) as T | null;
}

/** Clamp a user-supplied `limit` so a stray query can't ask for 10k rows. */
export function limitOf(v: string | null, fallback: number, max: number): number {
  const n = v ? parseInt(v, 10) : NaN;
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.min(n, max);
}
