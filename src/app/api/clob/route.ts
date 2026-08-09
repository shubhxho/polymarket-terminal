import { fail } from "@/lib/api-util";

/**
 * Thin server-side relay to the Polymarket CLOB.
 *
 * The CLOB's authenticated endpoints (`/auth/*`, `/order`) reject cross-origin
 * browser calls, and even the readable ones would need custom `POLY_*` headers
 * that trip CORS preflight. So the order client (`lib/clob.ts`) never talks to
 * the CLOB directly — it hands us the fully-formed request (path, method,
 * headers, body) and we forward it from the server, where CORS doesn't apply.
 *
 * The L2 request signature is an HMAC the client already computed over the
 * CLOB path and body, so we must forward both *verbatim* — no rewriting the
 * body or the path, or the signature stops matching. The secret itself never
 * reaches us; only the derived HMAC does.
 *
 * Unlike the read routes this is a bare passthrough — no cached `ok()`
 * envelope — because it carries live auth and must never be shared between
 * users or reused from cache.
 */

const CLOB = "https://clob.polymarket.com";

/** Only relay the endpoints the order flow actually uses. */
const ALLOWED = /^\/(auth\/|order$|orders$|order\/|data\/)/;

type RelayRequest = {
  path?: string;
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export async function POST(req: Request) {
  let relay: RelayRequest;
  try {
    relay = (await req.json()) as RelayRequest;
  } catch {
    return fail("invalid relay envelope", 400);
  }

  const path = relay.path ?? "";
  const method = (relay.method ?? "GET").toUpperCase();
  if (!path.startsWith("/") || !ALLOWED.test(path)) {
    return fail(`path not allowed: ${path}`, 400);
  }
  if (method !== "GET" && method !== "POST" && method !== "DELETE") {
    return fail(`method not allowed: ${method}`, 400);
  }

  try {
    const upstream = await fetch(`${CLOB}${path}`, {
      method,
      headers: { accept: "application/json", ...relay.headers },
      body: method === "GET" ? undefined : relay.body,
      cache: "no-store",
    });
    // Pass the CLOB's status and payload straight back so the client can read
    // its error messages unchanged.
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") ?? "application/json",
        "cache-control": "no-store",
      },
    });
  } catch (err) {
    return fail(err);
  }
}
