/**
 * Polymarket CLOB order path — signing + submission, hand-rolled.
 *
 * The read side of the exchange lives in `polymarket.ts` (books, prices, the
 * public websocket). This is the *write* side: turning a price/size ticket into
 * a signed order the CLOB will accept. There is no `@polymarket/clob-client`
 * here on purpose — the whole terminal keeps its dependency surface tiny, and
 * everything the order flow needs is reachable without one:
 *
 *   - the wallet signs both EIP-712 payloads (`eth_signTypedData_v4`), so we
 *     never hash a struct ourselves — we only assemble the typed-data JSON;
 *   - the L2 request signature is a plain HMAC-SHA256, which the browser's
 *     Web Crypto does natively;
 *   - CORS on the CLOB is dodged by relaying every call through our own
 *     `/api/clob` route (see `app/api/clob/route.ts`).
 *
 * Two EIP-712 payloads are involved. First an **L1 attestation** ("I control
 * this wallet") that mints a per-address API key. Then the **order** itself,
 * signed against the CTF Exchange contract. The API key drives the **L2** HMAC
 * headers on the actual `POST /order`.
 *
 * Account model: this assumes funds sit in a Polymarket **proxy wallet** (the
 * polymarket.com default). For a browser wallet like Phantom that proxy is a
 * Gnosis Safe, so `signatureType` defaults to 2 and `maker` is the Safe
 * address (the "funder"), while `signer` stays the connected EOA.
 */

/** Polygon — the chain Polymarket clears on. */
const CHAIN_ID = 137;

/** CTF Exchange contracts on Polygon. Neg-risk markets clear on their own. */
const EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E";
const NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a";
/** Both exchanges share this EIP-712 domain name; only the address differs. */
const EXCHANGE_DOMAIN_NAME = "Polymarket CTF Exchange";

const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000";

/** Signature schemes the exchange understands. */
export const SIG_TYPE = {
  /** EOA signs and is itself the maker. */
  EOA: 0,
  /** Email/magic proxy wallet. */
  POLY_PROXY: 1,
  /** Browser-wallet Gnosis Safe proxy — Phantom/MetaMask on polymarket.com. */
  POLY_GNOSIS_SAFE: 2,
} as const;

export type SignatureType = (typeof SIG_TYPE)[keyof typeof SIG_TYPE];

export type Side = "BUY" | "SELL";
export type OrderType = "GTC" | "FOK" | "GTD";

export type ApiCreds = { apiKey: string; secret: string; passphrase: string };

type SignFn = (typedData: unknown) => Promise<string>;

// ── typed-data domain declarations ──────────────────────────────────────────

/** Domain type *without* verifyingContract — the L1 auth domain omits it. */
const EIP712_DOMAIN_NO_CONTRACT = [
  { name: "name", type: "string" },
  { name: "version", type: "string" },
  { name: "chainId", type: "uint256" },
] as const;

/** Domain type for the exchange order (carries the verifying contract). */
const EIP712_DOMAIN = [
  { name: "name", type: "string" },
  { name: "version", type: "string" },
  { name: "chainId", type: "uint256" },
  { name: "verifyingContract", type: "address" },
] as const;

const CLOB_AUTH_TYPES = [
  { name: "address", type: "address" },
  { name: "timestamp", type: "string" },
  { name: "nonce", type: "uint256" },
  { name: "message", type: "string" },
] as const;

const ORDER_TYPES = [
  { name: "salt", type: "uint256" },
  { name: "maker", type: "address" },
  { name: "signer", type: "address" },
  { name: "taker", type: "address" },
  { name: "tokenId", type: "uint256" },
  { name: "makerAmount", type: "uint256" },
  { name: "takerAmount", type: "uint256" },
  { name: "expiration", type: "uint256" },
  { name: "nonce", type: "uint256" },
  { name: "feeRateBps", type: "uint256" },
  { name: "side", type: "uint8" },
  { name: "signatureType", type: "uint8" },
] as const;

// ── small numeric helpers ───────────────────────────────────────────────────

/** Round a price onto the market's tick grid and clamp inside (0, 1). */
export function roundToTick(price: number, tick: number): number {
  const t = tick > 0 ? tick : 0.001;
  const snapped = Math.round(price / t) * t;
  const clamped = Math.min(1 - t, Math.max(t, snapped));
  // Kill binary float dust so "0.63" doesn't sign as "0.6300000000000001".
  const decimals = Math.max(0, Math.ceil(-Math.log10(t)));
  return Number(clamped.toFixed(decimals));
}

/** USDC and outcome shares are both 6-decimal on Polymarket. */
const DECIMALS = 1_000_000;
const toBase = (x: number): string => String(Math.round(x * DECIMALS));

/**
 * Maker/taker amounts in 6-decimal base units.
 *
 * BUY: you put up USDC (maker) to receive shares (taker).
 * SELL: you put up shares (maker) to receive USDC (taker).
 */
function amounts(
  side: Side,
  price: number,
  size: number
): {
  makerAmount: string;
  takerAmount: string;
} {
  const shares = toBase(size);
  const usdc = toBase(price * size);
  return side === "BUY"
    ? { makerAmount: usdc, takerAmount: shares }
    : { makerAmount: shares, takerAmount: usdc };
}

/**
 * Random 52-bit salt.
 *
 * Kept under 2^53 so it survives as a JS number both in the signed struct
 * (stringified) and in the POST body (numeric) — the two must be the exact
 * same integer or the signature won't verify.
 */
function randomSalt(): number {
  const r = new Uint32Array(2);
  crypto.getRandomValues(r);
  const HALF = 67_108_864; // 2^26
  return (r[0] % HALF) * HALF + (r[1] % HALF);
}

// ── order construction ──────────────────────────────────────────────────────

export type OrderDraft = {
  side: Side;
  tokenId: string;
  /** 0..1, already snapped to the market tick. */
  price: number;
  /** Number of outcome shares. */
  size: number;
  /** Proxy/Safe wallet that holds the funds — the order's `maker`. */
  funder: string;
  /** Connected EOA that actually signs — the order's `signer`. */
  signer: string;
  negRisk: boolean;
  signatureType: SignatureType;
  /** Unix seconds; only meaningful for a GTD order. */
  expiration?: number;
};

/** The order object as it appears in the POST body (side as a string). */
export type OrderPost = {
  salt: number;
  maker: string;
  signer: string;
  taker: string;
  tokenId: string;
  makerAmount: string;
  takerAmount: string;
  expiration: string;
  nonce: string;
  feeRateBps: string;
  side: Side;
  signatureType: SignatureType;
};

export type BuiltOrder = {
  /** Full EIP-712 payload handed to the wallet. */
  typedData: unknown;
  /** POST-shaped order, still unsigned. */
  post: OrderPost;
  /** Derived economics, for the preview panel. */
  notional: number;
  makerAmount: string;
  takerAmount: string;
};

/**
 * Assemble the order two ways at once: the typed-data the wallet signs (with a
 * numeric `side` uint8) and the POST object the CLOB ingests (with a string
 * `side`). Both carry the identical salt/amounts so the signature stays valid.
 */
export function buildOrder(d: OrderDraft): BuiltOrder {
  const salt = randomSalt();
  const { makerAmount, takerAmount } = amounts(d.side, d.price, d.size);
  const expiration = d.expiration ? String(d.expiration) : "0";

  const shared = {
    salt: String(salt),
    maker: d.funder,
    signer: d.signer,
    taker: ZERO_ADDRESS,
    tokenId: d.tokenId,
    makerAmount,
    takerAmount,
    expiration,
    nonce: "0",
    feeRateBps: "0",
  };

  const typedData = {
    types: { EIP712Domain: EIP712_DOMAIN, Order: ORDER_TYPES },
    primaryType: "Order",
    domain: {
      name: EXCHANGE_DOMAIN_NAME,
      version: "1",
      chainId: CHAIN_ID,
      verifyingContract: d.negRisk ? NEG_RISK_EXCHANGE : EXCHANGE,
    },
    message: {
      ...shared,
      side: d.side === "BUY" ? 0 : 1,
      signatureType: d.signatureType,
    },
  };

  const post: OrderPost = {
    ...shared,
    salt,
    side: d.side,
    signatureType: d.signatureType,
  };

  return {
    typedData,
    post,
    notional: d.price * d.size,
    makerAmount,
    takerAmount,
  };
}

// ── base64url (for the L2 HMAC secret + signature) ──────────────────────────

function b64urlToBytes(s: string): Uint8Array {
  const b64 = s
    .replace(/-/g, "+")
    .replace(/_/g, "/")
    .padEnd(Math.ceil(s.length / 4) * 4, "=");
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function bytesToB64url(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

// ── relay through our own route (CORS-free) ─────────────────────────────────

type ClobResponse = { ok: boolean; status: number; data: unknown };

/**
 * Every CLOB call goes through `/api/clob`, which forwards it server-side. The
 * request signature (HMAC over `timestamp + method + path + body`) is computed
 * against the *CLOB* path here, so the relay must hit that exact path — it does,
 * because we hand it the path verbatim.
 */
async function clobFetch(
  path: string,
  method: string,
  headers: Record<string, string>,
  body?: string
): Promise<ClobResponse> {
  const res = await fetch("/api/clob", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ path, method, headers, body }),
  });
  const data = await res.json().catch(() => null);
  return { ok: res.ok, status: res.status, data };
}

/** Surface the CLOB's own error text when a relayed call fails. */
function errorText(r: ClobResponse): string {
  const d = r.data as { error?: string; message?: string } | null;
  return d?.error ?? d?.message ?? `CLOB ${r.status}`;
}

// ── L1: mint / re-derive the API key ────────────────────────────────────────

const CLOB_AUTH_MESSAGE = "This message attests that I control the given wallet";

async function signClobAuth(sign: SignFn, address: string, timestamp: number): Promise<string> {
  return sign({
    types: { EIP712Domain: EIP712_DOMAIN_NO_CONTRACT, ClobAuth: CLOB_AUTH_TYPES },
    primaryType: "ClobAuth",
    domain: { name: "ClobAuthDomain", version: "1", chainId: CHAIN_ID },
    message: {
      address,
      timestamp: String(timestamp),
      nonce: 0,
      message: CLOB_AUTH_MESSAGE,
    },
  });
}

function normalizeCreds(data: unknown): ApiCreds | null {
  const d = data as Record<string, string> | null;
  if (!d) return null;
  const apiKey = d.apiKey ?? d.api_key;
  const secret = d.secret;
  const passphrase = d.passphrase;
  if (!apiKey || !secret || !passphrase) return null;
  return { apiKey, secret, passphrase };
}

/**
 * Get usable API credentials for `address`, minting them if needed.
 *
 * `derive-api-key` deterministically returns the key an address already owns;
 * if none exists yet the CLOB 404s it, so we fall back to `create-api-key`.
 * One wallet prompt covers both attempts (same attestation).
 */
export async function ensureApiCreds(sign: SignFn, address: string): Promise<ApiCreds> {
  const ts = Math.floor(Date.now() / 1000);
  const signature = await signClobAuth(sign, address, ts);
  const l1: Record<string, string> = {
    POLY_ADDRESS: address,
    POLY_SIGNATURE: signature,
    POLY_TIMESTAMP: String(ts),
    POLY_NONCE: "0",
  };

  let r = await clobFetch("/auth/derive-api-key", "GET", l1);
  let creds = r.ok ? normalizeCreds(r.data) : null;
  if (!creds) {
    r = await clobFetch("/auth/api-key", "POST", l1);
    if (!r.ok) throw new Error(`could not create API key — ${errorText(r)}`);
    creds = normalizeCreds(r.data);
  }
  if (!creds) throw new Error("CLOB returned incomplete API credentials");
  return creds;
}

// ── L2: HMAC headers for an authenticated request ───────────────────────────

async function l2Headers(
  address: string,
  creds: ApiCreds,
  method: string,
  path: string,
  body: string
): Promise<Record<string, string>> {
  const ts = String(Math.floor(Date.now() / 1000));
  const message = ts + method + path + body;
  const key = await crypto.subtle.importKey(
    "raw",
    b64urlToBytes(creds.secret) as unknown as ArrayBuffer,
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  return {
    POLY_ADDRESS: address,
    POLY_SIGNATURE: bytesToB64url(new Uint8Array(mac)),
    POLY_TIMESTAMP: ts,
    POLY_API_KEY: creds.apiKey,
    POLY_PASSPHRASE: creds.passphrase,
  };
}

// ── submit ──────────────────────────────────────────────────────────────────

export type SubmitResult = { orderId?: string; status?: string; raw: unknown };

/**
 * Sign and post an order. `signature` is produced by the wallet over the built
 * typed-data, then the whole POST body is HMAC-signed with the API key.
 */
export async function submitOrder(
  sign: SignFn,
  address: string,
  creds: ApiCreds,
  built: BuiltOrder,
  orderType: OrderType
): Promise<SubmitResult> {
  const signature = await sign(built.typedData);

  const payload = {
    order: { ...built.post, signature },
    owner: creds.apiKey,
    orderType,
  };
  const body = JSON.stringify(payload);
  const headers = {
    ...(await l2Headers(address, creds, "POST", "/order", body)),
    "content-type": "application/json",
  };

  const r = await clobFetch("/order", "POST", headers, body);
  if (!r.ok) throw new Error(errorText(r));
  const d = r.data as { orderID?: string; orderId?: string; status?: string } | null;
  return { orderId: d?.orderID ?? d?.orderId, status: d?.status, raw: r.data };
}

// ── credential cache (avoid re-signing the L1 attestation every order) ───────

const CREDS_KEY = (address: string) => `pmt.clob.creds.${address.toLowerCase()}`;

export function loadCreds(address: string): ApiCreds | null {
  try {
    const raw = window.localStorage.getItem(CREDS_KEY(address));
    return raw ? (JSON.parse(raw) as ApiCreds) : null;
  } catch {
    return null;
  }
}

export function saveCreds(address: string, creds: ApiCreds): void {
  try {
    window.localStorage.setItem(CREDS_KEY(address), JSON.stringify(creds));
  } catch {
    // Private-mode / quota — the creds just won't persist across reloads.
  }
}

export function clearCreds(address: string): void {
  try {
    window.localStorage.removeItem(CREDS_KEY(address));
  } catch {
    // ignore
  }
}

/** Remembered funder (proxy) address — the same across every market. */
const FUNDER_KEY = "pmt.clob.funder";

export function loadFunder(): string {
  try {
    return window.localStorage.getItem(FUNDER_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveFunder(funder: string): void {
  try {
    window.localStorage.setItem(FUNDER_KEY, funder);
  } catch {
    // ignore
  }
}

/** Loose 0x-address check — enough to catch a fat-fingered funder. */
export function isAddress(s: string): boolean {
  return /^0x[0-9a-fA-F]{40}$/.test(s.trim());
}
