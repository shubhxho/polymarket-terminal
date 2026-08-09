/**
 * Peer-to-peer signal mesh — the wire protocol and merge logic.
 *
 * Terminals share the signals they computed locally over WebRTC data channels
 * (see `useSignalMesh`), so a desk of traders sees one another's model reads and
 * can tell when a call is a lone opinion or a consensus. This module is the pure
 * half: how a share is encoded, validated on the way in, merged across peers
 * with a freshness window, and reduced to a per-market consensus. No WebRTC, no
 * DOM, no clock of its own — `now` is always passed in — so it is fully testable
 * and the transport can be swapped without touching any of it.
 *
 * Everything crossing the wire is untrusted: a peer is another browser we do not
 * control, so `decodeMessage` rejects anything malformed or from a protocol
 * version we do not speak rather than trusting the shape.
 */

import type { Direction } from "./signals";

/** Bump when the wire shape changes incompatibly; older peers are then ignored. */
export const MESH_PROTOCOL_VERSION = 1;

/** How long a peer's share stays live before it is pruned as stale, in ms. */
export const PEER_TTL_MS = 45_000;

/** Hard caps on an inbound frame, so a hostile peer cannot flood or bloat us. */
export const MAX_SIGNALS = 200;
export const MAX_QUESTION_LEN = 120;
export const MAX_MARKET_ID_LEN = 96;

/** One market's read, compacted for the wire — only what a peer can act on. */
export type SharedSignal = {
  readonly marketId: string;
  readonly question: string;
  /** Model P(up), 0..1. */
  readonly prob: number;
  readonly direction: Direction;
  /** 0..1. */
  readonly conviction: number;
  /** 0..100 attention. */
  readonly heat: number;
  /** -100..100 directional. */
  readonly bias: number;
};

export type MeshMessage = {
  readonly v: number;
  readonly peer: string;
  readonly ts: number;
  readonly signals: readonly SharedSignal[];
};

/** What we hold for each connected peer: their latest share and when it landed. */
export type PeerState = { readonly ts: number; readonly signals: readonly SharedSignal[] };
export type MeshState = ReadonlyMap<string, PeerState>;

const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const clampUnit = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const clampRange = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);

function validSignal(s: unknown): s is SharedSignal {
  if (!s || typeof s !== "object") return false;
  const o = s as Record<string, unknown>;
  return (
    typeof o.marketId === "string" &&
    o.marketId.length > 0 &&
    typeof o.question === "string" &&
    isNum(o.prob) &&
    (o.direction === "bullish" || o.direction === "bearish" || o.direction === "neutral") &&
    isNum(o.conviction) &&
    isNum(o.heat) &&
    isNum(o.bias)
  );
}

/**
 * Validate a pasted WebRTC signaling blob before it is ever handed to the
 * connection. A peer's offer/answer arrives as a copy-pasted string that could
 * be truncated, doubled, or plain wrong; this rejects anything that is not a
 * `{ type: "offer" | "answer", sdp: string }` so the handshake fails with a clear
 * message instead of an opaque `setRemoteDescription` throw.
 */
export function parseSignaling(raw: string): { type: "offer" | "answer"; sdp: string } | null {
  let o: unknown;
  try {
    o = JSON.parse(raw.trim());
  } catch {
    return null;
  }
  if (!o || typeof o !== "object") return null;
  const d = o as Record<string, unknown>;
  if (
    (d.type === "offer" || d.type === "answer") &&
    typeof d.sdp === "string" &&
    d.sdp.length > 0
  ) {
    return { type: d.type, sdp: d.sdp };
  }
  return null;
}

// ── Latency / geo ────────────────────────────────────────────────────────────
// We hold no TURN infrastructure to route by region, so "where is this peer"
// is answered by the one geographic fact a browser will give us for free: how
// long a round trip takes. A ping/pong on the data channel measures it, and the
// tier below turns milliseconds into a rough distance a trader can read.

/** How often to ping a link for its round-trip time. */
export const PING_INTERVAL_MS = 5_000;

export type Control = { readonly kind: "ping" | "pong"; readonly t: number };

export function encodePing(t: number): string {
  return JSON.stringify({ v: MESH_PROTOCOL_VERSION, kind: "ping", t });
}

export function encodePong(t: number): string {
  return JSON.stringify({ v: MESH_PROTOCOL_VERSION, kind: "pong", t });
}

/**
 * Parse a control frame (ping/pong). Disjoint from `decodeMessage`: a control
 * frame has a `kind` and no `peer`, a signal frame has a `peer` and no `kind`,
 * so each decoder returns `null` on the other's frames and they never collide on
 * one channel.
 */
export function decodeControl(raw: string): Control | null {
  let o: unknown;
  try {
    o = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!o || typeof o !== "object") return null;
  const d = o as Record<string, unknown>;
  if (d.v !== MESH_PROTOCOL_VERSION) return null;
  if ((d.kind === "ping" || d.kind === "pong") && isNum(d.t)) return { kind: d.kind, t: d.t };
  return null;
}

export type Region = "local" | "regional" | "continental" | "intercontinental";

/**
 * Rough geographic tier from a round-trip time. Light-in-fibre plus routing puts
 * ~1ms on a LAN, tens of ms within a region, ~100ms across a continent and more
 * between them; these cutoffs are deliberately coarse — a distance a human reads,
 * not a claim of precision.
 */
export function regionFromRtt(ms: number): Region {
  if (ms < 25) return "local";
  if (ms < 75) return "regional";
  if (ms < 160) return "continental";
  return "intercontinental";
}

/** Build the wire payload for this terminal's current signals. */
export function encodeMessage(peer: string, ts: number, signals: readonly SharedSignal[]): string {
  const msg: MeshMessage = { v: MESH_PROTOCOL_VERSION, peer, ts, signals };
  return JSON.stringify(msg);
}

/**
 * Parse and validate an inbound frame. Returns `null` — never throws — on
 * anything we would not want to merge: bad JSON, a version we do not speak, a
 * missing peer id, or a signals array with junk in it. Individual bad signals
 * are dropped; the message survives if any remain.
 *
 * Because the sender is an untrusted browser, values are not just type-checked
 * but **bounded**: the array is capped before any work is done (flood defence),
 * strings are truncated, and every number is clamped to the range its consumer
 * assumes (`prob`/`conviction` to 0..1, `heat` to 0..100, `bias` to ±100). A
 * peer cannot make a market look 99999-hot or push a bias off the dial.
 */
export function decodeMessage(raw: string): MeshMessage | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const o = parsed as Record<string, unknown>;
  if (o.v !== MESH_PROTOCOL_VERSION) return null;
  if (typeof o.peer !== "string" || o.peer.length === 0 || o.peer.length > MAX_MARKET_ID_LEN) {
    return null;
  }
  if (!isNum(o.ts) || !Array.isArray(o.signals)) return null;
  const signals = o.signals
    .slice(0, MAX_SIGNALS) // cap before filtering so a huge array can't burn CPU
    .filter(validSignal)
    .map((s) => ({
      marketId: s.marketId.slice(0, MAX_MARKET_ID_LEN),
      question: s.question.slice(0, MAX_QUESTION_LEN),
      prob: clampUnit(s.prob),
      direction: s.direction,
      conviction: clampUnit(s.conviction),
      heat: clampRange(s.heat, 0, 100),
      bias: clampRange(s.bias, -100, 100),
    }));
  return { v: MESH_PROTOCOL_VERSION, peer: o.peer, ts: o.ts, signals };
}

/**
 * Fold a decoded message into the mesh state and prune peers that have gone
 * quiet past `PEER_TTL_MS`. Pure: returns a new map, so React state updates
 * cleanly and a dropped peer's stale opinions never linger on screen.
 *
 * A frame is accepted only if it is strictly newer than what we already hold for
 * that peer. That single rule does three jobs: it drops out-of-order frames,
 * makes replays idempotent, and — crucially — lets peers **gossip** each other's
 * frames without looping forever, because a relayed copy of a frame we have
 * already seen carries the same `ts` and is ignored. A terminal never merges its
 * own id, so its signals cannot echo back in through a neighbour.
 */
export function mergeRemote(
  state: MeshState,
  msg: MeshMessage,
  now: number,
  selfId: string
): MeshState {
  const next = new Map(state);
  if (msg.peer !== selfId) {
    const cur = next.get(msg.peer);
    if (!cur || msg.ts > cur.ts) next.set(msg.peer, { ts: msg.ts, signals: msg.signals });
  }
  for (const [peer, s] of next) {
    if (now - s.ts > PEER_TTL_MS) next.delete(peer);
  }
  return next;
}

/**
 * Every peer-frame this node should forward to a neighbour so the mesh becomes
 * transitive: a new link learns everyone we already know (minus the neighbour's
 * own frame, which it authored). Combined with the newer-only rule in
 * `mergeRemote`, a chain A–B–C gives A a view of C through B with no O(n²)
 * handshakes and no relay loops.
 */
export function relayFrames(state: MeshState, exceptPeer?: string): MeshMessage[] {
  const out: MeshMessage[] = [];
  for (const [peer, s] of state) {
    if (peer === exceptPeer) continue;
    out.push({ v: MESH_PROTOCOL_VERSION, peer, ts: s.ts, signals: s.signals });
  }
  return out;
}

/** Drop every peer that has not refreshed inside the TTL. */
export function prune(state: MeshState, now: number): MeshState {
  const next = new Map(state);
  for (const [peer, s] of next) if (now - s.ts > PEER_TTL_MS) next.delete(peer);
  return next;
}

export type Consensus = {
  readonly marketId: string;
  /** Peers whose model points bullish / bearish on this market. */
  readonly bullish: number;
  readonly bearish: number;
  /** Total peers with a directional view here. */
  readonly voters: number;
  /** Mean of the peers' probabilities, 0..1. */
  readonly meanProb: number;
  /** −1..1: net directional agreement, sign = side, magnitude = how lopsided. */
  readonly agreement: number;
};

/**
 * Reduce the whole mesh to a per-market consensus.
 *
 * Only committed views count — a peer sitting at prob≈0.5 abstains rather than
 * being counted as a half-hearted vote either way. The returned map is keyed by
 * market id, so the scanner can mark a row that the desk agrees on versus one
 * that is a single terminal's lone read.
 */
export function consensus(state: MeshState): Map<string, Consensus> {
  const acc = new Map<string, { bull: number; bear: number; probSum: number; n: number }>();
  for (const peer of state.values()) {
    for (const s of peer.signals) {
      const a = acc.get(s.marketId) ?? { bull: 0, bear: 0, probSum: 0, n: 0 };
      a.probSum += s.prob;
      a.n += 1;
      if (s.direction === "bullish") a.bull += 1;
      else if (s.direction === "bearish") a.bear += 1;
      acc.set(s.marketId, a);
    }
  }
  const out = new Map<string, Consensus>();
  for (const [marketId, a] of acc) {
    const voters = a.bull + a.bear;
    out.set(marketId, {
      marketId,
      bullish: a.bull,
      bearish: a.bear,
      voters,
      meanProb: a.n > 0 ? a.probSum / a.n : 0.5,
      agreement: voters > 0 ? (a.bull - a.bear) / voters : 0,
    });
  }
  return out;
}
