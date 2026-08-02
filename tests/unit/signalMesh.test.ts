import { describe, expect, test } from "vitest";
import {
  consensus,
  decodeControl,
  decodeMessage,
  encodeMessage,
  encodePing,
  encodePong,
  mergeRemote,
  MESH_PROTOCOL_VERSION,
  parseSignaling,
  regionFromRtt,
  PEER_TTL_MS,
  prune,
  relayFrames,
  type MeshState,
  type SharedSignal,
} from "@/lib/signalMesh";

const sig = (over: Partial<SharedSignal> = {}): SharedSignal => ({
  marketId: "m1",
  question: "Will it?",
  prob: 0.7,
  direction: "bullish",
  conviction: 0.4,
  heat: 60,
  bias: 40,
  ...over,
});

describe("encode / decode", () => {
  test("round-trips a well-formed message", () => {
    const raw = encodeMessage("peerA", 1000, [
      sig(),
      sig({ marketId: "m2", direction: "bearish", prob: 0.3 }),
    ]);
    const msg = decodeMessage(raw)!;
    expect(msg.peer).toBe("peerA");
    expect(msg.ts).toBe(1000);
    expect(msg.signals).toHaveLength(2);
    expect(msg.signals[1].marketId).toBe("m2");
  });

  test("rejects bad JSON and non-objects", () => {
    expect(decodeMessage("{not json")).toBeNull();
    expect(decodeMessage("42")).toBeNull();
    expect(decodeMessage("null")).toBeNull();
  });

  test("rejects a protocol-version mismatch", () => {
    const raw = JSON.stringify({ v: MESH_PROTOCOL_VERSION + 1, peer: "p", ts: 1, signals: [] });
    expect(decodeMessage(raw)).toBeNull();
  });

  test("rejects a missing or empty peer id", () => {
    expect(
      decodeMessage(JSON.stringify({ v: MESH_PROTOCOL_VERSION, peer: "", ts: 1, signals: [] }))
    ).toBeNull();
    expect(
      decodeMessage(JSON.stringify({ v: MESH_PROTOCOL_VERSION, ts: 1, signals: [] }))
    ).toBeNull();
  });

  test("drops individual malformed signals but keeps the good ones", () => {
    const raw = JSON.stringify({
      v: MESH_PROTOCOL_VERSION,
      peer: "p",
      ts: 1,
      signals: [sig(), { marketId: "bad" }, { ...sig(), prob: "x" }, sig({ marketId: "m3" })],
    });
    const msg = decodeMessage(raw)!;
    expect(msg.signals.map((s) => s.marketId)).toEqual(["m1", "m3"]);
  });

  test("clamps out-of-range prob/conviction from a hostile peer", () => {
    const raw = JSON.stringify({
      v: MESH_PROTOCOL_VERSION,
      peer: "p",
      ts: 1,
      signals: [sig({ prob: 5, conviction: -2 })],
    });
    const s = decodeMessage(raw)!.signals[0];
    expect(s.prob).toBe(1);
    expect(s.conviction).toBe(0);
  });
});

describe("mergeRemote / prune", () => {
  test("adds a peer's share and never merges the terminal's own echo", () => {
    let state = mergeRemote(new Map(), { v: 1, peer: "b", ts: 100, signals: [sig()] }, 100, "self");
    expect([...state.keys()]).toEqual(["b"]);
    state = mergeRemote(state, { v: 1, peer: "self", ts: 100, signals: [sig()] }, 100, "self");
    expect([...state.keys()]).toEqual(["b"]);
  });

  test("prunes peers that have gone quiet past the TTL", () => {
    const state = mergeRemote(new Map(), { v: 1, peer: "b", ts: 0, signals: [sig()] }, 0, "self");
    const later = prune(state, PEER_TTL_MS + 1);
    expect(later.size).toBe(0);
  });

  test("a fresh share from a stale peer revives it; other stale peers still drop", () => {
    let state = mergeRemote(new Map(), { v: 1, peer: "b", ts: 0, signals: [sig()] }, 0, "self");
    state = mergeRemote(state, { v: 1, peer: "c", ts: 0, signals: [sig()] }, 0, "self");
    // c refreshes at a time where b is already stale.
    state = mergeRemote(
      state,
      { v: 1, peer: "c", ts: PEER_TTL_MS + 5, signals: [sig()] },
      PEER_TTL_MS + 5,
      "self"
    );
    expect([...state.keys()]).toEqual(["c"]);
  });
});

describe("parseSignaling", () => {
  test("accepts a well-formed offer or answer, trimming whitespace", () => {
    expect(parseSignaling('  {"type":"offer","sdp":"v=0..."}  ')).toEqual({
      type: "offer",
      sdp: "v=0...",
    });
    expect(parseSignaling('{"type":"answer","sdp":"v=0"}')).toEqual({ type: "answer", sdp: "v=0" });
  });

  test("rejects junk, wrong type, or a missing/empty sdp", () => {
    expect(parseSignaling("not json")).toBeNull();
    expect(parseSignaling("null")).toBeNull();
    expect(parseSignaling('{"type":"pranswer","sdp":"x"}')).toBeNull();
    expect(parseSignaling('{"type":"offer"}')).toBeNull();
    expect(parseSignaling('{"type":"offer","sdp":""}')).toBeNull();
  });
});

describe("latency / geo control frames", () => {
  test("ping/pong round-trip through their own decoder", () => {
    expect(decodeControl(encodePing(1234))).toEqual({ kind: "ping", t: 1234 });
    expect(decodeControl(encodePong(5678))).toEqual({ kind: "pong", t: 5678 });
  });

  test("control and signal decoders are disjoint — neither accepts the other's frame", () => {
    // A signal frame is not a control frame...
    expect(decodeControl(encodeMessage("p", 1, [sig()]))).toBeNull();
    // ...and a control frame is not a signal frame.
    expect(decodeMessage(encodePing(1))).toBeNull();
  });

  test("decodeControl rejects junk and version mismatch", () => {
    expect(decodeControl("nope")).toBeNull();
    expect(
      decodeControl(JSON.stringify({ v: MESH_PROTOCOL_VERSION + 1, kind: "ping", t: 1 }))
    ).toBeNull();
    expect(decodeControl(JSON.stringify({ v: MESH_PROTOCOL_VERSION, kind: "ping" }))).toBeNull();
  });

  test("regionFromRtt tiers milliseconds into a coarse distance", () => {
    expect(regionFromRtt(2)).toBe("local");
    expect(regionFromRtt(24)).toBe("local");
    expect(regionFromRtt(25)).toBe("regional");
    expect(regionFromRtt(74)).toBe("regional");
    expect(regionFromRtt(120)).toBe("continental");
    expect(regionFromRtt(300)).toBe("intercontinental");
  });
});

describe("gossip / relay", () => {
  test("a newer frame updates a peer; an older or replayed frame is ignored", () => {
    let s: MeshState = mergeRemote(
      new Map(),
      { v: 1, peer: "c", ts: 10, signals: [sig({ prob: 0.8 })] },
      10,
      "self"
    );
    // Stale frame (older ts) — ignored.
    s = mergeRemote(s, { v: 1, peer: "c", ts: 5, signals: [sig({ prob: 0.1 })] }, 10, "self");
    expect(s.get("c")!.signals[0].prob).toBeCloseTo(0.8, 6);
    // Exact replay (same ts) — idempotent no-op.
    s = mergeRemote(s, { v: 1, peer: "c", ts: 10, signals: [sig({ prob: 0.1 })] }, 10, "self");
    expect(s.get("c")!.signals[0].prob).toBeCloseTo(0.8, 6);
    // Genuinely newer — accepted.
    s = mergeRemote(s, { v: 1, peer: "c", ts: 11, signals: [sig({ prob: 0.2 })] }, 11, "self");
    expect(s.get("c")!.signals[0].prob).toBeCloseTo(0.2, 6);
  });

  test("relayFrames forwards every peer except the neighbour's own frame", () => {
    let s: MeshState = new Map();
    s = mergeRemote(s, { v: 1, peer: "b", ts: 1, signals: [sig()] }, 1, "self");
    s = mergeRemote(s, { v: 1, peer: "c", ts: 1, signals: [sig({ marketId: "m2" })] }, 1, "self");
    const frames = relayFrames(s, "b");
    expect(frames.map((f) => f.peer)).toEqual(["c"]);
    // A relayed frame re-merges verbatim and, being same-ts, changes nothing —
    // the gossip loop terminates.
    const before = s;
    const after = mergeRemote(before, frames[0], 1, "self");
    expect(after.get("c")!.ts).toBe(before.get("c")!.ts);
  });

  test("transitive mesh: A learns C's read relayed through B", () => {
    // A already knows B; B relays C's frame to A; A now sees C.
    let a: MeshState = mergeRemote(new Map(), { v: 1, peer: "b", ts: 1, signals: [sig()] }, 1, "A");
    a = mergeRemote(
      a,
      { v: 1, peer: "c", ts: 2, signals: [sig({ direction: "bearish", prob: 0.3 })] },
      2,
      "A"
    );
    expect([...a.keys()].sort()).toEqual(["b", "c"]);
    expect(a.get("c")!.signals[0].direction).toBe("bearish");
  });
});

describe("consensus", () => {
  test("tallies direction and averages probability per market", () => {
    let state: MeshState = new Map();
    state = mergeRemote(
      state,
      { v: 1, peer: "a", ts: 1, signals: [sig({ prob: 0.8 })] },
      1,
      "self"
    );
    state = mergeRemote(
      state,
      { v: 1, peer: "b", ts: 1, signals: [sig({ prob: 0.6 })] },
      1,
      "self"
    );
    state = mergeRemote(
      state,
      { v: 1, peer: "c", ts: 1, signals: [sig({ direction: "bearish", prob: 0.4 })] },
      1,
      "self"
    );
    const c = consensus(state).get("m1")!;
    expect(c.bullish).toBe(2);
    expect(c.bearish).toBe(1);
    expect(c.voters).toBe(3);
    expect(c.meanProb).toBeCloseTo(0.6, 6);
    expect(c.agreement).toBeCloseTo((2 - 1) / 3, 6);
  });

  test("a fence-sitting neutral peer abstains from the vote", () => {
    const state = mergeRemote(
      new Map(),
      { v: 1, peer: "a", ts: 1, signals: [sig({ direction: "neutral", prob: 0.5 })] },
      1,
      "self"
    );
    const c = consensus(state).get("m1")!;
    expect(c.voters).toBe(0);
    expect(c.agreement).toBe(0);
  });
});
