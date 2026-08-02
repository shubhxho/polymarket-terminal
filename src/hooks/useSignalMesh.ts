"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  consensus,
  decodeControl,
  decodeMessage,
  encodeMessage,
  encodePing,
  encodePong,
  mergeRemote,
  parseSignaling,
  PING_INTERVAL_MS,
  prune,
  regionFromRtt,
  relayFrames,
  type Consensus,
  type MeshState,
  type Region,
  type SharedSignal,
} from "@/lib/signalMesh";

/**
 * WebRTC transport for the signal mesh — the impure half of `signalMesh.ts`.
 *
 * Terminals link directly over data channels with **manual signaling** (host
 * offer → guest answer → host, copy-pasted out of band). No server; a public
 * STUN only helps punch through NAT. It is a real **N-peer gossip mesh**: each
 * link relays the frames it receives to its other links and catches a fresh link
 * up on everyone it already knows, so a chain A–B–C gives all three a full view
 * without every pair having to shake hands. Loops terminate because a frame is
 * deduped by its origin peer and timestamp — a relayed copy of something already
 * seen is dropped, never forwarded again.
 *
 * ICE is gathered non-trickle so each signaling blob is a single self-contained
 * paste rather than a trickle of candidates.
 */

const STUN: RTCConfiguration = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };

function randId(prefix: string): string {
  return `${prefix}-${Math.floor(Math.random() * 0xffffff)
    .toString(16)
    .padStart(6, "0")}`;
}

export type MeshRole = "idle" | "host" | "guest";
export type MeshStatus = RTCPeerConnectionState | "idle";

type Conn = { pc: RTCPeerConnection; chan: RTCDataChannel | null };

/** Round-trip latency to a directly-linked peer, and the geo tier it implies. */
export type LinkStat = { peer: string; rttMs: number; region: Region };

export type SignalMesh = {
  selfId: string;
  role: MeshRole;
  status: MeshStatus;
  /** Established WebRTC links (may be fewer than peers, thanks to gossip relay). */
  links: number;
  /** Per-direct-link latency + region, newest RTT first computed. */
  linkStats: LinkStat[];
  peers: MeshState;
  consensusMap: Map<string, Consensus>;
  localBlob: string;
  /** Last handshake error, cleared when a new one starts. */
  error: string | null;
  createOffer: () => Promise<void>;
  acceptOffer: (remote: string) => Promise<void>;
  acceptAnswer: (remote: string) => Promise<void>;
  reset: () => void;
};

/** Resolve once ICE gathering finishes, so `localDescription` carries every candidate. */
function whenGathered(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
  });
}

export function useSignalMesh(localSignals: readonly SharedSignal[]): SignalMesh {
  const selfRef = useRef<string>("");
  if (!selfRef.current) selfRef.current = randId("term");
  const selfId = selfRef.current;

  // All live links, the pending handshake's link id, and the highest ts we have
  // accepted per origin peer (the gossip de-dup, kept in a ref so relaying never
  // waits on a React commit).
  const connsRef = useRef<Map<string, Conn>>(new Map());
  const pendingRef = useRef<string | null>(null);
  const seenRef = useRef<Map<string, number>>(new Map());
  const peersRef = useRef<MeshState>(new Map());
  // Which origin peer each direct link carries, and its latest measured RTT.
  const linkPeerRef = useRef<Map<string, string>>(new Map());
  const rttRef = useRef<Map<string, number>>(new Map());

  const [role, setRole] = useState<MeshRole>("idle");
  const [status, setStatus] = useState<MeshStatus>("idle");
  const [links, setLinks] = useState(0);
  const [linkStats, setLinkStats] = useState<LinkStat[]>([]);
  const [localBlob, setLocalBlob] = useState("");
  const [peers, setPeers] = useState<MeshState>(new Map());
  const [error, setError] = useState<string | null>(null);

  const localRef = useRef<readonly SharedSignal[]>(localSignals);
  localRef.current = localSignals;

  const setPeersBoth = useCallback((next: MeshState) => {
    peersRef.current = next;
    setPeers(next);
  }, []);

  /** Send our current signals (fresh ts) to every open link. */
  const broadcastOwn = useCallback(() => {
    const raw = encodeMessage(selfId, Date.now(), localRef.current);
    for (const c of connsRef.current.values()) {
      if (c.chan?.readyState === "open") c.chan.send(raw);
    }
  }, [selfId]);

  /** Record a fresh RTT for a link and rebuild the per-link latency table. */
  const recordRtt = useCallback((id: string, rtt: number) => {
    rttRef.current.set(id, rtt);
    const stats: LinkStat[] = [];
    for (const [connId, ms] of rttRef.current) {
      stats.push({
        peer: linkPeerRef.current.get(connId) ?? connId,
        rttMs: ms,
        region: regionFromRtt(ms),
      });
    }
    stats.sort((a, b) => a.rttMs - b.rttMs);
    setLinkStats(stats);
  }, []);

  const refreshStatus = useCallback(() => {
    const states = [...connsRef.current.values()].map((c) => c.pc.connectionState);
    setLinks(states.filter((s) => s === "connected").length);
    setStatus(
      states.includes("connected")
        ? "connected"
        : (states.find((s) => s === "connecting" || s === "new") ?? "idle")
    );
  }, []);

  /** Merge an inbound frame and, if it was genuinely new, gossip it onward. */
  const onFrame = useCallback(
    (raw: string, fromId: string) => {
      const msg = decodeMessage(raw);
      if (!msg || msg.peer === selfId) return;
      // First frame on a link is the neighbour's own signal (sent before any
      // relayed frames on open), so first-write-wins maps the link to its peer.
      if (!linkPeerRef.current.has(fromId)) linkPeerRef.current.set(fromId, msg.peer);
      const seen = seenRef.current.get(msg.peer);
      if (seen !== undefined && msg.ts <= seen) return; // stale/replay — stop the loop here
      seenRef.current.set(msg.peer, msg.ts);
      setPeersBoth(mergeRemote(peersRef.current, msg, Date.now(), selfId));
      for (const [id, c] of connsRef.current) {
        if (id !== fromId && c.chan?.readyState === "open") c.chan.send(raw);
      }
    },
    [selfId, setPeersBoth]
  );

  const wireChannel = useCallback(
    (id: string, ch: RTCDataChannel) => {
      const conn = connsRef.current.get(id);
      if (conn) conn.chan = ch;
      ch.onopen = () => {
        ch.send(encodeMessage(selfId, Date.now(), localRef.current));
        // Catch the new link up on everyone we already know.
        for (const f of relayFrames(peersRef.current)) ch.send(JSON.stringify(f));
        refreshStatus();
      };
      ch.onmessage = (e) => {
        const raw = typeof e.data === "string" ? e.data : "";
        const ctrl = decodeControl(raw);
        if (ctrl) {
          if (ctrl.kind === "ping") ch.send(encodePong(ctrl.t));
          else recordRtt(id, Math.max(0, Date.now() - ctrl.t));
          return;
        }
        onFrame(raw, id);
      };
    },
    [selfId, onFrame, refreshStatus, recordRtt]
  );

  const newConn = useCallback(
    (id: string) => {
      const pc = new RTCPeerConnection(STUN);
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === "failed" || pc.connectionState === "closed") {
          connsRef.current.delete(id);
        }
        refreshStatus();
      };
      pc.ondatachannel = (e) => wireChannel(id, e.channel);
      connsRef.current.set(id, { pc, chan: null });
      return pc;
    },
    [wireChannel, refreshStatus]
  );

  const reset = useCallback(() => {
    for (const c of connsRef.current.values()) {
      c.chan?.close();
      c.pc.close();
    }
    connsRef.current.clear();
    seenRef.current.clear();
    linkPeerRef.current.clear();
    rttRef.current.clear();
    pendingRef.current = null;
    setRole("idle");
    setStatus("idle");
    setLinks(0);
    setLinkStats([]);
    setLocalBlob("");
    setError(null);
    setPeersBoth(new Map());
  }, [setPeersBoth]);

  const createOffer = useCallback(async () => {
    setError(null);
    try {
      const id = randId("link");
      pendingRef.current = id;
      const pc = newConn(id);
      setRole("host");
      wireChannel(id, pc.createDataChannel("signals"));
      await pc.setLocalDescription(await pc.createOffer());
      await whenGathered(pc);
      setLocalBlob(JSON.stringify(pc.localDescription));
    } catch (e) {
      setError(e instanceof Error ? e.message : "could not create an offer");
    }
  }, [newConn, wireChannel]);

  const acceptOffer = useCallback(
    async (remote: string) => {
      const desc = parseSignaling(remote);
      if (!desc || desc.type !== "offer") {
        setError("That is not a valid offer — paste the host's offer JSON.");
        return;
      }
      setError(null);
      try {
        const id = randId("link");
        const pc = newConn(id);
        setRole("guest");
        await pc.setRemoteDescription(desc);
        await pc.setLocalDescription(await pc.createAnswer());
        await whenGathered(pc);
        setLocalBlob(JSON.stringify(pc.localDescription));
      } catch (e) {
        setError(e instanceof Error ? e.message : "handshake failed");
      }
    },
    [newConn]
  );

  const acceptAnswer = useCallback(async (remote: string) => {
    const desc = parseSignaling(remote);
    if (!desc || desc.type !== "answer") {
      setError("That is not a valid answer — paste the peer's answer JSON.");
      return;
    }
    const conn = pendingRef.current ? connsRef.current.get(pendingRef.current) : null;
    if (!conn) {
      setError("No pending offer to finish — make an offer first.");
      return;
    }
    setError(null);
    try {
      await conn.pc.setRemoteDescription(desc);
      pendingRef.current = null;
    } catch (e) {
      setError(e instanceof Error ? e.message : "handshake failed");
    }
  }, []);

  // Broadcast our own signals to every open link whenever they change.
  useEffect(() => {
    broadcastOwn();
  }, [localSignals, broadcastOwn]);

  // Liveness heartbeat: re-broadcast on a fixed cadence so this terminal stays
  // inside peers' TTL windows even when the scan poll pauses (a backgrounded tab
  // stops polling) or a quiet market produces no new signals. Well under the
  // 45s PEER_TTL_MS, so a live link never looks stale to the other side.
  useEffect(() => {
    const t = setInterval(broadcastOwn, 15000);
    return () => clearInterval(t);
  }, [broadcastOwn]);

  // Ping every open link to measure round-trip latency (the geo read).
  useEffect(() => {
    const t = setInterval(() => {
      const raw = encodePing(Date.now());
      for (const c of connsRef.current.values()) {
        if (c.chan?.readyState === "open") c.chan.send(raw);
      }
    }, PING_INTERVAL_MS);
    return () => clearInterval(t);
  }, []);

  // Expire peers that go quiet even if no closing frame ever arrives.
  useEffect(() => {
    const t = setInterval(() => setPeersBoth(prune(peersRef.current, Date.now())), 5000);
    return () => clearInterval(t);
  }, [setPeersBoth]);

  useEffect(() => () => reset(), [reset]);

  return {
    selfId,
    role,
    status,
    links,
    linkStats,
    peers,
    consensusMap: consensus(peers),
    localBlob,
    error,
    createOffer,
    acceptOffer,
    acceptAnswer,
    reset,
  };
}
