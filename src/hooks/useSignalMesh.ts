"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  consensus,
  decodeMessage,
  encodeMessage,
  mergeRemote,
  prune,
  relayFrames,
  type Consensus,
  type MeshState,
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

export type SignalMesh = {
  selfId: string;
  role: MeshRole;
  status: MeshStatus;
  /** Established WebRTC links (may be fewer than peers, thanks to gossip relay). */
  links: number;
  peers: MeshState;
  consensusMap: Map<string, Consensus>;
  localBlob: string;
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

  const [role, setRole] = useState<MeshRole>("idle");
  const [status, setStatus] = useState<MeshStatus>("idle");
  const [links, setLinks] = useState(0);
  const [localBlob, setLocalBlob] = useState("");
  const [peers, setPeers] = useState<MeshState>(new Map());

  const localRef = useRef<readonly SharedSignal[]>(localSignals);
  localRef.current = localSignals;

  const setPeersBoth = useCallback((next: MeshState) => {
    peersRef.current = next;
    setPeers(next);
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
      ch.onmessage = (e) => onFrame(typeof e.data === "string" ? e.data : "", id);
    },
    [selfId, onFrame, refreshStatus]
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
    pendingRef.current = null;
    setRole("idle");
    setStatus("idle");
    setLinks(0);
    setLocalBlob("");
    setPeersBoth(new Map());
  }, [setPeersBoth]);

  const createOffer = useCallback(async () => {
    const id = randId("link");
    pendingRef.current = id;
    const pc = newConn(id);
    setRole("host");
    wireChannel(id, pc.createDataChannel("signals"));
    await pc.setLocalDescription(await pc.createOffer());
    await whenGathered(pc);
    setLocalBlob(JSON.stringify(pc.localDescription));
  }, [newConn, wireChannel]);

  const acceptOffer = useCallback(
    async (remote: string) => {
      const id = randId("link");
      const pc = newConn(id);
      setRole("guest");
      await pc.setRemoteDescription(JSON.parse(remote) as RTCSessionDescriptionInit);
      await pc.setLocalDescription(await pc.createAnswer());
      await whenGathered(pc);
      setLocalBlob(JSON.stringify(pc.localDescription));
    },
    [newConn]
  );

  const acceptAnswer = useCallback(async (remote: string) => {
    const id = pendingRef.current;
    const conn = id ? connsRef.current.get(id) : null;
    await conn?.pc.setRemoteDescription(JSON.parse(remote) as RTCSessionDescriptionInit);
    pendingRef.current = null;
  }, []);

  // Broadcast our own signals to every open link whenever they change.
  useEffect(() => {
    const raw = encodeMessage(selfId, Date.now(), localSignals);
    for (const c of connsRef.current.values()) {
      if (c.chan?.readyState === "open") c.chan.send(raw);
    }
  }, [localSignals, selfId]);

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
    peers,
    consensusMap: consensus(peers),
    localBlob,
    createOffer,
    acceptOffer,
    acceptAnswer,
    reset,
  };
}
