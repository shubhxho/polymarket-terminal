"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  consensus,
  decodeMessage,
  encodeMessage,
  mergeRemote,
  prune,
  type Consensus,
  type MeshState,
  type SharedSignal,
} from "@/lib/signalMesh";

/**
 * WebRTC transport for the signal mesh — the impure half of `signalMesh.ts`.
 *
 * Two terminals link directly over a data channel with **manual signaling**: the
 * host makes an offer, the guest pastes it and returns an answer, the host pastes
 * that back. No server, no account — the offer/answer blobs are the whole
 * handshake, and a public STUN server only helps punch through NAT. Once open,
 * each side streams its `SharedSignal[]` whenever they change and merges whatever
 * the other sends through the pure, validated `mergeRemote`.
 *
 * ICE is gathered non-trickle (we wait for gathering to finish) so each blob is
 * a single self-contained paste rather than a stream of candidates the UI would
 * have to shuttle across by hand.
 */

const STUN: RTCConfiguration = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };

/** Short, human-copyable id for this terminal in the mesh. */
function makeSelfId(): string {
  const n = Math.floor(Math.random() * 0xffffff);
  return `term-${n.toString(16).padStart(6, "0")}`;
}

export type MeshRole = "idle" | "host" | "guest";
export type MeshStatus = RTCPeerConnectionState | "idle";

export type SignalMesh = {
  selfId: string;
  role: MeshRole;
  status: MeshStatus;
  peers: MeshState;
  consensusMap: Map<string, Consensus>;
  /** Signaling blob for the local side to hand across (offer if host, answer if guest). */
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
  if (!selfRef.current) selfRef.current = makeSelfId();
  const selfId = selfRef.current;

  const pcRef = useRef<RTCPeerConnection | null>(null);
  const chanRef = useRef<RTCDataChannel | null>(null);
  const [role, setRole] = useState<MeshRole>("idle");
  const [status, setStatus] = useState<MeshStatus>("idle");
  const [localBlob, setLocalBlob] = useState("");
  const [peers, setPeers] = useState<MeshState>(new Map());

  // Latest signals kept in a ref so the data-channel open handler always sends
  // the current set without being recreated on every tick.
  const localRef = useRef<readonly SharedSignal[]>(localSignals);
  localRef.current = localSignals;

  const send = useCallback(() => {
    const ch = chanRef.current;
    if (ch?.readyState === "open") {
      ch.send(encodeMessage(selfId, Date.now(), localRef.current));
    }
  }, [selfId]);

  const wireChannel = useCallback(
    (ch: RTCDataChannel) => {
      chanRef.current = ch;
      ch.onopen = () => send();
      ch.onmessage = (e) => {
        const msg = decodeMessage(typeof e.data === "string" ? e.data : "");
        if (msg) setPeers((prev) => mergeRemote(prev, msg, Date.now(), selfId));
      };
    },
    [send, selfId]
  );

  const newPeer = useCallback(() => {
    const pc = new RTCPeerConnection(STUN);
    pc.onconnectionstatechange = () => setStatus(pc.connectionState);
    pc.ondatachannel = (e) => wireChannel(e.channel);
    pcRef.current = pc;
    return pc;
  }, [wireChannel]);

  const reset = useCallback(() => {
    chanRef.current?.close();
    pcRef.current?.close();
    chanRef.current = null;
    pcRef.current = null;
    setRole("idle");
    setStatus("idle");
    setLocalBlob("");
    setPeers(new Map());
  }, []);

  const createOffer = useCallback(async () => {
    reset();
    const pc = newPeer();
    setRole("host");
    wireChannel(pc.createDataChannel("signals"));
    await pc.setLocalDescription(await pc.createOffer());
    await whenGathered(pc);
    setLocalBlob(JSON.stringify(pc.localDescription));
  }, [reset, newPeer, wireChannel]);

  const acceptOffer = useCallback(
    async (remote: string) => {
      reset();
      const pc = newPeer();
      setRole("guest");
      pc.setRemoteDescription(JSON.parse(remote) as RTCSessionDescriptionInit);
      await pc.setLocalDescription(await pc.createAnswer());
      await whenGathered(pc);
      setLocalBlob(JSON.stringify(pc.localDescription));
    },
    [reset, newPeer]
  );

  const acceptAnswer = useCallback(async (remote: string) => {
    await pcRef.current?.setRemoteDescription(JSON.parse(remote) as RTCSessionDescriptionInit);
  }, []);

  // Re-broadcast whenever the local signals change and the channel is up.
  useEffect(() => {
    send();
  }, [localSignals, send]);

  // Expire peers that go quiet even if they never send a closing frame.
  useEffect(() => {
    const t = setInterval(() => setPeers((prev) => prune(prev, Date.now())), 5000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => () => reset(), [reset]);

  return {
    selfId,
    role,
    status,
    peers,
    consensusMap: consensus(peers),
    localBlob,
    createOffer,
    acceptOffer,
    acceptAnswer,
    reset,
  };
}
