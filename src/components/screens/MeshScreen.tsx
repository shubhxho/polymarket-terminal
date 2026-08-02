"use client";

import { motion } from "motion/react";
import { useMemo, useState } from "react";
import type { SignalsPayload } from "@/app/api/signals/route";
import { Empty, Field, Panel } from "@/components/ui/Panel";
import { usePoll } from "@/hooks/usePoll";
import { useMesh } from "@/components/MeshProvider";
import { cents, truncate } from "@/lib/format";
import { panelVariants, staggerContainer } from "@/lib/motion";
import type { SharedSignal } from "@/lib/signalMesh";

const copy = (text: string) => void navigator.clipboard?.writeText(text);

/**
 * Signal mesh — share what this terminal computed with other terminals, directly.
 *
 * The whole point is to tell a lone read from a consensus: when three desks'
 * models all lean the same way on a market, that is different from one. The link
 * is peer-to-peer over WebRTC with no server — you hand an offer/answer pair
 * across out of band (paste it in chat), and from then on both sides stream their
 * signals to each other. Nothing here trades; it is a second opinion, and the
 * validation in `signalMesh.ts` treats every peer as untrusted.
 */
export default function MeshScreen() {
  const { data } = usePoll<SignalsPayload>("/api/signals", 20000);

  // Share the model's read on every scanned market — compact, only what a peer
  // can act on. Markets without a model read share as neutral and abstain from
  // the vote downstream.
  const local = useMemo<SharedSignal[]>(
    () =>
      (data?.markets ?? []).slice(0, 40).map((m) => ({
        marketId: m.market.id || m.market.slug,
        question: truncate(m.market.question, 60),
        prob: m.model?.prob ?? 0.5,
        direction: m.model?.direction ?? "neutral",
        conviction: m.model?.conviction ?? 0,
        heat: m.heat,
        bias: m.bias,
      })),
    [data]
  );

  const mesh = useMesh();
  const [offerIn, setOfferIn] = useState("");
  const [answerIn, setAnswerIn] = useState("");

  const byId = useMemo(() => new Map(local.map((s) => [s.marketId, s.question] as const)), [local]);
  const consensusRows = useMemo(
    () =>
      [...mesh.consensusMap.values()]
        .filter((c) => c.voters > 0)
        .sort((a, b) => Math.abs(b.agreement) * b.voters - Math.abs(a.agreement) * a.voters)
        .slice(0, 20),
    [mesh.consensusMap]
  );

  return (
    <motion.div
      className="flex h-full min-h-0 flex-col gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      <Panel
        title="Mesh"
        right={
          <span className="flex items-center gap-2">
            <span className="text-faint">{mesh.selfId}</span>
            <span className={mesh.status === "connected" ? "text-up" : "text-muted"}>
              {mesh.status}
            </span>
          </span>
        }
        className="shrink-0"
        flush
        animate
      >
        <div className="flex flex-wrap items-end gap-x-6 gap-y-2 px-2.5 py-2">
          <Field label="Role" value={mesh.role} />
          <Field label="Links" value={String(mesh.links)} />
          <Field label="Peers" value={String(mesh.peers.size)} />
          <Field label="Sharing" value={`${local.length} markets`} />
        </div>
        <p className="border-t border-edge px-2.5 py-1.5 text-[11px] leading-[15px] text-faint">
          Peer-to-peer over WebRTC, no server. Host makes an offer, the other side pastes it and
          returns an answer, host pastes that back. Signals stream both ways once connected. Nothing
          here places a trade — it is a second opinion.
        </p>
      </Panel>

      <motion.div
        variants={panelVariants}
        className="grid min-h-0 flex-1 grid-cols-1 gap-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]"
      >
        {/* ── Connect ──────────────────────────────────────────────────── */}
        <Panel title="Connect" className="min-h-0 overflow-auto">
          <div className="flex flex-col gap-2 p-0.5">
            <div className="flex gap-2">
              <button
                onClick={() => void mesh.createOffer()}
                className="border border-edge-strong px-2 py-1 text-tiny text-ink hover:bg-surface-2"
              >
                Host — make offer
              </button>
              <button
                onClick={mesh.reset}
                className="border border-edge px-2 py-1 text-tiny text-muted hover:bg-surface-2"
              >
                Reset
              </button>
            </div>

            {mesh.error ? (
              <div className="border border-down-weak px-2 py-1 text-[11px] text-down">
                {mesh.error}
              </div>
            ) : null}

            {mesh.localBlob ? (
              <div className="flex flex-col gap-1">
                <span className="eyebrow">
                  Your {mesh.role === "host" ? "offer" : "answer"} — send it to the peer
                </span>
                <textarea
                  readOnly
                  value={mesh.localBlob}
                  className="h-20 w-full resize-none border border-edge bg-surface-2 p-1 text-[10px] text-muted"
                />
                <button
                  onClick={() => copy(mesh.localBlob)}
                  className="self-start border border-edge px-2 py-0.5 text-tiny text-accent hover:bg-surface-2"
                >
                  copy
                </button>
              </div>
            ) : null}

            <div className="flex flex-col gap-1 border-t border-edge/60 pt-2">
              <span className="eyebrow">Join — paste a host&apos;s offer</span>
              <textarea
                value={offerIn}
                onChange={(e) => setOfferIn(e.target.value)}
                placeholder="paste offer JSON"
                className="h-16 w-full resize-none border border-edge bg-surface-2 p-1 text-[10px] text-ink"
              />
              <button
                onClick={() => offerIn.trim() && void mesh.acceptOffer(offerIn.trim())}
                className="self-start border border-edge-strong px-2 py-0.5 text-tiny text-ink hover:bg-surface-2"
              >
                accept offer → make answer
              </button>
            </div>

            {mesh.role === "host" ? (
              <div className="flex flex-col gap-1 border-t border-edge/60 pt-2">
                <span className="eyebrow">Finish — paste the peer&apos;s answer</span>
                <textarea
                  value={answerIn}
                  onChange={(e) => setAnswerIn(e.target.value)}
                  placeholder="paste answer JSON"
                  className="h-16 w-full resize-none border border-edge bg-surface-2 p-1 text-[10px] text-ink"
                />
                <button
                  onClick={() => answerIn.trim() && void mesh.acceptAnswer(answerIn.trim())}
                  className="self-start border border-edge-strong px-2 py-0.5 text-tiny text-ink hover:bg-surface-2"
                >
                  accept answer → connect
                </button>
              </div>
            ) : null}
          </div>
        </Panel>

        {/* ── Consensus ────────────────────────────────────────────────── */}
        <Panel
          title="Desk consensus"
          right={`${consensusRows.length}`}
          className="min-h-0 overflow-auto"
          flush
        >
          {consensusRows.length === 0 ? (
            <Empty
              text={
                mesh.peers.size === 0 ? "no peers connected" : "no shared directional views yet"
              }
            />
          ) : (
            <div className="text-tiny">
              <div className="sticky top-0 flex items-center gap-2 border-b border-edge bg-surface-2 px-2.5 py-[3px]">
                <span className="eyebrow min-w-0 flex-1">Market</span>
                <span className="eyebrow w-[70px] shrink-0 text-right">Bull / Bear</span>
                <span className="eyebrow w-[52px] shrink-0 text-right">Mean</span>
                <span className="eyebrow w-[54px] shrink-0 text-right">Agree</span>
              </div>
              {consensusRows.map((c) => {
                const q = byId.get(c.marketId) ?? c.marketId;
                return (
                  <div
                    key={c.marketId}
                    className="flex items-center gap-2 border-b border-edge/60 px-2.5 py-[4px] last:border-0"
                  >
                    <span className="min-w-0 flex-1 truncate text-ink" title={q}>
                      {q}
                    </span>
                    <span className="w-[70px] shrink-0 text-right">
                      <span className="text-up">{c.bullish}</span>
                      <span className="text-faint"> / </span>
                      <span className="text-down">{c.bearish}</span>
                    </span>
                    <span className="w-[52px] shrink-0 text-right text-muted">
                      {cents(c.meanProb)}
                    </span>
                    <span
                      className={`w-[54px] shrink-0 text-right ${
                        c.agreement > 0 ? "text-up" : c.agreement < 0 ? "text-down" : "text-faint"
                      }`}
                    >
                      {(c.agreement * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </Panel>
      </motion.div>
    </motion.div>
  );
}
