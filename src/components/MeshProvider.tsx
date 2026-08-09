"use client";

import { createContext, useContext, useMemo, type ReactNode } from "react";
import type { SignalsPayload } from "@/app/api/signals/route";
import { usePoll } from "@/hooks/usePoll";
import { useSignalMesh, type SignalMesh } from "@/hooks/useSignalMesh";
import { truncate } from "@/lib/format";
import type { SharedSignal } from "@/lib/signalMesh";

/**
 * App-wide signal mesh.
 *
 * Lifting the mesh above the screens is what lets the scanner mark a market the
 * desk agrees on while the MESH screen runs the handshake — one connection, one
 * consensus, read from anywhere. The scan poll here shares react-query's cache
 * with the scanner's own `usePoll("/api/signals")`, so hosting it app-wide costs
 * no extra fetch; it just keeps this terminal broadcasting its current read to
 * peers no matter which screen is open.
 */
const MeshContext = createContext<SignalMesh | null>(null);

export function MeshProvider({ children }: { children: ReactNode }) {
  const { data } = usePoll<SignalsPayload>("/api/signals", 20000);

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

  const mesh = useSignalMesh(local);
  return <MeshContext.Provider value={mesh}>{children}</MeshContext.Provider>;
}

export function useMesh(): SignalMesh {
  const ctx = useContext(MeshContext);
  if (!ctx) throw new Error("useMesh must be used within MeshProvider");
  return ctx;
}
