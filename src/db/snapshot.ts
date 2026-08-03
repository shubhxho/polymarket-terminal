/**
 * Persist a scan of `/api/signals` so the live signals can be studied later.
 *
 * `snapshotRow` is pure — it flattens a payload into the insert shape and is
 * unit-tested — while `recordSignalSnapshot` does the guarded, best-effort
 * write. Like the logger, it is a no-op without a database and never lets a
 * write failure reach the request.
 */

import { db } from "./client";
import { signalSnapshots, type SignalSnapshotRow } from "./schema";
import { logError } from "@/lib/logger";
import type { SignalsPayload } from "@/app/api/signals/route";

/** How many ranked markets to keep in the snapshot's compact `markets` blob. */
const KEEP = 30;

export function snapshotRow(p: SignalsPayload): SignalSnapshotRow {
  const s = p.stats;
  const top = p.markets[0];
  return {
    scanned: s.scanned,
    flagged: s.flagged,
    modeled: s.modeled,
    confirms: s.modelConfirms,
    conflicts: s.modelConflicts,
    bullish: s.bullish,
    bearish: s.bearish,
    arbs: p.arbs.length,
    drifts: p.drifts.length,
    blockNotional: s.blockNotional,
    topMarketId: top?.market.id ?? null,
    topHeat: top?.heat ?? null,
    markets: p.markets.slice(0, KEEP).map((m) => ({
      id: m.market.id || m.market.slug,
      q: m.market.question,
      heat: m.heat,
      bias: m.bias,
      prob: m.model?.prob ?? null,
    })),
  };
}

export async function recordSignalSnapshot(p: SignalsPayload): Promise<void> {
  const d = await db();
  if (!d) return;
  try {
    await d.insert(signalSnapshots).values(snapshotRow(p));
  } catch (err) {
    logError("snapshot.insert_failed", {
      message: err instanceof Error ? err.message : String(err),
    });
  }
}
