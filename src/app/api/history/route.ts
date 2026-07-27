import { fetchHistory } from "@/lib/polymarket";
import { fail, ok } from "@/lib/api-util";
import type { HistoryInterval } from "@/lib/types";

const INTERVALS: HistoryInterval[] = ["1h", "6h", "1d", "1w", "1m", "max"];

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const tokens = (q.get("tokens") ?? q.get("token") ?? "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean)
    .slice(0, 8);
  const raw = q.get("interval") as HistoryInterval | null;
  const interval: HistoryInterval = raw && INTERVALS.includes(raw) ? raw : "1d";
  if (tokens.length === 0) return ok([], 0);
  try {
    // One slow token shouldn't blank the whole chart — settle and drop failures.
    const settled = await Promise.allSettled(
      tokens.map((t) => fetchHistory(t, interval))
    );
    const series = settled.map((r, i) => ({
      tokenId: tokens[i],
      points: r.status === "fulfilled" ? r.value : [],
    }));
    return ok(series, 15);
  } catch (err) {
    return fail(err);
  }
}
