import { fetchTrades } from "@/lib/polymarket";
import { fail, limitOf, ok } from "@/lib/api-util";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  try {
    const trades = await fetchTrades({
      conditionId: q.get("condition") ?? undefined,
      user: q.get("user") ?? undefined,
      limit: limitOf(q.get("limit"), 60, 500),
      minSize: Number(q.get("min") ?? 0) || undefined,
    });
    return ok(trades, 2);
  } catch (err) {
    return fail(err);
  }
}
