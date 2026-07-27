import { searchPolymarket } from "@/lib/polymarket";
import { fail, limitOf, ok } from "@/lib/api-util";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const query = (q.get("q") ?? "").trim();
  if (!query) return ok({ events: [], markets: [] });
  try {
    return ok(await searchPolymarket(query, limitOf(q.get("limit"), 12, 40)), 15);
  } catch (err) {
    return fail(err);
  }
}
