import { fetchHolders } from "@/lib/polymarket";
import { fail, limitOf, ok } from "@/lib/api-util";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const condition = q.get("condition");
  if (!condition) return ok([], 0);
  try {
    return ok(await fetchHolders(condition, limitOf(q.get("limit"), 10, 50)), 30);
  } catch (err) {
    return fail(err);
  }
}
