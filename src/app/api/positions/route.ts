import { fetchPositions } from "@/lib/polymarket";
import { fail, limitOf, ok } from "@/lib/api-util";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const user = (q.get("user") ?? "").trim();
  if (!/^0x[a-fA-F0-9]{40}$/.test(user)) {
    return fail("expected a 0x-prefixed 40-character wallet address", 400);
  }
  try {
    return ok(await fetchPositions(user, limitOf(q.get("limit"), 50, 200)), 5);
  } catch (err) {
    return fail(err);
  }
}
