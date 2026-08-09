import { fetchMarkets } from "@/lib/polymarket";
import { fail, limitOf, ok } from "@/lib/api-util";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  try {
    const markets = await fetchMarkets({
      limit: limitOf(q.get("limit"), 60, 200),
      offset: Number(q.get("offset") ?? 0) || 0,
      order: q.get("order") ?? "volume24hr",
      ascending: q.get("ascending") === "true",
      tagId: q.get("tag") ?? undefined,
      slug: q.get("slug") ?? undefined,
      ids: q.getAll("id").filter(Boolean),
    });
    return ok(markets);
  } catch (err) {
    return fail(err);
  }
}
