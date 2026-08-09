import { fetchEvents } from "@/lib/polymarket";
import { fail, limitOf, ok } from "@/lib/api-util";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  try {
    const events = await fetchEvents({
      limit: limitOf(q.get("limit"), 30, 100),
      offset: Number(q.get("offset") ?? 0) || 0,
      order: q.get("order") ?? "volume24hr",
      ascending: q.get("ascending") === "true",
      tagId: q.get("tag") ?? undefined,
      slug: q.get("slug") ?? undefined,
    });
    return ok(events);
  } catch (err) {
    return fail(err);
  }
}
