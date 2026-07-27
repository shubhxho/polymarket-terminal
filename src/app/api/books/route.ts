import { fetchBooks } from "@/lib/polymarket";
import { fail, ok } from "@/lib/api-util";

export async function GET(req: Request) {
  const tokens = (new URL(req.url).searchParams.get("tokens") ?? "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean)
    .slice(0, 20);
  try {
    return ok(await fetchBooks(tokens), 0);
  } catch (err) {
    return fail(err);
  }
}
