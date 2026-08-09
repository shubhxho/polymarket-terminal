import { fetchMarkets } from "@/lib/polymarket";
import { fail, ok } from "@/lib/api-util";

export type TapeItem = {
  slug: string;
  label: string;
  price: number;
  chg24h: number;
  tokenId: string;
};

export type Summary = {
  volume24h: number;
  liquidity: number;
  openMarkets: number;
  advancers: number;
  decliners: number;
  unchanged: number;
  /** Most-moved markets, for the scrolling tape. */
  tape: TapeItem[];
};

/**
 * Market-wide breadth, powering the status bar and ticker tape.
 *
 * Derived from the top 200 markets by turnover rather than the whole book:
 * the long tail is thousands of near-zero-volume markets that would swamp the
 * advance/decline read without changing what a trader cares about.
 */
export async function GET() {
  try {
    const markets = await fetchMarkets({ limit: 200, order: "volume24hr" });

    let volume24h = 0;
    let liquidity = 0;
    let advancers = 0;
    let decliners = 0;
    let unchanged = 0;

    for (const m of markets) {
      volume24h += m.volume24h;
      liquidity += m.liquidity;
      const c = m.chg24h ?? 0;
      if (c > 0.05) advancers++;
      else if (c < -0.05) decliners++;
      else unchanged++;
    }

    const tape: TapeItem[] = markets
      .filter((m) => m.outcomes[0]?.tokenId && m.volume24h > 1000)
      .slice(0, 60)
      .map((m) => ({
        slug: m.eventSlug || m.slug,
        label: (m.groupItemTitle || m.question).slice(0, 44),
        price: m.last,
        chg24h: m.chg24h ?? 0,
        tokenId: m.outcomes[0].tokenId,
      }));

    const summary: Summary = {
      volume24h,
      liquidity,
      openMarkets: markets.length,
      advancers,
      decliners,
      unchanged,
      tape,
    };
    return ok(summary, 20);
  } catch (err) {
    return fail(err);
  }
}
