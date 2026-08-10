/**
 * Live smoke check against the real Gamma and Hyperliquid APIs.
 *
 * The unit suite runs on fixtures, so it cannot catch a changed upstream shape,
 * a renamed field, or a degenerate value that only appears in real data — both
 * the collapsed uncertainty band and the dust-hedge coverage blow-up were found
 * here and nowhere else. Run it before trusting a number the desk prints.
 *
 *   bun run verify:live
 *
 * Read-only: it fetches public endpoints and places no orders.
 */
import {
  getPerpContexts, getFundingComparison, getCandles,
  getPerpBook, getFundingHistory, intervalForHorizon, INTERVAL_MINUTES,
} from "../src/lib/hyperliquid";
import { fetchMarkets, fetchEvents } from "../src/lib/polymarket";
import { buildDesk, matchCoin, parseClaim } from "../src/lib/derivatives";
import { volSuite } from "../src/lib/options";

const ok = (l: string, v: unknown) => console.log(`  ✓ ${l}:`, v);

console.log("\n── HYPERLIQUID ──");
const perps = await getPerpContexts();
ok("perp universe", `${perps.length} coins`);
const btc = perps.find((p) => p.coin === "BTC");
if (!btc) throw new Error("BTC missing from HL universe");
ok("BTC oracle/mark", `${btc.oraclePx} / ${btc.markPx}`);
ok("BTC funding hourly", btc.fundingHourly);
ok("BTC impact bid/ask", `${btc.impactBid} / ${btc.impactAsk}`);

const funding = await getFundingComparison();
const btcF = funding.get("BTC");
ok("cross-venue coins", funding.size);
ok("BTC venues", btcF?.venues.map((v) => `${v.venue}@${v.intervalHours}h`).join(" "));
ok("BTC HL-vs-peers bps/h", btcF?.dislocationBpsPerHour?.toFixed(3));

const iv = intervalForHorizon(24 * 14);
const candles = await getCandles("BTC", iv, 240);
ok(`candles ${iv}`, `${candles.length} bars`);
const vs = volSuite(candles, INTERVAL_MINUTES[iv]);
ok("vol suite blended/spread", `${(vs.blended * 100).toFixed(1)}% / ${(vs.spread * 100).toFixed(1)}%`);

const book = await getPerpBook("BTC");
ok("L2 ladder", `${book.bids.length} bids / ${book.asks.length} asks, top ${book.bids[0]?.price}/${book.asks[0]?.price}`);
const hist = await getFundingHistory("BTC", 168);
ok("funding history", `${hist.length} prints`);

console.log("\n── GAMMA ──");
const markets = await fetchMarkets({ limit: 300 });
ok("markets", markets.length);
const events = await fetchEvents({ limit: 20 });
ok("events", events.length);
const crypto = markets.filter((m) => matchCoin(m.question) || matchCoin(m.eventTitle ?? ""));
ok("crypto-mentioning", crypto.length);
ok("parseable claims", crypto.filter((m) => parseClaim(m.question) || parseClaim(`${m.eventTitle ?? ""} ${m.groupItemTitle ?? ""}`)).length);

console.log("\n── DESK (end to end) ──");
const desk = await buildDesk(markets, { maxRows: 12 });
ok("priced rows", desk.rows.length);
ok("unparsed", desk.unparsed.length);
ok("perps used", desk.perps.map((p) => p.coin).join(" ") || "none");
for (const r of desk.rows) {
  const sane =
    r.modelProbability >= 0 && r.modelProbability <= 1 &&
    Number.isFinite(r.z) && Math.abs(r.z) < 1e4 &&
    r.band.width > 0 && r.vol.blended > 0;
  console.log(
    `  ${sane ? "✓" : "✗"} ${r.claim.coin} ${r.claim.style} ${r.claim.direction} $${r.claim.strike}`,
    `mkt ${(r.marketProbability * 100).toFixed(1)}% model ${(r.modelProbability * 100).toFixed(1)}%`,
    `z ${r.z.toFixed(2)} band ±${((r.band.width / 2) * 100).toFixed(2)}pp`,
    `σ ${(r.vol.blended * 100).toFixed(0)}% VR ${r.varianceRatio.toFixed(2)}`,
    `| ${r.book ? `hedge ${r.book.hedgeNotionalUsd.toFixed(0)} @ ${r.book.hedgeSlippageBps.toFixed(1)}bps cov ${r.book.depthCoverage.toFixed(1)}x` : "no book"}`,
    `| net ${r.netExpectedUsd === null ? "—" : r.netExpectedUsd.toFixed(0)}`,
  );
}
const bad = desk.rows.filter((r) => !Number.isFinite(r.z) || Math.abs(r.z) >= 1e4);
console.log(`\n${bad.length === 0 ? "PASS" : "FAIL"}: ${bad.length} row(s) with a degenerate z`);
