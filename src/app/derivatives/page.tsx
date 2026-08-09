import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import { DerivativesRow } from "@/components/derivatives-row";
import { buildDesk } from "@/lib/derivatives";
import { usd as fmtUsd } from "@/lib/format";
import { fetchMarkets } from "@/lib/polymarket";
import { fundingApr } from "@/lib/options";

export const metadata: Metadata = {
  title: "DERIVATIVES DESK",
  description:
    "Polymarket crypto claims priced as digital and one-touch options against Hyperliquid's forward and realized volatility. Not financial advice.",
};

/** How deep into the board to look for parseable crypto claims. */
const SCAN_DEPTH = 120;
const MAX_ROWS = 30;

const TAGS = [
  { slug: "crypto", label: "CRYPTO" },
  { slug: "", label: "ALL MARKETS" },
];

export default async function DerivativesPage({
  searchParams,
}: {
  searchParams: Promise<{ tag?: string }>;
}) {
  const { tag = "crypto" } = await searchParams;

  return (
    <main className="flex flex-1 flex-col gap-3">
      <nav className="flex items-center gap-1.5 text-xs text-muted">
        <Link href="/" className="hover:text-accent">
          ~/MARKETS
        </Link>
        <span>/</span>
        <span className="text-foreground/60">DERIVATIVES-DESK</span>
      </nav>

      <div className="border border-edge bg-panel p-4 panel-lit">
        <h1 className="flex items-center gap-2 text-base font-bold text-foreground">
          <span className="glow-soft text-accent">∫</span> DERIVATIVES DESK
        </h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-muted">
          A Polymarket contract on “BTC above $120,000 on Dec 31” is a{" "}
          <span className="text-foreground">cash-or-nothing digital call</span>, and its price is a
          risk-neutral probability. Hyperliquid quotes the same underlying continuously — so the
          same claim is priced twice, by two venues, and the disagreement is measurable. This desk
          takes <span className="text-cyan">HL funding</span> as cost-of-carry to build a forward,
          estimates <span className="text-cyan">σ from five realized-vol estimators</span> on HL
          candles, prices the claim as a digital (or a{" "}
          <span className="text-foreground">one-touch</span>, when the market says “hits” rather
          than “closes above”), and reports the gap against the model&apos;s own uncertainty band.{" "}
          <span className="text-muted/60">
            An edge smaller than its band is not an edge. Not financial advice.
          </span>
        </p>
      </div>

      <nav className="flex flex-wrap gap-px self-start border border-edge bg-edge">
        {TAGS.map((t) => (
          <Link
            key={t.slug}
            href={t.slug ? `/derivatives?tag=${t.slug}` : "/derivatives?tag="}
            className={`px-3 py-1.5 text-xs tracking-wider transition-colors ${
              t.slug === tag
                ? "bg-accent font-bold text-black shadow-[0_0_16px_var(--accent-dim)]"
                : "bg-panel text-muted hover:bg-panel-raised hover:text-accent"
            }`}
          >
            {t.label}
          </Link>
        ))}
      </nav>

      <Suspense key={tag} fallback={<DeskSkeleton />}>
        <Desk tag={tag} />
      </Suspense>

      <p className="text-[11px] leading-relaxed text-muted/50">
        <span className="text-accent/60">▪</span> MODEL = GBM WITH HL FUNDING AS DRIFT · σ BLENDED
        FROM YANG-ZHANG · EWMA · ROGERS-SATCHELL · GARMAN-KLASS · PARKINSON · CLOSE-TO-CLOSE · BAND
        = |VEGA| × ESTIMATOR SPREAD · JUMPS, SMILE AND EARLY RESOLUTION UNMODELED · DESCRIPTIVE, NOT
        A RECOMMENDATION
      </p>
    </main>
  );
}

async function Desk({ tag }: { tag: string }) {
  let desk: Awaited<ReturnType<typeof buildDesk>>;
  try {
    const markets = await fetchMarkets({ limit: SCAN_DEPTH, tagId: tag || undefined });
    desk = await buildDesk(markets, { maxRows: MAX_ROWS });
  } catch {
    return (
      <div className="border border-red/40 bg-panel p-5 panel-lit">
        <div className="flex items-center gap-2 border-b border-red/20 pb-2 text-[11px] tracking-widest text-red">
          <span className="glow-red">▲</span> PRICING FAILED · CODE 0x3C
        </div>
        <p className="mt-3 text-xs text-muted/70">
          → could not reach gamma-api.polymarket.com or api.hyperliquid.xyz — reload to re-price
          <span className="cursor-blink ml-1 text-red">▊</span>
        </p>
      </div>
    );
  }

  if (desk.rows.length === 0) {
    return (
      <div className="border border-edge bg-panel p-5 panel-lit">
        <div className="flex items-center gap-2 border-b border-edge pb-2 text-[11px] tracking-widest text-muted">
          <span className="text-amber">∅</span> NOTHING PRICEABLE
        </div>
        <p className="mt-3 text-xs leading-relaxed text-muted/70">
          → no market on this board resolves to a strike on a Hyperliquid-listed underlying. The
          desk prices claims of the form “&lt;COIN&gt; above / hits $&lt;STRIKE&gt; by &lt;DATE&gt;”
          and deliberately refuses to guess at anything else — a wrong payoff model is worse than no
          model.
          <span className="cursor-blink ml-1 text-accent">▊</span>
        </p>
        {desk.unparsed.length > 0 && <Unparsed rows={desk.unparsed} />}
      </div>
    );
  }

  const { rows } = desk;
  const strong = rows.filter((r) => Math.abs(r.z) >= 2).length;
  const cheap = rows.filter((r) => r.edge > 0).length;
  const unattainable = rows.filter((r) => r.unattainable).length;
  const medianVol =
    rows.length > 0
      ? rows.map((r) => r.vol.blended).toSorted((a, b) => a - b)[Math.floor(rows.length / 2)]
      : 0;

  return (
    <>
      <div className="grid grid-cols-2 gap-px border border-edge bg-edge sm:grid-cols-4">
        <Stat label="CLAIMS PRICED" value={String(rows.length)} accent />
        <Stat
          label="|Z| ≥ 2"
          value={String(strong)}
          tone={strong > 0 ? "text-accent" : undefined}
        />
        <Stat label="MODEL > MARKET" value={String(cheap)} tone="text-cyan" />
        <Stat label="MEDIAN σ" value={`${(medianVol * 100).toFixed(0)}%`} tone="text-amber" />
      </div>

      {desk.perps.length > 0 && <VenueStrip desk={desk} />}

      {unattainable > 0 && (
        <p className="border border-amber/30 bg-panel px-3 py-2 text-[11px] leading-relaxed text-amber/80 panel-lit">
          <span className="mr-1">▲</span>
          {unattainable} CLAIM{unattainable === 1 ? "" : "S"} PRICED WHERE NO VOLATILITY REPRODUCES
          THE QUOTE — the market is expressing a drift view, a jump, or information GBM cannot
          carry. Read those rows as “model does not apply”, not “free edge”.
        </p>
      )}

      <div className="grid gap-px border border-edge bg-edge lg:grid-cols-2">
        {rows.map((r) => (
          <DerivativesRow
            key={`${r.slug}-${r.claim.coin}-${r.claim.strike}-${r.claim.style}`}
            quote={r}
          />
        ))}
      </div>

      {desk.unparsed.length > 0 && (
        <div className="border border-edge bg-panel p-3 panel-lit">
          <Unparsed rows={desk.unparsed} />
        </div>
      )}
    </>
  );
}

/**
 * The underlying's own market, per coin. This is the honesty check on every row
 * above: if HL's basis is wide or its funding has decoupled from Binance/Bybit,
 * the forward those rows are priced off is idiosyncratic rather than corroborated.
 */
function VenueStrip({ desk }: { desk: Awaited<ReturnType<typeof buildDesk>> }) {
  return (
    <div className="border border-edge bg-panel panel-lit">
      <div className="flex items-center gap-2 border-b border-edge px-3 py-2 text-[10px] tracking-widest text-muted">
        <span className="live-dot inline-block h-1.5 w-1.5 rounded-full bg-cyan" />
        HYPERLIQUID · UNDERLYING VENUE
        <span className="ml-auto text-muted/50">
          ORACLE = VALIDATOR MEDIAN · FUNDING = NEXT-HOUR PRINT
        </span>
      </div>
      <div className="grid gap-px bg-edge sm:grid-cols-2 lg:grid-cols-3">
        {desk.perps.map((p) => {
          const apr = fundingApr(p.fundingHourly) * 100;
          const change = p.prevDayPx > 0 ? (p.markPx / p.prevDayPx - 1) * 100 : 0;
          const peers = desk.funding.get(p.coin)?.dislocationBpsPerHour ?? null;
          return (
            <div key={p.coin} className="bg-panel px-3 py-2">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-xs font-bold tracking-widest text-cyan">{p.coin}</span>
                <span className="tabular-nums text-sm font-bold text-foreground">
                  $
                  {p.oraclePx.toLocaleString("en-US", {
                    maximumFractionDigits: 2,
                  })}
                </span>
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] tabular-nums text-muted/60">
                <span className={change >= 0 ? "text-accent" : "text-red"}>
                  {change >= 0 ? "▲" : "▼"} {Math.abs(change).toFixed(2)}% 24H
                </span>
                <span title="Annualized funding. Positive = longs pay, i.e. an upward-sloping forward.">
                  FUND {apr >= 0 ? "+" : ""}
                  {apr.toFixed(1)}%
                </span>
                {peers !== null && (
                  <span title="HL funding minus the Binance/Bybit mean, per hour.">
                    VS PEERS {peers >= 0 ? "+" : ""}
                    {peers.toFixed(2)}BPS/H
                  </span>
                )}
                <span className="ml-auto">OI {fmtUsd(p.openInterest * p.oraclePx)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** What we refused to price, and why. Silence here would read as coverage. */
function Unparsed({ rows }: { rows: { slug: string; title: string; reason: string }[] }) {
  return (
    <details>
      <summary className="cursor-pointer text-[10px] tracking-widest text-muted/60 hover:text-accent">
        {rows.length} MARKET{rows.length === 1 ? "" : "S"} NOT PRICED — WHY
      </summary>
      <ul className="mt-2 flex flex-col gap-1 border-t border-edge pt-2">
        {rows.map((r) => (
          <li
            key={`${r.slug}-${r.title}`}
            className="flex items-baseline justify-between gap-3 text-[10px]"
          >
            <Link href={`/event/${r.slug}`} className="truncate text-muted/70 hover:text-accent">
              {r.title}
            </Link>
            <span className="shrink-0 tracking-widest text-amber/60">{r.reason}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

function Stat({
  label,
  value,
  accent = false,
  tone,
}: {
  label: string;
  value: string;
  accent?: boolean;
  tone?: string;
}) {
  return (
    <div className="relative bg-panel px-3 py-2.5 panel-lit">
      {accent && (
        <span className="absolute inset-x-0 top-0 h-px bg-accent/50 shadow-[0_0_8px_var(--accent-dim)]" />
      )}
      <div className="text-[10px] tracking-widest text-muted">{label}</div>
      <div
        className={`text-lg font-bold tabular-nums ${tone ?? (accent ? "text-accent glow-soft" : "text-foreground")}`}
      >
        {value}
      </div>
    </div>
  );
}

function DeskSkeleton() {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-px border border-edge bg-edge sm:grid-cols-4">
        {["CLAIMS PRICED", "|Z| ≥ 2", "MODEL > MARKET", "MEDIAN σ"].map((l) => (
          <div key={l} className="bg-panel px-3 py-2.5 panel-lit">
            <div className="text-[10px] tracking-widest text-muted/50">{l}</div>
            <div className="shimmer mt-1 h-5 w-16 rounded-sm bg-panel-raised" />
          </div>
        ))}
      </div>
      <div className="scan-sweep grid gap-px border border-edge bg-edge lg:grid-cols-2">
        {["pricing-a", "pricing-b", "pricing-c", "pricing-d"].map((k) => (
          <div key={k} className="flex flex-col gap-2 bg-panel px-3 py-3">
            <div className="shimmer h-3 w-32 rounded-sm bg-panel-raised" />
            <div className="shimmer h-3 w-3/4 rounded-sm bg-panel-raised" />
            <div className="shimmer h-6 w-full rounded-sm bg-panel-raised" />
            <div className="shimmer h-3 w-1/2 rounded-sm bg-panel-raised" />
          </div>
        ))}
      </div>
    </div>
  );
}
