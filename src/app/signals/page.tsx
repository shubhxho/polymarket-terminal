import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import { SignalCard } from "@/components/signal-card";
import { getTopEvents } from "@/lib/polymarket";
import { type EdgeKind, type EdgeSignal, scanEdges } from "@/lib/signals-plus";

export const metadata: Metadata = {
  title: "EDGE SCANNER",
  description:
    "Board-wide scan for arbitrage, momentum and liquidity signals on Polymarket. Not financial advice.",
};

const TAGS = [
  { slug: "", label: "ALL MARKETS" },
  { slug: "politics", label: "POLITICS" },
  { slug: "sports", label: "SPORTS" },
  { slug: "crypto", label: "CRYPTO" },
  { slug: "economy", label: "ECONOMY" },
  { slug: "geopolitics", label: "GEOPOLITICS" },
];

const KINDS = [
  { key: "all", label: "ALL" },
  { key: "arb", label: "ARB" },
  { key: "momentum", label: "MOMENTUM" },
  { key: "liquidity", label: "LIQUIDITY" },
  { key: "resolution", label: "RESOLUTION" },
] as const;

const SORTS = [
  { key: "score", label: "SCORE" },
  { key: "edge", label: "EDGE" },
  { key: "liq", label: "LIQUIDITY" },
] as const;

const HIGH_CONVICTION = 60;
const SCAN_DEPTH = 200;
const SHOW_LIMIT = 60;

interface SearchParams {
  tag?: string;
  kind?: string;
  sort?: string;
}

export default async function SignalsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const { tag = "", kind = "all", sort = "score" } = await searchParams;

  return (
    <main className="flex flex-1 flex-col gap-3">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-1.5 text-xs text-muted">
        <Link href="/" className="hover:text-accent">
          ~/MARKETS
        </Link>
        <span>/</span>
        <span className="text-foreground/60">EDGE-SCANNER</span>
      </nav>

      {/* Header */}
      <div className="border border-edge bg-panel p-4 panel-lit">
        <h1 className="flex items-center gap-2 text-base font-bold text-foreground">
          <span className="glow-soft text-accent">◈</span> EDGE SCANNER
        </h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-muted">
          Scans up to {SCAN_DEPTH} markets for four honestly-defined signals:{" "}
          <span className="text-cyan">ARB</span> — mutually-exclusive YES prices that don&apos;t sum
          to 100% — <span className="text-accent">MOMENTUM</span> — strong 24h moves read against
          the weekly trend — <span className="text-amber">LIQUIDITY</span> — deep books carrying a
          spread a maker could capture — and <span className="text-foreground">RESOLUTION</span> —
          markets settling within days while the outcome is still live. Every score rewards edge
          magnitude and discounts it by how liquid the book actually is.{" "}
          <span className="text-muted/60">Not financial advice.</span>
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <nav className="flex flex-wrap gap-px border border-edge bg-edge">
          {TAGS.map((t) => {
            const active = t.slug === tag;
            return (
              <Link
                key={t.slug}
                href={buildHref({ tag: t.slug, kind, sort })}
                className={`px-3 py-1.5 text-xs tracking-wider transition-colors ${
                  active
                    ? "bg-accent font-bold text-black shadow-[0_0_16px_var(--accent-dim)]"
                    : "bg-panel text-muted hover:bg-panel-raised hover:text-accent"
                }`}
              >
                {t.label}
              </Link>
            );
          })}
        </nav>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex gap-px border border-edge bg-edge">
            {KINDS.map((k) => {
              const active = k.key === kind;
              return (
                <Link
                  key={k.key}
                  href={buildHref({ tag, kind: k.key, sort })}
                  className={`px-3 py-1.5 text-[11px] tracking-wider transition-colors ${
                    active
                      ? "bg-accent font-bold text-black"
                      : "bg-panel text-muted hover:bg-panel-raised hover:text-accent"
                  }`}
                >
                  {k.label}
                </Link>
              );
            })}
          </div>
          <div className="flex items-center gap-px border border-edge bg-edge">
            <span className="bg-panel px-2 py-1.5 text-[10px] tracking-widest text-muted/50">
              SORT
            </span>
            {SORTS.map((sopt) => {
              const active = sopt.key === sort;
              return (
                <Link
                  key={sopt.key}
                  href={buildHref({ tag, kind, sort: sopt.key })}
                  className={`px-2.5 py-1.5 text-[11px] tracking-wider transition-colors ${
                    active
                      ? "bg-accent font-bold text-black"
                      : "bg-panel text-muted hover:bg-panel-raised hover:text-accent"
                  }`}
                >
                  {sopt.label}
                  {active ? " ▼" : ""}
                </Link>
              );
            })}
          </div>
        </div>
      </div>

      <Suspense key={`${tag}|${kind}|${sort}`} fallback={<ScanSkeleton />}>
        <ScanResults tag={tag} kind={kind} sort={sort} />
      </Suspense>

      <p className="text-[11px] text-muted/50">
        <span className="text-accent/60">▪</span> SCORE = EDGE MAGNITUDE × BOOK QUALITY · SIGNALS
        ARE DESCRIPTIVE, NOT RECOMMENDATIONS · PRICES = IMPLIED PROBABILITY
      </p>
    </main>
  );
}

function buildHref(p: { tag: string; kind: string; sort: string }): string {
  const q = new URLSearchParams();
  if (p.tag) q.set("tag", p.tag);
  if (p.kind && p.kind !== "all") q.set("kind", p.kind);
  if (p.sort && p.sort !== "score") q.set("sort", p.sort);
  const s = q.toString();
  return s ? `/signals?${s}` : "/signals";
}

async function ScanResults({ tag, kind, sort }: { tag: string; kind: string; sort: string }) {
  let signals: EdgeSignal[];
  try {
    const events = await getTopEvents(tag || undefined, SCAN_DEPTH, 0);
    signals = scanEdges(events, { limit: SHOW_LIMIT });
  } catch {
    return (
      <div className="border border-red/40 bg-panel p-5 panel-lit">
        <div className="flex items-center gap-2 border-b border-red/20 pb-2 text-[11px] tracking-widest text-red">
          <span className="glow-red">▲</span> SCAN FAILED · CODE 0x2B
        </div>
        <p className="mt-3 text-xs text-muted/70">
          → could not reach the market feed — reload to re-run the scan
          <span className="cursor-blink ml-1 text-red">▊</span>
        </p>
      </div>
    );
  }

  if (signals.length === 0) {
    return (
      <div className="border border-edge bg-panel p-5 panel-lit">
        <div className="flex items-center gap-2 border-b border-edge pb-2 text-[11px] tracking-widest text-muted">
          <span className="text-amber">∅</span> NO SIGNALS
        </div>
        <p className="mt-3 text-xs text-muted/70">
          → no market on this board cleared any signal threshold — try another tag
          <span className="cursor-blink ml-1 text-accent">▊</span>
        </p>
      </div>
    );
  }

  const arbCount = signals.filter((s) => s.kind === "ARB").length;
  const momCount = signals.filter((s) => s.kind === "MOMENTUM").length;
  const liqCount = signals.filter((s) => s.kind === "LIQUIDITY").length;
  const resCount = signals.filter((s) => s.kind === "RESOLUTION").length;
  const buyableArb = signals.filter((s) => s.kind === "ARB" && s.edgeBps > 0).length;
  const highConviction = signals.filter((s) => s.score >= HIGH_CONVICTION).length;
  const topScore = signals.reduce((m, s) => Math.max(m, s.score), 0);
  const bestEdge = signals.reduce((m, s) => (Math.abs(s.edgeBps) > Math.abs(m) ? s.edgeBps : m), 0);
  const total = signals.length;
  const pct = (n: number) => (n / total) * 100;
  const arbPct = pct(arbCount);
  const momPct = pct(momCount);
  const liqPct = pct(liqCount);
  const resPct = pct(resCount);

  const filtered =
    kind === "all" ? signals : signals.filter((s) => s.kind === (kind.toUpperCase() as EdgeKind));

  const sorted =
    sort === "edge"
      ? filtered.toSorted((a, b) => Math.abs(b.edgeBps) - Math.abs(a.edgeBps))
      : sort === "liq"
        ? filtered.toSorted((a, b) => b.liquidity - a.liquidity)
        : filtered; // scanEdges already returns score-descending

  return (
    <>
      {/* Summary stats */}
      <div className="grid grid-cols-2 gap-px border border-edge bg-edge sm:grid-cols-4">
        <ScanStat label="SIGNALS" value={String(total)} accent />
        <ScanStat label="ARB" value={String(arbCount)} tone="text-cyan" />
        <ScanStat label="MOMENTUM" value={String(momCount)} />
        <ScanStat label="LIQUIDITY" value={String(liqCount)} tone="text-amber" />
      </div>

      {/* Composition bar — how the board's edge splits by signal type */}
      <div className="border border-edge bg-panel px-3 py-2.5 panel-lit">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-[10px] tracking-widest text-muted">
          <span>COMPOSITION</span>
          <span className="flex flex-wrap items-center gap-3 tabular-nums text-muted/60">
            <span>{highConviction} HIGH-CONVICTION</span>
            <span className="text-cyan">{buyableArb} BUYABLE</span>
            <span>
              TOP {Math.round(topScore)} · BEST {bestEdge > 0 ? "+" : ""}
              {(bestEdge / 100).toFixed(1)}%
            </span>
          </span>
        </div>
        <div className="flex h-2 w-full overflow-hidden rounded-sm bg-panel-raised ring-1 ring-inset ring-edge">
          <div
            className="h-full bg-cyan/60 shadow-[0_0_8px_-1px_var(--cyan)] transition-all"
            style={{ width: `${arbPct}%` }}
          />
          <div className="h-full bg-accent/60 transition-all" style={{ width: `${momPct}%` }} />
          <div className="h-full bg-amber/60 transition-all" style={{ width: `${liqPct}%` }} />
          <div className="h-full bg-foreground/40 transition-all" style={{ width: `${resPct}%` }} />
        </div>
        <div className="mt-1.5 flex flex-wrap items-center gap-4 text-[10px] tabular-nums text-muted">
          <span className="flex items-center gap-1 text-cyan">
            <span className="inline-block h-1.5 w-1.5 rounded-sm bg-cyan/70" />
            ARB {arbPct.toFixed(0)}%
          </span>
          <span className="flex items-center gap-1 text-accent">
            <span className="inline-block h-1.5 w-1.5 rounded-sm bg-accent/70" />
            MOMENTUM {momPct.toFixed(0)}%
          </span>
          <span className="flex items-center gap-1 text-amber">
            <span className="inline-block h-1.5 w-1.5 rounded-sm bg-amber/70" />
            LIQUIDITY {liqPct.toFixed(0)}%
          </span>
          <span className="flex items-center gap-1 text-foreground/80">
            <span className="inline-block h-1.5 w-1.5 rounded-sm bg-foreground/50" />
            RESOLUTION {resPct.toFixed(0)}%
          </span>
        </div>
      </div>

      {sorted.length === 0 ? (
        <div className="border border-edge bg-panel p-5 text-xs text-muted/70 panel-lit">
          → no {kind.toUpperCase()} signals on this board — switch the filter above
        </div>
      ) : (
        <div className="grid gap-px border border-edge bg-edge sm:grid-cols-2 lg:grid-cols-3">
          {sorted.map((s, i) => (
            <SignalCard key={`${s.kind}-${s.slug}`} signal={s} rank={i + 1} />
          ))}
        </div>
      )}
    </>
  );
}

function ScanStat({
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

function ScanSkeleton() {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-px border border-edge bg-edge sm:grid-cols-4">
        {["SIGNALS", "ARB", "MOMENTUM", "LIQUIDITY"].map((l) => (
          <div key={l} className="bg-panel px-3 py-2.5 panel-lit">
            <div className="text-[10px] tracking-widest text-muted/50">{l}</div>
            <div className="shimmer mt-1 h-5 w-16 rounded-sm bg-panel-raised" />
          </div>
        ))}
      </div>
      <div className="border border-edge bg-panel px-3 py-2.5 panel-lit">
        <div className="mb-2 text-[10px] tracking-widest text-muted/50">COMPOSITION · SCANNING</div>
        <div className="shimmer h-2 w-full rounded-sm bg-panel-raised" />
      </div>
      <div className="scan-sweep grid gap-px border border-edge bg-edge sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }, (_, i) => `signal-card-${i}`).map((id) => (
          <div key={id} className="flex flex-col gap-2 bg-panel px-3 py-2.5 pl-4">
            <div className="shimmer h-3 w-24 rounded-sm bg-panel-raised" />
            <div className="shimmer h-3 w-3/4 rounded-sm bg-panel-raised" />
            <div className="shimmer h-1 w-full rounded-sm bg-panel-raised" />
            <div className="shimmer h-3 w-1/2 rounded-sm bg-panel-raised" />
          </div>
        ))}
      </div>
    </div>
  );
}
