import Link from "next/link";
import type { DerivativeQuote } from "@/lib/derivatives";
import { fmtUsd } from "@/lib/polymarket";

/**
 * One priced claim.
 *
 * The centrepiece is a single 0–100% probability track carrying THREE things at
 * once: where the market is, where the model is, and how wide the model's own
 * uncertainty band is. That last one is the whole point — an edge drawn without
 * its band is the most common way a desk lies to itself, so the band is a
 * first-class mark here, not a footnote.
 *
 * Identity is never colour-alone: the market and model marks are both directly
 * labelled, which is also what makes the phosphor palette legal for CVD readers
 * (green↔cyan clears the CVD floor, but the labels carry it regardless).
 */

const pct = (p: number) => `${(p * 100).toFixed(1)}%`;
const pp = (x: number) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}pp`;
const vol = (s: number | null) =>
  s == null || !Number.isFinite(s) ? "—" : `${(s * 100).toFixed(0)}%`;

/** Horizon, in the coarsest unit that still reads precisely. */
function horizon(hours: number): string {
  if (hours < 1) return `${Math.round(hours * 60)}M`;
  if (hours < 48) return `${hours.toFixed(1)}H`;
  return `${(hours / 24).toFixed(1)}D`;
}

/**
 * Conviction tiers. |z| is the edge measured in units of the model's own
 * uncertainty — below 1 the "edge" is entirely inside the noise of our vol
 * estimate, and the row says so in words rather than showing a green number.
 */
function conviction(z: number): { label: string; tone: string } {
  const a = Math.abs(z);
  if (a >= 2) return { label: "STRONG", tone: "text-accent" };
  if (a >= 1) return { label: "MARGINAL", tone: "text-amber" };
  return { label: "INSIDE NOISE", tone: "text-muted" };
}

export function DerivativesRow({ quote: q }: { quote: DerivativeQuote }) {
  const conv = conviction(q.z);
  const rich = q.edge > 0; // model above market ⇒ the YES side looks cheap
  const edgeTone = rich ? "text-accent" : "text-red";

  // Band geometry on the shared 0–100% track.
  const bandLeft = q.band.lo * 100;
  const bandWidth = Math.max((q.band.hi - q.band.lo) * 100, 0.5);
  const marketX = q.marketProbability * 100;
  const modelX = q.modelProbability * 100;

  const strike = q.claim.strike.toLocaleString("en-US", {
    maximumFractionDigits: 0,
  });

  return (
    <div className="flex flex-col gap-2 bg-panel px-3 py-3 panel-lit">
      {/* Claim header */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] tracking-widest">
        <span className="rounded-sm border border-cyan/30 px-1 py-0.5 text-cyan">
          {q.claim.coin}
        </span>
        <span className="rounded-sm border border-current/25 px-1 py-0.5 text-muted">
          {q.claim.style}
        </span>
        <span className="text-muted/70">
          {q.claim.direction === "UP" ? "≥" : "≤"} ${strike}
        </span>
        <span className="text-muted/40">·</span>
        <span className="text-muted/70">{horizon(q.hoursToExpiry)} LEFT</span>
        <span className={`ml-auto ${conv.tone}`}>
          {conv.label} · Z {q.z >= 0 ? "+" : ""}
          {q.z.toFixed(1)}
        </span>
      </div>

      <Link
        href={`/event/${q.slug}`}
        className="truncate text-xs text-foreground hover:text-accent"
      >
        {q.title}
      </Link>

      {/* Probability track: market mark, model mark, model uncertainty band */}
      <div className="mt-0.5">
        <div className="relative h-6 w-full rounded-sm bg-panel-raised ring-1 ring-inset ring-edge">
          {/* Model uncertainty band — the range no estimate can resolve inside */}
          <div
            className="absolute inset-y-0 bg-accent/15"
            style={{ left: `${bandLeft}%`, width: `${bandWidth}%` }}
          />
          {/* Market mark */}
          <div className="absolute inset-y-0 w-0.5 bg-cyan" style={{ left: `${marketX}%` }} />
          {/* Model mark — 2px surface gap keeps it readable over the band */}
          <div
            className="absolute inset-y-0 w-0.5 bg-accent shadow-[0_0_6px_var(--accent)] ring-2 ring-panel-raised"
            style={{ left: `${modelX}%` }}
          />
        </div>
        {/* Direct labels — identity never rests on colour alone */}
        <div className="mt-1 flex items-center gap-3 text-[10px] tabular-nums">
          <span className="flex items-center gap-1 text-muted">
            <span className="inline-block h-1.5 w-1.5 rounded-sm bg-cyan" />
            MARKET {pct(q.marketProbability)}
          </span>
          <span className="flex items-center gap-1 text-muted">
            <span className="inline-block h-1.5 w-1.5 rounded-sm bg-accent" />
            MODEL {pct(q.modelProbability)}
          </span>
          <span className="text-muted/50">BAND ±{((q.band.width / 2) * 100).toFixed(1)}pp</span>
          <span className={`ml-auto font-bold ${edgeTone}`}>
            {rich ? "▲" : "▼"} EDGE {pp(q.edge)}
          </span>
        </div>
      </div>

      {/* Model inputs — every number that produced the edge above */}
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] tabular-nums sm:grid-cols-4">
        <Field label="σ REALIZED" value={vol(q.vol.blended)} />
        <Field
          label="σ IMPLIED"
          value={q.unattainable ? "UNATTAINABLE" : vol(q.impliedVol)}
          tone={q.unattainable ? "text-amber" : undefined}
          title={
            q.unattainable
              ? "No volatility reproduces this price under GBM — the market is pricing a drift view, a jump, or a real edge."
              : "Volatility the market price implies, inverted through the same model."
          }
        />
        <Field
          label="σ SPREAD"
          value={vol(q.vol.spread)}
          title="Dispersion across the five realized-vol estimators — model risk, made visible."
        />
        <Field
          label="VR(4)"
          value={q.varianceRatio.toFixed(2)}
          tone={q.varianceRatio > 1.25 || q.varianceRatio < 0.75 ? "text-amber" : undefined}
          title="Lo–MacKinlay variance ratio. ≈1 random walk (GBM holds); >1 trending; <1 mean-reverting."
        />
        <Field
          label="SPOT"
          value={`$${q.spot.toLocaleString("en-US", { maximumFractionDigits: 2 })}`}
        />
        <Field
          label="FORWARD"
          value={`$${q.forward.toLocaleString("en-US", { maximumFractionDigits: 2 })}`}
          title="Spot compounded at Hyperliquid's realized hourly funding to the claim's expiry."
        />
        <Field
          label="FUNDING"
          value={`${q.fundingAprPct >= 0 ? "+" : ""}${q.fundingAprPct.toFixed(1)}% APR`}
          tone={q.fundingAprPct >= 0 ? undefined : "text-amber"}
          title="Week-averaged HL funding, annualized. This is the carry the forward is built from."
        />
        <Field
          label="KELLY"
          value={`${q.kelly >= 0 ? "+" : ""}${(q.kelly * 100).toFixed(1)}%`}
          tone={q.kelly >= 0 ? "text-accent" : "text-red"}
          title="Quarter-Kelly stake as a fraction of bankroll. Positive = buy YES, negative = sell."
        />
      </dl>

      {/* Venue context — is the underlying's own market corroborating? */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-edge/60 pt-1.5 text-[10px] tabular-nums text-muted/50">
        <span title="Perp mark vs validator-median oracle.">
          BASIS {q.basisBps >= 0 ? "+" : ""}
          {q.basisBps.toFixed(1)}BPS
        </span>
        <span title="Spread across HL's oracle, mark and mid — a staleness check.">
          DISPERSION {q.dispersionBps.toFixed(1)}BPS
        </span>
        {q.crossVenueBpsPerHour != null && (
          <span title="HL funding minus the mean of Binance/Bybit, per hour. A persistent gap is a real cross-venue carry trade.">
            HL−PEERS {q.crossVenueBpsPerHour >= 0 ? "+" : ""}
            {q.crossVenueBpsPerHour.toFixed(2)}BPS/H
          </span>
        )}
        {q.impactSpreadBps != null && (
          <span title="Cost of crossing HL's own book at $20k notional.">
            IMPACT {q.impactSpreadBps.toFixed(1)}BPS
          </span>
        )}
        <span className="ml-auto">HL 24H {fmtUsd(q.perpDayVolume)}</span>
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  tone,
  title,
}: {
  label: string;
  value: string;
  tone?: string;
  title?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2" title={title}>
      <dt className="tracking-widest text-muted/50">{label}</dt>
      <dd className={`font-bold ${tone ?? "text-foreground"}`}>{value}</dd>
    </div>
  );
}
