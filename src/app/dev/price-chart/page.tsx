"use client";

import { type ChartSeries, PriceChart } from "@/components/price-chart";

/**
 * Deterministic, network-free fixture route used by the Playwright E2E suite.
 * It renders PriceChart with series that share a display label (the real-world
 * "Los Cabos Open, Qualifier" collision) to guard against duplicate React keys.
 * Client-only + static data so tests never depend on the live Polymarket API.
 */

function ramp(base: number, drift: number, n = 24): { t: number; p: number }[] {
  // Deterministic pseudo-noise (no Math.random) so renders are stable.
  return Array.from({ length: n }, (_, i) => {
    const wobble = Math.sin(i * 1.3) * 0.04;
    const p = Math.min(0.99, Math.max(0.01, base + drift * (i / (n - 1)) + wobble));
    return { t: 1_700_000_000 + i * 3600, p };
  });
}

// Two series intentionally share the same label but have distinct ids.
const FIXTURE: ChartSeries[] = [
  {
    id: "tok-1001",
    label: "Los Cabos Open, Qualifier",
    points: ramp(0.42, 0.25),
  },
  {
    id: "tok-1002",
    label: "Los Cabos Open, Qualifier",
    points: ramp(0.31, -0.12),
  },
  { id: "tok-1003", label: "Miami Masters, Final", points: ramp(0.6, 0.1) },
];

export default function PriceChartFixture() {
  return (
    <main style={{ padding: 24 }}>
      <h1 data-testid="fixture-heading">PriceChart fixture</h1>
      <div data-testid="chart-wrap" style={{ maxWidth: 900 }}>
        <PriceChart series={FIXTURE} />
      </div>
    </main>
  );
}
