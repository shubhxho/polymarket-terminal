import { expect, test } from "@playwright/test";
import {
  brentSolve,
  type Candle,
  digitalCall,
  digitalProbabilityExtremum,
  digitalPut,
  edgeZ,
  erfc,
  expectedValue,
  forwardFromFunding,
  impliedVolFromDigitalCall,
  impliedVolFromTouch,
  kellyBinary,
  normCdf,
  normInv,
  pickImpliedVol,
  priceDispersionBps,
  probabilityBand,
  quantile,
  touchProbability,
  varianceRatio,
  volSuite,
  wilsonInterval,
  yangZhangVol,
} from "../../src/lib/quant";

/**
 * Unit tests for the quant kernel.
 *
 * The point of this file is that every number the desk prints can be checked
 * against something independent — a closed-form identity, a known constant, or
 * a round-trip through the inverse. Where a property is exact (put-call parity,
 * inversion) it is asserted exactly; where it is statistical (recovering σ from
 * a simulated path) the tolerance is stated and justified.
 */

/**
 * Deterministic normal deviates via a seeded LCG + Box-Muller. Tests must not
 * depend on `Math.random` — a vol estimator that passes one run in three is not
 * a passing test.
 */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1_664_525 + 1_013_904_223) >>> 0;
    return (s >>> 8) / 16_777_216;
  };
}

function gaussian(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    const u = Math.max(rng(), 1e-12);
    const v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

/**
 * Simulate a GBM path and bucket it into OHLC bars, so the range estimators
 * (which need real intrabar highs/lows) see a genuine path rather than a
 * synthetic box.
 */
function simulateCandles(opts: {
  sigmaAnnual: number;
  barMinutes: number;
  bars: number;
  seed: number;
  stepsPerBar?: number;
  spot?: number;
}): Candle[] {
  const { sigmaAnnual, barMinutes, bars, seed } = opts;
  const stepsPerBar = opts.stepsPerBar ?? 250;
  const norm = gaussian(makeRng(seed));
  const barsPerYear = (365 * 24 * 60) / barMinutes;
  const dt = 1 / (barsPerYear * stepsPerBar);
  const vol = sigmaAnnual * Math.sqrt(dt);

  let price = opts.spot ?? 100;
  const out: Candle[] = [];
  for (let b = 0; b < bars; b++) {
    const open = price;
    let high = price;
    let low = price;
    for (let s = 0; s < stepsPerBar; s++) {
      price *= Math.exp(-0.5 * vol * vol + vol * norm());
      if (price > high) high = price;
      if (price < low) low = price;
    }
    out.push({
      t: b * barMinutes * 60_000,
      o: open,
      h: high,
      l: low,
      c: price,
      v: 1,
    });
  }
  return out;
}

/**
 * A price series whose log returns follow AR(1): r_t = phi·r_{t-1} + noise.
 * `phi > 0` trends (variance grows faster than linearly with the horizon),
 * `phi < 0` mean-reverts. This is the process the variance ratio is designed
 * to detect, so it is what the VR tests are written against.
 */
function autocorrelated(phi: number, seed: number, bars = 600): Candle[] {
  const norm = gaussian(makeRng(seed));
  let r = 0;
  let price = 100;
  const out: Candle[] = [];
  for (let i = 0; i < bars; i++) {
    r = phi * r + 0.01 * norm();
    const open = price;
    price *= Math.exp(r);
    out.push({
      t: i * 3_600_000,
      o: open,
      h: Math.max(open, price),
      l: Math.min(open, price),
      c: price,
      v: 1,
    });
  }
  return out;
}

test.describe("normal distribution", () => {
  test("erfc matches known values inside its stated 1.2e-7 error", () => {
    expect(erfc(0)).toBeCloseTo(1, 12);
    // erfc(1) and erfc(2), standard references.
    expect(erfc(1)).toBeCloseTo(0.15729920705028513, 7);
    expect(erfc(2)).toBeCloseTo(0.004677734981063127, 9);
    // Symmetry identity erfc(-x) = 2 - erfc(x) must hold exactly in structure.
    expect(erfc(-1.3) + erfc(1.3)).toBeCloseTo(2, 12);
  });

  test("normCdf hits the textbook quantiles", () => {
    expect(normCdf(0)).toBeCloseTo(0.5, 12);
    expect(normCdf(1.959963984540054)).toBeCloseTo(0.975, 8);
    expect(normCdf(-1.2815515655446004)).toBeCloseTo(0.1, 8);
  });

  test("normInv inverts normCdf to ~1e-12 across the range", () => {
    for (const p of [0.001, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 0.999]) {
      expect(normCdf(normInv(p))).toBeCloseTo(p, 12);
    }
  });

  test("normInv is signed correctly at the tails", () => {
    expect(normInv(0.0001)).toBeLessThan(-3);
    expect(normInv(0.9999)).toBeGreaterThan(3);
    expect(normInv(0)).toBe(Number.NEGATIVE_INFINITY);
    expect(normInv(1)).toBe(Number.POSITIVE_INFINITY);
  });
});

test.describe("digital options", () => {
  const base = { spot: 100, strike: 100, years: 0.25, sigma: 0.6 };

  test("put-call parity holds exactly", () => {
    const c = digitalCall(base);
    const p = digitalPut(base);
    expect(c.probability + p.probability).toBeCloseTo(1, 12);
  });

  test("at-the-money forward, the digital prices BELOW 50%", () => {
    // d2 = -σ√T/2 < 0 when F = K: the lognormal median sits under the mean.
    const c = digitalCall(base);
    expect(c.probability).toBeLessThan(0.5);
    expect(c.probability).toBeCloseTo(normCdf(-0.5 * 0.6 * Math.sqrt(0.25)), 12);
  });

  test("zero vol or zero time collapses to the deterministic indicator", () => {
    // The limit is 1{F >= K}, so at-the-money forward it is 1, not 0.
    expect(digitalCall({ ...base, sigma: 0 }).probability).toBe(1);
    expect(digitalCall({ ...base, sigma: 0, strike: 90 }).probability).toBe(1);
    expect(digitalCall({ ...base, sigma: 0, strike: 110 }).probability).toBe(0);
    expect(digitalCall({ ...base, years: 0, strike: 110 }).probability).toBe(0);
  });

  test("probability rises monotonically as the strike falls", () => {
    let prev = -1;
    for (const strike of [140, 120, 110, 100, 90, 80]) {
      const p = digitalCall({ ...base, strike }).probability;
      expect(p).toBeGreaterThan(prev);
      prev = p;
    }
  });

  test("implied vol round-trips through the closed form", () => {
    const forward = 100;
    const strike = 110;
    const years = 0.1;
    const sigma = 0.6;
    const target = digitalCall({
      spot: forward,
      forward,
      strike,
      years,
      sigma,
    }).probability;

    const roots = impliedVolFromDigitalCall(target, forward, strike, years);
    expect(roots).not.toBeNull();
    const picked = pickImpliedVol(roots, sigma);
    expect(picked).not.toBeNull();
    expect(picked as number).toBeCloseTo(sigma, 6);
  });

  test("an out-of-the-money digital genuinely has TWO implied vols", () => {
    const forward = 100;
    const strike = 130;
    const years = 0.1;
    const roots = impliedVolFromDigitalCall(0.08, forward, strike, years);
    expect(roots).not.toBeNull();
    const { low, high } = roots as { low: number | null; high: number };
    expect(low).not.toBeNull();
    expect(high).toBeGreaterThan(low as number);
    // Both branches must reproduce the same price — that is what makes them roots.
    const pLow = digitalCall({
      spot: forward,
      forward,
      strike,
      years,
      sigma: low as number,
    }).probability;
    const pHigh = digitalCall({
      spot: forward,
      forward,
      strike,
      years,
      sigma: high,
    }).probability;
    expect(pLow).toBeCloseTo(0.08, 6);
    expect(pHigh).toBeCloseTo(0.08, 6);
  });

  test("a price below the OTM cap is unattainable, and the cap is exact", () => {
    const forward = 100;
    const strike = 130;
    const years = 0.1;
    const cap = digitalProbabilityExtremum(forward, strike, years);
    expect(cap).not.toBeNull();
    const { sigma, probability } = cap as { sigma: number; probability: number };

    // The cap is attained at σ*, and nothing exceeds it.
    const atPeak = digitalCall({
      spot: forward,
      forward,
      strike,
      years,
      sigma,
    }).probability;
    expect(atPeak).toBeCloseTo(probability, 10);
    for (const s of [0.05, 0.3, sigma * 0.5, sigma * 2, 5]) {
      const p = digitalCall({
        spot: forward,
        forward,
        strike,
        years,
        sigma: s,
      }).probability;
      expect(p).toBeLessThanOrEqual(probability + 1e-12);
    }
    // The cap is a CEILING: a quote ABOVE it has no σ at all, while one below
    // it has two. Getting this backwards is exactly the bug this test exists
    // to catch.
    expect(
      impliedVolFromDigitalCall(Math.min(probability * 1.5, 0.999), forward, strike, years),
    ).toBeNull();
    expect(impliedVolFromDigitalCall(probability * 0.5, forward, strike, years)).not.toBeNull();
    // In the money there is no interior peak.
    expect(digitalProbabilityExtremum(100, 90, years)).toBeNull();
  });
});

test.describe("one-touch barriers", () => {
  test("touching is strictly more likely than finishing above", () => {
    const spot = 100;
    const barrier = 115;
    const years = 0.25;
    const sigma = 0.6;
    const touch = touchProbability(spot, barrier, years, sigma, 0);
    const terminal = digitalCall({
      spot,
      forward: spot,
      strike: barrier,
      years,
      sigma,
    }).probability;
    expect(touch).toBeGreaterThan(terminal);
    // The reflection term roughly doubles it; this is the modelling error that
    // pricing a "will X hit Y" market as a terminal digital would introduce.
    expect(touch).toBeGreaterThan(terminal * 1.5);
  });

  test("driftless up-touch of the spot barrier is certain, and probabilities stay in [0,1]", () => {
    expect(touchProbability(100, 100, 0.25, 0.6, 0)).toBe(1);
    for (const barrier of [50, 80, 99, 101, 120, 400]) {
      const p = touchProbability(100, barrier, 0.3, 0.7, 0.02);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  test("touch probability is monotone increasing in sigma", () => {
    let prev = -1;
    for (const sigma of [0.1, 0.3, 0.6, 1.0, 2.0]) {
      const p = touchProbability(100, 130, 0.25, sigma, 0);
      expect(p).toBeGreaterThan(prev);
      prev = p;
    }
  });

  test("touch implied vol round-trips through Brent", () => {
    const spot = 100;
    const barrier = 125;
    const years = 0.2;
    const sigma = 0.55;
    const target = touchProbability(spot, barrier, years, sigma, 0);
    const recovered = impliedVolFromTouch(target, spot, barrier, years, 0);
    expect(recovered).not.toBeNull();
    expect(recovered as number).toBeCloseTo(sigma, 5);
  });

  test("a far barrier over no time is unreachable", () => {
    expect(touchProbability(100, 1000, 0, 0.6, 0)).toBe(0);
  });
});

test.describe("realized volatility", () => {
  test("Yang-Zhang recovers the simulated sigma within 15%", () => {
    const sigma = 0.8;
    const candles = simulateCandles({
      sigmaAnnual: sigma,
      barMinutes: 60,
      bars: 700,
      seed: 12345,
    });
    const estimate = yangZhangVol(candles, (365 * 24 * 60) / 60);
    // Sampling error on ~700 bars is a few percent; 15% is a loose band that
    // still fails hard if the estimator is mis-scaled (a wrong bars-per-year
    // constant would be off by multiples, not percent).
    expect(estimate).toBeGreaterThan(sigma * 0.85);
    expect(estimate).toBeLessThan(sigma * 1.15);
  });

  test("every estimator in the suite lands in the same neighbourhood", () => {
    const sigma = 0.5;
    const candles = simulateCandles({
      sigmaAnnual: sigma,
      barMinutes: 60,
      bars: 700,
      seed: 987,
    });
    const suite = volSuite(candles, 60);
    for (const [name, value] of Object.entries({
      closeToClose: suite.closeToClose,
      parkinson: suite.parkinson,
      garmanKlass: suite.garmanKlass,
      rogersSatchell: suite.rogersSatchell,
      yangZhang: suite.yangZhang,
      ewma: suite.ewma,
    })) {
      expect(value, `${name} = ${value}`).toBeGreaterThan(sigma * 0.7);
      expect(value, `${name} = ${value}`).toBeLessThan(sigma * 1.3);
    }
    expect(suite.blended).toBeGreaterThan(sigma * 0.8);
    expect(suite.blended).toBeLessThan(sigma * 1.2);
    // Spread is the dispersion of the estimators, so it must be small here.
    expect(suite.spread).toBeLessThan(sigma * 0.5);
    expect(suite.bars).toBe(candles.length);
  });

  test("the suite degrades to zero rather than NaN on empty input", () => {
    const suite = volSuite([], 60);
    expect(suite.blended).toBe(0);
    expect(Number.isFinite(suite.spread)).toBe(true);
  });

  test("variance ratio is ~1 on a random walk", () => {
    const candles = simulateCandles({
      sigmaAnnual: 0.6,
      barMinutes: 60,
      bars: 900,
      seed: 4242,
    });
    const vr = varianceRatio(candles, 4);
    expect(vr).toBeGreaterThan(0.75);
    expect(vr).toBeLessThan(1.25);
  });

  test("variance ratio exceeds 1 when returns are positively autocorrelated", () => {
    // A constant drift does NOT move VR: variance is measured about the mean,
    // so the drift cancels and a noisy ramp still scores ~1. What VR actually
    // detects is autocorrelation, so the series has to have some — an AR(1)
    // with a positive coefficient is the honest way to produce it.
    expect(varianceRatio(autocorrelated(0.6, 4321), 4)).toBeGreaterThan(1.5);
  });

  test("variance ratio falls below 1 when returns mean-revert", () => {
    // The mirror case: a negative AR(1) coefficient means each move is partly
    // given back, so aggregated variance grows slower than linearly.
    expect(varianceRatio(autocorrelated(-0.6, 4321), 4)).toBeLessThan(0.75);
  });
});

test.describe("carry and forwards", () => {
  test("positive funding lifts the forward above spot", () => {
    const f = forwardFromFunding(100, 0.0001, 24 * 30);
    expect(f).toBeGreaterThan(100);
    expect(f).toBeCloseTo(100 * 1.0001 ** 720, 9);
  });

  test("zero or invalid inputs return spot unchanged", () => {
    expect(forwardFromFunding(100, 0, 720)).toBe(100);
    expect(forwardFromFunding(100, Number.NaN, 720)).toBe(100);
    expect(forwardFromFunding(100, 0.0001, 0)).toBe(100);
  });
});

test.describe("sizing and uncertainty", () => {
  test("Kelly is signed by which side is cheap", () => {
    // Model 60%, market 50% ⇒ buy YES.
    expect(kellyBinary(0.6, 0.5)).toBeGreaterThan(0);
    // Model 40%, market 50% ⇒ sell.
    expect(kellyBinary(0.4, 0.5)).toBeLessThan(0);
    // No edge ⇒ no stake.
    expect(kellyBinary(0.5, 0.5)).toBeCloseTo(0, 12);
    // Closed form for the buy side: (p − q)/(1 − q).
    expect(kellyBinary(0.6, 0.5)).toBeCloseTo(0.2, 12);
    // Fractional Kelly scales linearly and never leaves [-1, 1].
    expect(kellyBinary(0.6, 0.5, 0.25)).toBeCloseTo(0.05, 12);
    expect(kellyBinary(1, 0.01)).toBeLessThanOrEqual(1);
    expect(kellyBinary(0, 0.99)).toBeGreaterThanOrEqual(-1);
  });

  test("expected value is zero exactly at fair value", () => {
    expect(expectedValue(0.5, 0.5)).toBeCloseTo(0, 12);
    expect(expectedValue(0.6, 0.5)).toBeCloseTo(0.2, 12);
  });

  test("the probability band widens with estimator disagreement", () => {
    const tight = probabilityBand(0.5, -0.4, 0.05);
    const wide = probabilityBand(0.5, -0.4, 0.4);
    expect(wide.width).toBeGreaterThan(tight.width);
    // Bands are clamped into [0,1] — a probability cannot sit outside it.
    const clamped = probabilityBand(0.02, -5, 2);
    expect(clamped.lo).toBeGreaterThanOrEqual(0);
    expect(clamped.hi).toBeLessThanOrEqual(1);
  });

  test("edge z-score reports conviction relative to the band", () => {
    expect(edgeZ(0.06, 0.03)).toBeCloseTo(2, 12);
    expect(edgeZ(-0.06, 0.03)).toBeCloseTo(-2, 12);
    // A zero-width band cannot divide; the sentinel must stay finite.
    expect(Number.isFinite(edgeZ(0.05, 0))).toBe(true);
    expect(edgeZ(0, 0)).toBe(0);
  });

  test("Wilson interval stays inside [0,1] and tightens with n", () => {
    const small = wilsonInterval(0.02, 10);
    const large = wilsonInterval(0.02, 10_000);
    expect(small.lo).toBeGreaterThanOrEqual(0);
    expect(small.hi).toBeLessThanOrEqual(1);
    expect(large.hi - large.lo).toBeLessThan(small.hi - small.lo);
  });

  test("price dispersion is measured in bps of the median", () => {
    expect(priceDispersionBps([100, 100, 100])).toBeCloseTo(0, 12);
    // 1% spread around 100 is 100bps.
    expect(priceDispersionBps([99.5, 100, 100.5])).toBeCloseTo(100, 6);
    // Junk feeds are dropped, not averaged in.
    expect(priceDispersionBps([100, Number.NaN, 0, -5])).toBe(0);
  });

  test("quantile interpolates linearly", () => {
    const xs = [1, 2, 3, 4];
    expect(quantile(xs, 0)).toBe(1);
    expect(quantile(xs, 1)).toBe(4);
    expect(quantile(xs, 0.5)).toBeCloseTo(2.5, 12);
    // It must not mutate the caller's array.
    const original = [3, 1, 2];
    quantile(original, 0.5);
    expect(original).toEqual([3, 1, 2]);
  });
});

test.describe("root finding", () => {
  test("Brent solves a bracketed root", () => {
    const root = brentSolve((x) => x * x - 2, { lo: 0, hi: 4 });
    expect(root).not.toBeNull();
    expect(root as number).toBeCloseTo(Math.SQRT2, 9);
  });

  test("Brent returns null rather than inventing an unbracketed root", () => {
    expect(brentSolve((x) => x * x + 1, { lo: 0, hi: 4 })).toBeNull();
  });
});
