/**
 * Quant kernel — the pure math the whole terminal is built on.
 *
 * No I/O, no fetch, no framework. Every function here is deterministic and
 * unit-tested (`tests/unit/options.spec.ts`), so the derivative desk can be
 * audited number-by-number instead of trusted.
 *
 * WHAT THIS EXISTS TO DO
 * ----------------------
 * A Polymarket contract on "BTC above $120,000 on Dec 31" is, in options
 * language, a CASH-OR-NOTHING DIGITAL CALL that settles $1. Its price is a
 * risk-neutral probability. Hyperliquid quotes the same underlying continuously
 * — spot, perp mark, funding — and Chainlink on Base publishes an independent
 * on-chain reference for it. So the same claim is priced three ways by three
 * venues on three chains, and the disagreement is measurable.
 *
 * To measure it we need, in order:
 *   1. a forward price      — from Hyperliquid perp funding as cost-of-carry
 *   2. a volatility         — from Hyperliquid OHLC candles (5 estimators)
 *   3. a model probability  — GBM digital / one-touch barrier under (1) and (2)
 *   4. an implied vol       — inverting the Polymarket price back through (3)
 *   5. a risk-sized stake   — Kelly on the binary payoff, after spread
 *
 * Every one of those steps lives below.
 *
 * CONVENTIONS
 *   - Volatility σ is ALWAYS annualized, in decimal (0.65 = 65%/yr).
 *   - Time T is ALWAYS in years (365-day, since crypto trades 24/7/365).
 *   - Probabilities are decimals in [0,1]; "bps" are 1e-4 of a probability point
 *     unless a doc comment says otherwise.
 *   - Rates: `r` is a continuously-compounded discount rate. Prediction markets
 *     are fully collateralized in USDC, so we default r = 0 and let the perp
 *     funding curve carry all the drift. That is the honest choice — see
 *     `forwardFromFunding`.
 *
 * Not financial advice. These are descriptive models of public market data.
 */

/* ────────────────────────────── primitives ────────────────────────────── */

export const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));

/** Year length used everywhere. Crypto has no market holidays. */
export const YEAR_MS = 365 * 24 * 3_600_000;
export const YEAR_HOURS = 365 * 24;

export const msToYears = (ms: number): number => ms / YEAR_MS;

/** Sample mean. Returns 0 for an empty series. */
export function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/**
 * Sample variance with Bessel's correction (n−1). We use the unbiased form
 * because vol estimates here are computed on short windows (tens of bars),
 * where the 1/n estimator is meaningfully biased low.
 */
export function variance(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return s / (xs.length - 1);
}

export const stdev = (xs: number[]): number => Math.sqrt(variance(xs));

/** Linear-interpolated quantile, q ∈ [0,1]. Used for vol-cone percentiles. */
export function quantile(xs: number[], q: number): number {
  if (xs.length === 0) return Number.NaN;
  const s = xs.toSorted((a, b) => a - b);
  const pos = clamp(q, 0, 1) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return s[lo];
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/* ─────────────────────────── normal distribution ─────────────────────────── */

const INV_SQRT_2PI = 0.3989422804014327;

/** Standard normal PDF φ(x). */
export const normPdf = (x: number): number => INV_SQRT_2PI * Math.exp(-0.5 * x * x);

/**
 * Complementary error function, Numerical-Recipes `erfc` (Chebyshev fit).
 * Fractional error < 1.2e-7 everywhere — three orders of magnitude tighter than
 * the tick size of any book we price against, and it stays accurate deep in the
 * tail where cheap rational fits blow up. Tail accuracy matters: a 2¢ market is
 * a 2-sigma event and that is exactly where these signals live.
 */
export function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 2 / (2 + z);
  const ty = 4 * t - 2;
  const cof = [
    // NB: 17-significant-digit literals silently round to the nearest double.
    // These are written at the precision a double can actually hold, so the
    // source matches the value the fit is evaluated with.
    -1.3026537197817094, 6.419697923564902e-1, 1.9476473204185836e-2, -9.561514786808631e-3,
    -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5, -2.0278578112534e-5,
    -1.624290004647e-6, 1.30365583558e-6, 1.5626441722e-8, -8.5238095915e-8, 6.529054439e-9,
    5.059343495e-9, -9.91364156e-10, -2.27365122e-10, 9.6467911e-11, 2.394038e-12, -6.886027e-12,
    8.94487e-13, 3.13092e-13, -1.12708e-13, 3.81e-16, 7.106e-15,
  ];
  let d = 0;
  let dd = 0;
  for (let j = cof.length - 1; j > 0; j--) {
    const tmp = d;
    d = ty * d - dd + cof[j];
    dd = tmp;
  }
  const ans = t * Math.exp(-z * z + 0.5 * (cof[0] + ty * d) - dd);
  return x >= 0 ? ans : 2 - ans;
}

export const erf = (x: number): number => 1 - erfc(x);

/** Standard normal CDF Φ(x). */
export const normCdf = (x: number): number => 0.5 * erfc(-x / Math.SQRT2);

/**
 * Inverse standard normal CDF Φ⁻¹(p) — Acklam's rational approximation with one
 * Halley refinement step, giving ~1e-15 relative accuracy. This is the workhorse
 * of the whole desk: it is how a market price (a probability) becomes a
 * z-score, and therefore how a Polymarket quote becomes an implied vol.
 */
export function normInv(p: number): number {
  if (p <= 0) return Number.NEGATIVE_INFINITY;
  if (p >= 1) return Number.POSITIVE_INFINITY;

  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2,
    -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783,
  ];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let x: number;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    x =
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= pHigh) {
    const q = p - 0.5;
    const r = q * q;
    x =
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x =
      -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }

  // One Halley step against the true CDF kills the last ~9 digits of error.
  const e = normCdf(x) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
  return x - u / (1 + (x * u) / 2);
}

/* ──────────────────────────── root finding ──────────────────────────── */

export interface SolveOptions {
  lo?: number;
  hi?: number;
  tol?: number;
  maxIter?: number;
}

/**
 * Brent's method — bracketing root finder used wherever a closed-form inverse
 * doesn't exist (one-touch implied vol, most notably). Returns null rather than
 * a garbage root when the bracket doesn't straddle zero, so callers can render
 * "unattainable" honestly instead of printing a made-up number.
 */
export function brentSolve(f: (x: number) => number, opts: SolveOptions = {}): number | null {
  const { lo = 1e-6, hi = 10, tol = 1e-10, maxIter = 128 } = opts;
  let a = lo;
  let b = hi;
  let fa = f(a);
  let fb = f(b);
  if (!Number.isFinite(fa) || !Number.isFinite(fb)) return null;
  if (fa === 0) return a;
  if (fb === 0) return b;
  if (fa * fb > 0) return null; // no sign change ⇒ no bracketed root

  let c = a;
  let fc = fa;
  let d = b - a;
  let e = d;

  for (let i = 0; i < maxIter; i++) {
    if (fb * fc > 0) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }
    if (Math.abs(fc) < Math.abs(fb)) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    const tol1 = 2 * Number.EPSILON * Math.abs(b) + 0.5 * tol;
    const xm = 0.5 * (c - b);
    if (Math.abs(xm) <= tol1 || fb === 0) return b;

    if (Math.abs(e) >= tol1 && Math.abs(fa) > Math.abs(fb)) {
      const s = fb / fa;
      let p: number;
      let q: number;
      if (a === c) {
        p = 2 * xm * s;
        q = 1 - s;
      } else {
        const qq = fa / fc;
        const r = fb / fc;
        p = s * (2 * xm * qq * (qq - r) - (b - a) * (r - 1));
        q = (qq - 1) * (r - 1) * (s - 1);
      }
      if (p > 0) q = -q;
      p = Math.abs(p);
      if (2 * p < Math.min(3 * xm * q - Math.abs(tol1 * q), Math.abs(e * q))) {
        e = d;
        d = p / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }
    a = b;
    fa = fb;
    b += Math.abs(d) > tol1 ? d : xm > 0 ? tol1 : -tol1;
    fb = f(b);
  }
  return b;
}

/* ───────────────────────── forwards and carry ───────────────────────── */

/**
 * Turn a Hyperliquid hourly funding rate into a forward price.
 *
 * A perp has no expiry, so it has no forward curve of its own — but funding IS
 * the market-clearing cost of holding the position, which is precisely a
 * cost-of-carry. If longs pay `f` per hour, holding the exposure to horizon
 * `hours` costs `(1+f)^hours`, so the break-even future price a perp long needs
 * is `F = S · (1+f)^hours`. This is the drift we feed the GBM: it is the
 * market's own carry, not a guess.
 *
 * We compound discretely because HL settles funding hourly on the hour, which
 * is what actually hits the account.
 */
export function forwardFromFunding(spot: number, hourlyFunding: number, hours: number): number {
  if (!(spot > 0) || !Number.isFinite(hourlyFunding) || !(hours > 0)) return spot;
  return spot * (1 + hourlyFunding) ** hours;
}

/** Hourly funding → simple APR (rate × 24 × 365), the number desks quote. */
export const fundingApr = (hourlyFunding: number): number => hourlyFunding * YEAR_HOURS;

/** Hourly funding → compounded APY. Diverges hard from APR past ~50%/yr. */
export const fundingApy = (hourlyFunding: number): number => (1 + hourlyFunding) ** YEAR_HOURS - 1;

/**
 * Continuously-compounded carry implied by a forward: μ = ln(F/S)/T. This is
 * the drift term the digital/barrier formulas want.
 */
export function carryDrift(spot: number, forward: number, years: number): number {
  if (!(spot > 0) || !(forward > 0) || !(years > 0)) return 0;
  return Math.log(forward / spot) / years;
}

/** Perp basis vs its oracle, in bps. Positive = perp rich to index. */
export function basisBps(markPx: number, oraclePx: number): number {
  if (!(oraclePx > 0)) return 0;
  return ((markPx - oraclePx) / oraclePx) * 10_000;
}

/* ───────────────────── digital (binary) option pricing ───────────────────── */

export interface DigitalInputs {
  /** Spot price of the underlying. */
  spot: number;
  /** Strike / threshold the claim is about. */
  strike: number;
  /** Time to settlement, in years. */
  years: number;
  /** Annualized volatility (decimal). */
  sigma: number;
  /** Forward price. Defaults to spot (zero carry). */
  forward?: number;
  /** Continuously-compounded discount rate. Default 0 — see file header. */
  rate?: number;
}

export interface DigitalQuote {
  /** Risk-neutral probability the claim settles YES (undiscounted). */
  probability: number;
  /** PV of a $1 cash-or-nothing payout = e^{-rT} · probability. */
  price: number;
  d1: number;
  d2: number;
  /** ∂price/∂σ. Near-ATM digitals have vega ≈ 0 — the sign flips through the strike. */
  vega: number;
  /** ∂price/∂S. Peaks at the strike; this is why digitals are hard to hedge. */
  delta: number;
}

/**
 * Cash-or-nothing digital CALL — "underlying ≥ strike at expiry".
 *
 * Under GBM with forward F, ln(S_T/F) ~ N(−σ²T/2, σ²T), so
 *
 *     P(S_T ≥ K) = Φ(d2),   d2 = [ln(F/K) − σ²T/2] / (σ√T)
 *
 * and the $1 digital is worth e^{-rT}Φ(d2). The greeks come from
 * ∂d2/∂σ = −d1/σ and ∂d2/∂S = 1/(Sσ√T).
 *
 * Degenerate cases are handled explicitly rather than by returning NaN: zero
 * vol or zero time collapses to the deterministic indicator 1{F ≥ K}, which is
 * the correct limit and keeps the desk from rendering blanks at expiry.
 */
export function digitalCall(inp: DigitalInputs): DigitalQuote {
  const { spot, strike, years, sigma } = inp;
  const rate = inp.rate ?? 0;
  const forward = inp.forward ?? spot;
  const df = Math.exp(-rate * years);

  if (!(strike > 0) || !(forward > 0)) {
    return { probability: 0, price: 0, d1: 0, d2: 0, vega: 0, delta: 0 };
  }
  if (!(sigma > 0) || !(years > 0)) {
    const p = forward >= strike ? 1 : 0;
    return {
      probability: p,
      price: df * p,
      d1: p ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY,
      d2: p ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY,
      vega: 0,
      delta: 0,
    };
  }

  const vt = sigma * Math.sqrt(years);
  const d1 = (Math.log(forward / strike) + 0.5 * sigma * sigma * years) / vt;
  const d2 = d1 - vt;
  const probability = normCdf(d2);
  const price = df * probability;
  const vega = -df * normPdf(d2) * (d1 / sigma);
  const delta = spot > 0 ? (df * normPdf(d2)) / (spot * vt) : 0;
  return { probability, price, d1, d2, vega, delta };
}

/** Cash-or-nothing digital PUT — "underlying < strike at expiry". Parity: 1 − call. */
export function digitalPut(inp: DigitalInputs): DigitalQuote {
  const c = digitalCall(inp);
  const df = Math.exp(-(inp.rate ?? 0) * inp.years);
  return {
    probability: 1 - c.probability,
    price: df - c.price,
    d1: c.d1,
    d2: c.d2,
    vega: -c.vega,
    delta: -c.delta,
  };
}

export interface ImpliedVolRoots {
  /** Low-vol branch. Null when only one positive root exists. */
  low: number | null;
  /** High-vol branch. */
  high: number;
}

/**
 * Invert a digital CALL price back to the volatility that produces it —
 * closed form, no iteration.
 *
 *   Φ(d2) = p  ⇒  d2 = Φ⁻¹(p)
 *   d2·σ√T = ln(F/K) − σ²T/2
 *   ⇒ (T/2)σ² + (d2√T)σ − ln(F/K) = 0
 *
 * a quadratic in σ:  σ± = [−d2√T ± √(d2²T + 2T·ln(F/K))] / T
 *
 * TWO SUBTLETIES that make this function look more complicated than the algebra:
 *
 * 1. A digital's price is NOT monotone in σ when the claim is out of the money.
 *    d2(σ) = m/(σ√T) − σ√T/2 with m = ln(F/K) < 0 rises, peaks at
 *    σ* = √(−2m/T), then falls. So an OTM price has TWO implied vols — e.g. a
 *    5% quote 22% out of the money over 5 weeks is reproduced by BOTH 40% vol
 *    and 1000% vol. Returning one of them silently would be a lie by omission;
 *    we return both and let `pickImpliedVol` choose the branch nearest realized.
 *    In the money (m > 0) only the high root is positive and it is unambiguous.
 *
 * 2. That peak is a CEILING, so prices ABOVE it are the unreachable ones. The
 *    OTM price rises to Φ(−√(−2m)) at σ*, then falls back toward 0 as σ→∞, so
 *    every price strictly below the cap has two roots and every price above it
 *    has none — the discriminant d2²T + 2Tm goes negative exactly there. A
 *    market printing OVER that cap is pricing something GBM cannot express — a
 *    drift view, a jump, or a real edge. That returns null, and the desk labels
 *    it UNATTAINABLE rather than inventing a number.
 */
export function impliedVolFromDigitalCall(
  probability: number,
  forward: number,
  strike: number,
  years: number
): ImpliedVolRoots | null {
  if (!(years > 0) || !(forward > 0) || !(strike > 0)) return null;
  const p = clamp(probability, 1e-9, 1 - 1e-9);
  const d2 = normInv(p);
  const m = Math.log(forward / strike); // log-moneyness
  const disc = d2 * d2 * years + 2 * years * m;
  if (disc < 0) return null; // price unreachable for any σ ≥ 0
  const root = Math.sqrt(disc);
  const high = (-d2 * Math.sqrt(years) + root) / years;
  const lowCandidate = (-d2 * Math.sqrt(years) - root) / years;
  if (!(high > 0) || !Number.isFinite(high)) return null;
  return { low: lowCandidate > 1e-9 ? lowCandidate : null, high };
}

/**
 * Choose which implied-vol branch to report. When both roots exist we take the
 * one closer to the realized-vol reference — the market is far likelier to be
 * pricing a 45% vol than a 1000% one, and anchoring on realized is the only
 * non-arbitrary tie-break available.
 */
export function pickImpliedVol(roots: ImpliedVolRoots | null, reference: number): number | null {
  if (!roots) return null;
  if (roots.low === null) return roots.high;
  if (!(reference > 0)) return roots.low;
  return Math.abs(roots.low - reference) <= Math.abs(roots.high - reference)
    ? roots.low
    : roots.high;
}

/**
 * The highest probability a digital call can reach for ANY σ ≥ 0 — i.e. the
 * ceiling above which a quote is unattainable under GBM.
 *
 * Out of the money (m = ln(F/K) < 0) the price peaks at σ* = √(−2m/T), where
 * d2 = −√(−2m) and so the cap is Φ(−√(−2m)) — notably independent of T. In the
 * money d2 is monotone decreasing in σ, so there is no interior turning point
 * and the supremum is 1 as σ→0; we return null there because "the cap is 1" is
 * not a useful thing to render.
 */
export function digitalProbabilityExtremum(
  forward: number,
  strike: number,
  years: number
): { sigma: number; probability: number } | null {
  if (!(years > 0) || !(forward > 0) || !(strike > 0)) return null;
  const m = Math.log(forward / strike);
  if (m >= 0) return null; // monotone in σ; sup = 1 at σ → 0
  const sigma = Math.sqrt((-2 * m) / years);
  return { sigma, probability: normCdf(-Math.sqrt(-2 * m)) };
}

/* ─────────────────────── one-touch barrier probability ─────────────────────── */

/**
 * Probability the underlying EVER trades through a barrier before expiry.
 *
 * This matters because most Polymarket crypto markets are phrased "will BTC
 * HIT $X by <date>", not "close above $X on <date>". Pricing a touch market
 * with a terminal digital systematically UNDER-prices it — the path can cross
 * and come back. Using the wrong one is the single biggest modelling error
 * available here, so we model both and pick by parsed market phrasing.
 *
 * Reflection principle for X_t = νt + σW_t with ν = μ − σ²/2 and b = ln(B/S):
 *
 *   up-touch (B > S):
 *     P = Φ((νT − b)/(σ√T)) + e^{2νb/σ²}·Φ((−b − νT)/(σ√T))
 *   down-touch (B < S):
 *     P = Φ((b − νT)/(σ√T)) + e^{2νb/σ²}·Φ((b + νT)/(σ√T))
 *
 * The exponential term is the reflected path contribution — the mass that
 * touched and came back. Ignore it and you lose roughly half the probability
 * for a near barrier.
 */
export function touchProbability(
  spot: number,
  barrier: number,
  years: number,
  sigma: number,
  drift = 0
): number {
  if (!(spot > 0) || !(barrier > 0)) return 0;
  if (barrier === spot) return 1;
  if (!(years > 0) || !(sigma > 0)) {
    return (barrier > spot ? spot >= barrier : spot <= barrier) ? 1 : 0;
  }
  const b = Math.log(barrier / spot);
  const nu = drift - 0.5 * sigma * sigma;
  const sT = sigma * Math.sqrt(years);
  const expo = (2 * nu * b) / (sigma * sigma);
  // exp() of a large positive number would overflow; the paired Φ term is
  // vanishing there, so clamp the product instead of returning Infinity.
  const reflect = expo > 700 ? Number.POSITIVE_INFINITY : Math.exp(expo);

  if (barrier > spot) {
    const t1 = normCdf((nu * years - b) / sT);
    const t2 = normCdf((-b - nu * years) / sT);
    const term = Number.isFinite(reflect) ? reflect * t2 : t2 > 0 ? 1 : 0;
    return clamp(t1 + term, 0, 1);
  }
  const t1 = normCdf((b - nu * years) / sT);
  const t2 = normCdf((b + nu * years) / sT);
  const term = Number.isFinite(reflect) ? reflect * t2 : t2 > 0 ? 1 : 0;
  return clamp(t1 + term, 0, 1);
}

/**
 * Implied vol from a one-touch price. No closed form (the barrier formula isn't
 * invertible in σ), so we bracket-solve. Touch probability IS monotone in σ for
 * a fixed barrier, which makes Brent well-behaved here.
 */
export function impliedVolFromTouch(
  probability: number,
  spot: number,
  barrier: number,
  years: number,
  drift = 0
): number | null {
  const p = clamp(probability, 1e-6, 1 - 1e-6);
  return brentSolve((s) => touchProbability(spot, barrier, years, s, drift) - p, {
    lo: 1e-4,
    hi: 8,
  });
}

/* ───────────────────────── realized volatility ───────────────────────── */

export interface Candle {
  /** Bar open time, ms. */
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  /** Base-asset volume. */
  v: number;
}

/** Bars per year for a given bar length in minutes. 24/7 market. */
export const barsPerYear = (barMinutes: number): number => (365 * 24 * 60) / barMinutes;

/**
 * Close-to-close realized vol — the textbook estimator, and the only one that
 * is unbiased under the *actual* discrete sampling we do. Slow to converge:
 * its variance is ~4× Garman-Klass on the same data.
 */
export function closeToCloseVol(candles: Candle[], perYear: number): number {
  if (candles.length < 3) return 0;
  const rets: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const a = candles[i - 1].c;
    const b = candles[i].c;
    if (a > 0 && b > 0) rets.push(Math.log(b / a));
  }
  return Math.sqrt(variance(rets) * perYear);
}

/**
 * Parkinson (1980) — uses the high-low range, which contains far more
 * information about diffusion than two closes do. ~5× more efficient than
 * close-to-close, but assumes zero drift and NO overnight gaps. On a 24/7 perp
 * there are no gaps, which is exactly why it works well on Hyperliquid data.
 *
 *   σ² = 1/(4 ln2 · n) · Σ ln(H/L)²
 */
export function parkinsonVol(candles: Candle[], perYear: number): number {
  const use = candles.filter((k) => k.h > 0 && k.l > 0);
  if (use.length < 2) return 0;
  let s = 0;
  for (const k of use) s += Math.log(k.h / k.l) ** 2;
  const v = s / (4 * Math.LN2 * use.length);
  return Math.sqrt(v * perYear);
}

/**
 * Garman–Klass (1980) — Parkinson plus the open/close information.
 *
 *   σ² = mean[ 0.5·ln(H/L)² − (2ln2 − 1)·ln(C/O)² ]
 *
 * The most efficient of the classical estimators when drift is small; it can go
 * slightly negative on pathological bars, so we floor at zero.
 */
export function garmanKlassVol(candles: Candle[], perYear: number): number {
  const use = candles.filter((k) => k.h > 0 && k.l > 0 && k.o > 0 && k.c > 0);
  if (use.length < 2) return 0;
  let s = 0;
  for (const k of use) {
    s += 0.5 * Math.log(k.h / k.l) ** 2 - (2 * Math.LN2 - 1) * Math.log(k.c / k.o) ** 2;
  }
  return Math.sqrt(Math.max(0, s / use.length) * perYear);
}

/**
 * Rogers–Satchell (1991) — the drift-robust range estimator. Garman-Klass
 * assumes a driftless process and inflates when the asset trends; RS does not.
 * On a strongly trending week (very common in crypto) RS is the one to trust.
 *
 *   σ² = mean[ ln(H/C)ln(H/O) + ln(L/C)ln(L/O) ]
 */
export function rogersSatchellVol(candles: Candle[], perYear: number): number {
  const use = candles.filter((k) => k.h > 0 && k.l > 0 && k.o > 0 && k.c > 0);
  if (use.length < 2) return 0;
  let s = 0;
  for (const k of use) {
    s += Math.log(k.h / k.c) * Math.log(k.h / k.o) + Math.log(k.l / k.c) * Math.log(k.l / k.o);
  }
  return Math.sqrt(Math.max(0, s / use.length) * perYear);
}

/**
 * Yang–Zhang (2000) — the best available: minimum-variance combination of
 * overnight (close→open), open→close, and Rogers-Satchell range variance.
 * Drift-independent AND gap-robust.
 *
 *   σ² = σ²_overnight + k·σ²_openclose + (1−k)·σ²_RS
 *   k  = 0.34 / (1.34 + (n+1)/(n−1))
 *
 * We report this as the headline realized vol.
 */
export function yangZhangVol(candles: Candle[], perYear: number): number {
  const use = candles.filter((k) => k.h > 0 && k.l > 0 && k.o > 0 && k.c > 0);
  const n = use.length - 1;
  if (n < 3) return closeToCloseVol(candles, perYear);

  const overnight: number[] = [];
  const openClose: number[] = [];
  let rs = 0;
  for (let i = 1; i < use.length; i++) {
    const p = use[i - 1];
    const k = use[i];
    overnight.push(Math.log(k.o / p.c));
    openClose.push(Math.log(k.c / k.o));
    rs += Math.log(k.h / k.c) * Math.log(k.h / k.o) + Math.log(k.l / k.c) * Math.log(k.l / k.o);
  }
  const vOvernight = variance(overnight);
  const vOpenClose = variance(openClose);
  const vRs = rs / n;
  const kCoef = 0.34 / (1.34 + (n + 1) / (n - 1));
  const v = vOvernight + kCoef * vOpenClose + (1 - kCoef) * vRs;
  return Math.sqrt(Math.max(0, v) * perYear);
}

/**
 * EWMA / RiskMetrics vol, λ = 0.94 for daily-equivalent decay.
 *
 *   σ²_t = λσ²_{t−1} + (1−λ)r²_t
 *
 * Unlike the window estimators this has no cliff: a shock decays smoothly
 * instead of dropping out of the sample on some arbitrary bar. It is the right
 * estimator for "what is vol RIGHT NOW", which is what a short-dated digital
 * actually cares about.
 */
export function ewmaVol(candles: Candle[], perYear: number, lambda = 0.94): number {
  if (candles.length < 3) return 0;
  const rets: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const a = candles[i - 1].c;
    const b = candles[i].c;
    if (a > 0 && b > 0) rets.push(Math.log(b / a));
  }
  if (rets.length < 2) return 0;
  // Seed with the sample variance of the first third, then decay through.
  const seed = Math.max(3, Math.floor(rets.length / 3));
  let v = variance(rets.slice(0, seed));
  for (let i = seed; i < rets.length; i++) {
    v = lambda * v + (1 - lambda) * rets[i] * rets[i];
  }
  return Math.sqrt(Math.max(0, v) * perYear);
}

export interface VolSuite {
  closeToClose: number;
  parkinson: number;
  garmanKlass: number;
  rogersSatchell: number;
  yangZhang: number;
  ewma: number;
  /** Headline estimate fed to the pricer — a blend, see `blendedVol`. */
  blended: number;
  /** Dispersion across estimators — model risk, made visible. */
  spread: number;
  bars: number;
}

/**
 * Blend the estimators rather than picking one.
 *
 * Each has a different failure mode (drift bias, gap bias, window cliff), and
 * they fail on different data. A weighted mean with more weight on the two
 * robust estimators (Yang-Zhang, EWMA) is materially more stable out-of-sample
 * than any single one — and the SPREAD between them is itself the honest
 * measure of how much model risk sits under the probability we print.
 */
export function volSuite(candles: Candle[], barMinutes: number): VolSuite {
  const perYear = barsPerYear(barMinutes);
  const closeToClose = closeToCloseVol(candles, perYear);
  const parkinson = parkinsonVol(candles, perYear);
  const garmanKlass = garmanKlassVol(candles, perYear);
  const rogersSatchell = rogersSatchellVol(candles, perYear);
  const yangZhang = yangZhangVol(candles, perYear);
  const ewma = ewmaVol(candles, perYear);

  const parts: [number, number][] = [
    [yangZhang, 0.3],
    [ewma, 0.3],
    [rogersSatchell, 0.15],
    [garmanKlass, 0.1],
    [parkinson, 0.1],
    [closeToClose, 0.05],
  ];
  let num = 0;
  let den = 0;
  for (const [v, w] of parts) {
    if (v > 0 && Number.isFinite(v)) {
      num += v * w;
      den += w;
    }
  }
  const blended = den > 0 ? num / den : 0;
  const live = parts.map(([v]) => v).filter((v) => v > 0 && Number.isFinite(v));
  const spread = live.length > 1 ? Math.max(...live) - Math.min(...live) : 0;

  return {
    closeToClose,
    parkinson,
    garmanKlass,
    rogersSatchell,
    yangZhang,
    ewma,
    blended,
    spread,
    bars: candles.length,
  };
}

/**
 * Lo–MacKinlay variance ratio, VR(q) = Var(q-period return) / (q · Var(1-period)).
 *
 *   VR ≈ 1  random walk — GBM assumption holds, trust the digital price
 *   VR > 1  trending / positively autocorrelated — GBM UNDER-states touch odds
 *   VR < 1  mean-reverting — GBM OVER-states them
 *
 * This is the diagnostic that tells you when to distrust your own model, which
 * is why it ships next to the price rather than in a footnote.
 */
export function varianceRatio(candles: Candle[], q = 4): number {
  if (candles.length < q * 4) return 1;
  const rets: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const a = candles[i - 1].c;
    const b = candles[i].c;
    if (a > 0 && b > 0) rets.push(Math.log(b / a));
  }
  if (rets.length < q * 3) return 1;
  const v1 = variance(rets);
  if (!(v1 > 0)) return 1;
  const agg: number[] = [];
  for (let i = 0; i + q <= rets.length; i += q) {
    let s = 0;
    for (let j = 0; j < q; j++) s += rets[i + j];
    agg.push(s);
  }
  const vq = variance(agg);
  return vq / (q * v1);
}

/* ────────────────────────── sizing and edge ────────────────────────── */

/**
 * Kelly fraction for a binary contract.
 *
 * Buying YES at price `q` pays $1 on YES: win (1−q)/q per $1 staked, lose 1.
 * Kelly f* = (p·b − (1−p))/b with b = (1−q)/q simplifies to the clean form
 *
 *     f* = (p − q) / (1 − q)
 *
 * Selling YES (equivalently buying NO at 1−q) with true probability p gives
 *
 *     f* = (q − p) / q
 *
 * Returns the signed fraction of bankroll: positive = buy YES, negative = sell.
 * Full Kelly is the growth-optimal but wildly volatile bet; `fraction` scales it
 * (0.25 is the desk default — quarter-Kelly gives ~94% of the growth at ~44% of
 * the drawdown, and it is the right answer when p itself is estimated).
 */
export function kellyBinary(p: number, q: number, fraction = 1): number {
  const pp = clamp(p, 0, 1);
  const qq = clamp(q, 1e-6, 1 - 1e-6);
  const f = pp > qq ? (pp - qq) / (1 - qq) : (pp - qq) / qq;
  return clamp(f * fraction, -1, 1);
}

/** Expected value per $1 staked buying YES at `q` when true prob is `p`. */
export const expectedValue = (p: number, q: number): number => (q > 0 ? p / q - 1 : 0);

/**
 * Wilson score interval — a confidence band on a probability estimate.
 *
 * Used to answer "is this edge bigger than the noise in my own vol estimate?".
 * Preferred over the normal (Wald) interval because it stays inside [0,1] and
 * doesn't collapse to zero width at p near 0 or 1 — precisely where cheap
 * prediction-market contracts live.
 */
export function wilsonInterval(p: number, n: number, z = 1.96): { lo: number; hi: number } {
  if (!(n > 0)) return { lo: 0, hi: 1 };
  const pp = clamp(p, 0, 1);
  const d = 1 + (z * z) / n;
  const centre = (pp + (z * z) / (2 * n)) / d;
  const half = (z * Math.sqrt((pp * (1 - pp)) / n + (z * z) / (4 * n * n))) / d;
  return { lo: clamp(centre - half, 0, 1), hi: clamp(centre + half, 0, 1) };
}

/**
 * Irreducible model uncertainty, as a half-width in probability. Nothing this
 * desk prices is known to better than roughly half a point.
 */
export const MIN_BAND_HALF_WIDTH = 0.005;

/**
 * Translate the vol estimator spread into a probability band on the model.
 *
 * A digital's sensitivity to vol is its vega. If our estimators disagree by
 * Δσ, the model probability is uncertain by roughly |vega| · Δσ. Reporting the
 * probability without this band is the most common way a desk lies to itself:
 * a "6-point edge" on a contract whose model band is ±9 points is not an edge.
 */
export function probabilityBand(
  probability: number,
  vega: number,
  sigmaSpread: number
): { lo: number; hi: number; width: number } {
  // Vega-derived width alone goes to ZERO deep in or out of the money, where a
  // digital stops caring about vol. Taken literally that claims the model is
  // infinitely certain, and any edge divided by it explodes — a 0.5pp
  // disagreement on a 99.5c contract scored z = 3e7 against live data and
  // swamped the whole desk ranking.
  //
  // Vol dispersion is not the only thing we can be wrong about. Jumps, the
  // discreteness of a 1c tick, a stale oracle and the GBM assumption itself all
  // survive at zero vega, so the band carries a floor representing that
  // irreducible model risk.
  const half = Math.max(Math.abs(vega) * Math.max(0, sigmaSpread) * 0.5, MIN_BAND_HALF_WIDTH);
  return {
    lo: clamp(probability - half, 0, 1),
    hi: clamp(probability + half, 0, 1),
    width: 2 * half,
  };
}

/** Signed z-score of an edge given its uncertainty band half-width. */
export function edgeZ(edge: number, bandHalfWidth: number): number {
  if (!(bandHalfWidth > 0)) return edge === 0 ? 0 : Math.sign(edge) * 99;
  return edge / bandHalfWidth;
}

/**
 * Dispersion across independent price feeds, in bps of the median.
 *
 * Three venues on three chains should agree to a few bps. When they don't, one
 * of them is stale, thin, or being pushed — and any model built on a single one
 * of them is quietly wrong. This is the cross-chain sanity check.
 */
export function priceDispersionBps(prices: number[]): number {
  const good = prices.filter((p) => Number.isFinite(p) && p > 0);
  if (good.length < 2) return 0;
  const med = quantile(good, 0.5);
  if (!(med > 0)) return 0;
  return ((Math.max(...good) - Math.min(...good)) / med) * 10_000;
}

/** Median of a numeric series — the robust consensus across venues. */
export const median = (xs: number[]): number => quantile(xs, 0.5);
