import type { BookLevel, OrderBook } from "./types";

/**
 * Order-book fill simulation — the honest version of "execution".
 *
 * This models what it would COST to actually take a signal's edge by walking
 * Polymarket's live CLOB order book. It places no orders and signs nothing; it
 * just answers "if I market-bought $X right now, what average price would I get,
 * how much of the quoted edge survives slippage, and where's the depth wall?"
 *
 * The two shapes that matter for our signals:
 *   DIRECTIONAL  buy $X of one outcome (a MOMENTUM bet) — report avg fill and
 *                slippage vs the touch.
 *   ARB BASKET   buy one share of every mutually-exclusive outcome (an ARB) so
 *                exactly one resolves to $1. The guaranteed payout is the SMALLEST
 *                per-leg share count (worst case, the winner is your thinnest leg),
 *                so slippage and depth on any single leg erodes the whole edge.
 */

/** Return asks sorted best-first (lowest price) — the order a buyer walks. */
export function asksAscending(book: OrderBook): BookLevel[] {
  return book.asks.toSorted((a, b) => a.price - b.price);
}

/** Return bids sorted best-first (highest price) — the order a seller walks. */
export function bidsDescending(book: OrderBook): BookLevel[] {
  return book.bids.toSorted((a, b) => b.price - a.price);
}

export interface Fill {
  shares: number;
  spent: number;
  /** Volume-weighted average price paid. 0 if nothing filled. */
  avgPrice: number;
  /** Best (lowest) ask before the walk — the "touch". */
  touch: number;
  /** avgPrice vs touch, in basis points. */
  slippageBps: number;
  /** False if the book ran dry before hitting the target. */
  filled: boolean;
}

const EMPTY_FILL: Fill = {
  shares: 0,
  spent: 0,
  avgPrice: 0,
  touch: 0,
  slippageBps: 0,
  filled: false,
};

function finalize(shares: number, spent: number, touch: number, filled: boolean): Fill {
  if (shares <= 0) return { ...EMPTY_FILL, touch };
  const avgPrice = spent / shares;
  const slippageBps = touch > 0 ? ((avgPrice - touch) / touch) * 10000 : 0;
  return { shares, spent, avgPrice, touch, slippageBps, filled };
}

/** Walk the bids selling up to `targetShares`; stop early if depth runs out. */
export function sellShares(book: OrderBook, targetShares: number): Fill {
  const bids = bidsDescending(book);
  const touch = bids[0]?.price ?? 0;
  let shares = 0;
  let proceeds = 0;
  for (const lvl of bids) {
    if (shares >= targetShares) break;
    const take = Math.min(lvl.size, targetShares - shares);
    shares += take;
    proceeds += take * lvl.price;
  }
  // For a sell, slippage is how far the avg falls BELOW the touch.
  if (shares <= 0) return { ...EMPTY_FILL, touch };
  const avgPrice = proceeds / shares;
  const slippageBps = touch > 0 ? ((touch - avgPrice) / touch) * 10000 : 0;
  return {
    shares,
    spent: proceeds,
    avgPrice,
    touch,
    slippageBps,
    filled: shares >= targetShares - 1e-9,
  };
}

/** Walk the book buying up to `targetShares`; stop early if depth runs out. */
export function buyShares(book: OrderBook, targetShares: number): Fill {
  const asks = asksAscending(book);
  const touch = asks[0]?.price ?? 0;
  let shares = 0;
  let spent = 0;
  for (const lvl of asks) {
    if (shares >= targetShares) break;
    const take = Math.min(lvl.size, targetShares - shares);
    shares += take;
    spent += take * lvl.price;
  }
  return finalize(shares, spent, touch, shares >= targetShares - 1e-9);
}

/** Walk the book spending up to `dollars`; partial-fills the marginal level. */
export function buyDollars(book: OrderBook, dollars: number): Fill {
  const asks = asksAscending(book);
  const touch = asks[0]?.price ?? 0;
  let shares = 0;
  let spent = 0;
  for (const lvl of asks) {
    if (spent >= dollars) break;
    const levelCost = lvl.size * lvl.price;
    if (spent + levelCost <= dollars) {
      shares += lvl.size;
      spent += levelCost;
    } else {
      const affordable = (dollars - spent) / lvl.price;
      shares += affordable;
      spent += affordable * lvl.price;
      break;
    }
  }
  return finalize(shares, spent, touch, spent >= dollars - 1e-9);
}

export interface DirectionalResult {
  kind: "DIRECTIONAL";
  fill: Fill;
  budget: number;
  /** Breakeven = avg fill: resolves $1 (YES) or $0 (NO). */
  upsidePct: number; // return if it resolves YES
}

export function simulateDirectional(book: OrderBook, budget: number): DirectionalResult {
  const fill = buyDollars(book, budget);
  const upsidePct = fill.avgPrice > 0 ? (1 / fill.avgPrice - 1) * 100 : 0;
  return { kind: "DIRECTIONAL", fill, budget, upsidePct };
}

export interface ArbLegResult {
  label: string;
  fill: Fill;
}

export interface ArbResult {
  kind: "ARB";
  budget: number;
  legs: ArbLegResult[];
  /** Baskets actually buyable = min per-leg shares (guarantee is worst leg). */
  baskets: number;
  cost: number;
  /** Guaranteed payout = baskets × $1 (exactly one leg resolves YES). */
  payout: number;
  net: number;
  /** net / cost, in %. Negative once vig + slippage exceed the book edge. */
  realizedPct: number;
  /** True if every leg filled its target — otherwise the basket is incomplete. */
  complete: boolean;
  missingLegs: number;
}

/**
 * Buy-all-outcomes basket. Targets `budget / topOfBookBasketCost` baskets, then
 * walks each leg for real to expose slippage and depth walls.
 *
 * THE SIZING SUBTLETY: legs run dry at different depths, so a first pass fills
 * unequal share counts. Only `min(shares)` of those are actually a basket — the
 * excess on the deeper legs is a NAKED DIRECTIONAL position, not arbitrage.
 * Costing the guarantee against all of it (including the naked overhang) prices
 * a trade nobody would place. So we discover the achievable basket count, then
 * re-walk every leg for exactly that many shares. `net` and `realizedPct` then
 * describe a real, executable, fully-hedged basket.
 */
export function simulateArbBasket(
  legs: { label: string; book: OrderBook }[],
  budget: number
): ArbResult | null {
  if (legs.length === 0) return null;
  const touches = legs.map((l) => asksAscending(l.book)[0]?.price ?? 0);
  const basketCost = touches.reduce((s, p) => s + p, 0);
  if (basketCost <= 0) return null;

  const targetBaskets = Math.max(budget / basketCost, 0);
  const probe = legs.map((l) => buyShares(l.book, targetBaskets));

  // The guarantee is set by the thinnest leg.
  const baskets = Math.min(...probe.map((f) => f.shares));
  // Re-walk at the achievable size so cost covers only hedged shares.
  const fills = baskets > 0 ? legs.map((l) => buyShares(l.book, baskets)) : probe;

  const cost = fills.reduce((s, f) => s + f.spent, 0);
  const payout = baskets * 1;
  const net = payout - cost;
  const realizedPct = cost > 0 ? (net / cost) * 100 : 0;
  // A leg is "missing" if it couldn't reach the size the budget asked for.
  const missingLegs = probe.filter((f) => !f.filled).length;

  return {
    kind: "ARB",
    budget,
    legs: legs.map((l, i) => ({ label: l.label, fill: fills[i] })),
    baskets,
    cost,
    payout,
    net,
    realizedPct,
    complete: missingLegs === 0,
    missingLegs,
  };
}

export interface SellArbResult {
  kind: "SELL_ARB";
  budget: number;
  legs: ArbLegResult[];
  /** Sets sold = min per-leg shares (a set is one share of every outcome). */
  sets: number;
  /** Cash collected across all legs by selling into the bids. */
  proceeds: number;
  /** $1 owed per set when the winning outcome resolves. */
  liability: number;
  net: number;
  realizedPct: number;
  complete: boolean;
  missingLegs: number;
}

/**
 * Sell-all-outcomes basket — the OVERROUND capture. Selling one share of every
 * outcome into the bids collects `sum(bid)`; exactly one resolves YES so you owe
 * $1 per set. Net = proceeds − sets. Requires holding/minting the complete set
 * (this is not a naked short); it's the honest counterpart to the buy-all arb.
 * `budget` is capital at risk = $1 per set, so target sets = budget.
 */
export function simulateSellBasket(
  legs: { label: string; book: OrderBook }[],
  budget: number
): SellArbResult | null {
  if (legs.length === 0) return null;
  const targetSets = Math.max(budget, 0);
  const probe = legs.map((l) => sellShares(l.book, targetSets));

  // Same truncation as the buy side, and here it matters MORE: counting
  // proceeds from shares beyond `sets` while charging liability only on `sets`
  // books revenue from an unhedged naked short as if it were arbitrage, which
  // overstates net. Re-walk every leg at the achievable set count.
  const sets = Math.min(...probe.map((f) => f.shares));
  const fills = sets > 0 ? legs.map((l) => sellShares(l.book, sets)) : probe;

  const proceeds = fills.reduce((s, f) => s + f.spent, 0);
  const liability = sets * 1;
  const net = proceeds - liability;
  const realizedPct = liability > 0 ? (net / liability) * 100 : 0;
  const missingLegs = probe.filter((f) => !f.filled).length;

  return {
    kind: "SELL_ARB",
    budget,
    legs: legs.map((l, i) => ({ label: l.label, fill: fills[i] })),
    sets,
    proceeds,
    liability,
    net,
    realizedPct,
    complete: missingLegs === 0,
    missingLegs,
  };
}
