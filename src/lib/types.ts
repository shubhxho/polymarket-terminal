/** Domain types for the Polymarket terminal. Shapes are normalized from the
 *  Gamma / CLOB / Data APIs so screens never touch raw upstream payloads. */

export type Outcome = {
  label: string;
  price: number;
  tokenId: string;
};

export type Market = {
  id: string;
  slug: string;
  question: string;
  /** Short label when the market is one leg of a multi-outcome event. */
  groupItemTitle?: string;
  conditionId: string;
  icon?: string;
  description?: string;
  outcomes: Outcome[];
  /** Price of the first ("Yes") outcome, 0..1. */
  last: number;
  bestBid?: number;
  bestAsk?: number;
  spread?: number;
  /** Absolute change in probability points over 1h / 24h / 1w. */
  chg1h?: number;
  chg24h?: number;
  chg1w?: number;
  volume: number;
  volume24h: number;
  volume1w: number;
  liquidity: number;
  openInterest?: number;
  endDate?: string;
  startDate?: string;
  active: boolean;
  closed: boolean;
  acceptingOrders: boolean;
  negRisk: boolean;
  tickSize: number;
  eventId?: string;
  eventTitle?: string;
  eventSlug?: string;
  eventTicker?: string;
  tags: string[];
};

export type EventSummary = {
  id: string;
  ticker: string;
  slug: string;
  title: string;
  icon?: string;
  volume: number;
  volume24h: number;
  liquidity: number;
  openInterest?: number;
  endDate?: string;
  /** 0..1 measure of how contested the event is. */
  competitive?: number;
  markets: Market[];
  tags: string[];
};

export type BookLevel = { price: number; size: number };

export type OrderBook = {
  tokenId: string;
  bids: BookLevel[];
  asks: BookLevel[];
  timestamp: number;
};

export type PricePoint = { t: number; p: number };

export type Trade = {
  id: string;
  wallet: string;
  name?: string;
  side: "BUY" | "SELL";
  outcome: string;
  outcomeIndex: number;
  size: number;
  price: number;
  timestamp: number;
  title: string;
  slug?: string;
  conditionId: string;
  asset: string;
};

export type Holder = {
  wallet: string;
  name?: string;
  amount: number;
  outcomeIndex: number;
};

export type Position = {
  conditionId: string;
  asset: string;
  title: string;
  slug?: string;
  outcome: string;
  size: number;
  avgPrice: number;
  curPrice: number;
  value: number;
  cashPnl: number;
  percentPnl: number;
  realizedPnl: number;
  redeemable: boolean;
  endDate?: string;
};

export type HistoryInterval = "1h" | "6h" | "1d" | "1w" | "1m" | "max";

export type Alert = {
  id: string;
  tokenId: string;
  marketId: string;
  label: string;
  /** Fires when price crosses above (`gte`) or below (`lte`) `target`. */
  op: "gte" | "lte";
  target: number;
  createdAt: number;
  triggeredAt?: number;
};
