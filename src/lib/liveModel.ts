/**
 * Live model re-scoring, between the polls.
 *
 * The scan endpoint is a 20-second poll — fine for the heavy legs (upstream
 * fetches, the cross-section, arbitrage), but far too slow for the one number
 * that actually moves tick-by-tick: the model's read on where price goes next.
 * The payload carries the exact window the server-side model scored
 * (`MarketSignals.recent`); the client seeds a rolling buffer from it, appends
 * the socket's last trade on every tick, and re-runs the forward pass — which is
 * microseconds, so it stays on the main thread. A Web Worker would only add a
 * `postMessage` copy in each direction and make the round-trip *slower*, so
 * there isn't one.
 *
 * Everything here is pure and synchronous; the React glue lives in
 * `useLiveModel`, and it is thin precisely because the arithmetic is here where
 * it can be tested without a socket or a DOM.
 */

import { MODEL_WINDOW, modelSignalFromPrices, type ModelRead } from "./mlSignal";
import { blendedScore, type MarketSignals } from "./signals";

/**
 * Append a live price to a rolling look-back window.
 *
 * A tick equal to the current head is dropped and the same array returned by
 * reference — the model only saw a fresh print when the price actually moved,
 * and re-scoring an unchanged series just burns cycles for an identical answer.
 * The window never grows past `MODEL_WINDOW`; the oldest print falls off the
 * front exactly as it did in training.
 */
export function appendTick(tail: readonly number[], price: number): number[] {
  if (tail.length > 0 && tail[tail.length - 1] === price) return tail as number[];
  const start = tail.length >= MODEL_WINDOW ? tail.length - MODEL_WINDOW + 1 : 0;
  const next = tail.slice(start);
  next.push(price);
  return next;
}

export type LiveRead = {
  readonly model: ModelRead;
  /** `blendedScore` recomputed with the live model against the poll's heat/bias. */
  readonly blended: number;
};

/**
 * Re-score one market from its current live window.
 *
 * Only the model read and the blended rank value are refreshed — `heat` and
 * `bias` stay as the poll computed them, because they depend on the book, the
 * tape and the whole cross-section, none of which this cheap path has. Returns
 * `null` on a window that is not yet full, the same silence the model keeps.
 */
export function liveModelFor(row: MarketSignals, tail: readonly number[]): LiveRead | null {
  const model = modelSignalFromPrices(tail);
  if (!model) return null;
  return { model, blended: blendedScore({ ...row, model }) };
}
