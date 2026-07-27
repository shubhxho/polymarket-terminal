"use client";

import { useCallback, useMemo, useSyncExternalStore } from "react";

/**
 * A ticking wall clock.
 *
 * Implemented as an external store rather than state-in-an-effect: time is a
 * genuinely external source, and `getServerSnapshot` returning null gives a
 * matching server render for free. Returns `null` on the server and during
 * hydration, so every caller must handle the null frame.
 */
export function useClock(intervalMs = 1000): Date | null {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const t = setInterval(onChange, intervalMs);
      return () => clearInterval(t);
    },
    [intervalMs]
  );

  // Bucketed to the interval so the snapshot is referentially stable between
  // ticks — an unbucketed Date.now() would loop React forever.
  const ms = useSyncExternalStore(
    subscribe,
    () => Math.floor(Date.now() / intervalMs) * intervalMs,
    () => null
  );

  return useMemo(() => (ms === null ? null : new Date(ms)), [ms]);
}
