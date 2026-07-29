"use client";

import { useQuery } from "@tanstack/react-query";
import { useCallback } from "react";

type Envelope<T> = { ok: boolean; data?: T; error?: string; ts: number };

export type PollState<T> = {
  data: T | null;
  error: string | null;
  /** True only before the first result; refreshes keep showing stale data. */
  loading: boolean;
  /** True while a background refresh is in flight. */
  refreshing: boolean;
  updatedAt: number | null;
  refresh: () => void;
};

/**
 * Polls a terminal API endpoint on an interval.
 *
 * Backed by TanStack Query, so identical URLs mounted by several panels share
 * one in-flight request and one cache entry instead of each hammering upstream.
 * The public shape is unchanged from the hand-rolled version it replaced —
 * callers still get `{ data, error, loading, refreshing, updatedAt, refresh }`.
 *
 * The behaviours the terminal depends on are all native to React Query and
 * configured here: the last good data stays visible when a refresh fails (a
 * blank panel is worse than a stale one), polling suspends while the tab is
 * hidden (`refetchIntervalInBackground: false`), a re-focus fires an immediate
 * refetch (provider-level `refetchOnWindowFocus`), and a stale response from a
 * previous URL can never overwrite a newer one because each URL is its own
 * query key.
 *
 * Pass `url = null` to disable — useful when a screen has no selection yet.
 */
export function usePoll<T>(url: string | null, intervalMs = 5000): PollState<T> {
  const query = useQuery<T | null, Error>({
    queryKey: ["poll", url],
    enabled: url !== null,
    refetchInterval: intervalMs,
    // A backgrounded tab shouldn't keep polling; the provider's focus refetch
    // brings it current the instant the trader returns.
    refetchIntervalInBackground: false,
    // No `placeholderData`: a genuine URL switch must resolve to `undefined`
    // (→ loading), never paint the previous URL's data for a frame — the same
    // guarantee the hand-rolled version made by resetting state on url change.
    queryFn: async ({ signal }) => {
      const res = await fetch(url as string, { signal });
      const json = (await res.json()) as Envelope<T>;
      if (!json.ok) throw new Error(json.error ?? `HTTP ${res.status}`);
      return (json.data ?? null) as T | null;
    },
  });

  const refresh = useCallback(() => {
    void query.refetch();
  }, [query]);

  return {
    data: query.data ?? null,
    error: query.error ? query.error.message : null,
    // "No result yet for an enabled URL" — matches the old derived flag.
    loading: url !== null && query.data === undefined && !query.isError,
    refreshing: query.isFetching,
    updatedAt: query.dataUpdatedAt || null,
    refresh,
  };
}
