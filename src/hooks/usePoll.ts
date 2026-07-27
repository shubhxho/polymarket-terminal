"use client";

import { useCallback, useEffect, useRef, useState } from "react";

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
 * Screens stay mounted for a long time, so this deliberately: keeps the last
 * good data visible when a refresh fails (a blank panel is worse than a stale
 * one), suspends polling while the tab is hidden, and fires immediately on
 * re-focus so a returning user never reads a minutes-old quote.
 *
 * Pass `url = null` to disable — useful when a screen has no selection yet.
 */
export function usePoll<T>(url: string | null, intervalMs = 5000): PollState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);

  // Reset on URL change during render rather than in an effect, so the stale
  // result of the previous URL is never painted for a frame.
  const [prevUrl, setPrevUrl] = useState(url);
  if (prevUrl !== url) {
    setPrevUrl(url);
    setData(null);
    setError(null);
    setUpdatedAt(null);
  }

  // Guards against a slow response from a previous URL landing after a newer
  // one and overwriting it.
  const requestSeq = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async () => {
    if (!url) return;
    const seq = ++requestSeq.current;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRefreshing(true);
    try {
      const res = await fetch(url, { signal: ctrl.signal });
      const json = (await res.json()) as Envelope<T>;
      if (seq !== requestSeq.current) return;
      if (!json.ok) throw new Error(json.error ?? `HTTP ${res.status}`);
      setData((json.data ?? null) as T | null);
      setError(null);
      setUpdatedAt(Date.now());
    } catch (err) {
      if (seq !== requestSeq.current) return;
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (seq === requestSeq.current) setRefreshing(false);
    }
  }, [url]);

  useEffect(() => {
    if (!url) return;
    const seq = requestSeq;

    // The first fetch is kicked off after the commit rather than inline: the
    // effect body itself performs no state updates, so mounting a screen with
    // a dozen panels doesn't cascade a dozen extra renders.
    const kickoff = setTimeout(run, 0);

    let timer: ReturnType<typeof setInterval> | null = null;
    const start = () => {
      if (timer === null) timer = setInterval(run, intervalMs);
    };
    const stop = () => {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    };

    const onVisibility = () => {
      if (document.hidden) {
        stop();
      } else {
        run();
        start();
      }
    };

    if (!document.hidden) start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      clearTimeout(kickoff);
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
      abortRef.current?.abort();
      // Any in-flight response is now stale.
      seq.current++;
    };
  }, [url, intervalMs, run]);

  return {
    data,
    error,
    // Derived, not stored: "no result yet for an enabled URL".
    loading: url !== null && data === null && error === null,
    refreshing,
    updatedAt,
    refresh: run,
  };
}
