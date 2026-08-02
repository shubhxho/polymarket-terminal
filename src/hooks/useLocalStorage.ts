"use client";

import { useCallback, useMemo, useSyncExternalStore } from "react";

/** Listeners per key, so a write in one hook instance updates every other one
 *  mounted against the same key in this tab. */
const listeners = new Map<string, Set<() => void>>();

function notify(key: string) {
  listeners.get(key)?.forEach((fn) => fn());
}

function subscribeTo(key: string, onChange: () => void): () => void {
  let set = listeners.get(key);
  if (!set) {
    set = new Set();
    listeners.set(key, set);
  }
  set.add(onChange);

  // `storage` only fires in *other* tabs, which is exactly the case the
  // in-process listener set can't cover.
  const onStorage = (e: StorageEvent) => {
    if (e.key === key || e.key === null) onChange();
  };
  window.addEventListener("storage", onStorage);

  return () => {
    set?.delete(onChange);
    window.removeEventListener("storage", onStorage);
  };
}

/**
 * State mirrored into localStorage, SSR-safe and synchronised across tabs.
 *
 * Modelled as an external store rather than state hydrated in an effect: the
 * server snapshot is `null`, so the first client render matches the server and
 * the real value arrives without a flash-of-empty or a cascading re-render.
 */
export function useLocalStorage<T>(
  key: string,
  initial: T
): [T, (v: T | ((prev: T) => T)) => void] {
  const subscribe = useCallback((onChange: () => void) => subscribeTo(key, onChange), [key]);

  // Snapshot is the raw string: stable by value between writes, which is what
  // useSyncExternalStore requires. Parsing happens in the memo below.
  const raw = useSyncExternalStore(
    subscribe,
    () => {
      try {
        return window.localStorage.getItem(key);
      } catch {
        return null;
      }
    },
    () => null
  );

  const value = useMemo<T>(() => {
    if (raw === null) return initial;
    try {
      return JSON.parse(raw) as T;
    } catch {
      // Corrupt entry — fall back rather than crash the whole terminal.
      return initial;
    }
    // `initial` is intentionally excluded: callers pass a fresh literal every
    // render, and re-parsing on identity change alone would thrash.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raw]);

  const update = useCallback(
    (v: T | ((prev: T) => T)) => {
      let current: T = initial;
      try {
        const existing = window.localStorage.getItem(key);
        if (existing !== null) current = JSON.parse(existing) as T;
      } catch {
        // Unreadable — treat as unset.
      }
      const next = typeof v === "function" ? (v as (p: T) => T)(current) : v;
      try {
        window.localStorage.setItem(key, JSON.stringify(next));
      } catch {
        // Quota or private-mode failure: nothing sensible to do but carry on.
      }
      notify(key);
    },
    // Reads `initial` as a first-write fallback but keys only on `key`: callers
    // pass a fresh `initial` literal each render, so depending on it would churn.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [key]
  );

  return [value, update];
}
