"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useLocalStorage } from "@/hooks/useLocalStorage";
import type { Screen } from "@/lib/commands";
import type { Alert } from "@/lib/types";

export type WatchItem = {
  /** Event slug — what DES navigates by. */
  slug: string;
  label: string;
  tokenId: string;
  marketId: string;
  addedAt: number;
};

export type Toast = { id: number; text: string; tone: "info" | "warn" | "error" };

type TerminalCtx = {
  screen: Screen;
  /** Reverse-chronological command log, newest first. */
  history: string[];
  canBack: boolean;
  canForward: boolean;
  go: (screen: Screen, commandText?: string) => void;
  back: () => void;
  forward: () => void;

  watchlist: WatchItem[];
  isWatched: (tokenId: string) => boolean;
  toggleWatch: (item: WatchItem) => void;

  alerts: Alert[];
  addAlert: (a: Omit<Alert, "id" | "createdAt">) => void;
  removeAlert: (id: string) => void;
  markAlertTriggered: (id: string) => void;

  toasts: Toast[];
  toast: (text: string, tone?: Toast["tone"]) => void;
  dismissToast: (id: number) => void;
};

const Ctx = createContext<TerminalCtx | null>(null);

const HOME: Screen = { fn: "MON" };
const MAX_HISTORY = 40;

export function TerminalProvider({ children }: { children: ReactNode }) {
  // A single stack with a cursor gives browser-style back/forward without
  // touching the URL, which would otherwise remount the whole workspace.
  // Stack and cursor live in one state object so a navigation can move both
  // atomically — updating them separately would render a torn intermediate.
  const [nav, setNav] = useState<{ stack: Screen[]; cursor: number }>({
    stack: [HOME],
    cursor: 0,
  });
  const [history, setHistory] = useState<string[]>([]);

  const [watchlist, setWatchlist] = useLocalStorage<WatchItem[]>("pmt.watchlist", []);
  const [alerts, setAlerts] = useLocalStorage<Alert[]>("pmt.alerts", []);

  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastSeq = useRef(0);

  const dismissToast = useCallback((id: number) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const toast = useCallback(
    (text: string, tone: Toast["tone"] = "info") => {
      const id = ++toastSeq.current;
      setToasts((t) => [...t.slice(-4), { id, text, tone }]);
      setTimeout(() => dismissToast(id), tone === "error" ? 6000 : 4000);
    },
    [dismissToast]
  );

  const go = useCallback((screen: Screen, commandText?: string) => {
    setNav(({ stack, cursor }) => {
      // Navigating after going back truncates the forward branch.
      const truncated = stack.slice(0, cursor + 1);
      return {
        stack: [...truncated, screen].slice(-MAX_HISTORY),
        cursor: Math.min(truncated.length, MAX_HISTORY - 1),
      };
    });
    if (commandText) {
      setHistory((h) => [commandText, ...h.filter((x) => x !== commandText)].slice(0, MAX_HISTORY));
    }
  }, []);

  const back = useCallback(
    () => setNav((n) => ({ ...n, cursor: Math.max(0, n.cursor - 1) })),
    []
  );
  const forward = useCallback(
    () => setNav((n) => ({ ...n, cursor: Math.min(n.stack.length - 1, n.cursor + 1) })),
    []
  );

  const isWatched = useCallback(
    (tokenId: string) => watchlist.some((w) => w.tokenId === tokenId),
    [watchlist]
  );

  const toggleWatch = useCallback(
    (item: WatchItem) => {
      setWatchlist((prev) => {
        const exists = prev.some((w) => w.tokenId === item.tokenId);
        if (exists) {
          toast(`removed ${item.label} from watchlist`);
          return prev.filter((w) => w.tokenId !== item.tokenId);
        }
        toast(`added ${item.label} to watchlist`);
        return [...prev, item];
      });
    },
    [setWatchlist, toast]
  );

  const addAlert = useCallback(
    (a: Omit<Alert, "id" | "createdAt">) => {
      const alert: Alert = {
        ...a,
        id: `${a.tokenId}-${a.op}-${a.target}-${Date.now()}`,
        createdAt: Date.now(),
      };
      setAlerts((prev) => {
        // Re-arming the same threshold should replace, not duplicate.
        const dedup = prev.filter(
          (p) => !(p.tokenId === a.tokenId && p.op === a.op && p.target === a.target)
        );
        return [...dedup, alert];
      });
      toast(
        `alert armed · ${a.label} ${a.op === "gte" ? "≥" : "≤"} ${(a.target * 100).toFixed(1)}¢`
      );
    },
    [setAlerts, toast]
  );

  const removeAlert = useCallback(
    (id: string) => setAlerts((prev) => prev.filter((a) => a.id !== id)),
    [setAlerts]
  );

  const markAlertTriggered = useCallback(
    (id: string) => {
      setAlerts((prev) =>
        prev.map((a) => (a.id === id ? { ...a, triggeredAt: Date.now() } : a))
      );
    },
    [setAlerts]
  );

  const value = useMemo<TerminalCtx>(
    () => ({
      screen: nav.stack[nav.cursor] ?? HOME,
      history,
      canBack: nav.cursor > 0,
      canForward: nav.cursor < nav.stack.length - 1,
      go,
      back,
      forward,
      watchlist,
      isWatched,
      toggleWatch,
      alerts,
      addAlert,
      removeAlert,
      markAlertTriggered,
      toasts,
      toast,
      dismissToast,
    }),
    [
      nav,
      history,
      go,
      back,
      forward,
      watchlist,
      isWatched,
      toggleWatch,
      alerts,
      addAlert,
      removeAlert,
      markAlertTriggered,
      toasts,
      toast,
      dismissToast,
    ]
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTerminal(): TerminalCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useTerminal must be used inside <TerminalProvider>");
  return ctx;
}
