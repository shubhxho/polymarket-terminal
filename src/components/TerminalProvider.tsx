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

/**
 * One workspace tab. Each carries its own navigation stack, so flipping
 * between a chart you were reading and a scanner you were sorting restores
 * both exactly — the behaviour every terminal emulator and Bloomberg's
 * four-panel Launchpad share, and the reason tabs beat a single back button.
 */
export type Tab = {
  id: string;
  stack: Screen[];
  cursor: number;
};

type TerminalCtx = {
  screen: Screen;
  /** Reverse-chronological command log, newest first. */
  history: string[];
  canBack: boolean;
  canForward: boolean;
  go: (screen: Screen, commandText?: string) => void;
  back: () => void;
  forward: () => void;

  tabs: Tab[];
  activeTab: number;
  openTab: (screen?: Screen) => void;
  closeTab: (index: number) => void;
  selectTab: (index: number) => void;

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
/** Beyond this the strip stops being scannable and starts being a menu. */
const MAX_TABS = 8;

let tabSeq = 0;
const newTab = (screen: Screen = HOME): Tab => ({
  id: `t${++tabSeq}`,
  stack: [screen],
  cursor: 0,
});

export function TerminalProvider({ children }: { children: ReactNode }) {
  // Stacks live per tab, and tabs plus the active index live in one state
  // object so a navigation moves both atomically — updating them separately
  // would render a torn intermediate frame.
  const [nav, setNav] = useState<{ tabs: Tab[]; active: number }>(() => ({
    tabs: [newTab()],
    active: 0,
  }));
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

  /** Applies `fn` to the active tab, leaving every other tab untouched. */
  const updateActive = useCallback((fn: (t: Tab) => Tab) => {
    setNav((n) => ({
      ...n,
      tabs: n.tabs.map((t, i) => (i === n.active ? fn(t) : t)),
    }));
  }, []);

  const go = useCallback(
    (screen: Screen, commandText?: string) => {
      updateActive(({ id, stack, cursor }) => {
        // Navigating after going back truncates the forward branch.
        const truncated = stack.slice(0, cursor + 1);
        return {
          id,
          stack: [...truncated, screen].slice(-MAX_HISTORY),
          cursor: Math.min(truncated.length, MAX_HISTORY - 1),
        };
      });
      if (commandText) {
        setHistory((h) =>
          [commandText, ...h.filter((x) => x !== commandText)].slice(0, MAX_HISTORY)
        );
      }
    },
    [updateActive]
  );

  const back = useCallback(
    () => updateActive((t) => ({ ...t, cursor: Math.max(0, t.cursor - 1) })),
    [updateActive]
  );
  const forward = useCallback(
    () => updateActive((t) => ({ ...t, cursor: Math.min(t.stack.length - 1, t.cursor + 1) })),
    [updateActive]
  );

  const openTab = useCallback((screen: Screen = HOME) => {
    setNav((n) => {
      if (n.tabs.length >= MAX_TABS) return n;
      const tabs = [...n.tabs, newTab(screen)];
      return { tabs, active: tabs.length - 1 };
    });
  }, []);

  const closeTab = useCallback((index: number) => {
    setNav((n) => {
      // The last tab is never closed — an empty workspace has nothing to show
      // and no way back.
      if (n.tabs.length <= 1) return n;
      const tabs = n.tabs.filter((_, i) => i !== index);
      // Closing at or before the cursor shifts focus left, matching how every
      // browser and editor behaves.
      const active = index < n.active ? n.active - 1 : Math.min(n.active, tabs.length - 1);
      return { tabs, active };
    });
  }, []);

  const selectTab = useCallback((index: number) => {
    setNav((n) => ({ ...n, active: clampIndex(index, n.tabs.length) }));
  }, []);

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

  const active = nav.tabs[nav.active] ?? nav.tabs[0];

  const value = useMemo<TerminalCtx>(
    () => ({
      screen: active.stack[active.cursor] ?? HOME,
      history,
      canBack: active.cursor > 0,
      canForward: active.cursor < active.stack.length - 1,
      go,
      back,
      forward,
      tabs: nav.tabs,
      activeTab: nav.active,
      openTab,
      closeTab,
      selectTab,
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
      active,
      openTab,
      closeTab,
      selectTab,
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

function clampIndex(i: number, len: number): number {
  return Math.max(0, Math.min(len - 1, i));
}

export function useTerminal(): TerminalCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useTerminal must be used inside <TerminalProvider>");
  return ctx;
}
