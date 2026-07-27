"use client";

import { useEffect } from "react";
import { useTerminal } from "@/components/TerminalProvider";
import { screenTitle } from "@/lib/commands";

/**
 * Workspace tabs.
 *
 * Each tab owns its own navigation stack, so a user can keep a chart open in
 * one and sort a scanner in another without either losing its place — the
 * single most-missed thing when a terminal has only one viewport.
 *
 * Bindings follow the convention every browser, editor and terminal emulator
 * already shares, so none of it has to be learned: ⌘T new, ⌘W close, ⌘1–9
 * jump, ⌥⌘←/→ cycle.
 */
export function TabStrip() {
  const { tabs, activeTab, openTab, closeTab, selectTab, screen } = useTerminal();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;

      if (e.key === "t") {
        e.preventDefault();
        openTab();
        return;
      }
      if (e.key === "w") {
        e.preventDefault();
        closeTab(activeTab);
        return;
      }
      // ⌥⌘← / ⌥⌘→ cycle tabs. Plain ⌥← is already back/forward within a tab.
      if (e.altKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        e.preventDefault();
        const delta = e.key === "ArrowRight" ? 1 : -1;
        selectTab((activeTab + delta + tabs.length) % tabs.length);
        return;
      }
      const digit = /^[1-9]$/.exec(e.key);
      if (digit) {
        const index = parseInt(digit[0], 10) - 1;
        if (index < tabs.length) {
          e.preventDefault();
          selectTab(index);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tabs.length, activeTab, openTab, closeTab, selectTab]);

  // A lone tab is just chrome with nothing to choose between.
  if (tabs.length <= 1 && screen.fn === "MON") return null;

  return (
    <div className="flex h-[32px] shrink-0 items-end gap-1 border-b border-edge bg-canvas px-3">
      {tabs.map((tab, i) => {
        const current = tab.stack[tab.cursor];
        const active = i === activeTab;
        return (
          <div
            key={tab.id}
            onClick={() => selectTab(i)}
            role="tab"
            aria-selected={active}
            title={`${screenTitle(current)} — ⌘${i + 1}`}
            className={`group flex h-[26px] max-w-[190px] min-w-0 cursor-pointer items-center gap-1.5 rounded-t-md border border-b-0 px-2.5 text-tiny transition-colors ${
              active
                ? "border-edge bg-surface font-medium text-ink"
                : "border-transparent text-muted hover:bg-surface-2"
            }`}
          >
            <span className={`dot ${active ? "text-accent" : "text-edge-strong"}`} />
            <span className="min-w-0 flex-1 truncate">{screenTitle(current)}</span>
            {tabs.length > 1 ? (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  closeTab(i);
                }}
                aria-label="Close tab"
                className="shrink-0 text-faint opacity-0 group-hover:opacity-100 hover:text-ink"
              >
                ×
              </button>
            ) : null}
          </div>
        );
      })}

      {tabs.length < 8 ? (
        <button
          onClick={() => openTab()}
          title="New tab (⌘T)"
          aria-label="New tab"
          className="mb-[3px] flex h-[20px] w-[20px] items-center justify-center rounded-sm text-muted hover:bg-surface-2 hover:text-ink"
        >
          +
        </button>
      ) : null}
    </div>
  );
}
