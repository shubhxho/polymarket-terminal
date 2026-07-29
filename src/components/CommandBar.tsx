"use client";

import { useCallback, useEffect, useState } from "react";
import { CommandPalette } from "@/components/CommandPalette";
import { useTerminal } from "@/components/TerminalProvider";
import { cn } from "@/lib/cn";

/** Bare letters already bound to actions on the focused grid row. */
const ROW_ACTION_KEYS = new Set(["w"]);

/**
 * The always-visible search field, ami's pattern: a soft pill that looks like
 * an input but is really a trigger. Clicking it — or ⌘K, or `/`, or simply
 * starting to type — opens the palette, so there is exactly one search surface
 * rather than a command line competing with an overlay for the same keystrokes.
 */
export function CommandBar() {
  const { canBack, canForward, back, forward } = useTerminal();
  const [open, setOpen] = useState(false);
  const [seed, setSeed] = useState("");

  const close = useCallback(() => {
    setOpen(false);
    setSeed("");
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const typing = el?.tagName === "INPUT" || el?.tagName === "TEXTAREA";

      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen(true);
        return;
      }
      if (!typing && e.key === "/") {
        e.preventDefault();
        setOpen(true);
        return;
      }

      // Type-anywhere, the way TradingView and Superhuman work: any printable
      // character with nothing focused opens the palette seeded with it, so
      // searching never begins by hunting for the search box. Single letters
      // already bound to row actions (w) are excluded, and modifiers are left
      // alone so browser and OS shortcuts still work.
      if (
        !typing &&
        !e.metaKey &&
        !e.ctrlKey &&
        !e.altKey &&
        e.key.length === 1 &&
        /[a-z0-9]/i.test(e.key) &&
        !ROW_ACTION_KEYS.has(e.key.toLowerCase())
      ) {
        e.preventDefault();
        setSeed(e.key);
        setOpen(true);
      }
      // Plain Alt+arrows move within the active tab's history. The tab strip
      // owns the ⌥⌘ variants, which cycle between tabs.
      if (e.altKey && !e.metaKey && !e.ctrlKey) {
        if (e.key === "ArrowLeft" && canBack) {
          e.preventDefault();
          back();
        } else if (e.key === "ArrowRight" && canForward) {
          e.preventDefault();
          forward();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [back, forward, canBack, canForward]);

  return (
    <>
      <div className="flex h-[42px] shrink-0 items-center gap-2 border-b border-edge bg-canvas px-3">
        <div className="flex shrink-0 items-center gap-0.5">
          <NavButton label="←" disabled={!canBack} onClick={back} title="Back (Alt+←)" />
          <NavButton label="→" disabled={!canForward} onClick={forward} title="Forward (Alt+→)" />
        </div>

        <button
          onClick={() => {
            setSeed("");
            setOpen(true);
          }}
          className="flex h-[28px] min-w-0 flex-1 items-center gap-2 rounded-md border border-edge bg-surface px-2.5 text-left transition-colors hover:border-edge-strong"
        >
          <span className="shrink-0 text-faint">⌘</span>
          <span className="min-w-0 flex-1 truncate text-tiny text-faint">
            Search markets, or type a function code…
          </span>
          <kbd className="hidden shrink-0 rounded-sm border border-edge px-1 text-[10px] text-faint sm:inline">
            ⌘K
          </kbd>
        </button>
      </div>

      <CommandPalette open={open} seed={seed} onClose={close} />
    </>
  );
}

function NavButton({
  label,
  disabled,
  onClick,
  title,
}: {
  label: string;
  disabled: boolean;
  onClick: () => void;
  title: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cn(
        "flex h-[24px] w-[24px] items-center justify-center rounded-md text-[13px]",
        disabled ? "text-edge-strong" : "text-muted hover:bg-surface-2 hover:text-ink"
      )}
    >
      {label}
    </button>
  );
}
