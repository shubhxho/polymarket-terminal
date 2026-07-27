"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTerminal } from "@/components/TerminalProvider";
import { COMMANDS, parseCommand, type CommandSpec } from "@/lib/commands";

/**
 * The command line. Everything in the terminal is reachable from here.
 *
 * Key routing is deliberate: when the input is empty the arrow keys belong to
 * the grid behind it (the primary surface), and when the user has typed
 * something they belong to the suggestion list. History recall is on Ctrl+↑/↓
 * so it never competes with either.
 */
export function CommandBar() {
  const { go, toast, history, canBack, canForward, back, forward } = useTerminal();
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const [historyIdx, setHistoryIdx] = useState(-1);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const suggestions = useMemo<CommandSpec[]>(() => {
    const token = value.trim().split(/\s+/)[0]?.toUpperCase() ?? "";
    if (!token) return [];
    const hits = COMMANDS.filter(
      (c) =>
        c.code.startsWith(token) ||
        c.aliases.some((a) => a.startsWith(token)) ||
        c.title.includes(token)
    );
    // An exact single match with args already typed is not worth a dropdown.
    if (hits.length === 1 && value.trim().length > hits[0].code.length) return [];
    return hits.slice(0, 6);
  }, [value]);

  // Reset the dropdown cursor whenever the query changes. Done during render
  // rather than in an effect so the highlight can never lag a frame behind the
  // suggestion list it indexes into.
  const [prevValue, setPrevValue] = useState(value);
  if (prevValue !== value) {
    setPrevValue(value);
    setHighlight(0);
  }

  const submit = useCallback(
    (raw: string) => {
      const text = raw.trim();
      if (!text) return;
      const result = parseCommand(text);
      if (result.kind === "error") {
        toast(result.message, "error");
        return;
      }
      go(result.screen, text.toUpperCase());
      setValue("");
      setHistoryIdx(-1);
    },
    [go, toast]
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      // Ctrl/Cmd + arrows: command history, always available.
      if ((e.ctrlKey || e.metaKey) && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
        e.preventDefault();
        e.stopPropagation();
        if (history.length === 0) return;
        const next =
          e.key === "ArrowUp"
            ? Math.min(history.length - 1, historyIdx + 1)
            : Math.max(-1, historyIdx - 1);
        setHistoryIdx(next);
        setValue(next === -1 ? "" : history[next]);
        return;
      }

      if (e.key === "Enter") {
        e.preventDefault();
        e.stopPropagation();
        const picked = suggestions[highlight];
        // A highlighted suggestion with no argument typed yet completes rather
        // than submits, so `S` + Enter becomes `SRCH ` awaiting a query.
        if (picked && !value.trim().includes(" ") && picked.args) {
          setValue(`${picked.code} `);
          return;
        }
        submit(picked && !value.trim().includes(" ") ? picked.code : value);
        return;
      }

      if (e.key === "Escape") {
        e.preventDefault();
        setValue("");
        setHistoryIdx(-1);
        inputRef.current?.blur();
        return;
      }

      if (e.key === "Tab" && suggestions[highlight]) {
        e.preventDefault();
        const spec = suggestions[highlight];
        setValue(spec.args ? `${spec.code} ` : spec.code);
        return;
      }

      // With text present the arrows drive the dropdown; stop them here so the
      // grid's window-level listener doesn't also move its selection.
      if ((e.key === "ArrowUp" || e.key === "ArrowDown") && suggestions.length > 0) {
        e.preventDefault();
        e.stopPropagation();
        setHighlight((h) => {
          const next = e.key === "ArrowDown" ? h + 1 : h - 1;
          return Math.max(0, Math.min(suggestions.length - 1, next));
        });
      }
    },
    [suggestions, highlight, value, submit, history, historyIdx]
  );

  // Global shortcuts that focus or bypass the command line.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const typing = el?.tagName === "INPUT" || el?.tagName === "TEXTAREA";

      if (e.altKey && e.key === "ArrowLeft" && canBack) {
        e.preventDefault();
        back();
        return;
      }
      if (e.altKey && e.key === "ArrowRight" && canForward) {
        e.preventDefault();
        forward();
        return;
      }
      if (!typing && (e.key === "/" || (e.key === "k" && (e.metaKey || e.ctrlKey)))) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [back, forward, canBack, canForward]);

  return (
    <div className="relative flex h-[42px] shrink-0 items-center gap-2 border-b border-edge bg-canvas px-3">
      <div className="flex shrink-0 items-center gap-0.5">
        <NavButton label="←" disabled={!canBack} onClick={back} title="Back (Alt+←)" />
        <NavButton label="→" disabled={!canForward} onClick={forward} title="Forward (Alt+→)" />
      </div>

      {/* ami's search field: a soft pill that lifts to the accent on focus. */}
      <div
        className={`flex h-[28px] min-w-0 flex-1 items-center gap-2 rounded-md border bg-surface px-2.5 transition-colors ${
          focused ? "border-accent bg-canvas" : "border-edge hover:border-edge-strong"
        }`}
      >
        <span className={`shrink-0 text-[12px] ${focused ? "text-accent" : "text-faint"}`}>
          ⌘
        </span>
        <input
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => {
            setFocused(false);
            // Delay so a mousedown on a suggestion still lands.
            setTimeout(() => setHighlight(0), 120);
          }}
          spellCheck={false}
          autoComplete="off"
          aria-label="Command line"
          placeholder="Search markets, or type a function code — press / to focus"
          className="min-w-0 flex-1 text-tiny placeholder:text-faint"
        />
        {!value ? (
          <kbd className="hidden shrink-0 rounded-sm border border-edge px-1 text-[10px] text-faint sm:inline">
            /
          </kbd>
        ) : (
          <button
            onClick={() => submit(value)}
            className="shrink-0 rounded-sm bg-accent-soft px-2 py-[2px] text-[11px] font-medium text-[#1a0e1e]"
          >
            Go
          </button>
        )}
      </div>

      {focused && suggestions.length > 0 ? (
        <ul className="absolute top-full left-[68px] z-40 mt-1 w-full max-w-[520px] overflow-hidden rounded-md border border-edge bg-canvas shadow-[var(--shadow-pop)]">
          {suggestions.map((s, i) => (
            <li key={s.code}>
              <button
                // mousedown fires before blur, so the click isn't swallowed.
                onMouseDown={(e) => {
                  e.preventDefault();
                  if (s.args) {
                    setValue(`${s.code} `);
                    inputRef.current?.focus();
                  } else {
                    submit(s.code);
                  }
                }}
                onMouseEnter={() => setHighlight(i)}
                className={`flex w-full items-baseline gap-2 px-2.5 py-[6px] text-left text-tiny ${
                  i === highlight ? "bg-surface-2" : ""
                }`}
              >
                <span className="w-[44px] shrink-0 font-medium text-accent">{s.code}</span>
                {s.args ? (
                  <span className="mono shrink-0 text-[11px] text-faint">{s.args}</span>
                ) : null}
                <span className="truncate text-muted">{s.blurb}</span>
                {s.fkey ? (
                  <kbd className="ml-auto shrink-0 rounded-sm border border-edge px-1 text-[10px] text-faint">
                    F{s.fkey}
                  </kbd>
                ) : null}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
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
      className={`flex h-[24px] w-[24px] items-center justify-center rounded-md text-[13px] ${
        disabled ? "text-edge-strong" : "text-muted hover:bg-surface-2 hover:text-ink"
      }`}
    >
      {label}
    </button>
  );
}
