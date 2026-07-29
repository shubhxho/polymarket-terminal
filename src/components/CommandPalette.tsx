"use client";

import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTerminal } from "@/components/TerminalProvider";
import { usePoll } from "@/hooks/usePoll";
import { useWallet } from "@/hooks/useWallet";
import {
  COMMANDS,
  defaultScreenFor,
  lookupCommand,
  parseCommand,
  SECTORS,
  screenTitle,
  type Screen,
} from "@/lib/commands";
import { cn } from "@/lib/cn";
import { compact, truncate } from "@/lib/format";
import { fuzzyMatch, highlight } from "@/lib/fuzzy";
import { popVariants, rowVariants, scrimVariants, staggerContainer } from "@/lib/motion";
import type { EventSummary, Market } from "@/lib/types";

type Row = {
  id: string;
  group: "Run" | "Commands" | "Sectors" | "Watchlist" | "Markets" | "Recent";
  label: string;
  hint?: string;
  meta?: string;
  positions: number[];
  score: number;
  screen: Screen | null;
  /** Commands needing an argument prefill the input instead of navigating. */
  prefill?: string;
};

const GROUP_ORDER: Row["group"][] = [
  "Run",
  "Commands",
  "Recent",
  "Watchlist",
  "Sectors",
  "Markets",
];

/**
 * Score floor for the literal interpretation of the typed line, high enough
 * that no fuzzy hit can outrank it. A user who typed a whole command meant it.
 */
const RUN_ROW_SCORE = 1e6;

/**
 * ⌘K palette — the terminal's primary way in.
 *
 * Unlike a plain command menu this searches *live markets* alongside the
 * function codes, so "fed cut" and "SIG" are the same gesture. Remote results
 * are debounced and merged under the local ones, which keeps the list stable
 * while typing instead of reshuffling on every keystroke.
 */
export function CommandPalette({
  open,
  seed = "",
  onClose,
}: {
  open: boolean;
  /** First character typed when the palette was opened by type-anywhere. */
  seed?: string;
  onClose: () => void;
}) {
  const { go, openTab, watchlist, history, toast } = useTerminal();
  const { address } = useWallet();
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [sel, setSel] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  // Reset on each open during render rather than in an effect, so the palette
  // never paints one frame of the previous session's query.
  const [wasOpen, setWasOpen] = useState(open);
  if (wasOpen !== open) {
    setWasOpen(open);
    if (open) {
      setQuery(seed);
      setDebounced("");
      setSel(0);
    }
  }

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
  }, [open]);

  // Debounce the remote leg only. Local matching stays instant.
  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 180);
    return () => clearTimeout(t);
  }, [query]);

  const remote = usePoll<{ events: EventSummary[]; markets: Market[] }>(
    open && debounced.length >= 2 ? `/api/search?q=${encodeURIComponent(debounced)}&limit=8` : null,
    60_000
  );

  const rows = useMemo<Row[]>(() => {
    const q = query.trim();
    const out: Row[] = [];

    const push = (group: Row["group"], id: string, label: string, extra: Partial<Row> = {}) => {
      const m = fuzzyMatch(q, label);
      // Remote market rows are already relevance-ranked upstream; keep them
      // even when the local fuzzy pass can't align the characters.
      if (!m && group !== "Markets") return;
      out.push({
        id,
        group,
        label,
        positions: m?.positions ?? [],
        score: m?.score ?? 0,
        screen: null,
        ...extra,
      });
    };

    // A line with an argument ("SRCH bitcoin", "PORT 0x…") has exactly one
    // meaning, and it is not whichever market happens to fuzzy-match the
    // literal string. Without this row, Enter ran the top fuzzy hit and
    // `SRCH bitcoin` silently opened an unrelated "Arch Network" market.
    if (q.includes(" ")) {
      const parsed = parseCommand(q);
      if (parsed.kind === "screen") {
        out.push({
          id: "run",
          group: "Run",
          label: screenTitle(parsed.screen),
          meta: `Run “${q}”`,
          positions: [],
          score: RUN_ROW_SCORE,
          screen: parsed.screen,
        });
      }
    }

    for (const c of COMMANDS) {
      const parsed = parseCommand(c.code);
      // A connected wallet supplies PORT's argument, so its row navigates
      // rather than prefilling — same resolution the sidebar uses.
      const preset = defaultScreenFor(c.code, { walletAddress: address });
      const screen = preset ?? (c.args ? null : parsed.kind === "screen" ? parsed.screen : null);
      push("Commands", `cmd-${c.code}`, `${c.code} · ${c.title}`, {
        hint: c.fkey ? `F${c.fkey}` : undefined,
        meta: c.blurb,
        screen,
        prefill: !screen && c.args ? `${c.code} ` : undefined,
      });
    }

    for (const s of SECTORS) {
      push("Sectors", `sec-${s.key}`, s.label, {
        meta: "Browse sector",
        screen: { fn: "CAT", tag: s.tag, label: s.label },
      });
    }

    for (const w of watchlist) {
      push("Watchlist", `w-${w.tokenId}`, w.label, {
        meta: "Pinned",
        screen: { fn: "DES", slug: w.slug, kind: "event" },
      });
    }

    if (!q) {
      for (const h of history.slice(0, 5)) {
        const parsed = parseCommand(h);
        if (parsed.kind !== "screen") continue;
        out.push({
          id: `r-${h}`,
          group: "Recent",
          label: h,
          meta: screenTitle(parsed.screen),
          positions: [],
          score: 0,
          screen: parsed.screen,
        });
      }
    }

    if (remote.data) {
      for (const ev of remote.data.events.slice(0, 6)) {
        push("Markets", `e-${ev.id}`, ev.title, {
          meta: `${ev.markets.length} legs · $${compact(ev.volume24h)} 24h`,
          screen: { fn: "DES", slug: ev.slug, kind: "event" },
        });
      }
      for (const m of remote.data.markets.slice(0, 6)) {
        const slug = m.eventSlug || m.slug;
        push("Markets", `m-${m.id}`, m.groupItemTitle || m.question, {
          meta: `$${compact(m.volume24h)} 24h`,
          screen: { fn: "DES", slug, kind: m.eventSlug ? "event" : "market" },
        });
      }
    }

    out.sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return GROUP_ORDER.indexOf(a.group) - GROUP_ORDER.indexOf(b.group);
    });
    return out.slice(0, 24);
  }, [query, watchlist, history, remote.data, address]);

  // Every keystroke is a new question, so the cursor returns to the top rather
  // than sitting on whatever row happened to be under it before.
  const [prevQuery, setPrevQuery] = useState(query);
  if (prevQuery !== query) {
    setPrevQuery(query);
    setSel(0);
  }

  // Clamp when the result set shrinks under the cursor — remote results
  // arriving and leaving can do that without the query changing.
  const [prevLen, setPrevLen] = useState(rows.length);
  if (prevLen !== rows.length) {
    setPrevLen(rows.length);
    if (sel > rows.length - 1) setSel(Math.max(0, rows.length - 1));
  }

  const run = useCallback(
    (row: Row, newTab: boolean) => {
      if (row.prefill) {
        setQuery(row.prefill);
        return;
      }
      if (!row.screen) return;
      if (newTab) openTab(row.screen);
      else go(row.screen, row.label);
      onClose();
    },
    [go, openTab, onClose]
  );

  const submitRaw = useCallback(() => {
    const parsed = parseCommand(query);
    if (parsed.kind === "error") {
      // A bare arg-taking code can still resolve from session state — typing
      // "PORT" with a wallet connected opens that wallet's book, not an error.
      const head = query.trim().split(/\s+/)[0];
      const spec = lookupCommand(head);
      const preset = spec ? defaultScreenFor(spec.code, { walletAddress: address }) : null;
      if (preset) {
        go(preset, `${spec!.code} ${address}`);
        onClose();
        return;
      }
      toast(parsed.message, "error");
      return;
    }
    go(parsed.screen, query.trim().toUpperCase());
    onClose();
  }, [query, address, go, onClose, toast]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
      return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setSel((s) => {
        const next = e.key === "ArrowDown" ? s + 1 : s - 1;
        return Math.max(0, Math.min(rows.length - 1, next));
      });
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      const row = rows[sel];
      if (row) run(row, e.metaKey || e.ctrlKey);
      else submitRaw();
    }
  };

  useEffect(() => {
    listRef.current
      ?.querySelector<HTMLElement>(`[data-row="${sel}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  let lastGroup: Row["group"] | null = null;

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          key="pal-scrim"
          variants={scrimVariants}
          initial="initial"
          animate="animate"
          exit="exit"
          className="fixed inset-0 z-50 flex items-start justify-center bg-ink/20 px-3 pt-[12vh] backdrop-blur-[2px]"
          onMouseDown={onClose}
          role="dialog"
          aria-modal="true"
          aria-label="Command palette"
        >
          <motion.div
            variants={popVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            onMouseDown={(e) => e.stopPropagation()}
            className="flex max-h-[62vh] w-full max-w-[620px] flex-col overflow-hidden rounded-lg border border-edge bg-canvas shadow-[var(--shadow-pop)]"
          >
            <div className="flex shrink-0 items-center gap-2.5 border-b border-edge px-3.5 py-3">
              <span className="text-faint">⌘</span>
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onKeyDown}
                spellCheck={false}
                autoComplete="off"
                aria-label="Command line"
                role="combobox"
                aria-expanded
                aria-controls="pal-list"
                aria-activedescendant={rows[sel] ? `pal-opt-${sel}` : undefined}
                placeholder="Search markets, or type a function code…"
                className="min-w-0 flex-1 text-sm2 placeholder:text-faint"
              />
              {remote.refreshing ? <span className="dot animate-pulse text-accent" /> : null}
            </div>

            <motion.div
              ref={listRef}
              id="pal-list"
              role="listbox"
              aria-label="Results"
              className="min-h-0 flex-1 overflow-y-auto py-1"
              variants={staggerContainer}
              initial="initial"
              animate="animate"
            >
              {rows.length === 0 ? (
                <div className="px-3.5 py-6 text-center text-tiny text-faint">
                  No matches — press Enter to search markets for “{query}”
                </div>
              ) : (
                rows.map((row, i) => {
                  const header = row.group !== lastGroup ? row.group : null;
                  lastGroup = row.group;
                  return (
                    <motion.div key={row.id} variants={rowVariants}>
                      {header ? <div className="eyebrow px-3.5 pt-2 pb-1">{header}</div> : null}
                      <button
                        data-row={i}
                        id={`pal-opt-${i}`}
                        role="option"
                        aria-selected={i === sel}
                        onMouseEnter={() => setSel(i)}
                        // Preventing mousedown keeps focus in the search field, so
                        // a prefilling row leaves the caret ready for its argument.
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={(e) => run(row, e.metaKey || e.ctrlKey)}
                        className={cn(
                          "flex w-full items-center gap-2.5 px-3.5 py-[6px] text-left",
                          i === sel && "row-sel"
                        )}
                      >
                        <span className="min-w-0 flex-1 truncate text-tiny">
                          {highlight(row.label, row.positions).map((part, k) => (
                            <span
                              key={k}
                              className={part.hit ? "font-semibold text-accent" : undefined}
                            >
                              {part.text}
                            </span>
                          ))}
                        </span>
                        {row.meta ? (
                          <span className="shrink-0 text-[11px] text-faint">
                            {truncate(row.meta, 34)}
                          </span>
                        ) : null}
                        {row.hint ? (
                          <kbd className="shrink-0 rounded-sm border border-edge px-1 text-[10px] text-faint">
                            {row.hint}
                          </kbd>
                        ) : null}
                      </button>
                    </motion.div>
                  );
                })
              )}
            </motion.div>

            <div className="flex shrink-0 items-center gap-3 border-t border-edge px-3.5 py-2 text-[11px] text-faint">
              <Hint keys="↑↓" label="navigate" />
              <Hint keys="⏎" label="open" />
              <Hint keys="⌘⏎" label="open in new tab" />
              <Hint keys="esc" label="close" />
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

function Hint({ keys, label }: { keys: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <kbd className="rounded-sm border border-edge px-1 text-[10px]">{keys}</kbd>
      {label}
    </span>
  );
}
