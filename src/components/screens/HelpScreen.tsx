"use client";

import { motion } from "motion/react";
import { useTerminal } from "@/components/TerminalProvider";
import { Panel } from "@/components/ui/Panel";
import { COMMANDS, SECTORS, type Screen, type ScreenName } from "@/lib/commands";
import { staggerContainer, tapScale } from "@/lib/motion";

/** Keyboard map. Documented here only — the handlers live with their widgets. */
const KEYS: { keys: string[]; action: string }[] = [
  { keys: ["Enter"], action: "Run the command on the command line" },
  {
    keys: ["↑", "↓"],
    action: "Move the grid selection · move the suggestion cursor once you type",
  },
  { keys: ["Enter"], action: "Open the highlighted grid row" },
  { keys: ["W"], action: "Toggle the highlighted market on your watchlist" },
  { keys: ["Esc"], action: "Close the command palette" },
  { keys: ["Alt+←", "Alt+→"], action: "Navigate back / forward through screens" },
  { keys: ["/"], action: "Focus the command line from anywhere" },
  { keys: ["F1–F10"], action: "Jump straight to the function in that slot" },
];

const EXAMPLES: { line: string; note: string }[] = [
  { line: "SRCH fed cut", note: "search every event and market for a phrase" },
  { line: "DES us-recession-2026", note: "open one event's analytics launchpad" },
  { line: "PORT 0x1a2b3c…", note: "pull any wallet's positions and P&L" },
  { line: "Connect", note: "link Phantom in the masthead — PORT then loads your own book" },
];

/** Screens reachable with no arguments, so HELP can launch them on click. */
function screenFor(code: ScreenName): Screen | null {
  switch (code) {
    case "MON":
      return { fn: "MON" };
    case "MOV":
      return { fn: "MOV" };
    case "WATCH":
      return { fn: "WATCH" };
    case "TAS":
      return { fn: "TAS" };
    case "ALRT":
      return { fn: "ALRT" };
    case "HELP":
      return { fn: "HELP" };
    default:
      return null;
  }
}

/**
 * Command reference. Every row is generated from the `COMMANDS` / `SECTORS`
 * registries, so a new function code shows up here the moment it is registered
 * and can never drift out of sync with the parser.
 */
export default function HelpScreen() {
  const { go } = useTerminal();

  return (
    <motion.div
      className="flex h-full min-h-0 flex-col gap-2"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      <motion.div
        variants={staggerContainer}
        className="grid min-h-0 flex-1 grid-cols-1 gap-2 overflow-auto lg:grid-cols-3"
      >
        <Panel title="Function Codes" right={`${COMMANDS.length} codes`} flush animate>
          <div className="text-tiny">
            <div className="sticky top-0 z-10 flex items-center gap-2 border-b border-edge-strong bg-surface-2 px-1 py-[3px] text-[10px] tracking-wide text-accent-weak uppercase">
              <span className="w-[40px] shrink-0">Code</span>
              <span className="w-[26px] shrink-0">Key</span>
              <span className="w-[64px] shrink-0">Args</span>
              <span className="min-w-0 flex-1">Description</span>
            </div>
            {COMMANDS.map((c) => {
              const screen = c.args ? null : screenFor(c.code);
              return (
                <div
                  key={c.code}
                  onClick={() => {
                    if (screen) go(screen, c.code);
                  }}
                  title={c.args ? `${c.code} ${c.args}` : c.title}
                  className={`flex items-baseline gap-2 border-b border-edge/40 px-1 py-[2px] hover:bg-surface-2 ${
                    screen ? "cursor-pointer" : ""
                  }`}
                >
                  <span className="w-[40px] shrink-0 font-bold text-accent">{c.code}</span>
                  <span className="w-[26px] shrink-0 text-[10px] text-info">
                    {c.fkey ? `F${c.fkey}` : ""}
                  </span>
                  <span className="w-[64px] shrink-0 truncate text-[10px] text-muted italic">
                    {c.args ?? ""}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-ink/85">{c.blurb}</span>
                </div>
              );
            })}
          </div>
        </Panel>

        <Panel title="Keyboard" flush animate>
          <div className="text-tiny">
            {KEYS.map((k) => (
              <div
                key={`${k.keys.join("+")}-${k.action}`}
                className="flex items-baseline gap-2 border-b border-edge/40 px-1 py-[2px]"
              >
                <span className="flex w-[92px] shrink-0 flex-wrap gap-1">
                  {k.keys.map((key) => (
                    <span
                      key={key}
                      className="border border-edge-strong bg-surface-2 px-1 text-[10px] text-accent"
                    >
                      {key}
                    </span>
                  ))}
                </span>
                <span className="min-w-0 flex-1 text-ink/85">{k.action}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Sectors" right={`${SECTORS.length} tags`} animate>
          <div className="flex flex-wrap gap-1">
            {SECTORS.map((s) => (
              <motion.button
                key={s.key}
                whileTap={tapScale}
                onClick={() => go({ fn: "CAT", tag: s.tag, label: s.label }, `CAT ${s.key}`)}
                title={`Browse ${s.label}`}
                className="border border-edge-strong bg-surface-2 px-1.5 py-[2px] text-tiny tracking-wide text-info uppercase hover:border-accent-weak hover:text-accent"
              >
                {s.key}
              </motion.button>
            ))}
          </div>
          <p className="mt-2 border-t border-edge pt-1.5 text-[10px] leading-relaxed text-muted">
            A sector opens the CAT screen filtered to that Gamma tag. Type{" "}
            <span className="text-accent">CAT crypto</span> for the same result from the command
            line.
          </p>
        </Panel>
      </motion.div>

      <Panel title="Getting Started" className="shrink-0" flush animate>
        <div className="text-tiny">
          {EXAMPLES.map((e) => (
            <div
              key={e.line}
              className="flex items-baseline gap-2 border-b border-edge/40 px-1 py-[2px] last:border-0"
            >
              <span className="shrink-0 text-accent">{">"}</span>
              <span className="shrink-0 text-ink">{e.line}</span>
              <span className="shrink-0 text-faint">{"<GO>"}</span>
              <span className="min-w-0 flex-1 truncate text-muted">— {e.note}</span>
            </div>
          ))}
        </div>
      </Panel>
    </motion.div>
  );
}
