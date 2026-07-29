"use client";

import { useEffect, useMemo } from "react";
import { motion } from "motion/react";
import { useTerminal } from "@/components/TerminalProvider";
import { useWallet } from "@/hooks/useWallet";
import { cn } from "@/lib/cn";
import {
  COMMANDS,
  defaultScreenFor,
  parseCommand,
  SECTORS,
  type CommandSpec,
} from "@/lib/commands";
import { rowVariants, staggerContainer, tapScale } from "@/lib/motion";

/**
 * Primary navigation, modelled on ami.dev's project sidebar: grouped sections
 * with a small run-in label, one quiet row per item, a status dot on the left
 * and its shortcut trailing on the right.
 *
 * Functions that need an argument (DES, SRCH, PORT) can't be launched blind, so
 * their row prefills the command line instead of navigating.
 */
const GROUPS: { label: string; codes: string[] }[] = [
  { label: "Markets", codes: ["MON", "MOV", "CAT"] },
  { label: "Research", codes: ["SIG", "TAS", "DES", "SRCH"] },
  { label: "Positions", codes: ["WATCH", "ALRT", "PORT"] },
];

export function Sidebar() {
  const { screen, go, toast } = useTerminal();
  const { address } = useWallet();

  const byCode = useMemo(() => new Map(COMMANDS.map((c) => [c.code, c])), []);

  // Argument-taking functions open the palette rather than navigating blind.
  // Dispatching the shortcut is how the sidebar reaches it without either
  // component holding a reference to the other.
  const prefill = (spec: CommandSpec) => {
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true, bubbles: true }));
    toast(`${spec.code} takes ${spec.args} — search for it`);
  };

  const launch = (spec: CommandSpec) => {
    // An arg-taking function whose argument can come from session state (PORT ←
    // connected wallet) goes straight there instead of opening the palette.
    const preset = defaultScreenFor(spec.code, { walletAddress: address });
    if (preset) return go(preset, `${spec.code} ${address}`);
    if (spec.args) return prefill(spec);
    const parsed = parseCommand(spec.code);
    if (parsed.kind === "screen") go(parsed.screen, spec.code);
    else toast(parsed.message, "error");
  };

  // F-keys stay bound even though the strip is gone — muscle memory outlives
  // any redesign, and the shortcut is printed on every row.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const match = /^F(\d{1,2})$/.exec(e.key);
      if (!match) return;
      const spec = COMMANDS.find((c) => c.fkey === parseInt(match[1], 10));
      if (!spec) return;
      e.preventDefault();
      launch(spec);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // `launch` closes over `go`/`toast`, both stable from the provider.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [go, toast]);

  return (
    <motion.nav
      variants={staggerContainer}
      initial="initial"
      animate="animate"
      className="hidden w-[184px] shrink-0 flex-col gap-4 overflow-y-auto border-r border-edge bg-surface px-2 py-3 md:flex"
    >
      {GROUPS.map((group) => (
        <motion.div key={group.label} variants={rowVariants}>
          <div className="eyebrow px-1.5 pb-1">{group.label}</div>
          <ul className="flex flex-col gap-px">
            {group.codes.map((code) => {
              const spec = byCode.get(code as CommandSpec["code"]);
              if (!spec) return null;
              const active = screen.fn === spec.code;
              return (
                <li key={code}>
                  <motion.button
                    whileTap={tapScale}
                    onClick={() => launch(spec)}
                    title={`${spec.title} — ${spec.blurb}`}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "flex w-full items-center gap-2 rounded-sm px-1.5 py-[5px] text-left text-tiny transition-colors",
                      active
                        ? "bg-accent/10 font-medium text-accent"
                        : "text-ink/85 hover:bg-surface-2"
                    )}
                  >
                    <span
                      className={cn("dot", active ? "text-accent" : "text-edge-strong")}
                      aria-hidden
                    />
                    <span className="min-w-0 flex-1 truncate">{spec.short ?? spec.title}</span>
                    <span
                      className={cn(
                        "shrink-0 text-[10px]",
                        active ? "text-accent/70" : "text-faint"
                      )}
                    >
                      {spec.fkey ? `F${spec.fkey}` : spec.code}
                    </span>
                  </motion.button>
                </li>
              );
            })}
          </ul>
        </motion.div>
      ))}

      <motion.div variants={rowVariants} className="mt-auto">
        <div className="eyebrow px-1.5 pb-1">Sectors</div>
        <div className="flex flex-wrap gap-1 px-1">
          {SECTORS.slice(0, 6).map((s) => (
            <motion.button
              key={s.key}
              whileTap={tapScale}
              onClick={() => go({ fn: "CAT", tag: s.tag, label: s.label }, `CAT ${s.key}`)}
              className="rounded-sm border border-edge px-1.5 py-[1px] text-[10px] text-muted hover:border-accent-weak hover:text-accent"
            >
              {s.label}
            </motion.button>
          ))}
        </div>
      </motion.div>
    </motion.nav>
  );
}
