"use client";

import { AnimatePresence, motion } from "motion/react";
import { useTerminal } from "@/components/TerminalProvider";
import { popVariants } from "@/lib/motion";

/** Transient notices, stacked bottom-right above the function-key strip. */
export function Toasts() {
  const { toasts, dismissToast } = useTerminal();
  if (toasts.length === 0) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="false"
      className="pointer-events-none fixed right-2 bottom-11 z-50 flex w-[300px] flex-col gap-1"
    >
      <AnimatePresence initial={false}>
        {toasts.map((t) => {
          const tone =
            t.tone === "error"
              ? "border-down text-down"
              : t.tone === "warn"
                ? "border-warn text-warn"
                : "border-edge-strong text-ink";
          // A severity mark, so an error/warning reads as one without relying on
          // colour alone; info keeps the quiet chevron.
          const info = t.tone !== "error" && t.tone !== "warn";
          return (
            <motion.button
              key={t.id}
              layout
              variants={popVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              onClick={() => dismissToast(t.id)}
              className={`pointer-events-auto border bg-canvas px-2 py-1 text-left text-tiny ${tone}`}
            >
              <span className={`mr-1 ${info ? "text-faint" : "font-bold"}`} aria-hidden>
                {info ? "›" : "!"}
              </span>
              {t.text}
            </motion.button>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
