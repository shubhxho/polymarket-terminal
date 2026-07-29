"use client";

import { useTerminal } from "@/components/TerminalProvider";

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
      {toasts.map((t) => {
        const tone =
          t.tone === "error"
            ? "border-down text-down"
            : t.tone === "warn"
              ? "border-accent text-accent"
              : "border-edge-strong text-ink";
        return (
          <button
            key={t.id}
            onClick={() => dismissToast(t.id)}
            className={`pointer-events-auto border bg-canvas px-2 py-1 text-left text-tiny ${tone}`}
          >
            <span className="mr-1 text-faint">›</span>
            {t.text}
          </button>
        );
      })}
    </div>
  );
}
