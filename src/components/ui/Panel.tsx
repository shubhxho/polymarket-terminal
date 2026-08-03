"use client";

import { motion } from "motion/react";
import type { ReactNode } from "react";
import { panelVariants, tapScale } from "@/lib/motion";

/**
 * Standard panel chrome: a hairline card with a quiet header row.
 *
 * The header is deliberately understated — small grey uppercase label on the
 * card's own surface, not a filled bar. A screen shows six of these at once,
 * and six saturated title bars would fight the data for attention when the
 * data is the only thing worth looking at.
 */
export function Panel({
  title,
  right,
  children,
  className = "",
  bodyClassName = "",
  flush = false,
  animate = false,
}: {
  title: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  /** Drop body padding — for tables that draw their own gutters. */
  flush?: boolean;
  /**
   * Opt into the shared panel mount animation (a fade-and-lift via
   * `panelVariants`). Off by default, so every existing caller renders the
   * identical plain `<section>` — no public prop changes behaviour.
   *
   * When on, the panel becomes a `motion.section` that carries the variants but
   * *inherits its animation state* from the nearest ancestor motion container
   * (typically a `staggerContainer` at the screen root), rather than driving
   * its own initial/animate. That keeps the same element — className, flex
   * sizing and all — so no wrapper node disturbs the layout, and it lets a row
   * of panels stagger in together. Outside a motion container it is an inert
   * no-op.
   */
  animate?: boolean;
}) {
  const content = (
    <>
      <header className="flex h-[26px] shrink-0 items-center justify-between gap-2 border-b border-edge px-2.5">
        <span className="eyebrow truncate">{title}</span>
        {right ? <span className="shrink-0 text-[11px] text-faint">{right}</span> : null}
      </header>
      <div
        className={`min-h-0 flex-1 overflow-auto ${flush ? "" : "px-2.5 py-2"} ${bodyClassName}`}
      >
        {children}
      </div>
    </>
  );

  const base =
    "flex min-h-0 min-w-0 flex-col overflow-hidden rounded-md border border-edge bg-surface";

  if (animate) {
    return (
      <motion.section variants={panelVariants} className={`${base} ${className}`}>
        {content}
      </motion.section>
    );
  }

  return <section className={`${base} ${className}`}>{content}</section>;
}

/**
 * Dim label / bright value pair used across the detail screens.
 *
 * `title` is worth supplying on anything derived: a reader who doesn't already
 * know what "trend quality" means has nowhere else to find out, and a label
 * long enough to explain itself would break the column.
 */
export function Field({
  label,
  value,
  tone = "text-ink",
  title,
}: {
  label: string;
  value: ReactNode;
  tone?: string;
  title?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2 border-b border-edge/60 py-[5px] last:border-0">
      <span
        className={`shrink-0 text-[11px] text-muted ${title ? "cursor-help decoration-edge-strong decoration-dotted underline-offset-2 hover:underline" : ""}`}
        title={title}
      >
        {label}
      </span>
      <span className={`truncate text-right text-tiny font-medium ${tone}`}>{value}</span>
    </div>
  );
}

/**
 * The one refreshing indicator, for a Panel's `right` header. Every screen was
 * rolling its own — two glyphs (`···`/`sync…`) across four colours — for the
 * same "a background refetch is in flight" concept; this makes it one mark.
 * Renders nothing when idle.
 */
export function Refreshing({ show }: { show: boolean }) {
  return show ? (
    <span className="text-faint" title="Refreshing…" aria-hidden>
      ···
    </span>
  ) : null;
}

/**
 * Empty slot. `hint` opts into a two-tier message — a plain headline over a
 * fainter line of guidance — the way polished empty states separate "nothing
 * here" from "here's how to fill it". Without it, the single faint line renders
 * exactly as before, so existing callers are untouched.
 */
export function Empty({ text = "no data", hint }: { text?: string; hint?: string }) {
  return (
    <div className="flex h-full min-h-[64px] flex-col items-center justify-center gap-1 px-4 text-center">
      <span className={`text-[11px] ${hint ? "text-muted" : "text-faint"}`}>{text}</span>
      {hint ? <span className="text-[10px] text-faint">{hint}</span> : null}
    </div>
  );
}

export function Loading({ text = "loading" }: { text?: string }) {
  return (
    <div className="flex h-full min-h-[64px] items-center justify-center gap-2 text-[11px] text-faint">
      <span className="dot animate-pulse text-accent" />
      {text}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-sm border border-down-weak bg-down/5 px-2.5 py-2 text-tiny text-down">
      <span className="dot mt-1.5" />
      <span className="min-w-0">
        <span className="font-medium">Feed error · </span>
        {message}
      </span>
    </div>
  );
}

/**
 * ami's segmented control: a pill track with the active segment lifted onto a
 * white chip. Used for every interval / timeframe / filter switch so those all
 * read as one control rather than a dozen ad-hoc button rows.
 */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  size = "sm",
}: {
  options: readonly { value: T; label: string; title?: string }[];
  value: T;
  onChange: (v: T) => void;
  size?: "sm" | "xs";
}) {
  const pad = size === "xs" ? "px-1.5 py-[1px] text-[11px]" : "px-2 py-[3px] text-[11px]";
  return (
    <div className="inline-flex items-center gap-0.5 rounded-md border border-edge bg-surface-2 p-0.5">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <motion.button
            key={o.value}
            whileTap={tapScale}
            onClick={() => onChange(o.value)}
            title={o.title}
            className={`rounded-sm font-medium transition-colors ${pad} ${
              active
                ? "bg-canvas text-ink shadow-[0_1px_2px_rgba(16,16,20,0.06)]"
                : "text-muted hover:text-ink"
            }`}
          >
            {o.label}
          </motion.button>
        );
      })}
    </div>
  );
}

/** Small bordered tag. `tone` maps to the palette's semantic roles. */
export function Chip({
  children,
  tone = "neutral",
  active = false,
  onClick,
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "up" | "down" | "accent" | "info" | "warn";
  active?: boolean;
  onClick?: () => void;
  title?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "border-edge text-muted",
    up: "border-up-weak text-up",
    down: "border-down-weak text-down",
    accent: "border-accent/40 text-accent",
    info: "border-info-weak text-info",
    warn: "border-warn/40 text-warn",
  };
  const cls = `inline-flex shrink-0 items-center gap-1 rounded-sm border px-1.5 py-[1px] text-[11px] font-medium whitespace-nowrap ${
    active ? "border-accent bg-accent/8 text-accent" : tones[tone]
  } ${onClick ? "hover:border-accent-weak hover:text-accent" : ""}`;

  if (onClick) {
    return (
      <motion.button whileTap={tapScale} onClick={onClick} title={title} className={cls}>
        {children}
      </motion.button>
    );
  }
  return (
    <span title={title} className={cls}>
      {children}
    </span>
  );
}
