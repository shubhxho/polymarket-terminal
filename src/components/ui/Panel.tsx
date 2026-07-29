"use client";

import type { ReactNode } from "react";

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
}: {
  title: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  /** Drop body padding — for tables that draw their own gutters. */
  flush?: boolean;
}) {
  return (
    <section
      className={`flex min-h-0 min-w-0 flex-col overflow-hidden rounded-md border border-edge bg-surface ${className}`}
    >
      <header className="flex h-[26px] shrink-0 items-center justify-between gap-2 border-b border-edge px-2.5">
        <span className="eyebrow truncate">{title}</span>
        {right ? <span className="shrink-0 text-[11px] text-faint">{right}</span> : null}
      </header>
      <div
        className={`min-h-0 flex-1 overflow-auto ${flush ? "" : "px-2.5 py-2"} ${bodyClassName}`}
      >
        {children}
      </div>
    </section>
  );
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

export function Empty({ text = "no data" }: { text?: string }) {
  return (
    <div className="flex h-full min-h-[64px] items-center justify-center px-4 text-center text-[11px] text-faint">
      {text}
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
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            title={o.title}
            className={`rounded-sm font-medium transition-colors ${pad} ${
              active
                ? "bg-canvas text-ink shadow-[0_1px_2px_rgba(16,16,20,0.06)]"
                : "text-muted hover:text-ink"
            }`}
          >
            {o.label}
          </button>
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
  const Tag = onClick ? "button" : "span";
  return (
    <Tag
      onClick={onClick}
      title={title}
      className={`inline-flex shrink-0 items-center gap-1 rounded-sm border px-1.5 py-[1px] text-[11px] font-medium whitespace-nowrap ${
        active ? "border-accent bg-accent/8 text-accent" : tones[tone]
      } ${onClick ? "hover:border-accent-weak hover:text-accent" : ""}`}
    >
      {children}
    </Tag>
  );
}
