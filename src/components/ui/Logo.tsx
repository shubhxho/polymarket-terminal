import { cn } from "@/lib/cn";

/**
 * Brand mark for the terminal — a rounded badge holding a stylised probability
 * curve that resolves toward 1 (a market finding its answer), with the two
 * candlestick ticks that make it read as a trading surface rather than a plain
 * monogram. Draws in `currentColor` so it inherits the masthead's ink/canvas
 * tokens and flips cleanly between light and dark. Purely decorative.
 */
export function LogoMark({ size = 24, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      role="img"
      aria-label="Polymarket Terminal"
      className={cn("shrink-0", className)}
    >
      <rect x="0.75" y="0.75" width="22.5" height="22.5" rx="6" className="fill-ink" />
      {/* probability curve settling toward the top rail */}
      <path
        d="M4.5 16.5 C7.5 16.2 8.6 9.8 12 9.2 C15.4 8.6 16.6 6.4 19.5 6.3"
        className="stroke-canvas"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* aggressor ticks under the curve */}
      <rect
        x="8.1"
        y="14.4"
        width="1.7"
        height="3.6"
        rx="0.85"
        className="fill-canvas"
        opacity="0.55"
      />
      <rect
        x="14.2"
        y="11.2"
        width="1.7"
        height="6.8"
        rx="0.85"
        className="fill-canvas"
        opacity="0.8"
      />
      {/* resolution node at the top rail */}
      <circle cx="19.5" cy="6.3" r="1.7" className="fill-canvas" />
      <circle cx="19.5" cy="6.3" r="0.7" className="fill-ink" />
    </svg>
  );
}

/**
 * Full lockup: mark + two-tone wordmark. "Polymarket" carries the weight,
 * "Terminal" trails as a muted micro-caps tag so the surname reads as a product
 * label, not a second brand.
 */
export function Logo({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <LogoMark size={22} />
      <span className="flex items-baseline gap-1.5 leading-none">
        <span className="text-sm2 font-semibold tracking-[-0.015em]">Polymarket</span>
        <span className="text-[10px] font-medium uppercase tracking-[0.14em] text-faint">
          Terminal
        </span>
      </span>
    </div>
  );
}
