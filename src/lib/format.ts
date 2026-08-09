/** Fixed-width formatters. Terminal columns align only if every cell in a
 *  column renders the same number of glyphs, so these bias toward padding. */

/** Rendered wherever a value is genuinely absent. */
const EMPTY = "\u2014";

export function num(n: number | undefined, digits = 2): string {
  if (n === undefined || n === null || Number.isNaN(n)) return EMPTY;
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** 1_234_567 -> "1.23M". Keeps 3 significant-ish chars so columns stay tight. */
export function compact(n: number | undefined): string {
  if (n === undefined || n === null || Number.isNaN(n)) return EMPTY;
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (abs >= 1e9) return `${sign}${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(1)}K`;
  return `${sign}${abs.toFixed(abs < 10 ? 2 : 0)}`;
}

export function usd(n: number | undefined): string {
  if (n === undefined || n === null || Number.isNaN(n)) return EMPTY;
  // Sign goes outside the currency symbol: "-$23", never "$-23".
  return `${n < 0 ? "-" : ""}$${compact(Math.abs(n))}`;
}

/** Probability 0..1 rendered as a Polymarket-style cent price: 0.4231 -> "42.3". */
export function cents(p: number | undefined, digits = 1): string {
  if (p === undefined || p === null || Number.isNaN(p)) return EMPTY;
  return (p * 100).toFixed(digits);
}

export function signed(n: number | undefined, digits = 1): string {
  if (n === undefined || n === null || Number.isNaN(n)) return EMPTY;
  const s = n > 0 ? "+" : n < 0 ? "" : " ";
  return `${s}${n.toFixed(digits)}`;
}

export function pad(s: string, width: number, align: "l" | "r" = "l"): string {
  const t = s.length > width ? s.slice(0, width) : s;
  return align === "l" ? t.padEnd(width) : t.padStart(width);
}

export function truncate(s: string, width: number): string {
  if (!s) return "";
  return s.length <= width ? s : `${s.slice(0, Math.max(0, width - 1))}…`;
}

/** Wall clock in UTC, terminal-style: "14:32:07". */
export function clock(d: Date = new Date()): string {
  return d.toISOString().slice(11, 19);
}

export function dateShort(iso?: string): string {
  if (!iso) return EMPTY;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return EMPTY;
  const months = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(" ");
  return `${String(d.getUTCDate()).padStart(2, "0")}${months[d.getUTCMonth()]}${String(
    d.getUTCFullYear()
  ).slice(2)}`;
}

export function timeOfDay(unixSeconds: number): string {
  return new Date(unixSeconds * 1000).toISOString().slice(11, 19);
}

/** Countdown to resolution, e.g. "12d 04h" / "03:22:10" when inside a day. */
export function timeToExpiry(iso?: string): string {
  if (!iso) return EMPTY;
  const end = new Date(iso).getTime();
  if (Number.isNaN(end)) return EMPTY;
  const ms = end - Date.now();
  if (ms <= 0) return "EXPIRED";
  const d = Math.floor(ms / 86400000);
  const h = Math.floor((ms % 86400000) / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  if (d > 0) return `${d}d ${String(h).padStart(2, "0")}h`;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function relTime(unixSeconds: number): string {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - unixSeconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

export function shortAddr(a?: string): string {
  if (!a) return EMPTY;
  return `${a.slice(0, 6)}..${a.slice(-4)}`;
}

/** Direction class for tinting a delta. Zero is neutral, not green. */
export function dirClass(n: number | undefined): string {
  if (n === undefined || Number.isNaN(n) || n === 0) return "text-muted";
  return n > 0 ? "text-up" : "text-down";
}
