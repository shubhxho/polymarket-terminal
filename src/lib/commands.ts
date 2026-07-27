/** Function-code registry. Single source of truth for the parser, the HELP
 *  screen and the autocomplete dropdown, so they can never drift apart. */

export type ScreenName =
  | "MON"
  | "SIG"
  | "MOV"
  | "DES"
  | "SRCH"
  | "WATCH"
  | "TAS"
  | "PORT"
  | "ALRT"
  | "CAT"
  | "HELP";

export type Screen =
  | { fn: "MON" }
  | { fn: "SIG" }
  | { fn: "MOV" }
  | { fn: "DES"; slug: string; kind: "event" | "market" }
  | { fn: "SRCH"; q: string }
  | { fn: "WATCH" }
  | { fn: "TAS" }
  | { fn: "PORT"; user: string }
  | { fn: "ALRT" }
  | { fn: "CAT"; tag: string; label: string }
  | { fn: "HELP" };

export type CommandSpec = {
  code: ScreenName;
  aliases: string[];
  title: string;
  /** Shown in HELP and in the autocomplete row. */
  blurb: string;
  /** Sidebar label. Falls back to `title`, which is too long for the rail. */
  short?: string;
  args?: string;
  /** Function-key slot, 1-indexed, if bound to F1..F12. */
  fkey?: number;
};

export const COMMANDS: CommandSpec[] = [
  {
    code: "MON",
    short: "Monitor",
    aliases: ["M", "MONITOR", "TOP"],
    title: "MARKET MONITOR",
    blurb: "Live grid of the highest-turnover markets",
    fkey: 2,
  },
  {
    code: "SIG",
    short: "Signals",
    aliases: ["SIGNAL", "SIGNALS", "SCAN"],
    title: "SIGNAL SCANNER",
    blurb: "Ranked signals, block flow and basket arbitrage",
    fkey: 3,
  },
  {
    code: "MOV",
    short: "Movers",
    aliases: ["MOVERS", "MOVE"],
    title: "MOVERS",
    blurb: "Biggest probability gainers and losers",
    fkey: 4,
  },
  {
    code: "DES",
    short: "Market Detail",
    aliases: ["D", "GO", "Q"],
    title: "MARKET DESCRIPTION",
    blurb: "Full analytics launchpad for one market",
    args: "<slug>",
    fkey: 5,
  },
  {
    code: "SRCH",
    short: "Search",
    aliases: ["S", "SEARCH", "FIND"],
    title: "SEARCH",
    blurb: "Full-text search across events and markets",
    args: "<query>",
    fkey: 6,
  },
  {
    code: "WATCH",
    short: "Watchlist",
    aliases: ["W", "WL"],
    title: "WATCHLIST",
    blurb: "Your pinned markets, quoted live",
    fkey: 7,
  },
  {
    code: "TAS",
    short: "Trade Tape",
    aliases: ["T", "TAPE", "WHALE"],
    title: "TRADE & SALES",
    blurb: "Consolidated print tape with size filter",
    fkey: 8,
  },
  {
    code: "ALRT",
    short: "Alerts",
    aliases: ["A", "ALERT", "ALERTS"],
    title: "ALERTS",
    blurb: "Price alerts that fire on live quotes",
    fkey: 9,
  },
  {
    code: "CAT",
    short: "Sectors",
    aliases: ["C", "SECTOR", "TAG"],
    title: "CATEGORY",
    blurb: "Browse a sector: politics, sports, crypto…",
    args: "<sector>",
    fkey: 10,
  },
  {
    // No F-key: it cannot be launched without an address anyway.
    code: "PORT",
    short: "Portfolio",
    aliases: ["P", "POS", "PORTFOLIO"],
    title: "PORTFOLIO",
    blurb: "Open positions and P&L for any wallet",
    args: "<0x wallet>",
  },
  {
    code: "HELP",
    short: "Help",
    aliases: ["H", "?", "MENU"],
    title: "HELP",
    blurb: "Command reference and keyboard map",
    fkey: 1,
  },
];

/** Curated sectors for the CAT screen. Tag ids are stable in Gamma. */
export const SECTORS: { key: string; label: string; tag: string }[] = [
  { key: "POLITICS", label: "Politics", tag: "2" },
  { key: "CRYPTO", label: "Crypto", tag: "21" },
  { key: "SPORTS", label: "Sports", tag: "1" },
  { key: "ECONOMICS", label: "Economics", tag: "100328" },
  { key: "GEOPOLITICS", label: "Geopolitics", tag: "100265" },
  { key: "TECH", label: "Tech", tag: "1401" },
  { key: "CULTURE", label: "Culture", tag: "596" },
  { key: "ELECTIONS", label: "Elections", tag: "100196" },
];

const BY_TOKEN = new Map<string, CommandSpec>();
for (const c of COMMANDS) {
  BY_TOKEN.set(c.code, c);
  for (const a of c.aliases) BY_TOKEN.set(a, c);
}

export function lookupCommand(token: string): CommandSpec | undefined {
  return BY_TOKEN.get(token.trim().toUpperCase());
}

export function commandByCode(code: ScreenName): CommandSpec {
  return COMMANDS.find((c) => c.code === code)!;
}

export type ParseResult =
  | { kind: "screen"; screen: Screen }
  | { kind: "error"; message: string };

/**
 * Parses a command line into a screen.
 *
 * Accepts Bloomberg-ish forms: a bare function code (`MON`), a code with
 * arguments (`SRCH fed cut`), a trailing `<GO>` which is stripped, or — when
 * the first token matches nothing — the whole line as a search query, which is
 * what a user typing a market name actually wants.
 */
export function parseCommand(input: string): ParseResult {
  const line = input.trim().replace(/\s*<?\bGO\b>?\s*$/i, "").trim();
  if (!line) return { kind: "error", message: "empty command" };

  const [head, ...rest] = line.split(/\s+/);
  const arg = rest.join(" ").trim();
  const spec = lookupCommand(head);

  if (!spec) {
    // Unknown leading token: treat the entire line as a search.
    return { kind: "screen", screen: { fn: "SRCH", q: line } };
  }

  switch (spec.code) {
    case "SRCH":
      if (!arg) return { kind: "error", message: "SRCH needs a query, e.g. SRCH fed" };
      return { kind: "screen", screen: { fn: "SRCH", q: arg } };

    case "DES": {
      if (!arg) return { kind: "error", message: "DES needs a market slug" };
      // Accept a pasted polymarket.com URL as well as a bare slug.
      const m = arg.match(/polymarket\.com\/(event|market)\/([^/?#\s]+)/i);
      if (m) {
        return {
          kind: "screen",
          screen: { fn: "DES", slug: m[2], kind: m[1].toLowerCase() === "market" ? "market" : "event" },
        };
      }
      return { kind: "screen", screen: { fn: "DES", slug: arg, kind: "event" } };
    }

    case "PORT": {
      const addr = arg.trim();
      if (!/^0x[a-fA-F0-9]{40}$/.test(addr)) {
        return { kind: "error", message: "PORT needs a 0x wallet address" };
      }
      return { kind: "screen", screen: { fn: "PORT", user: addr } };
    }

    case "CAT": {
      if (!arg) return { kind: "screen", screen: { fn: "CAT", tag: SECTORS[0].tag, label: SECTORS[0].label } };
      const want = arg.toUpperCase();
      const sector =
        SECTORS.find((s) => s.key === want) ??
        SECTORS.find((s) => s.key.startsWith(want)) ??
        SECTORS.find((s) => s.label.toUpperCase().includes(want));
      if (!sector) {
        return {
          kind: "error",
          message: `unknown sector "${arg}" — try ${SECTORS.map((s) => s.key).join(", ")}`,
        };
      }
      return { kind: "screen", screen: { fn: "CAT", tag: sector.tag, label: sector.label } };
    }

    case "MON":
      return { kind: "screen", screen: { fn: "MON" } };
    case "SIG":
      return { kind: "screen", screen: { fn: "SIG" } };
    case "MOV":
      return { kind: "screen", screen: { fn: "MOV" } };
    case "WATCH":
      return { kind: "screen", screen: { fn: "WATCH" } };
    case "TAS":
      return { kind: "screen", screen: { fn: "TAS" } };
    case "ALRT":
      return { kind: "screen", screen: { fn: "ALRT" } };
    case "HELP":
      return { kind: "screen", screen: { fn: "HELP" } };
  }
}

export function screenTitle(screen: Screen): string {
  const base = commandByCode(screen.fn).title;
  switch (screen.fn) {
    case "SRCH":
      return `${base} · "${screen.q}"`;
    case "DES":
      return `${base} · ${screen.slug}`;
    case "PORT":
      return `${base} · ${screen.user.slice(0, 10)}…`;
    case "CAT":
      return `${base} · ${screen.label.toUpperCase()}`;
    default:
      return base;
  }
}
