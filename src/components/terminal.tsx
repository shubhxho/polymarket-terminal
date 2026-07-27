"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import {
  type OrderBook,
  simulateArbBasket,
  simulateDirectional,
  simulateSellBasket,
} from "@/lib/execution";
import { fmtCents, fmtUsd } from "@/lib/polymarket";
import type { ScanOptions, Signal, SignalKind } from "@/lib/signals";
import type { ScanRequest, ScanResponse } from "@/workers/signals.worker";

const TAGS = [
  "trending",
  "politics",
  "sports",
  "crypto",
  "economy",
  "geopolitics",
  "tech",
  "pop-culture",
];
const SORT_KEYS = ["vol", "liq", "oi", "change", "ends", "odds"];
const THEMES = ["green", "amber", "cyan", "red"];
const COMMANDS = [
  "help",
  "tags",
  "tag",
  "search",
  "sort",
  "page",
  "next",
  "prev",
  "open",
  "edge",
  "scan",
  "signal",
  "watch",
  "sim",
  "simulate",
  "export",
  "theme",
  "home",
  "back",
  "refresh",
  "clear",
  "exit",
];

type Kind = "cmd" | "out" | "ok" | "err" | "dim";

interface Line {
  id: number;
  kind: Kind;
  text: string;
}

let lineId = 0;
function L(kind: Kind, text: string): Line {
  return { id: ++lineId, kind, text };
}

const KIND_COLOR: Record<Kind, string> = {
  cmd: "text-foreground",
  out: "text-muted",
  ok: "text-accent",
  err: "text-red",
  dim: "text-muted/60",
};

const HELP: string[] = [
  "COMMANDS",
  "  TAG <name>        SWITCH BOARD — TAGS LISTS THEM",
  "  SEARCH <query>    FIND MARKETS",
  "  SORT <key> [DIR]  VOL|LIQ|OI|CHANGE|ENDS|ODDS · ASC|DESC",
  "  PAGE <n>          JUMP TO PAGE · NEXT · PREV",
  "  OPEN <#|E#|slug>  OPEN MARKET BY ROW #, SIGNAL E#, OR SLUG",
  "  EDGE [ARB|MOM] [N]  SCAN BOARD FOR SIGNALS (WEB-WORKER KERNEL) · OPEN E# TO JUMP",
  "  WATCH [SEC] [ARB|MOM]  LIVE RE-SCAN, FLAG NEW/RISING EDGES · WATCH OFF STOPS",
  "  SIM E<#> [$AMT]   FILL-SIM A SIGNAL ON LIVE CLOB DEPTH · SLIPPAGE & P&L",
  "  EXPORT            DOWNLOAD CURRENT VIEW AS CSV",
  "  THEME <name>      GREEN|AMBER|CYAN|RED PHOSPHOR",
  "  HOME · BACK · REFRESH · CLEAR · EXIT",
];

/** Quick-action buttons — the "instructor" bar. Each runs a command on click. */
const QUICK: { label: string; cmd: string; title: string }[] = [
  { label: "SCAN", cmd: "edge", title: "Scan the board for all signals" },
  {
    label: "ARB",
    cmd: "edge arb",
    title: "Arbitrage / mispricing signals only",
  },
  { label: "MOM", cmd: "edge mom", title: "Momentum signals only" },
  { label: "WATCH", cmd: "watch", title: "Live re-scan every 30s" },
  { label: "STOP", cmd: "watch off", title: "Stop the live watch" },
  { label: "EXPORT", cmd: "export", title: "Download current board as CSV" },
  { label: "HELP", cmd: "help", title: "List every command" },
];

export function Terminal() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [open, setOpen] = useState(false);
  const [lines, setLines] = useState<Line[]>(() => [
    L("ok", "POLYMARKET TERMINAL — COMMAND INTERFACE READY"),
    L(
      "dim",
      'TYPE "HELP" FOR COMMANDS · TAB COMPLETES · ↑↓ HISTORY · ESC CLOSES',
    ),
  ]);
  const [value, setValue] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const histIdx = useRef(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const scanId = useRef(0);
  const lastSignals = useRef<Signal[]>([]);
  const watchTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const watchSeen = useRef<Map<string, { score: number; title: string }>>(
    new Map(),
  );

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      if (e.key === "`" && tag !== "INPUT" && tag !== "TEXTAREA") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Tear down the worker and any watch timer when the terminal unmounts.
  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
      if (watchTimer.current !== null) clearInterval(watchTimer.current);
    };
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll when the log grows
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [lines, open]);

  function print(...ls: Line[]) {
    setLines((prev) => [...prev.slice(-300), ...ls]);
  }

  /** Board params to mutate: current ones on the board, fresh elsewhere. */
  function boardParams(): URLSearchParams {
    return pathname === "/"
      ? new URLSearchParams(searchParams.toString())
      : new URLSearchParams();
  }

  function go(p: URLSearchParams, note: string) {
    const qs = p.toString();
    const href = qs ? `/?${qs}` : "/";
    router.push(href);
    print(L("ok", note), L("dim", `→ ${href}`));
  }

  function applyTheme(name: string) {
    try {
      if (name === "green") {
        delete document.documentElement.dataset.theme;
        localStorage.removeItem("pm-theme");
      } else {
        document.documentElement.dataset.theme = name;
        localStorage.setItem("pm-theme", name);
      }
    } catch {
      // localStorage unavailable — theme still applies for this page
    }
  }

  async function doExport() {
    const p = boardParams();
    const tag = p.get("tag") ?? "";
    const q = p.get("q") ?? "";
    print(L("dim", "CONNECTING TO GAMMA API_"));
    try {
      const params = new URLSearchParams();
      if (tag) params.set("tag", tag);
      if (q) params.set("q", q);
      const res = await fetch(`/api/export?${params}`);
      if (!res.ok) throw new Error("export failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `polymarket-${q ? `search-${q.slice(0, 20)}` : tag || "trending"}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      print(L("ok", "EXPORT COMPLETE — DATA SECURED"));
    } catch {
      print(L("err", "EXPORT FAILED — RETRY"));
    }
  }

  /** Lazily spin up the compute worker; correlate replies by request id. */
  function scan(events: unknown[], opts: ScanOptions): Promise<ScanResponse> {
    if (!workerRef.current) {
      workerRef.current = new Worker(
        new URL("../workers/signals.worker.ts", import.meta.url),
      );
    }
    const worker = workerRef.current;
    const id = ++scanId.current;
    return new Promise<ScanResponse>((resolve, reject) => {
      const onMessage = (e: MessageEvent<ScanResponse>) => {
        if (e.data.id !== id) return;
        worker.removeEventListener("message", onMessage);
        resolve(e.data);
      };
      worker.addEventListener("message", onMessage);
      const req: ScanRequest = {
        id,
        events: events as ScanRequest["events"],
        opts,
      };
      worker.postMessage(req);
      setTimeout(() => {
        worker.removeEventListener("message", onMessage);
        reject(new Error("kernel timeout"));
      }, 15000);
    });
  }

  function printSignals(signals: Signal[]) {
    lastSignals.current = signals;
    if (signals.length === 0) {
      print(L("out", "NO SIGNALS CLEAR THRESHOLD ON THIS BOARD"));
      return;
    }
    const arbN = signals.filter((s) => s.kind === "ARB").length;
    const momN = signals.length - arbN;
    const avg = Math.round(
      signals.reduce((a, s) => a + s.score, 0) / signals.length,
    );
    const best = signals[0];
    print(
      L(
        "ok",
        `${arbN} ARB / ${momN} MOM · AVG SCORE ${avg} · TOP ${Math.round(best.score)} (${best.kind}) — ${best.title.toUpperCase().slice(0, 46)}`,
      ),
    );
    for (const [i, s] of signals.entries()) {
      const handle = `E${String(i + 1).padStart(2, "0")}`;
      const bar = "█".repeat(Math.max(1, Math.round(s.score / 10)));
      const edge =
        s.kind === "ARB"
          ? `${s.edgeBps > 0 ? "+" : ""}${(s.edgeBps / 100).toFixed(2)}%`
          : `${s.edgeBps > 0 ? "+" : ""}${(s.edgeBps / 100).toFixed(1)}%`;
      const spr =
        s.spreadBps != null ? `${(s.spreadBps / 100).toFixed(1)}% SPR` : "—";
      print(
        L(
          s.score >= 60 ? "ok" : "out",
          `${handle} ${s.kind.padEnd(8)} ${bar.padEnd(10)} ${String(
            Math.round(s.score),
          ).padStart(
            3,
          )} · EDGE ${edge.padStart(7)} · ${spr.padStart(9)} · ${fmtUsd(s.liquidity).padStart(6)} LIQ`,
        ),
      );
      print(L("dim", `   ${s.title.toUpperCase()} — ${s.detail}`));
    }
    print(
      L(
        "dim",
        "OPEN E<#> TO INSPECT A MARKET · SCORE 0–100 = EDGE × BOOK QUALITY",
      ),
    );
    print(
      L(
        "dim",
        "EDGE IS A MISPRICING SIGNAL, NOT A FILLABLE QUOTE — DEPTH, SLIPPAGE & FEES UNMODELED · INFORMATIONAL ONLY, NOT FINANCIAL ADVICE",
      ),
    );
  }

  /** Fetch the current board and run the kernel over it. Throws on failure. */
  async function runScan(
    limit: number,
    kinds?: SignalKind[],
  ): Promise<ScanResponse> {
    const p = boardParams();
    const params = new URLSearchParams();
    const tag = p.get("tag") ?? "";
    const q = p.get("q") ?? "";
    if (tag) params.set("tag", tag);
    if (q) params.set("q", q);
    const res = await fetch(`/api/signals?${params}`);
    if (!res.ok) throw new Error("feed error");
    const { events } = (await res.json()) as { events: unknown[] };
    return scan(events, { limit, kinds });
  }

  async function doScan(limit: number, kinds?: SignalKind[]) {
    const label = kinds ? kinds.join("+") : "ALL";
    print(
      L("dim", `SCANNING BOARD [${label}] — DISPATCHING KERNEL TO WEB WORKER_`),
    );
    try {
      const out = await runScan(limit, kinds);
      print(
        L(
          "ok",
          `SCANNED ${out.scanned} EVENTS IN ${out.ms.toFixed(1)}MS · ${out.signals.length} SIGNALS`,
        ),
      );
      printSignals(out.signals);
    } catch {
      print(L("err", "SCAN FAILED — RETRY"));
    }
  }

  /** Poll the kernel on an interval, surfacing only new / strengthening edges. */
  function startWatch(seconds: number, kinds?: SignalKind[]) {
    stopWatch();
    watchSeen.current = new Map();
    const label = kinds ? kinds.join("+") : "ALL";
    print(
      L("ok", `WATCH ON [${label}] · EVERY ${seconds}S · "WATCH OFF" TO STOP`),
    );
    const tick = async () => {
      try {
        const out = await runScan(12, kinds);
        lastSignals.current = out.signals;
        const prev = watchSeen.current;
        const next = new Map<string, { score: number; title: string }>();
        const rising: Line[] = [];
        const falling: Line[] = [];
        for (const s of out.signals) {
          const now = Math.round(s.score);
          const title = s.title.toUpperCase();
          next.set(s.slug, { score: now, title });
          const before = prev.get(s.slug)?.score;
          if (before === undefined) {
            rising.push(L("ok", `WATCH · NEW  ${s.kind} ${now} — ${title}`));
          } else if (now - before >= 5) {
            rising.push(
              L("ok", `WATCH · ▲${now - before} ${s.kind} ${now} — ${title}`),
            );
          } else if (before - now >= 5) {
            falling.push(
              L("out", `WATCH · ▼${before - now} ${s.kind} ${now} — ${title}`),
            );
          }
        }
        for (const [slug, was] of prev) {
          if (!next.has(slug)) {
            falling.push(L("dim", `WATCH · GONE ${was.score} — ${was.title}`));
          }
        }
        watchSeen.current = next;
        const changes = [...rising, ...falling];
        if (changes.length === 0) {
          print(
            L(
              "dim",
              `WATCH · ${out.scanned} EVENTS · ${out.ms.toFixed(0)}MS · ${out.signals.length} SIGNALS · NO CHANGE`,
            ),
          );
        } else {
          print(...changes);
        }
      } catch {
        print(L("err", "WATCH · SCAN FAILED — WILL RETRY"));
      }
    };
    void tick();
    watchTimer.current = setInterval(() => void tick(), seconds * 1000);
  }

  function stopWatch(announce = false) {
    if (watchTimer.current !== null) {
      clearInterval(watchTimer.current);
      watchTimer.current = null;
      if (announce) print(L("ok", "WATCH OFF"));
    } else if (announce) {
      print(L("out", "WATCH IS NOT RUNNING"));
    }
  }

  const slip = (bps: number) => `${bps >= 0 ? "+" : ""}${bps.toFixed(0)}BPS`;

  /** Pull live CLOB depth for a signal and simulate walking the book. */
  async function doSim(handle: string, budget: number) {
    const idx = Number(handle.replace(/^e/i, "")) - 1;
    const s = lastSignals.current[idx];
    if (!s) {
      print(L("err", `NO SIGNAL ${handle.toUpperCase()} — RUN EDGE FIRST`));
      return;
    }
    const tokens = s.legs.map((l) => l.tokenId);
    if (tokens.length === 0) {
      print(L("err", "SIGNAL HAS NO TRADEABLE LEGS"));
      return;
    }
    print(
      L(
        "dim",
        `SIMULATING $${budget} FILL · PULLING CLOB DEPTH FOR ${tokens.length} LEG${tokens.length > 1 ? "S" : ""}_`,
      ),
    );
    try {
      const res = await fetch(`/api/book?tokens=${tokens.join(",")}`);
      if (!res.ok) throw new Error("book fetch");
      const { books } = (await res.json()) as { books: OrderBook[] };
      const byToken = new Map(books.map((b) => [b.tokenId, b]));

      print(
        L("ok", `SIM ${handle.toUpperCase()} · $${budget} · ${s.kind}`),
        L("dim", `   ${s.title.toUpperCase()}`),
      );

      if (s.kind === "MOMENTUM") {
        const book = byToken.get(s.legs[0].tokenId);
        if (!book || book.asks.length === 0) {
          print(L("err", "NO LIVE ASKS ON THIS BOOK — CANNOT FILL"));
          return;
        }
        const r = simulateDirectional(book, budget);
        const notFilled = r.fill.filled
          ? ""
          : ` · DEPTH DRY, ONLY $${r.fill.spent.toFixed(0)} FILLED`;
        print(
          L(
            "out",
            `   FILLED ${r.fill.shares.toFixed(0)} SH @ AVG ${fmtCents(r.fill.avgPrice)} (TOUCH ${fmtCents(r.fill.touch)}, SLIP ${slip(r.fill.slippageBps)})${notFilled}`,
          ),
          L(
            "out",
            `   RESOLVES YES → +${r.upsidePct.toFixed(1)}% · NO → -100% · BREAKEVEN ${fmtCents(r.fill.avgPrice)}`,
          ),
        );
      } else {
        const legs = s.legs
          .map((l) => ({ label: l.label, book: byToken.get(l.tokenId) }))
          .filter((l): l is { label: string; book: OrderBook } =>
            Boolean(l.book && l.book.asks.length > 0),
          );
        if (legs.length === 0) {
          print(L("err", "NO LIVE ASKS ON ANY LEG — CANNOT FILL"));
          return;
        }
        const r = simulateArbBasket(legs, budget);
        if (!r) {
          print(L("err", "BASKET COST IS ZERO — CANNOT SIMULATE"));
          return;
        }
        const avgSlip =
          r.legs.reduce((a, l) => a + l.fill.slippageBps, 0) / r.legs.length;
        const fillNote = r.complete
          ? "ALL LEGS FILLED"
          : `INCOMPLETE — ${r.missingLegs} LEG(S) SHORT, ARB NOT GUARANTEED`;
        if (s.edgeBps >= 0) {
          // Underround: buying every YES is the real trade. Positive net = a
          // captured arb; slippage/spread usually eats most of the mid-price edge.
          print(
            L(
              "out",
              `   BUY-ALL · ${legs.length} LEGS · ${r.baskets.toFixed(1)} BASKETS · COST $${r.cost.toFixed(2)}`,
            ),
            L(
              r.net >= 0 ? "ok" : "err",
              `   GUARANTEED PAYOUT $${r.payout.toFixed(2)} · NET ${r.net >= 0 ? "+" : ""}$${r.net.toFixed(2)} · REALIZED ${r.realizedPct >= 0 ? "+" : ""}${r.realizedPct.toFixed(2)}%`,
            ),
            L(
              "dim",
              `   MID-EDGE ${(s.edgeBps / 100).toFixed(2)}% → ${r.realizedPct.toFixed(2)}% AT THE ASK · AVG SLIP ${slip(avgSlip)} · ${fillNote}`,
            ),
          );
        } else {
          // Overround: the capture is SELLING the basket into the bids, not
          // buying. Walk the real bid depth we already fetched.
          const sell = simulateSellBasket(legs, budget);
          if (!sell) {
            print(L("err", "NO LIVE BIDS — CANNOT MODEL SELL-SIDE CAPTURE"));
          } else {
            const avgSlip =
              sell.legs.reduce((a, l) => a + l.fill.slippageBps, 0) /
              sell.legs.length;
            const fillNote = sell.complete
              ? "ALL LEGS FILLED"
              : `INCOMPLETE — ${sell.missingLegs} LEG(S) SHORT, NOT FULLY HEDGED`;
            print(
              L(
                "out",
                `   OVERROUND · CAPTURE = SELL-ALL INTO THE BIDS (${legs.length} LEGS)`,
              ),
              L(
                "out",
                `   ${sell.sets.toFixed(1)} SETS · COLLECT $${sell.proceeds.toFixed(2)} · OWE $${sell.liability.toFixed(2)}`,
              ),
              L(
                sell.net >= 0 ? "ok" : "err",
                `   NET ${sell.net >= 0 ? "+" : ""}$${sell.net.toFixed(2)} · REALIZED ${sell.realizedPct >= 0 ? "+" : ""}${sell.realizedPct.toFixed(2)}% (MID VIG ${(s.edgeBps / -100).toFixed(2)}%)`,
              ),
              L(
                "dim",
                `   AVG SLIP ${slip(avgSlip)} · ${fillNote} · NEEDS MINTED SET / NEG-RISK CONVERT, NOT A NAKED SHORT`,
              ),
            );
          }
        }
      }
      print(
        L("dim", "SIMULATION ONLY · NO ORDER PLACED · NOT FINANCIAL ADVICE"),
      );
    } catch {
      print(L("err", "SIM FAILED — BOOK FEED UNAVAILABLE"));
    }
  }

  function run(raw: string) {
    const input = raw.trim();
    if (!input) return;
    print(L("cmd", `> ${input.toUpperCase()}`));
    setHistory((h) => [...h, input]);
    histIdx.current = -1;

    const parts = input.split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const arg = parts.slice(1).join(" ");
    const argLower = arg.toLowerCase();

    switch (cmd) {
      case "help":
      case "?":
        print(...HELP.map((t) => L("out", t)));
        break;

      case "tags":
      case "ls":
        print(L("out", TAGS.join(" · ").toUpperCase()));
        break;

      case "tag":
      case "go": {
        if (!TAGS.includes(argLower)) {
          print(
            L(
              "err",
              `UNKNOWN TAG "${arg.toUpperCase()}" — TRY: ${TAGS.join(", ").toUpperCase()}`,
            ),
          );
          break;
        }
        const p = boardParams();
        p.delete("q");
        p.delete("page");
        if (argLower === "trending") p.delete("tag");
        else p.set("tag", argLower);
        go(p, `BOARD → ${argLower.toUpperCase()}`);
        break;
      }

      case "search":
      case "find":
      case "q": {
        if (!arg) {
          print(L("err", "USAGE: SEARCH <QUERY>"));
          break;
        }
        const p = new URLSearchParams();
        p.set("q", arg);
        go(p, `SEARCHING "${arg.toUpperCase()}"`);
        break;
      }

      case "sort": {
        const [key, dir = "desc"] = argLower.split(/\s+/);
        if (!SORT_KEYS.includes(key) || !["asc", "desc"].includes(dir)) {
          print(
            L(
              "err",
              `USAGE: SORT <${SORT_KEYS.join("|").toUpperCase()}> [ASC|DESC]`,
            ),
          );
          break;
        }
        const p = boardParams();
        p.delete("page");
        p.set("sort", key);
        p.set("dir", dir);
        go(p, `SORTED BY ${key.toUpperCase()} ${dir === "asc" ? "▲" : "▼"}`);
        break;
      }

      case "page":
      case "next":
      case "prev": {
        const current = Number(boardParams().get("page")) || 1;
        const target =
          cmd === "next"
            ? current + 1
            : cmd === "prev"
              ? current - 1
              : Number(arg);
        if (!Number.isInteger(target) || target < 1) {
          print(
            L("err", cmd === "prev" ? "ALREADY ON PAGE 1" : "USAGE: PAGE <N>"),
          );
          break;
        }
        const p = boardParams();
        p.set("page", String(target));
        go(p, `PAGE ${target}`);
        break;
      }

      case "open": {
        if (!arg) {
          print(
            L("err", "USAGE: OPEN <ROW #> · OPEN E<#> (SIGNAL) · OPEN <SLUG>"),
          );
          break;
        }
        const sig = argLower.match(/^e(\d+)$/);
        if (sig) {
          const s = lastSignals.current[Number(sig[1]) - 1];
          if (!s) {
            print(L("err", `NO SIGNAL ${arg.toUpperCase()} — RUN EDGE FIRST`));
            break;
          }
          router.push(`/event/${s.slug}`);
          print(
            L("ok", `OPENING ${arg.toUpperCase()} · ${s.kind}`),
            L("dim", `→ /event/${s.slug}`),
          );
          break;
        }
        if (/^\d+$/.test(arg)) {
          const slug = document
            .querySelector(`[data-term-index="${Number(arg)}"]`)
            ?.getAttribute("data-term-row");
          if (!slug) {
            print(L("err", `ROW ${arg} NOT ON SCREEN — CHECK THE # COLUMN`));
            break;
          }
          router.push(`/event/${slug}`);
          print(L("ok", `OPENING ROW ${arg}`), L("dim", `→ /event/${slug}`));
        } else {
          router.push(`/event/${encodeURIComponent(argLower)}`);
          print(L("ok", `OPENING ${argLower.toUpperCase()}`));
        }
        break;
      }

      case "edge":
      case "scan":
      case "signal": {
        const tokens = argLower.split(/\s+/).filter(Boolean);
        let kinds: SignalKind[] | undefined;
        let limit = 12;
        for (const t of tokens) {
          if (t === "arb") kinds = ["ARB"];
          else if (t === "mom" || t === "momentum") kinds = ["MOMENTUM"];
          else {
            const n = Number(t);
            if (Number.isInteger(n) && n > 0 && n <= 40) limit = n;
          }
        }
        void doScan(limit, kinds);
        break;
      }

      case "watch": {
        const tokens = argLower.split(/\s+/).filter(Boolean);
        if (tokens.includes("off") || tokens.includes("stop")) {
          stopWatch(true);
          break;
        }
        let kinds: SignalKind[] | undefined;
        let seconds = 30;
        for (const t of tokens) {
          if (t === "arb") kinds = ["ARB"];
          else if (t === "mom" || t === "momentum") kinds = ["MOMENTUM"];
          else {
            const n = Number(t);
            if (Number.isFinite(n) && n >= 10 && n <= 3600) seconds = n;
          }
        }
        startWatch(seconds, kinds);
        break;
      }

      case "sim":
      case "simulate": {
        const tokens = argLower.split(/\s+/).filter(Boolean);
        const handle = tokens.find((t) => /^e\d+$/.test(t));
        if (!handle) {
          print(L("err", "USAGE: SIM E<#> [$AMOUNT] — RUN EDGE FIRST"));
          break;
        }
        const amt = tokens
          .map((t) => Number(t.replace(/^\$/, "")))
          .find((n) => Number.isFinite(n) && n >= 1 && n <= 1_000_000);
        void doSim(handle, amt ?? 500);
        break;
      }

      case "export":
        void doExport();
        break;

      case "theme": {
        if (!THEMES.includes(argLower)) {
          print(L("err", `USAGE: THEME <${THEMES.join("|").toUpperCase()}>`));
          break;
        }
        applyTheme(argLower);
        print(L("ok", `PHOSPHOR SET TO ${argLower.toUpperCase()}`));
        break;
      }

      case "home":
      case "top":
        router.push("/");
        print(L("ok", "BOARD → TRENDING"));
        break;

      case "back":
        router.back();
        print(L("dim", "← BACK"));
        break;

      case "refresh":
      case "r":
        router.refresh();
        print(L("ok", "FEED REFRESHED"));
        break;

      case "clear":
        setLines([]);
        break;

      case "exit":
      case "close":
      case "quit":
        stopWatch();
        setOpen(false);
        break;

      case "whoami":
        print(L("out", "GUEST · READ-ONLY FEED · NOT FINANCIAL ADVICE"));
        break;

      default:
        print(L("err", `UNKNOWN COMMAND: ${cmd.toUpperCase()} — TYPE "HELP"`));
    }
  }

  function complete() {
    const parts = value.split(/\s+/);
    if (parts.length === 1) {
      const matches = COMMANDS.filter((c) =>
        c.startsWith(parts[0].toLowerCase()),
      );
      if (matches.length === 1) setValue(`${matches[0]} `);
      else if (matches.length > 1)
        print(L("dim", matches.join(" · ").toUpperCase()));
      return;
    }
    const pool =
      {
        tag: TAGS,
        go: TAGS,
        sort: SORT_KEYS,
        theme: THEMES,
        edge: ["arb", "mom"],
        scan: ["arb", "mom"],
        signal: ["arb", "mom"],
        watch: ["arb", "mom", "off"],
      }[parts[0].toLowerCase()] ?? [];
    const last = parts[parts.length - 1].toLowerCase();
    const matches = pool.filter((c) => c.startsWith(last));
    if (matches.length === 1)
      setValue([...parts.slice(0, -1), matches[0]].join(" "));
    else if (matches.length > 1)
      print(L("dim", matches.join(" · ").toUpperCase()));
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      run(value);
      setValue("");
    } else if (e.key === "Tab") {
      e.preventDefault();
      complete();
    } else if (e.key === "Escape") {
      setOpen(false);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (history.length === 0) return;
      const next =
        histIdx.current === -1
          ? history.length - 1
          : Math.max(0, histIdx.current - 1);
      histIdx.current = next;
      setValue(history[next]);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (histIdx.current === -1) return;
      const next = histIdx.current + 1;
      if (next >= history.length) {
        histIdx.current = -1;
        setValue("");
      } else {
        histIdx.current = next;
        setValue(history[next]);
      }
    }
  }

  const ctx = pathname.startsWith("/event/")
    ? `~${pathname}`
    : searchParams.get("q")
      ? `~/search/${searchParams.get("q")}`
      : `~/markets/${searchParams.get("tag") || "trending"}`;

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="Open command terminal (`)"
        className="sheen fixed bottom-3 right-3 z-50 flex items-center gap-1.5 overflow-hidden border border-edge bg-panel px-2.5 py-1.5 text-[10px] tracking-widest text-muted shadow-[0_4px_20px_-4px_rgba(0,0,0,0.7)] hover:border-accent/50 hover:text-accent"
      >
        <span className="text-accent">&gt;_</span> TERMINAL
        <kbd className="border border-edge bg-panel-raised px-1 text-muted/70">
          `
        </kbd>
      </button>
    );
  }

  return (
    <div className="terminal-slide fixed inset-x-0 bottom-0 z-50 border-t-2 border-accent/50 bg-panel/95 shadow-[0_-8px_40px_rgba(0,0,0,0.6)] backdrop-blur-sm">
      <div className="mx-auto flex max-w-7xl flex-col px-4">
        <div className="flex items-center justify-between border-b border-edge py-1.5 text-[10px] tracking-widest text-muted">
          <span className="flex items-center gap-1.5">
            <span className="live-dot inline-block h-1.5 w-1.5 rounded-full bg-accent" />
            PM/TERM — COMMAND INTERFACE
          </span>
          <span className="hidden sm:block">
            TAB COMPLETES · ↑↓ HISTORY · ESC CLOSES
          </span>
        </div>
        <div className="flex items-center gap-1.5 overflow-x-auto border-b border-edge py-1.5">
          <span className="shrink-0 select-none text-[10px] tracking-widest text-muted/50">
            RUN
          </span>
          {QUICK.map((q) => (
            <button
              key={q.cmd}
              type="button"
              title={q.title}
              onClick={() => {
                run(q.cmd);
                inputRef.current?.focus();
              }}
              className="lift shrink-0 border border-edge bg-panel-raised px-2 py-0.5 text-[10px] tracking-widest text-muted hover:border-accent/50 hover:text-accent"
            >
              {q.label}
            </button>
          ))}
        </div>
        <div
          ref={logRef}
          className="max-h-[38vh] min-h-36 overflow-y-auto whitespace-pre-wrap py-2 text-xs leading-5"
        >
          {lines.map((l) => (
            <div key={l.id} className={KIND_COLOR[l.kind]}>
              {l.text}
            </div>
          ))}
        </div>
        <div className="flex items-center gap-2 border-t border-edge py-2">
          <span className="shrink-0 text-xs text-accent">{ctx}&gt;</span>
          <input
            ref={inputRef}
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={onKeyDown}
            spellCheck={false}
            autoComplete="off"
            placeholder='TYPE "HELP"_'
            aria-label="Terminal command input"
            className="w-full bg-transparent text-xs uppercase tracking-wider text-foreground placeholder:text-muted/50 focus:outline-none"
          />
          <span className="cursor-blink shrink-0 text-accent">▊</span>
        </div>
      </div>
    </div>
  );
}
