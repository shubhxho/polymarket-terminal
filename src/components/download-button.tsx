"use client";

import { useRef, useState } from "react";

interface Step {
  text: string;
  amber?: boolean;
}

const LOADING_STEPS: Step[] = [
  { text: "CONNECTING TO GAMMA API_" },
  { text: "PULLING MARKET FEED_" },
  { text: "ENCODING TO CSV_" },
  { text: "CRAFTING PAYLOAD_", amber: true },
  { text: "PACKAGED." },
];

// Financial advice cycles through these messages in sequence
const FINANCIAL_CYCLE = [
  "!! DISCLAIMER: NOT. FINANCIAL. ADVICE. !!",
  "!! PAST PERFORMANCE ≠ FUTURE RETURNS !!",
  "!! THESE ARE PROBABILITIES. NOT PROPHECIES. !!",
  "!! SEEK A LICENSED PROFESSIONAL. IMMEDIATELY. !!",
];

type Variant = "done" | "warn" | "error";

interface EggEntry {
  msg: string;
  variant: Variant;
  download: boolean;
  cycle?: boolean; // cycle through FINANCIAL_CYCLE instead
}

const EGGS: EggEntry[] = [
  { msg: "WAIT — YOU ALREADY HAVE THIS FILE", variant: "done", download: true },
  { msg: "CSV HOARDING PROTOCOL ENGAGED", variant: "done", download: true },
  {
    msg: "⚠ COMPULSIVE DOWNLOADING DETECTED ⚠",
    variant: "warn",
    download: false,
  },
  { msg: "", variant: "error", download: false, cycle: true },
  {
    msg: "CONSULT AN ACTUAL LICENSED PROFESSIONAL",
    variant: "error",
    download: false,
  },
  {
    msg: "THIS IS A PREDICTION TERMINAL NOT THERAPY",
    variant: "error",
    download: false,
  },
];

type Status = "idle" | "loading" | "done" | "warn" | "error";

export function DownloadButton({
  tag,
  query,
}: {
  tag?: string;
  query?: string;
}) {
  const [status, setStatus] = useState<Status>("idle");
  const [label, setLabel] = useState("");
  const [stepAmber, setStepAmber] = useState(false);
  const [count, setCount] = useState(0);

  const abortRef = useRef(false);
  const cycleRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cycleIdx = useRef(0);

  function clearCycle() {
    if (cycleRef.current) {
      clearInterval(cycleRef.current);
      cycleRef.current = null;
    }
  }

  function startCycle(onDone: () => void) {
    cycleIdx.current = 0;
    setLabel(FINANCIAL_CYCLE[0]);
    cycleRef.current = setInterval(() => {
      cycleIdx.current++;
      if (cycleIdx.current >= FINANCIAL_CYCLE.length) {
        clearCycle();
        onDone();
        return;
      }
      setLabel(FINANCIAL_CYCLE[cycleIdx.current]);
    }, 800);
  }

  async function handleClick() {
    if (status === "loading") return;

    const next = count + 1;
    setCount(next);
    abortRef.current = false;

    const egg = next >= 2 ? EGGS[Math.min(next - 2, EGGS.length - 1)] : null;

    if (egg) {
      clearCycle();

      if (egg.cycle) {
        setStatus("error");
        startCycle(() => {
          setStatus("idle");
          setLabel("");
        });
        return;
      }

      setStatus(egg.variant);
      setLabel(egg.msg);
      const dur =
        egg.variant === "error" ? 2800 : egg.variant === "warn" ? 2400 : 2200;
      setTimeout(() => {
        setStatus("idle");
        setLabel("");
      }, dur);

      if (!egg.download) return;
    }

    setStatus("loading");
    setLabel(LOADING_STEPS[0].text);
    setStepAmber(false);

    let stepIndex = 1;
    const stepTimer = setInterval(() => {
      if (abortRef.current || stepIndex >= LOADING_STEPS.length - 1) {
        clearInterval(stepTimer);
        return;
      }
      const step = LOADING_STEPS[stepIndex++];
      setLabel(step.text);
      setStepAmber(!!step.amber);
    }, 350);

    try {
      const params = new URLSearchParams();
      if (tag) params.set("tag", tag);
      if (query) params.set("q", query);

      const res = await fetch(`/api/export?${params}`);
      clearInterval(stepTimer);
      if (!res.ok) throw new Error("export failed");

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `polymarket-${query ? `search-${query.slice(0, 20)}` : tag || "trending"}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      if (!abortRef.current) {
        setStepAmber(false);
        setStatus("done");
        setLabel(`EXPORT #${next} — DATA SECURED`);
        setTimeout(() => {
          setStatus("idle");
          setLabel("");
        }, 2600);
      }
    } catch {
      clearInterval(stepTimer);
      if (!abortRef.current) {
        setStepAmber(false);
        setStatus("error");
        setLabel("EXPORT FAILED — RETRY");
        setTimeout(() => {
          setStatus("idle");
          setLabel("");
        }, 2200);
      }
    }
  }

  const borderColor =
    status === "loading"
      ? stepAmber
        ? "border-amber/40"
        : "border-accent/30"
      : status === "done"
        ? "border-accent"
        : status === "warn"
          ? "border-amber"
          : status === "error"
            ? "border-red"
            : "border-edge hover:border-edge-bright";

  const textColor =
    status === "loading"
      ? stepAmber
        ? "text-amber/70"
        : "text-accent/50"
      : status === "done"
        ? "text-accent"
        : status === "warn"
          ? "text-amber"
          : status === "error"
            ? "text-red"
            : "text-muted hover:text-accent";

  const cursor = status === "loading" ? "cursor-not-allowed" : "cursor-pointer";

  const icon =
    status === "loading" ? (
      <span
        className={`cursor-blink ${stepAmber ? "text-amber" : "text-accent"}`}
      >
        ▊
      </span>
    ) : status === "done" ? (
      <span>✓</span>
    ) : status === "warn" ? (
      <span>▲</span>
    ) : status === "error" ? (
      <span>!!</span>
    ) : (
      <span>↓</span>
    );

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={status === "loading"}
      title="Export current view as CSV"
      className={`flex items-center gap-1.5 border px-3 py-1.5 text-xs tracking-wider transition-colors bg-panel ${borderColor} ${textColor} ${cursor}`}
    >
      {icon}
      <span>{status === "idle" ? "EXPORT CSV" : label}</span>
    </button>
  );
}
