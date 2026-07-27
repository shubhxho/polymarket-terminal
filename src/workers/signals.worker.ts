/// <reference lib="webworker" />
import type { GammaEvent } from "@/lib/polymarket";
import { type ScanOptions, type Signal, scanSignals } from "@/lib/signals";

export interface ScanRequest {
  id: number;
  events: GammaEvent[];
  opts?: ScanOptions;
}

export interface ScanResponse {
  id: number;
  signals: Signal[];
  scanned: number;
  /** Compute time in ms — surfaced so the terminal can report kernel latency. */
  ms: number;
}

self.addEventListener("message", (e: MessageEvent<ScanRequest>) => {
  const { id, events, opts } = e.data;
  const started = performance.now();
  const signals = scanSignals(events, opts);
  const res: ScanResponse = {
    id,
    signals,
    scanned: events.length,
    ms: performance.now() - started,
  };
  (self as DedicatedWorkerGlobalScope).postMessage(res);
});
