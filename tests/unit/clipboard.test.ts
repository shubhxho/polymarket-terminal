import { afterEach, describe, expect, it } from "bun:test";
import { copyToClipboard } from "@/lib/clipboard";

type ToastCall = [string, ("info" | "warn" | "error")?];

const savedNavigator = globalThis.navigator;

function setClipboard(writeText: ((text: string) => Promise<void>) | null) {
  Object.defineProperty(globalThis, "navigator", {
    value: writeText ? { clipboard: { writeText } } : {},
    configurable: true,
    writable: true,
  });
}

function makeToast() {
  const calls: ToastCall[] = [];
  const toast = (message: string, tone?: "info" | "warn" | "error") => {
    calls.push([message, tone]);
  };
  return { calls, toast };
}

afterEach(() => {
  Object.defineProperty(globalThis, "navigator", {
    value: savedNavigator,
    configurable: true,
    writable: true,
  });
});

describe("copyToClipboard", () => {
  it("warns when the clipboard API is unavailable", () => {
    setClipboard(null);
    const { calls, toast } = makeToast();
    copyToClipboard("0xabc", toast);
    expect(calls).toEqual([["clipboard unavailable", "warn"]]);
  });

  it("toasts success with the default label on write", async () => {
    let written = "";
    setClipboard(async (t) => {
      written = t;
    });
    const { calls, toast } = makeToast();
    copyToClipboard("0xabc", toast);
    await new Promise((r) => setTimeout(r, 0));
    expect(written).toBe("0xabc");
    expect(calls).toEqual([["address copied", undefined]]);
  });

  it("uses a custom label in the success toast", async () => {
    setClipboard(async () => {});
    const { calls, toast } = makeToast();
    copyToClipboard("hash123", toast, "tx");
    await new Promise((r) => setTimeout(r, 0));
    expect(calls).toEqual([["tx copied", undefined]]);
  });

  it("reports an error toast when writeText rejects", async () => {
    setClipboard(async () => {
      throw new Error("denied");
    });
    const { calls, toast } = makeToast();
    copyToClipboard("hash123", toast, "tx");
    await new Promise((r) => setTimeout(r, 0));
    expect(calls).toEqual([["could not copy tx", "error"]]);
  });
});
