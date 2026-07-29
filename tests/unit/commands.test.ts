import { describe, expect, it } from "vitest";
import { defaultScreenFor, lookupCommand, parseCommand, screenTitle } from "@/lib/commands";

/** Narrowing helper — every happy-path case expects a screen, not an error. */
function screen(input: string) {
  const r = parseCommand(input);
  if (r.kind !== "screen") throw new Error(`expected a screen, got: ${r.message}`);
  return r.screen;
}

describe("parseCommand — codes and aliases", () => {
  it("resolves a bare code, case-insensitively", () => {
    expect(screen("MON")).toEqual({ fn: "MON" });
    expect(screen("mon")).toEqual({ fn: "MON" });
  });

  it("strips a trailing <GO> the way a Bloomberg line does", () => {
    expect(screen("MON <GO>")).toEqual({ fn: "MON" });
    expect(screen("MON GO")).toEqual({ fn: "MON" });
  });

  it("errors on an empty line", () => {
    const r = parseCommand("   ");
    expect(r.kind).toBe("error");
  });
});

describe("parseCommand — search", () => {
  it("takes the argument as the query", () => {
    expect(screen("SRCH fed cut")).toEqual({ fn: "SRCH", q: "fed cut" });
  });

  it("treats an unknown leading token as a full-line search", () => {
    expect(screen("banana futures")).toEqual({ fn: "SRCH", q: "banana futures" });
  });

  it("requires a query", () => {
    expect(parseCommand("SRCH").kind).toBe("error");
  });
});

describe("parseCommand — DES", () => {
  it("accepts a bare slug as an event", () => {
    expect(screen("DES us-recession-2026")).toEqual({
      fn: "DES",
      slug: "us-recession-2026",
      kind: "event",
    });
  });

  it("extracts slug and kind from a pasted polymarket URL", () => {
    expect(screen("DES https://polymarket.com/event/foo?tid=1")).toEqual({
      fn: "DES",
      slug: "foo",
      kind: "event",
    });
    expect(screen("DES https://polymarket.com/market/bar")).toEqual({
      fn: "DES",
      slug: "bar",
      kind: "market",
    });
  });
});

describe("parseCommand — PORT", () => {
  const addr = `0x${"a".repeat(40)}`;

  it("accepts a well-formed 0x address", () => {
    expect(screen(`PORT ${addr}`)).toEqual({ fn: "PORT", user: addr });
  });

  it("rejects anything that is not a 40-hex address", () => {
    expect(parseCommand("PORT 0x123").kind).toBe("error");
    expect(parseCommand("PORT vitalik.eth").kind).toBe("error");
  });
});

describe("parseCommand — CAT", () => {
  it("maps a sector name to its Gamma tag", () => {
    expect(screen("CAT crypto")).toEqual({ fn: "CAT", tag: "21", label: "Crypto" });
  });

  it("defaults to the first sector with no argument", () => {
    expect(screen("CAT")).toEqual({ fn: "CAT", tag: "2", label: "Politics" });
  });

  it("errors on an unknown sector", () => {
    expect(parseCommand("CAT atlantis").kind).toBe("error");
  });
});

describe("defaultScreenFor", () => {
  const addr = `0x${"a".repeat(40)}`;

  it("resolves PORT to the connected wallet's book", () => {
    expect(defaultScreenFor("PORT", { walletAddress: addr })).toEqual({
      fn: "PORT",
      user: addr,
    });
  });

  it("is null without a wallet, so callers fall back to prompting", () => {
    expect(defaultScreenFor("PORT", { walletAddress: null })).toBeNull();
    expect(defaultScreenFor("PORT", {})).toBeNull();
  });

  it("does not default codes that have no session-state source", () => {
    expect(defaultScreenFor("SRCH", { walletAddress: addr })).toBeNull();
    expect(defaultScreenFor("MON", { walletAddress: addr })).toBeNull();
  });
});

describe("lookupCommand / screenTitle", () => {
  it("resolves aliases to their spec", () => {
    expect(lookupCommand("M")?.code).toBe("MON");
    expect(lookupCommand("whale")?.code).toBe("TAS");
    expect(lookupCommand("nope")).toBeUndefined();
  });

  it("titles a screen with its argument", () => {
    expect(screenTitle({ fn: "SRCH", q: "fed" })).toContain('"fed"');
    expect(screenTitle({ fn: "MON" })).toBe("MARKET MONITOR");
  });
});
