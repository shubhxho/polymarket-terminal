import { describe, expect, it } from "vitest";
import {
  cents,
  compact,
  dateShort,
  dirClass,
  num,
  pad,
  shortAddr,
  signed,
  truncate,
  usd,
} from "@/lib/format";

const EMPTY = "—";

describe("compact", () => {
  it("scales into K/M/B with tight precision", () => {
    expect(compact(1_234_567)).toBe("1.23M");
    expect(compact(2.5e9)).toBe("2.50B");
    expect(compact(1_500)).toBe("1.5K");
    expect(compact(999)).toBe("999");
    expect(compact(5)).toBe("5.00");
  });

  it("carries the sign and never emits $-", () => {
    expect(compact(-1_234_567)).toBe("-1.23M");
  });

  it("renders an em dash for absent or NaN input", () => {
    expect(compact(undefined)).toBe(EMPTY);
    expect(compact(Number.NaN)).toBe(EMPTY);
  });
});

describe("usd", () => {
  it("puts the sign outside the currency symbol", () => {
    expect(usd(0)).toBe("$0.00");
    expect(usd(-23)).toBe("-$23");
    expect(usd(1_500_000)).toBe("$1.50M");
  });
});

describe("cents", () => {
  it("renders a probability as a cent price", () => {
    expect(cents(0.4231)).toBe("42.3");
    expect(cents(0.5, 0)).toBe("50");
    expect(cents(undefined)).toBe(EMPTY);
  });
});

describe("signed", () => {
  it("prefixes + for gains, - for losses, a space for flat", () => {
    expect(signed(1.23)).toBe("+1.2");
    expect(signed(-1.23)).toBe("-1.2");
    expect(signed(0)).toBe(" 0.0");
  });
});

describe("truncate / pad", () => {
  it("truncates with an ellipsis only past the width", () => {
    expect(truncate("hello", 10)).toBe("hello");
    expect(truncate("hello", 3)).toBe("he…");
    expect(truncate("", 5)).toBe("");
  });

  it("pads to a fixed column width on either side", () => {
    expect(pad("ab", 5)).toBe("ab   ");
    expect(pad("ab", 5, "r")).toBe("   ab");
    expect(pad("abcdef", 3)).toBe("abc");
  });
});

describe("dirClass", () => {
  it("treats zero as neutral, not green", () => {
    expect(dirClass(0)).toBe("text-muted");
    expect(dirClass(5)).toBe("text-up");
    expect(dirClass(-5)).toBe("text-down");
    expect(dirClass(undefined)).toBe("text-muted");
  });
});

describe("num / shortAddr / dateShort", () => {
  it("formats with grouping and fixed fraction digits", () => {
    expect(num(1234.5)).toBe("1,234.50");
    expect(num(undefined)).toBe(EMPTY);
  });

  it("elides the middle of an address", () => {
    expect(shortAddr("0x1234567890abcdef")).toBe("0x1234..cdef");
    expect(shortAddr(undefined)).toBe(EMPTY);
  });

  it("renders a terminal-style UTC date", () => {
    expect(dateShort("2026-01-15T00:00:00Z")).toBe("15JAN26");
    expect(dateShort(undefined)).toBe(EMPTY);
    expect(dateShort("not-a-date")).toBe(EMPTY);
  });
});
