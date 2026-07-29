import { describe, expect, it } from "bun:test";
import { clock, relTime, timeOfDay, timeToExpiry } from "@/lib/format";

const EMPTY = "—";
const nowSec = () => Math.floor(Date.now() / 1000);

describe("clock / timeOfDay", () => {
  it("renders a UTC wall clock as HH:MM:SS", () => {
    expect(clock(new Date(0))).toBe("00:00:00");
    expect(clock(new Date(Date.UTC(2026, 0, 1, 14, 32, 7)))).toBe("14:32:07");
  });

  it("timeOfDay reads unix seconds as a UTC clock", () => {
    expect(timeOfDay(0)).toBe("00:00:00");
    expect(timeOfDay(3661)).toBe("01:01:01");
  });
});

describe("relTime", () => {
  it("buckets an age into s / m / h / d", () => {
    expect(relTime(nowSec() - 5)).toMatch(/^\d+s$/);
    expect(relTime(nowSec() - 90)).toBe("1m");
    expect(relTime(nowSec() - 2 * 3600)).toBe("2h");
    expect(relTime(nowSec() - 3 * 86400)).toBe("3d");
  });

  it("never goes negative for a future stamp", () => {
    expect(relTime(nowSec() + 100)).toBe("0s");
  });
});

describe("timeToExpiry", () => {
  it("is EMPTY for a missing or unparseable date", () => {
    expect(timeToExpiry(undefined)).toBe(EMPTY);
    expect(timeToExpiry("not-a-date")).toBe(EMPTY);
  });

  it("reads EXPIRED once the end is in the past", () => {
    expect(timeToExpiry(new Date(Date.now() - 1000).toISOString())).toBe("EXPIRED");
  });

  it("shows days+hours beyond a day out", () => {
    const iso = new Date(Date.now() + 3 * 86400_000 + 4 * 3600_000).toISOString();
    expect(timeToExpiry(iso)).toMatch(/^3d \d\dh$/);
  });

  it("shows a HH:MM:SS countdown inside a day", () => {
    const iso = new Date(Date.now() + 2 * 3600_000).toISOString();
    expect(timeToExpiry(iso)).toMatch(/^\d\d:\d\d:\d\d$/);
  });
});
