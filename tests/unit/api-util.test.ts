import { describe, expect, it } from "vitest";
import { fail, limitOf, ok } from "@/lib/api-util";

describe("ok", () => {
  it("wraps data in a success envelope with a timestamp", async () => {
    const res = ok({ hi: 1 });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data).toEqual({ hi: 1 });
    expect(typeof body.ts).toBe("number");
    expect(body.ts).toBeGreaterThan(0);
  });

  it("defaults cache-control to s-maxage=5 with 6x stale-while-revalidate", () => {
    const cc = ok(null).headers.get("cache-control");
    expect(cc).toBe("public, s-maxage=5, stale-while-revalidate=30");
  });

  it("honors a custom sMaxAge and scales the swr window", () => {
    const cc = ok(null, 10).headers.get("cache-control");
    expect(cc).toBe("public, s-maxage=10, stale-while-revalidate=60");
  });

  it("preserves falsy payloads verbatim", async () => {
    const body = await ok(0).json();
    expect(body.data).toBe(0);
  });
});

describe("fail", () => {
  it("defaults to a 502 and lifts the message off an Error", async () => {
    const res = fail(new Error("upstream boom"));
    expect(res.status).toBe(502);
    const body = await res.json();
    expect(body.ok).toBe(false);
    expect(body.error).toBe("upstream boom");
    expect(typeof body.ts).toBe("number");
  });

  it("stringifies non-Error throwables", async () => {
    const body = await fail("plain string").json();
    expect(body.error).toBe("plain string");
  });

  it("honors a custom status", () => {
    expect(fail("bad", 400).status).toBe(400);
  });
});

describe("limitOf", () => {
  it("returns the fallback for null", () => {
    expect(limitOf(null, 20, 100)).toBe(20);
  });

  it("returns the fallback for non-numeric or non-positive input", () => {
    expect(limitOf("abc", 20, 100)).toBe(20);
    expect(limitOf("0", 20, 100)).toBe(20);
    expect(limitOf("-5", 20, 100)).toBe(20);
  });

  it("parses a valid limit", () => {
    expect(limitOf("42", 20, 100)).toBe(42);
  });

  it("clamps to max", () => {
    expect(limitOf("10000", 20, 100)).toBe(100);
  });

  it("parses leading integers like parseInt", () => {
    expect(limitOf("30rows", 20, 100)).toBe(30);
  });
});
