import { describe, expect, it } from "vitest";
import { fuzzyMatch, highlight } from "@/lib/fuzzy";

describe("fuzzyMatch", () => {
  it("matches an empty query with a zero score", () => {
    expect(fuzzyMatch("", "anything")).toEqual({ score: 0, positions: [] });
  });

  it("returns null when the query is not a subsequence", () => {
    expect(fuzzyMatch("xyz", "abc")).toBeNull();
  });

  it("matches case-insensitively and reports matched positions", () => {
    const m = fuzzyMatch("mon", "MON");
    expect(m).not.toBeNull();
    expect(m?.positions).toEqual([0, 1, 2]);
  });

  it("ranks a tight acronym above a buried mid-token match", () => {
    const tight = fuzzyMatch("mon", "MON");
    const buried = fuzzyMatch("mon", "Market Monitor Overview");
    expect(tight).not.toBeNull();
    expect(buried).not.toBeNull();
    expect(tight!.score).toBeGreaterThan(buried!.score);
  });

  it("rewards word-boundary matches over interior ones", () => {
    const boundary = fuzzyMatch("fc", "Fed Cut");
    const interior = fuzzyMatch("fc", "Affect");
    expect(boundary!.score).toBeGreaterThan(interior!.score);
  });
});

describe("highlight", () => {
  it("splits text into alternating matched and unmatched runs", () => {
    expect(highlight("MON", [0, 2])).toEqual([
      { text: "M", hit: true },
      { text: "O", hit: false },
      { text: "N", hit: true },
    ]);
  });

  it("returns a single unmatched run when nothing hit", () => {
    expect(highlight("abc", [])).toEqual([{ text: "abc", hit: false }]);
  });
});
