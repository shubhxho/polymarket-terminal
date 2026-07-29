import { describe, expect, it } from "bun:test";
import { cn } from "@/lib/cn";

describe("cn", () => {
  it("joins truthy classes and drops falsy ones", () => {
    expect(cn("a", false, null, undefined, "b")).toBe("a b");
  });

  it("flattens conditional objects and arrays", () => {
    expect(cn("base", { active: true, hidden: false }, ["x", "y"])).toBe("base active x y");
  });

  it("lets a later Tailwind utility win a conflict instead of keeping both", () => {
    // The reason twMerge is here: without it both land in the DOM.
    expect(cn("text-muted", "text-accent")).toBe("text-accent");
    expect(cn("px-2 py-1", "px-4")).toBe("py-1 px-4");
  });
});
