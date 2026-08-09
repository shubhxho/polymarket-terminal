import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * Vitest is the voidzero test runner, standardised on alongside oxlint + oxfmt.
 *
 * The unit suite is pure logic — the signal engine, the model port, formatting
 * and quant helpers — so it runs in the plain `node` environment with no DOM
 * shim. The `@/` alias mirrors the one in tsconfig so tests import exactly the
 * way the app does. Playwright owns the browser-level e2e specs under
 * `tests/e2e` and is excluded here so the two runners never collide.
 */
export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
    exclude: ["tests/e2e/**", "node_modules/**", "automaton/**"],
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
});
