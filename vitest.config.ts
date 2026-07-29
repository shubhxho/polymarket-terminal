import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * Unit layer. These exercise the pure `lib/` logic — formatters, the quant
 * primitives, the fuzzy ranker, the command parser — in a plain Node runtime
 * with no DOM. Anything that touches the browser or the network is an e2e
 * concern and lives under `tests/e2e` with Playwright instead, so this suite
 * stays instant and deterministic.
 */
export default defineConfig({
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
    globals: false,
  },
});
