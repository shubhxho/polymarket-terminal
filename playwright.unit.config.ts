import { defineConfig } from "@playwright/test";

/**
 * Unit config for the pure kernels (`src/lib/quant.ts`, `src/lib/derivatives.ts`).
 *
 * Deliberately separate from the E2E config: these tests import functions
 * directly and touch no browser and no network, so they must not pay for a dev
 * server boot. Keeping them in their own config is what lets `test:unit` run in
 * well under a second, which is the only way a math kernel actually gets
 * re-tested while it is being edited.
 */
export default defineConfig({
  testDir: "./tests/unit",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? "github" : "list",
});
