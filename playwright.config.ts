import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end layer. Drives the real terminal in a real Chromium against a
 * production build, so what the suite asserts is what a user gets — routing,
 * keyboard shortcuts, the command line, the wallet control and theming.
 *
 * The tests deliberately assert on shell behaviour, never on live Polymarket
 * data: the upstream feed is not ours to make deterministic, so binding
 * assertions to it would trade real coverage for flakes. The HELP screen and
 * navigation are fully client-side, which is exactly where e2e earns its keep.
 */
const PORT = 3100;
const BASE_URL = `http://localhost:${PORT}`;

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  // A production build, not `next dev`: it's the artifact that ships, and it
  // starts clean without the dev overlay intercepting keystrokes.
  webServer: {
    command: `npm run build && npm run start -- --port ${PORT}`,
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
