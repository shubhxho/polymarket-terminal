import { expect, test } from "@playwright/test";

/**
 * Per-tab smoke coverage.
 *
 * Every argument-free function code is reached through the command palette and
 * its screen must mount with the right masthead title. The title is client-side
 * chrome driven by the nav state, so this holds regardless of whether the live
 * Polymarket feeds resolve — true to the suite's rule of never asserting on live
 * data. Screens that need an argument (DES/SRCH/PORT/CAT) are covered elsewhere
 * with fixtures; this is the "does every tab open" guard.
 */
const TABS: { code: string; title: string }[] = [
  { code: "MON", title: "MARKET MONITOR" },
  { code: "SIG", title: "SIGNAL SCANNER" },
  { code: "MOV", title: "MOVERS" },
  { code: "WATCH", title: "WATCHLIST" },
  { code: "TAS", title: "TRADE & SALES" },
  { code: "ALRT", title: "ALERTS" },
  { code: "MESH", title: "SIGNAL MESH" },
  { code: "HELP", title: "HELP" },
];

test.describe("tabs open with the right masthead", () => {
  for (const { code, title } of TABS) {
    test(`${code} → ${title}`, async ({ page }) => {
      await page.goto("/");
      // Wait for hydration before firing the global ⌘K shortcut.
      await expect(page.getByRole("banner").getByText("MARKET MONITOR")).toBeVisible();

      await page.keyboard.press("ControlOrMeta+k");
      await expect(page.getByRole("dialog", { name: "Command palette" })).toBeVisible();
      await page.getByRole("combobox", { name: "Command line" }).fill(code);
      await page.keyboard.press("Enter");

      await expect(page.getByRole("banner").getByText(title, { exact: true })).toBeVisible();
    });
  }
});
