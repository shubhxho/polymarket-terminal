import { expect, test } from "@playwright/test";

/**
 * The terminal shell: it must paint its chrome and default screen before any
 * market feed resolves, because the feed is remote and the frame is not allowed
 * to wait on it.
 */
test.describe("shell", () => {
  test("renders the masthead and the default Monitor screen", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Polymarket Terminal")).toBeVisible();
    // TopBar prints the active screen's title; MON is home. Scope to the
    // masthead — the screen body echoes the same title in a panel.
    await expect(page.getByRole("banner").getByText("MARKET MONITOR")).toBeVisible();
  });

  test("the theme toggle inverts the palette", async ({ page }) => {
    await page.goto("/");
    const html = page.locator("html");
    await expect(html).toHaveAttribute("data-theme", "light");
    await page.getByRole("button", { name: "Toggle theme" }).click();
    await expect(html).toHaveAttribute("data-theme", "dark");
    await page.getByRole("button", { name: "Toggle theme" }).click();
    await expect(html).toHaveAttribute("data-theme", "light");
  });

  test("exposes a wallet control, offering install when Phantom is absent", async ({ page }) => {
    await page.goto("/");
    // Exact — a live market row like "BUY Connecticut…" otherwise substring-matches.
    const connect = page.getByRole("button", { name: "Connect", exact: true });
    await expect(connect).toBeVisible();
    await expect(connect).toHaveAttribute("title", /Phantom not detected/i);
  });
});
