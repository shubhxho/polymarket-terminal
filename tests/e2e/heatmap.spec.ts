import { expect, test } from "@playwright/test";

/**
 * The Monitor heatmap toggle. Drives the real component in Chromium — which has
 * WebGPU — so this exercises the GPU path when available and the Canvas2D
 * fallback otherwise; either way the canvas and its backend badge must appear.
 */
test.describe("market heatmap", () => {
  test("toggles the Monitor board to the GPU heatmap", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("banner").getByText("MARKET MONITOR")).toBeVisible();

    // Flip the board's Grid/Heat segmented control.
    await page.getByRole("button", { name: "Heat", exact: true }).click();

    const canvas = page.getByRole("img", { name: /Heatmap of \d+ markets/ });
    await expect(canvas).toBeVisible();
    // The badge reports which backend actually rendered (GPU or 2D).
    await expect(page.getByText(/^(GPU|2D) · \d+$/)).toBeVisible();
  });
});
