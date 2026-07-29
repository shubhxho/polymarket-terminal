import { expect, test } from "@playwright/test";

/**
 * Navigation is entirely client-side, so it is both the safest and the most
 * valuable thing e2e can assert: the command palette, the function-code
 * parser, the sidebar and the back stack all have to agree on which screen is
 * live.
 */
test.describe("navigation", () => {
  test("the ⌘K palette runs a function code", async ({ page }) => {
    await page.goto("/");
    // Wait for hydration before a global keydown, or the shortcut is lost.
    await expect(page.getByRole("banner").getByText("MARKET MONITOR")).toBeVisible();
    await page.keyboard.press("ControlOrMeta+k");

    const palette = page.getByRole("dialog", { name: "Command palette" });
    await expect(palette).toBeVisible();

    await page.getByRole("textbox", { name: "Command line" }).fill("HELP");
    await page.keyboard.press("Enter");

    await expect(page.getByText("Function Codes")).toBeVisible();
    await expect(page.getByText("Keyboard", { exact: true })).toBeVisible();
  });

  test("the sidebar navigates and the back button unwinds it", async ({ page }) => {
    await page.goto("/");
    // Sidebar rows carry their F-key in the accessible name ("Movers F4").
    const nav = page.getByRole("navigation");
    const banner = page.getByRole("banner");

    await nav.getByRole("button", { name: /Movers/ }).click();
    await expect(banner.getByText("MOVERS")).toBeVisible();

    await nav.getByRole("button", { name: /Monitor/ }).click();
    await expect(banner.getByText("MARKET MONITOR")).toBeVisible();

    // Alt+← walks back through this tab's history to the Movers screen.
    await page.keyboard.press("Alt+ArrowLeft");
    await expect(banner.getByText("MOVERS")).toBeVisible();
  });

  test("typing a bare market name falls through to search", async ({ page }) => {
    await page.goto("/");
    // Wait for hydration — the type-anywhere keydown listener mounts in an
    // effect, and a key pressed before then is simply lost.
    await expect(page.getByRole("banner").getByText("MARKET MONITOR")).toBeVisible();
    // Type-anywhere seeds the palette with the first key.
    await page.keyboard.press("f");
    const input = page.getByRole("textbox", { name: "Command line" });
    await expect(input).toBeVisible();
    await input.fill("SRCH bitcoin");
    await page.keyboard.press("Enter");
    // The masthead title echoes the query for the SEARCH screen.
    await expect(page.getByRole("banner").getByText(/SEARCH · "bitcoin"/i)).toBeVisible();
  });
});
