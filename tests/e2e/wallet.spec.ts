import { expect, test, type Page } from "@playwright/test";

/**
 * Phantom wallet connect, driven against a stubbed EIP-1193 provider.
 *
 * A real wallet can't be automated, so we inject the same surface Phantom
 * injects — `window.phantom.ethereum`, `isPhantom`, `eth_requestAccounts` — and
 * assert the terminal's own reaction to it: the address in the masthead and the
 * jump into that wallet's portfolio. The stub answers on Polygon (0x89), the
 * chain Polymarket settles on.
 */
const ADDRESS = "0x1111111111111111111111111111111111111111";

async function injectPhantom(page: Page) {
  await page.addInitScript((address) => {
    const provider = {
      isPhantom: true,
      request: async ({ method }: { method: string }) => {
        if (method === "eth_requestAccounts" || method === "eth_accounts") return [address];
        if (method === "eth_chainId") return "0x89";
        return null;
      },
      on: () => {},
      removeListener: () => {},
    };
    (window as unknown as { phantom: unknown }).phantom = { ethereum: provider };
  }, ADDRESS);
}

test.describe("wallet", () => {
  test("connects and surfaces the address in the masthead", async ({ page }) => {
    await injectPhantom(page);
    await page.goto("/");

    await page.getByRole("button", { name: "Connect", exact: true }).click();

    // 0x1111…1111 — the short form the pill renders once connected. Scope to
    // the masthead so the connect toast's copy doesn't also match.
    await expect(page.getByRole("banner").getByText(/0x1111…1111/)).toBeVisible();
  });

  test("jumps from the wallet menu into its own portfolio", async ({ page }) => {
    await injectPhantom(page);
    await page.goto("/");

    await page.getByRole("button", { name: "Connect", exact: true }).click();
    // Open the popover on the connected pill, then view the book.
    await page
      .getByRole("banner")
      .getByText(/0x1111…1111/)
      .click();
    await page.getByRole("button", { name: "View my portfolio" }).click();

    // The portfolio header shows the full address and the "You" ownership badge.
    await expect(page.getByText(ADDRESS)).toBeVisible();
    await expect(page.getByText("You", { exact: true })).toBeVisible();
  });

  test("typing PORT in the palette resolves to the connected wallet", async ({ page }) => {
    await injectPhantom(page);
    await page.goto("/");
    await page.getByRole("button", { name: "Connect", exact: true }).click();
    await expect(page.getByRole("banner").getByText(/0x1111…1111/)).toBeVisible();

    // Bare "PORT" would normally error for want of an address; a connected
    // wallet supplies it, so the palette opens that wallet's book instead.
    await page.keyboard.press("ControlOrMeta+k");
    await page.getByRole("textbox", { name: "Command line" }).fill("PORT");
    await page.keyboard.press("Enter");

    await expect(page.getByText(ADDRESS)).toBeVisible();
    await expect(page.getByText("You", { exact: true })).toBeVisible();
  });
});
