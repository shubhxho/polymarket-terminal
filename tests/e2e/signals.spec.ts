import { expect, test } from "@playwright/test";

/**
 * The signal scanner, driven deterministically.
 *
 * True to the suite's rule of never asserting on live Polymarket data, this
 * stubs `/api/signals` with a fixed payload and checks that the screen renders
 * what the engine produced — the ranked table, the blended model column and the
 * header — rather than whatever the feed happens to be doing. The live socket
 * overlay is additive and never connects here, so the poll fixture is exactly
 * what paints. It is the one place e2e can prove the model actually reaches the
 * screen end to end.
 */

const PAYLOAD = {
  ok: true,
  ts: 0,
  data: {
    arbs: [],
    drifts: [],
    markets: [
      {
        market: {
          id: "mkt-e2e-alpha",
          slug: "e2e-alpha",
          conditionId: "cond-alpha",
          question: "E2E ALPHA MARKET — will the fixture render?",
          eventTitle: "Fixture Event",
          last: 0.42,
          chg24h: 6,
          volume24h: 120_000,
          outcomes: [{ label: "Yes", price: 0.42, tokenId: "tok-alpha" }],
          endDate: "2030-01-01T00:00:00Z",
        },
        signals: [
          {
            kind: "MOMENTUM",
            direction: "bullish",
            strength: 71,
            confidence: 0.8,
            headline: "trend +0.9σ",
            detail: "Persistent upward drift relative to its own volatility.",
          },
        ],
        bias: 55,
        heat: 71,
        conviction: 62,
        model: { prob: 0.71, direction: "bullish", conviction: 0.42, auc: 0.6502 },
        recent: [0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.415, 0.42],
        stats: {
          realisedVol: 3.1,
          driftPerDay: 0.9,
          trendQuality: 0.29,
          autocorrelation: 0.1,
          bandZ: 1.2,
          volCompression: 0.8,
          book: {
            imbalance: 0.1,
            microLean: 0.2,
            costToMoveOneCent: 1200,
            liquidityScore: 0.6,
            bidNotional: 8000,
            askNotional: 7000,
          },
          volumeZ: 1.5,
          moveZ: 1.1,
        },
      },
    ],
    stats: {
      scanned: 180,
      flagged: 1,
      bullish: 1,
      bearish: 0,
      deepScanned: 1,
      byKind: { MOMENTUM: 1 },
      blockNotional: 250_000,
      modeled: 1,
      modelConfirms: 1,
      modelConflicts: 0,
    },
  },
};

test.describe("signal scanner", () => {
  test("renders the ranked table and the blended model column from a fixed scan", async ({
    page,
  }) => {
    await page.route("**/api/signals**", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(PAYLOAD) })
    );

    await page.goto("/");
    await expect(page.getByRole("banner").getByText("MARKET MONITOR")).toBeVisible();

    // Reach the scanner through the command palette, the same path a user takes.
    await page.keyboard.press("ControlOrMeta+k");
    await page.getByRole("combobox", { name: "Command line" }).fill("SIG");
    await page.keyboard.press("Enter");

    await expect(page.getByRole("banner").getByText("SIGNAL SCANNER")).toBeVisible();

    // The ranked table, its Model column, and the market from the fixture.
    await expect(page.getByText("Ranked signals")).toBeVisible();
    await expect(page.getByText("Model", { exact: true })).toBeVisible();
    await expect(page.getByText("E2E ALPHA MARKET", { exact: false })).toBeVisible();

    // The model's up-probability (0.71 → 71%) is rendered in the model column.
    await expect(page.getByText("71%").first()).toBeVisible();
  });

  test("selecting the market opens its detail rail with the model panel", async ({ page }) => {
    await page.route("**/api/signals**", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(PAYLOAD) })
    );

    await page.goto("/");
    await page.keyboard.press("ControlOrMeta+k");
    await page.getByRole("combobox", { name: "Command line" }).fill("SIG");
    await page.keyboard.press("Enter");

    await page.getByText("E2E ALPHA MARKET", { exact: false }).click();

    // The detail rail's Model panel states the out-of-sample AUC caveat.
    await expect(page.getByText("AUC", { exact: false }).first()).toBeVisible();
  });
});
