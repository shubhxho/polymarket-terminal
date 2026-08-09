import { expect, test } from "@playwright/test";

const FIXTURE_PATH = "/dev/price-chart";

// Substrings React uses for the non-unique-key warning across versions.
const KEY_WARNING_RE = /same key|unique .*key|Encountered two children/i;

/**
 * Collect every console message + uncaught page error for a test so we can
 * assert React emitted no warnings while rendering the chart.
 */
function captureConsole(page: import("@playwright/test").Page) {
  const messages: string[] = [];
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      messages.push(msg.text());
    }
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return { messages, errors };
}

test.describe("PriceChart", () => {
  test("renders duplicate-label series without a React key warning", async ({ page }) => {
    const { messages, errors } = captureConsole(page);

    await page.goto(FIXTURE_PATH);
    await expect(page.getByTestId("fixture-heading")).toBeVisible();
    // Chart is fully painted once the SVG is present.
    await expect(page.locator("svg[aria-label='Price history chart']")).toBeVisible();

    const keyWarnings = messages.filter((m) => KEY_WARNING_RE.test(m));
    expect(keyWarnings, `console:\n${messages.join("\n")}`).toEqual([]);
    expect(errors, "uncaught page errors").toEqual([]);
  });

  test("draws one line + legend entry per series", async ({ page }) => {
    await page.goto(FIXTURE_PATH);

    const svg = page.locator("svg[aria-label='Price history chart']");
    await expect(svg).toBeVisible();

    // Each legend row carries a change badge ending in "pp" — one per series.
    await expect(page.getByText(/pp$/)).toHaveCount(3);

    // One stroked line-path (fill=none) per series inside a glow group.
    await expect(svg.locator("path[stroke][fill='none']")).toHaveCount(3);
  });

  test("hover shows a crosshair and per-series markers", async ({ page }) => {
    await page.goto(FIXTURE_PATH);
    const svg = page.locator("svg[aria-label='Price history chart']");
    await expect(svg).toBeVisible();

    const box = await svg.boundingBox();
    if (!box) throw new Error("svg has no bounding box");
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);

    // Dashed crosshair line appears on hover.
    await expect(svg.locator("line[stroke-dasharray='2 3']")).toHaveCount(1);
    // One hover marker circle per series (stroke=var(--panel)).
    await expect(svg.locator("circle[stroke='var(--panel)']")).toHaveCount(3);
  });
});
