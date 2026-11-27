import { expect, test } from "@playwright/test";

test("spaces long overview prose and preserves paragraph breaks in a walkthrough", async ({ page }) => {
  const summary = "The worker checks one batch at a time so the amount of pending work stays predictable. ".repeat(35).trim();
  await page.route("**/api/explain", (route) => route.fulfill({ json: {
    title: "Process the queue", summary, concepts: [],
    walkthrough: [{ heading: "Pick one batch", explanation: "Read the pending jobs.\n\nProcess only the selected items.", source_ranges: [] }],
    assumptions: [], visuals: [], visual_note: "",
  } }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input" }).fill("runBatch(queue);");
  await page.getByRole("button", { name: "Help me understand" }).click();
  const paragraphs = page.locator("p.result-summary");
  await expect(paragraphs.first()).toBeVisible();
  expect(await paragraphs.count()).toBeGreaterThan(2);
  expect((await paragraphs.allTextContents()).join(" ")).toBe(summary);
  const first = (await paragraphs.first().boundingBox())!;
  const second = (await paragraphs.nth(1).boundingBox())!;
  expect(second.y - (first.y + first.height)).toBeGreaterThan(5);
  await page.getByRole("navigation", { name: "Explanation chapters" }).getByRole("button", { name: /Pick one batch/ }).click();
  await expect(page.locator("p.result-body")).toHaveText(["Read the pending jobs.", "Process only the selected items."]);
});
