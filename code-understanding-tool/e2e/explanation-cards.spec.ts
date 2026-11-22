import { expect, test } from "@playwright/test";

test("uses optional audience and depth cards without generating until requested", async ({ page }) => {
  const requests: Record<string, unknown>[] = [];
  await page.route("**/api/explain", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({ json: { title: "Card result", summary: "A summary.", concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "" } });
  });
  await page.route("**/api/document-export", (route) => route.fulfill({ json: { available: false } }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input", exact: true }).fill("const value = 42;");
  const panel = page.getByRole("region", { name: "Explanation preferences", exact: true });
  await expect(panel.getByRole("radio", { name: "Beginner", exact: true })).toBeChecked();
  await expect(page.getByRole("button", { name: "Help me understand", exact: false })).toBeEnabled();
  await panel.getByRole("radio", { name: "Intermediate", exact: true }).check();
  await panel.getByRole("button", { name: "2. Depth: Step-by-step", exact: true }).click();
  await panel.getByRole("radio", { name: "Deep dive", exact: true }).check();
  await panel.getByRole("button", { name: "1. Audience: Intermediate", exact: true }).click();
  await expect(panel.getByRole("radio", { name: "Intermediate", exact: true })).toBeChecked();
  await expect(panel.getByRole("button", { name: "2. Depth: Deep dive", exact: true })).toBeVisible();
  expect(requests).toHaveLength(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
  await page.getByRole("button", { name: "Help me understand", exact: false }).click();
  await expect(page.getByRole("heading", { name: "Card result", exact: true })).toBeVisible();
  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({ learner_level: "intermediate", explanation_depth: "detailed" });
});
