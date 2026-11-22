import { expect, test, type Page } from "@playwright/test";

const explanation = {
  title: "Read the configured value",
  summary: "This document describes a default setting.",
  concepts: [{ term: "Default", explanation: "Used when no value is supplied.", source_ranges: [{ start_line: 1, end_line: 1 }] }],
  walkthrough: Array.from({ length: 6 }, (_, index) => ({
    heading: `Step ${index + 1}`, explanation: "Check the default setting.", source_ranges: [{ start_line: 1, end_line: 1 }],
  })),
  assumptions: [], visuals: [], visual_note: "",
};
const source = "The default limit is 42.";
const recoveryLabel = "Keep source and completed explanation after refresh in this tab";
const title = (page: Page) => page.getByRole("heading", { name: explanation.title, exact: true });

async function explainDocument(page: Page) {
  let calls = 0;
  await page.route("**/api/explain", (route) => {
    calls += 1;
    return route.fulfill({ json: explanation });
  });
  await page.goto("/");
  await page.getByRole("tab", { name: /Documentation/ }).click();
  await page.getByRole("textbox", { name: "Documentation input" }).fill(source);
  await page.getByRole("button", { name: "Help me understand" }).click();
  await expect(title(page)).toBeVisible();
  return () => calls;
}

test("restores the source and active chapter in the same tab without generation", async ({ page, context }) => {
  const calls = await explainDocument(page);
  const outline = page.getByRole("navigation", { name: "Explanation chapters", exact: true });
  await outline.getByRole("button", { name: /Key ideas/ }).click();
  await expect(page.getByRole("heading", { name: "Key ideas", exact: true })).toBeVisible();

  await page.reload();
  await expect(title(page)).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Documentation input" })).toHaveValue(source);
  await expect(page.getByRole("heading", { name: "Key ideas", exact: true })).toBeVisible();
  expect(calls()).toBe(1);

  const otherTab = await context.newPage();
  await otherTab.goto("/");
  await expect(title(otherTab)).toHaveCount(0);
  await expect(otherTab.getByRole("button", { name: "Resume saved study", exact: true })).toHaveCount(0);
  await otherTab.close();

  await page.getByRole("button", { name: "Clear input", exact: true }).click();
  await expect(title(page)).toHaveCount(0);
  await page.reload();
  await expect(title(page)).toHaveCount(0);
  await expect(page.getByRole("textbox", { name: "Code input" })).toHaveValue("");
});

test("turning recovery off removes the temporary result without deleting a manual save", async ({ page }) => {
  const calls = await explainDocument(page);
  await page.getByRole("button", { name: "Save study locally", exact: true }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();

  const recovery = page.getByRole("checkbox", { name: recoveryLabel, exact: true });
  await recovery.uncheck();
  await expect(title(page)).toBeVisible();
  await page.reload();
  await expect(recovery).not.toBeChecked();
  await expect(title(page)).toHaveCount(0);
  await page.getByRole("button", { name: "Resume saved study", exact: true }).click();
  await expect(title(page)).toBeVisible();
  expect(calls()).toBe(1);
});
