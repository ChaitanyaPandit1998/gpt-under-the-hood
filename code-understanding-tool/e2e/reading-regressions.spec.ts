import { expect, test } from "@playwright/test";

const explanation = {
  title: "Read the configured value", summary: "The input sets a value.",
  concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "",
};

test("keeps visual drafts isolated when cached chapter variants have identical text", async ({ page }) => {
  await page.route("**/api/explain", (route) => route.fulfill({ json: explanation }));
  await page.route("**/api/visual-brief", (route) => route.fulfill({ json: { brief: {
    disposition: "render", visualType: "comparison", takeaway: "Read the value.", composition: "Use one label.",
    content: [{ label: "Value", role: "illustrative", statement: "Show the configured value.", chapterLines: [] }],
    guardrails: [], caption: "A configured value.", altText: "One label for the value.",
  } } }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input" }).fill("const value = 42;");
  await page.getByRole("button", { name: "Help me understand" }).click();
  await expect(page.getByRole("navigation", { name: "Explanation chapters", exact: true })).toBeVisible();
  const refine = page.getByRole("region", { name: "Customize Overview" });
  const variant = refine.getByRole("article", { name: "Adapted section" });
  await refine.getByText("Customize explanation", { exact: true }).click();
  const focus = refine.getByRole("combobox", { name: "Chapter focus" });
  for (const option of ["behaviour", "example"]) {
    await focus.selectOption(option);
    await refine.getByRole("button", { name: "Generate section variant", exact: true }).click();
    await expect(variant).toBeVisible();
  }

  const designer = variant.locator("details.visual-brief-panel");
  await designer.locator(":scope > summary").click();
  await designer.getByRole("button", { name: "Generate visual brief" }).click();
  await expect(designer.getByRole("region", { name: "Visual design brief" })).toBeVisible();
  await focus.selectOption("behaviour");
  await expect(variant).toBeVisible();
  await expect(designer.getByRole("region", { name: "Visual design brief", includeHidden: true })).toHaveCount(0);
  await designer.locator(":scope > summary").click();
  await expect(designer.getByRole("button", { name: "Generate visual brief", exact: true })).toBeVisible();
  await designer.getByRole("button", { name: "Generate visual brief" }).click();
  await expect(designer.getByRole("region", { name: "Visual design brief" })).toBeVisible();
  const regenerate = refine.getByRole("button", { name: "Regenerate section variant", exact: true });
  await regenerate.click();
  await expect(regenerate).toBeEnabled();
  await expect(designer.getByRole("region", { name: "Visual design brief", includeHidden: true })).toHaveCount(0);
});

test("opens citations at the highlight even after very long context lines", async ({ page }) => {
  await page.route("**/api/explain", (route) => route.fulfill({ json: {
    ...explanation,
    concepts: [{ term: "Value", explanation: "This is the configured value.", source_ranges: [{ start_line: 2, end_line: 2 }] }],
  } }));
  await page.goto("/");
  await page.getByRole("tab", { name: /Documentation/ }).click();
  await page.getByRole("textbox", { name: "Documentation input" }).fill(`${"Context text. ".repeat(300)}\nConfigured value: 42.\nAfter the citation.`);
  await page.getByRole("button", { name: "Help me understand" }).click();
  await page.getByRole("navigation", { name: "Explanation chapters", exact: true }).getByRole("button", { name: /Key ideas/ }).click();
  const citation = page.getByRole("button", { name: "View source lines 2 through 2", exact: true });
  await citation.click();
  const excerpt = page.getByRole("region", { name: "Source excerpt" });
  const highlight = excerpt.locator("mark");
  await expect(highlight).toContainText("Configured value: 42.");
  const visible = (await excerpt.boundingBox())!;
  const cited = (await highlight.boundingBox())!;
  expect(cited.y).toBeGreaterThanOrEqual(visible.y - 1);
  expect(cited.y + cited.height).toBeLessThanOrEqual(visible.y + visible.height + 1);
  await expect.poll(() => excerpt.evaluate((node) => node.scrollTop)).toBeGreaterThan(0);
  await page.keyboard.press("Escape");
  await expect(citation).toBeFocused();
});
