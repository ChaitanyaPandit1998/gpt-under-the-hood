import { expect, test } from "@playwright/test";
import type { PullRequestPlan } from "../src/domain/pull-request-plan";

for (const mode of ["explanation", "pull request"] as const) {
  test(`${mode}: keeps long chapter outlines accessible while reading`, async ({ page }, testInfo) => {
    const steps = Array.from({ length: 25 }, (_, i) => ({
      heading: `Batching step ${i + 1}`, explanation: "Read the next group.", source_ranges: [],
    }));
    const pr = mode === "pull request";
    const summary = "The reader processes items in batches and preserves their original order. ".repeat(145);
    const plan: PullRequestPlan = {
      title: "Batching study", url: "https://github.com/acme/demo/pull/7", headSha: "a".repeat(40),
      totalFiles: 0, filesComplete: true, files: [], warnings: [],
      chapters: steps.map((step, i) => ({ id: `batch-${i}`, title: step.heading, files: [], sourceText: "Process a batch." })),
    };
    await page.route("**/api/pull-request-plan", (route) => route.fulfill({ json: { plan } }));
    await page.route("**/api/explain", (route) => route.fulfill({ json: {
      title: "Batching explanation", summary, concepts: [], walkthrough: pr ? [] : steps, assumptions: [], visuals: [], visual_note: "",
    } }));

    await page.goto("/");
    if (pr) {
      await page.getByRole("tab", { name: "Pull request" }).click();
      await page.getByRole("textbox", { name: "GitHub pull request URL" }).fill(plan.url);
      await page.getByRole("button", { name: "Load PR", exact: true }).click();
      await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
    } else {
      await page.getByRole("textbox", { name: "Code input" }).fill("processBatch(items);");
      await page.getByRole("button", { name: "Help me understand" }).click();
    }
    await expect(page.getByRole("heading", { name: "Batching explanation", exact: true })).toBeVisible();
    const outline = page.getByRole("navigation", { name: pr ? "PR chapter outline" : "Explanation chapters", exact: true });
    const list = outline.getByRole("list");
    await expect(outline.locator('[aria-current="step"]')).toHaveText(pr ? /Batching step 1$/ : /Overview$/);
    expect(await list.evaluate((node) => node.scrollHeight > node.clientHeight)).toBe(true);
    await outline.evaluate((node) => window.scrollBy(0, node.getBoundingClientRect().top + 175));
    if (testInfo.project.name === "desktop") {
      await expect.poll(async () => (await outline.boundingBox())?.y).toBe(12);
    } else {
      expect(await outline.evaluate((node) => getComputedStyle(node).position)).toBe("static");
      expect((await list.boundingBox())!.height).toBeLessThanOrEqual(180);
    }

    const last = outline.getByRole("button", { name: /Batching step 25$/ });
    await last.click();
    await expect(last).toHaveAttribute("aria-current", "step");
    await expect(page.getByRole("heading", { name: "Batching step 25", exact: true }).first()).toBeFocused();
    const listBox = (await list.boundingBox())!;
    const itemBox = (await last.boundingBox())!;
    expect(itemBox.y).toBeGreaterThanOrEqual(listBox.y - 1);
    expect(itemBox.y + itemBox.height).toBeLessThanOrEqual(listBox.y + listBox.height + 1);
  });
}
