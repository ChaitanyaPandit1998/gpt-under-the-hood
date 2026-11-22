import { expect, test } from "@playwright/test";

test("labels each reading scope and plays only the selected chapter", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "speechSynthesis", { configurable: true, value: {
      getVoices: () => [], addEventListener: () => {}, removeEventListener: () => {},
      speak: (utterance: SpeechSynthesisUtterance) => { document.body.dataset.spoken = utterance.text; },
      cancel: () => { delete document.body.dataset.spoken; },
    } });
  });
  await page.route("**/api/explain", (route) => route.fulfill({ json: {
    title: "Batching records", summary: "Start with a queue.", concepts: [],
    walkthrough: [{ heading: "Collect a group", explanation: "Read up to four records.", source_ranges: [] }],
    assumptions: [{ statement: "Ordering is unspecified.", reason: "The source does not define it." }],
    visuals: [], visual_note: "",
  } }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input" }).fill("collect(queue);");
  await page.getByRole("button", { name: "Help me understand" }).click();
  const whole = page.getByRole("group", { name: "Explanation tools", exact: true });
  await expect(whole.getByText("Whole explanation", { exact: true })).toBeVisible();
  await whole.getByRole("link", { name: "Refine explanation", exact: true }).click();
  await expect(page.getByRole("region", { name: "Explanation preferences" })).toBeFocused();
  await whole.getByText("Read aloud", { exact: true }).click();
  await whole.getByRole("button", { name: "Read explanation aloud", exact: true }).click();
  await expect(page.locator("body")).toHaveAttribute("data-spoken", /Start with a queue[\s\S]*Read up to four records/);

  await page.getByRole("button", { name: "Chapters", exact: true }).click();
  const chapter = page.getByRole("group", { name: "Chapter tools", exact: true });
  await expect(chapter.getByText("This chapter: Overview", { exact: true })).toBeVisible();
  await chapter.getByText("Read aloud", { exact: true }).click();
  await expect(page.locator("body")).toHaveAttribute("data-spoken", /Read up to four records/);
  await chapter.getByRole("button", { name: "Read chapter aloud", exact: true }).click();
  await expect(page.locator("body")).toHaveAttribute("data-spoken", "Batching records: Overview. Start with a queue.");
  await expect(whole.getByRole("button", { name: "Read explanation aloud", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.locator("body")).not.toHaveAttribute("data-spoken");
  await expect(chapter.getByText("This chapter: Collect a group", { exact: true })).toBeVisible();
});
