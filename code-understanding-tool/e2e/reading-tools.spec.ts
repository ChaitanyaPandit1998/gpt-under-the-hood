import { expect, test } from "@playwright/test";
import { openStudyActions, studyActions } from "./support/study-actions";

test("jumps from a long explanation title to the study toolbar without opening tools", async ({ page }) => {
  let generations = 0;
  await page.route("**/api/explain", (route) => {
    generations += 1;
    return route.fulfill({ json: {
      title: "Processing batches", summary: "A batch keeps the worker's memory use predictable. ".repeat(130),
      concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "",
    } });
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input" }).fill("processBatch(queue);");
  await page.getByRole("button", { name: "Help me understand" }).click();
  const title = page.getByRole("heading", { name: "Processing batches", exact: true });
  await title.focus();
  await page.keyboard.press("Tab");
  const jump = page.getByRole("link", { name: "Study tools", exact: true });
  await expect(jump).toBeFocused();
  const toolbar = page.getByRole("group", { name: "Explanation tools", exact: true });
  await expect(toolbar).not.toBeInViewport();
  await page.keyboard.press("Enter");
  await expect(toolbar).toBeFocused();
  await expect(toolbar).toBeInViewport();
  await expect(studyActions(page)).not.toHaveAttribute("open", "");
  await page.keyboard.press("Tab");
  await expect(toolbar.getByRole("link", { name: "Refine explanation", exact: true })).toBeFocused();
  expect(generations).toBe(1);
});

test("collapses idle tools without losing drafts and keeps active visual work open", async ({ page }) => {
  await page.route("**/api/explain", (route) => route.fulfill({ json: {
    title: "A configured value", summary: "The value is fixed.", concepts: [], walkthrough: [],
    assumptions: [], visuals: [], visual_note: "", quality_review_status: "passed", quality_review_note: "No blocking findings.",
  } }));
  await page.route("**/api/document-export", (route) => route.fulfill({ json: { available: true } }));
  let release!: () => void;
  const held = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/api/visual-brief", async (route) => {
    await held;
    await route.fulfill({ json: { brief: {
      disposition: "text-only", visualType: "none", takeaway: "The value is fixed.",
      composition: "Text is sufficient.", content: [], guardrails: [], caption: "A fixed value.", altText: "A fixed value.",
    } } }).catch(() => {});
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input" }).fill("const value = 3;");
  await page.getByRole("button", { name: "Help me understand" }).click();
  const tools = page.getByRole("group", { name: "Explanation tools", exact: true });
  const visual = tools.locator(":scope > details.visual-brief-panel");
  const speech = tools.locator(":scope > details.speech-tools");
  const actions = studyActions(page);
  const studyTrigger = actions.locator(":scope > summary");
  await expect(page.getByText("No blocking findings.", { exact: true })).not.toBeVisible();
  await expect(visual).not.toHaveAttribute("open", "");
  await expect(visual.locator(":scope > summary .tool-active-indicator")).not.toBeVisible();
  await expect(studyTrigger).toHaveAccessibleName("Save & export");
  await expect(actions.getByRole("button", { name: "Save study locally" })).not.toBeVisible();
  await expect(actions.getByRole("button", { name: "PDF preview" })).not.toBeVisible();
  await studyTrigger.press("Enter");
  await expect(actions.getByRole("button", { name: "Save study locally" })).toBeVisible();
  await studyTrigger.press("Enter");
  await expect(actions).not.toHaveAttribute("open", "");
  await visual.locator(":scope > summary").click();
  await visual.getByText("Customize visual", { exact: false }).click();
  const question = visual.getByRole("textbox", { name: "What should the visual explain?", exact: false });
  await question.fill("Show the configured value.");
  await speech.locator(":scope > summary").click();
  await expect(visual).not.toHaveAttribute("open", "");
  await visual.locator(":scope > summary").click();
  await expect(question).toHaveValue("Show the configured value.");
  await expect(speech).not.toHaveAttribute("open", "");
  await visual.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await expect(visual).toHaveAttribute("data-tool-active", "true");
  await openStudyActions(page);
  await expect(actions.getByRole("button", { name: "PDF preview" })).toBeVisible();
  await expect(visual).toHaveAttribute("open", "");
  await expect(visual.getByRole("button", { name: "Cancel visual brief", exact: true })).toBeVisible();
  await visual.locator(":scope > summary").click();
  await expect(visual).toHaveAttribute("data-tool-active", "true");
  await expect(visual.locator(":scope > summary .tool-active-indicator")).toBeVisible();
  await visual.locator(":scope > summary").click();
  await visual.getByRole("button", { name: "Cancel visual brief", exact: true }).click();
  await expect(visual).toHaveAttribute("data-tool-active", "false");
  release();
});

test("labels each reading scope and plays only the selected chapter", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "speechSynthesis", { configurable: true, value: {
      getVoices: () => [], addEventListener: () => {}, removeEventListener: () => {},
      speak: (utterance: SpeechSynthesisUtterance) => { document.body.dataset.spoken = utterance.text; },
      cancel: () => { delete document.body.dataset.spoken; },
    } });
  });
  await page.route("**/api/explain", (route) => route.fulfill({ json: {
    title: "Batching records", summary: "Start with a **queue**.", concepts: [],
    walkthrough: [{ heading: "Collect a group", explanation: "Read up to four records.", source_ranges: [] }],
    assumptions: [{ statement: "Ordering is unspecified.", reason: "The source does not define it." }],
    visuals: [], visual_note: "",
  } }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input" }).fill("collect(queue);");
  await page.getByRole("button", { name: "Help me understand" }).click();
  const whole = page.getByRole("group", { name: "Explanation tools", exact: true });
  await expect(whole.getByText("Whole explanation", { exact: true })).toBeVisible();
  const chapter = page.getByRole("group", { name: "Chapter tools", exact: true });
  const summary = page.getByText("Start with a queue.", { exact: true });
  const mode = page.getByRole("group", { name: "Chapter explanation style", exact: true });
  await expect(summary).toBeVisible();
  expect((await mode.boundingBox())!.y).toBeLessThan((await summary.boundingBox())!.y);
  expect((await summary.boundingBox())!.y).toBeLessThan((await chapter.boundingBox())!.y);
  expect((await chapter.boundingBox())!.y).toBeLessThan((await whole.boundingBox())!.y);
  await whole.getByRole("link", { name: "Refine explanation", exact: true }).click();
  await expect(page.getByRole("region", { name: "Explanation preferences" })).toBeFocused();
  await whole.locator("summary").filter({ hasText: /^Read aloud/ }).click();
  await whole.getByRole("button", { name: "Read explanation aloud", exact: true }).click();
  await expect(page.locator("body")).toHaveAttribute("data-spoken", /Start with a queue[\s\S]*Read up to four records/);

  await expect(page.getByRole("navigation", { name: "Explanation chapters", exact: true })).toBeVisible();
  await expect(chapter.getByText("This chapter: Overview", { exact: true })).toBeVisible();
  await chapter.locator("summary").filter({ hasText: /^Read aloud/ }).click();
  await expect(page.locator("body")).toHaveAttribute("data-spoken", /Read up to four records/);
  await chapter.getByRole("button", { name: "Read chapter aloud", exact: true }).click();
  await expect(page.locator("body")).toHaveAttribute("data-spoken", "Batching records: Overview. Start with a queue.");
  await expect(whole.getByRole("button", { name: "Read explanation aloud", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.locator("body")).not.toHaveAttribute("data-spoken");
  await expect(chapter.getByText("This chapter: Collect a group", { exact: true })).toBeVisible();
});
