import { expect, test, type Page } from "@playwright/test";
import { openStudyActions } from "./support/study-actions";

const source = "const limit = 3;";
const explanation = {
  title: "A fixed limit", summary: "The direct overview sets a limit.", concepts: [], assumptions: [], visuals: [], visual_note: "",
  walkthrough: [{ heading: "Set the value", explanation: "The value is three.", source_ranges: [] }],
};
const story = {
  title: "The final ticket", setup: "A clerk is counting tickets.", takeaway: "The supply is fixed.",
  scenes: [{ heading: "Counting", narrative: "Three tickets sit on a counter.", meaning: "The constant is set to three.", source_ranges: [{ start_line: 1, end_line: 1 }] }],
};
const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=", "base64");

async function openChapters(page: Page) {
  await page.route("**/api/explain", (route) => route.fulfill({ json: explanation }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input" }).fill(source);
  await page.getByRole("button", { name: "Help me understand" }).click();
  await expect(page.getByRole("navigation", { name: "Explanation chapters", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Full explanation", exact: true })).toHaveCount(0);
}

test("generates only on request, links evidence, and reuses the same chapter story", async ({ page }) => {
  const bodies: unknown[] = [];
  await page.route("**/api/chapter-story", (route) => {
    bodies.push(route.request().postDataJSON());
    return route.fulfill({ json: { story } });
  });
  await openChapters(page);
  expect(bodies).toHaveLength(0);
  const direct = page.getByText(explanation.summary, { exact: true });
  await expect(direct).toBeVisible();
  await page.getByRole("button", { name: "Story", exact: true }).click();
  await expect(page.getByRole("heading", { name: story.title })).toBeVisible();
  await expect(direct).toBeHidden();
  expect(bodies[0]).toMatchObject({ source, mode: "code", depth: "balanced", learnerLevel: "beginner", context: expect.stringContaining(explanation.summary) });
  await page.getByRole("button", { name: "View source lines 1 through 1" }).click();
  await expect(page.getByRole("region", { name: "Source excerpt" }).locator("mark")).toContainText(source);
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Direct", exact: true }).click();
  await expect(direct).toBeVisible();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByRole("button", { name: "Previous", exact: true }).click();
  await page.getByRole("button", { name: "Story", exact: true }).click();
  await expect(page.getByRole("heading", { name: story.title })).toBeVisible();
  expect(bodies).toHaveLength(1);
});

test("keeps the direct chapter available after cancellation or a review failure", async ({ page }) => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => { release = resolve; });
  let requests = 0;
  await page.route("**/api/chapter-story", async (route) => {
    if (++requests === 1) {
      await held;
      await route.fulfill({ json: { story } }).catch(() => {});
    } else {
      await route.fulfill({ status: 502, json: { error: "The story did not pass review." } });
    }
  });
  await openChapters(page);
  const generate = page.getByRole("button", { name: "Story", exact: true });
  await generate.click();
  await expect.poll(() => requests).toBe(1);
  await page.getByRole("button", { name: "Next", exact: true }).click();
  release();
  await expect(page.getByRole("heading", { name: story.title })).toBeHidden();
  await page.getByRole("button", { name: "Previous", exact: true }).click();
  await generate.click();
  await expect(page.getByRole("alert").filter({ hasText: "did not pass review" })).toBeVisible();
  await expect(page.getByText(explanation.summary, { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Retry story" })).toBeVisible();
});

test("stops chapter read-aloud when the story replaces the direct chapter", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "speechSynthesis", { configurable: true, value: {
      getVoices: () => [], addEventListener: () => {}, removeEventListener: () => {},
      speak: () => { document.body.dataset.spoken = "chapter"; },
      cancel: () => { delete document.body.dataset.spoken; },
    } });
  });
  await page.route("**/api/chapter-story", (route) => route.fulfill({ json: { story } }));
  await openChapters(page);
  const chapterTools = page.getByRole("group", { name: "Chapter tools" });
  await chapterTools.locator("summary").filter({ hasText: /^Read aloud/ }).click();
  await chapterTools.getByRole("button", { name: "Read chapter aloud", exact: true }).click();
  await expect(page.locator("body")).toHaveAttribute("data-spoken", "chapter");
  await page.getByRole("button", { name: "Story", exact: true }).click();
  await expect(page.getByRole("heading", { name: story.title })).toBeVisible();
  await expect(page.locator("body")).not.toHaveAttribute("data-spoken");
});

test("saves accepted scene images separately and resumes without generating them again", async ({ page }) => {
  const briefs: Record<string, unknown>[] = [];
  const renders: Record<string, unknown>[] = [];
  let stories = 0;
  await page.route("**/api/chapter-story", (route) => {
    stories += 1;
    return route.fulfill({ json: { story: {
      ...story, scenes: [...story.scenes, { heading: "Another scene", narrative: "A second clerk walks past.", meaning: "There is no other action.", source_ranges: [] }],
    } } });
  });
  await page.route("**/api/visual-brief", (route) => {
    const body = route.request().postDataJSON();
    briefs.push(body);
    const caption = briefs.length === 3 ? "An unaccepted alternative." : body.purpose === "factual-diagram" ? "A factual limit schematic." : "A drawing of the fictional clerk.";
    return route.fulfill({ json: { brief: {
      disposition: "render", visualType: "concept-map", takeaway: caption, composition: caption,
      content: [{ label: "Limit", role: "background", statement: "The constant is set to three.", chapterLines: [] }],
      guardrails: [], caption, altText: caption,
    } } });
  });
  let plans = 0;
  await page.route("**/api/visual-plan", (route) => route.fulfill({ json: { scene: {
    background: "#ffffff", items: [{ kind: "text", text: "Limit", x: ++plans === 1 ? 999 : 40, y: 60, size: 24, color: "#202033", align: "left" }],
  } } }));
  await page.route("**/api/visual-render", (route) => {
    renders.push(route.request().postDataJSON());
    return route.fulfill({ contentType: "image/png", body: png });
  });
  await openChapters(page);
  await page.getByRole("button", { name: "Story", exact: true }).click();
  const scenes = page.getByRole("article", { name: story.title }).locator(".chapter-story-scenes > li");
  const first = scenes.nth(0);
  await first.getByText("Illustrate this scene", { exact: true }).click();
  const storyChoice = first.getByRole("button", { name: "Story illustration", exact: true });
  await storyChoice.click();
  expect(briefs).toHaveLength(0);
  const generate = first.getByRole("button", { name: "Generate visual brief", exact: true });
  const render = first.getByRole("button", { name: "Render image", exact: true });
  await generate.click();
  await first.getByRole("radio", { name: "Full HD", exact: true }).check();
  await render.click();
  await expect(first.getByRole("alert")).toContainText("may be clipped");
  expect(renders).toHaveLength(0);
  await first.getByRole("button", { name: "Retry drawing plan", exact: true }).click();
  const fictionalImage = first.getByRole("figure", { name: "Story illustration: fictional analogy" }).getByRole("img");
  await expect(fictionalImage).toHaveAccessibleName("A drawing of the fictional clerk.");
  await first.getByRole("button", { name: "Factual diagram", exact: true }).click();
  await generate.click();
  await render.click();
  const diagram = first.getByRole("figure", { name: "Factual diagram: generated schematic" }).getByRole("img");
  await expect(diagram).toHaveAccessibleName("A factual limit schematic.");
  await expect(fictionalImage).toBeVisible();
  await expect(scenes.nth(1).locator("figure")).toHaveCount(0);
  expect(briefs).toHaveLength(2);
  expect(briefs[0]).toMatchObject({ purpose: "story-illustration", preferences: { form: "conceptual-illustration", style: "expressive" } });
  expect(briefs[0].context).toContain(story.scenes[0].narrative);
  expect(briefs[1]).toMatchObject({ purpose: "factual-diagram", preferences: { style: "technical", includeSource: false } });
  expect(briefs[1]).not.toHaveProperty("source");
  expect(briefs[1].context).toContain(story.scenes[0].meaning);
  expect(briefs[1].context).not.toContain(story.scenes[0].narrative);
  expect(renders.map((item) => item.resolution)).toEqual(["full-hd", "standard"]);
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByRole("button", { name: "Previous", exact: true }).click();
  await page.getByRole("button", { name: "Story", exact: true }).click();
  await expect(fictionalImage).toBeVisible();
  await expect(diagram).toBeVisible();
  expect(renders).toHaveLength(2);

  await first.getByText("Illustrate this scene", { exact: true }).click();
  await storyChoice.click();
  await generate.click();
  await render.click();
  const candidate = first.getByRole("group", { name: "Compare illustrations" }).getByRole("img", { name: "An unaccepted alternative." });
  await expect(candidate).toBeVisible();
  await expect(page.getByRole("button", { name: "Resume saved study", exact: true })).toHaveCount(0);
  await (await openStudyActions(page)).getByRole("button", { name: "Save study locally", exact: true }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  await page.getByRole("checkbox", { name: "Keep source and completed explanation after refresh in this tab" }).uncheck();
  await page.reload();
  await page.getByRole("button", { name: "Resume saved study", exact: true }).click();
  await expect(page.getByRole("button", { name: "Story", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(fictionalImage).toHaveAccessibleName("A drawing of the fictional clerk.");
  await expect(diagram).toHaveAccessibleName("A factual limit schematic.");
  await expect(candidate).toHaveCount(0);
  expect(stories).toBe(1);
  expect(renders).toHaveLength(3);
});

test("does not publish an image that finishes after leaving its scene", async ({ page }) => {
  await page.route("**/api/chapter-story", (route) => route.fulfill({ json: { story } }));
  await page.route("**/api/visual-brief", (route) => route.fulfill({ json: { brief: {
    disposition: "render", visualType: "conceptual-illustration", takeaway: "A fixed supply.", composition: "Draw tickets.",
    content: [{ label: "Tickets", role: "illustrative", statement: "Three tickets.", chapterLines: [] }],
    guardrails: [], caption: "A fictional supply.", altText: "Three tickets.",
  } } }));
  await page.route("**/api/visual-plan", (route) => route.fulfill({ json: { scene: {
    background: "#ffffff", items: [{ kind: "text", text: "Tickets", x: 40, y: 60, size: 24, color: "#202033", align: "left" }],
  } } }));
  let finish!: () => void;
  const hold = new Promise<void>((resolve) => { finish = resolve; });
  await page.route("**/api/visual-render", async (route) => {
    await hold;
    await route.fulfill({ contentType: "image/png", body: png }).catch(() => {});
  });
  await openChapters(page);
  await page.getByRole("button", { name: "Story", exact: true }).click();
  await page.getByText("Illustrate this scene", { exact: true }).click();
  await page.getByRole("button", { name: "Story illustration", exact: true }).click();
  await page.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  const requestStarted = page.waitForRequest("**/api/visual-render");
  await page.getByRole("button", { name: "Render image", exact: true }).click();
  await requestStarted;
  await page.getByRole("button", { name: "Direct", exact: true }).click();
  finish();
  await page.getByRole("button", { name: "Story", exact: true }).click();
  await expect(page.getByRole("heading", { name: story.title })).toBeVisible();
  await expect(page.getByRole("figure", { name: "Story illustration: fictional analogy" })).toHaveCount(0);
});
