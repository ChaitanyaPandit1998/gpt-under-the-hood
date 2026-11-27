import { expect, test, type Page } from "@playwright/test";
import type { PullRequestPlan } from "../src/domain/pull-request-plan";

const headSha = "a".repeat(40);
const plan: PullRequestPlan = {
  title: "Add request validation", url: "https://github.com/acme/demo/pull/7", headSha,
  totalFiles: 2, filesComplete: true,
  files: [
    { path: "src/app.ts", status: "modified", additions: 1, deletions: 1, coverage: "included", page: 1 },
    { path: "src/large.ts", status: "modified", additions: 200, deletions: 10, coverage: "omitted_budget", page: 1 },
  ],
  warnings: ["1 file excluded from the initial evidence budget."],
  chapters: [
    { id: "overview", title: "Overview and change inventory", files: [], sourceText: "PR overview\nInventory only\nNo implementation evidence" },
    { id: "chapter-1", title: "Implementation: src", files: ["src/app.ts"], sourceText: "Implementation evidence\n+validate(request);\n+return success;" },
  ],
};

async function workspace(page: Page) {
  const state = {
    generations: [] as Record<string, unknown>[], evidence: [] as Record<string, unknown>[], imports: 0, visualRequests: 0, stories: 0,
    covers: [] as Record<string, unknown>[], holdCover: false, releaseCover: () => {},
    holdGeneration: false, holdReview: false, holdEvidence: false, evidenceError: "",
    releaseGeneration: () => {}, releaseReview: () => {}, releaseEvidence: () => {},
  };
  // No GitHub or model requests leave the browser during these regression tests.
  await page.route("**/api/**", async (route) => {
    const endpoint = new URL(route.request().url()).pathname;
    if (endpoint === "/api/document-export") return route.fulfill({ json: { available: false } });
    if (endpoint === "/api/pull-request-plan") { state.imports += 1; return route.fulfill({ json: { plan } }); }
    if (endpoint === "/api/explain-progress") return route.fulfill({ json: { phase: "drafting" } });
    if (endpoint === "/api/chapter-story") {
      state.stories++;
      return route.fulfill({ json: { story: {
        title: "A checkpoint in the patch", setup: "A clerk receives a form.", takeaway: "Read the patch for actual behavior.",
        scenes: [{ heading: "The checkpoint", narrative: "The clerk checks the form.", meaning: "The change validates the request.", source_ranges: [{ start_line: 2, end_line: 2 }] }],
      } } });
    }
    if (endpoint === "/api/visual-brief") {
      state.visualRequests++;
      return route.fulfill({ json: { brief: { disposition: "render", visualType: "comparison", takeaway: "Compare two cases.",
        composition: "Two labels.", caption: "Chapter illustration caption.", altText: "Chapter illustration.", guardrails: [],
        content: [{ label: "Example", role: "illustrative", statement: "Compare cases.", chapterLines: [] }] } } });
    }
    if (endpoint === "/api/visual-plan") {
      state.visualRequests++;
      return route.fulfill({ json: { scene: { background: "#ffffff", items: [{ kind: "text", text: "Example", x: 30, y: 70, color: "#123456", size: 24, align: "left" }] } } });
    }
    if (endpoint === "/api/visual-render") {
      state.visualRequests++;
      return route.fulfill({ contentType: "image/png", body: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=", "base64") });
    }
    if (endpoint === "/api/chapter-cover") {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      state.covers.push(body);
      if (state.holdCover) await new Promise<void>((resolve) => { state.releaseCover = resolve; });
      await route.fulfill({ json: { theme: body.theme, model: "test-image-model", latencyMs: 10, dataUrl: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=" } }).catch(() => {});
      return;
    }
    if (endpoint === "/api/explain") {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      state.generations.push(body);
      const number = state.generations.length;
      if (state.holdGeneration) await new Promise<void>((resolve) => { state.releaseGeneration = resolve; });
      if (body.review_mode === "review" && state.holdReview) await new Promise<void>((resolve) => { state.releaseReview = resolve; });
      await route.fulfill({ json: {
        title: `Explanation ${number}`, summary: "This chapter explains only the supplied source.",
        concepts: body.review_mode === "review" ? [] : [{ term: "Validation", explanation: "Checks the request.", source_ranges: [{ start_line: 2, end_line: 3 }] }],
        walkthrough: [], assumptions: [], visuals: [], visual_note: "",
        quality_review_status: body.review_mode === "draft" ? "unreviewed" : "passed",
        quality_review_note: body.review_mode === "draft" ? "Awaiting review." : "Checked against chapter evidence.",
      } }).catch(() => {}); // Cancellation intentionally closes pending requests.
      return;
    }
    if (endpoint === "/api/pull-request-evidence") {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      state.evidence.push(body);
      if (state.holdEvidence) await new Promise<void>((resolve) => { state.releaseEvidence = resolve; });
      if (state.evidenceError) return route.fulfill({ status: 400, json: { error: state.evidenceError } });
      const offset = Number(body.offset);
      await route.fulfill({ json: {
        filename: "src/large.ts", offset, endOffset: offset + 12,
        nextOffset: offset === 0 ? 12 : null, totalCharacters: 24,
        chapter: { id: `evidence:src/large.ts:${offset}`, title: `Large file section ${offset === 0 ? 1 : 2}`, files: ["src/large.ts"], sourceText: `PR chapter section\n+part_${offset};\n+validation;` },
      } }).catch(() => {});
      return;
    }
    await route.fulfill({ status: 501, json: { error: `Unexpected API endpoint: ${endpoint}` } });
  });
  await page.goto("/");
  await page.getByRole("tab", { name: "Pull request" }).click();
  await page.getByRole("textbox", { name: "GitHub pull request URL" }).fill(plan.url);
  await page.getByRole("button", { name: "Load PR", exact: true }).click();
  await expect(page.getByRole("region", { name: "Pull request study plan" })).toBeVisible();
  return state;
}

test("imports an outline without generation and navigates its boundaries", async ({ page }) => {
  const state = await workspace(page);
  await expect(page.getByText("2 / 2 files listed", { exact: false })).toBeVisible();
  await expect(page.getByRole("button", { name: "Previous chapter", exact: true })).toBeDisabled();
  expect(state.generations).toHaveLength(0);
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Implementation: src", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Next chapter", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Previous chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Overview and change inventory", exact: true })).toBeVisible();
  expect(state.generations).toHaveLength(0);
});

test("resumes a cached PR chapter variant with its story and illustration", async ({ page }) => {
  const state = await workspace(page);
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("navigation", { name: "Explanation chapters", exact: true })).toBeVisible();
  const storyButton = page.getByRole("button", { name: "Story", exact: true });
  await storyButton.click();
  const story = page.getByRole("article", { name: "A checkpoint in the patch" });
  await expect(story).toBeVisible();
  expect(state.visualRequests).toBe(0);
  await story.getByText("Illustrate this scene", { exact: true }).click();
  await story.getByRole("button", { name: "Factual diagram", exact: true }).click();
  expect(state.visualRequests).toBe(0);
  await story.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await story.getByRole("button", { name: "Render image", exact: true }).click();
  const image = story.getByRole("figure", { name: "Factual diagram: generated schematic" }).getByRole("img");
  await expect(image).toBeVisible();
  await page.getByRole("button", { name: "Previous chapter", exact: true }).click();
  await page.getByRole("button", { name: "Save PR study locally", exact: true }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Resume saved study", exact: true }).click();
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  await expect(storyButton).toHaveAttribute("aria-pressed", "true");
  await expect(image).toHaveAccessibleName("Chapter illustration.");
  expect(state.generations).toHaveLength(1);
  expect(state.stories).toBe(1);
  expect(state.visualRequests).toBe(3);
});

test("saves, resumes and exports a pinned PR with cached variants and patch evidence", async ({ page }, testInfo) => {
  const state = await workspace(page);
  const pdfButton = page.getByRole("button", { name: "PR study PDF preview", exact: true });
  await expect(pdfButton).toBeDisabled();
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  const chapterTools = page.getByRole("group", { name: "PR chapter tools", exact: true });
  await expect(chapterTools.getByText("This PR chapter", { exact: true })).toBeVisible();
  await chapterTools.getByRole("link", { name: "Refine PR chapter", exact: true }).click();
  await expect(page.getByRole("group", { name: "PR chapter preferences", exact: true })).toBeFocused();
  await page.getByRole("button", { name: "Simplify", exact: true }).click();
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 2", exact: true })).toBeVisible();
  await page.getByText("Design a visual", { exact: false }).click();
  await page.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await page.getByRole("button", { name: "Render image", exact: true }).click();
  await expect(page.getByRole("img", { name: "Chapter illustration.", exact: true })).toBeVisible();
  const manifest = page.getByText("File manifest and evidence coverage", { exact: true });
  const load = page.getByRole("button", { name: "Load patch section for src/large.ts", exact: true });
  await manifest.click();
  await load.click();
  await expect(page.getByRole("heading", { name: "Large file section 1", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 3", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Save PR study locally", exact: true }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Resume saved study", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Large file section 1", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Explanation 3", exact: true })).toBeVisible();
  await expect(page.getByText(`Snapshot: ${headSha}.`, { exact: false })).toBeVisible();
  await pdfButton.click();
  const pdf = page.getByRole("dialog", { name: "PDF preview" });
  const paper = pdf.getByRole("article", { name: "Selected PDF content" });
  await expect(paper.getByText("Review: unreviewed.", { exact: true }).first()).toBeVisible();
  const overview = pdf.getByRole("checkbox", { name: "Overview and change inventory", exact: true });
  const evidence = pdf.getByRole("checkbox", { name: "Large file section 1", exact: true });
  const illustration = pdf.getByRole("checkbox", { name: "Overview and change inventory illustrations", exact: true });
  const source = pdf.getByRole("checkbox", { name: "Include source appendices for selected chapters", exact: true });
  await expect(overview).toBeChecked();
  await expect(evidence).toBeChecked();
  await expect(pdf.getByRole("checkbox", { name: "Implementation: src", exact: true })).toBeDisabled();
  await expect(paper.getByText(headSha, { exact: false })).toBeVisible();
  await expect(paper.getByText(plan.warnings[0], { exact: true })).toBeVisible();
  await expect(paper.getByText("Explanation 2", { exact: true })).toBeVisible();
  await expect(paper.getByText("Explanation 3", { exact: true })).toBeVisible();
  await expect(paper.getByText("Explanation 1", { exact: true })).toHaveCount(0);
  await expect(paper.getByText("Chapter source lines: 2–3.", { exact: true })).toHaveCount(2);
  await expect(paper.getByRole("heading", { name: "Chapter source appendix" })).toHaveCount(0);
  await expect(paper.getByRole("img", { name: "Chapter illustration.", exact: true })).toBeVisible();
  await illustration.uncheck();
  await expect(paper.getByRole("img")).toHaveCount(0);
  await illustration.check();
  await evidence.uncheck();
  await expect(paper.getByText("Explanation 3", { exact: true })).toHaveCount(0);
  await overview.uncheck();
  await expect(pdf.getByRole("button", { name: "Print / Save as PDF" })).toBeDisabled();
  await overview.check();
  await evidence.check();
  await source.check();
  await expect(paper.getByRole("heading", { name: "Chapter source appendix" })).toHaveCount(2);
  await source.uncheck();
  if (testInfo.project.name === "desktop") {
    const path = testInfo.outputPath("selected-pr-study.pdf");
    await page.pdf({ path, preferCSSPageSize: true });
    await testInfo.attach("selected PR study PDF", { path, contentType: "application/pdf" });
  }
  await pdf.getByRole("button", { name: "Close PDF preview" }).click();
  expect(state.visualRequests).toBe(3);
  await page.getByRole("button", { name: /Overview and change inventory/ }).click();
  await expect(page.getByRole("heading", { name: "Explanation 2", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Go deeper", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  expect(state.generations).toHaveLength(3);
  expect(state.imports).toBe(1);
  expect(state.evidence).toHaveLength(1);
  await manifest.click();
  await load.click();
  await expect(page.getByRole("heading", { name: "Large file section 2", exact: true })).toBeVisible();
  expect(state.evidence[1]).toMatchObject({ headSha, offset: 12 });
});

test("defaults to text and restores both saved variants when diagrams are opted in", async ({ page }) => {
  const state = await workspace(page);
  const diagrams = page.getByRole("checkbox", { name: "Include diagrams" });
  const generate = page.getByRole("button", { name: "Generate this chapter", exact: true });
  const next = page.getByRole("button", { name: "Next chapter", exact: true });
  const previous = page.getByRole("button", { name: "Previous chapter", exact: true });
  await next.click();
  await expect(diagrams).not.toBeChecked();
  await generate.click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  await diagrams.check();
  await expect(generate).toBeVisible();
  expect(state.generations).toHaveLength(1);
  await generate.click();
  await expect(page.getByRole("heading", { name: "Explanation 2", exact: true })).toBeVisible();
  expect(state.generations.map(request => request.visual_preference)).toEqual(["no", "auto"]);
  await previous.click();
  await expect(diagrams).not.toBeChecked();
  await next.click();
  await expect(diagrams).toBeChecked();
  await expect(page.getByRole("heading", { name: "Explanation 2", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Save PR study locally", exact: true }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Resume saved study", exact: true }).click();
  await expect(diagrams).toBeChecked();
  await expect(page.getByRole("heading", { name: "Explanation 2", exact: true })).toBeVisible();
  await diagrams.uncheck();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  expect(state.generations).toHaveLength(2);
});

test("shows an unreviewed draft before optional review and keeps it on cancellation", async ({ page }) => {
  const state = await workspace(page);
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  const draft = page.getByText("Unreviewed draft", { exact: true });
  await expect(draft).toBeVisible();
  expect(state.generations.map(request => request.review_mode)).toEqual(["draft"]);
  state.holdReview = true;
  const review = page.getByRole("button", { name: "Review this chapter", exact: true });
  await review.click();
  await expect.poll(() => state.generations[1]?.review_mode).toBe("review");
  expect(state.generations[1]?.draft).toMatchObject({ title: "Explanation 1", quality_review_status: "unreviewed" });
  await expect(draft).toBeVisible();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  state.holdReview = false;
  state.releaseReview();
  await expect(draft).toBeVisible();
  await page.getByRole("navigation", { name: "Explanation chapters", exact: true }).getByRole("button", { name: /Key ideas/ }).click();
  await review.click();
  await expect(page.getByText("Second-pass review · passed", { exact: true })).toBeVisible();
  await expect(draft).not.toBeVisible();
});

test("automatic review shows the draft before the review returns", async ({ page }) => {
  const state = await workspace(page);
  await page.getByRole("checkbox", { name: "Review automatically" }).check();
  state.holdReview = true;
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect.poll(() => state.generations.map(request => request.review_mode)).toEqual(["draft", "review"]);
  await expect(page.getByText("Unreviewed draft", { exact: true })).toBeVisible();
  state.releaseReview();
  await expect(page.getByText("Second-pass review · passed", { exact: true })).toBeVisible();
  await page.reload();
  await page.getByRole("tab", { name: "Pull request" }).click();
  await page.getByRole("textbox", { name: "GitHub pull request URL" }).fill(plan.url);
  await page.getByRole("button", { name: "Load PR", exact: true }).click();
  await expect(page.getByRole("checkbox", { name: "Review automatically" })).toBeChecked();
});

test("generates only selected source and restores cached results by preferences", async ({ page }) => {
  const state = await workspace(page);
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  expect(state.generations[0].source_text).toBe(plan.chapters[1].sourceText);
  await page.getByRole("button", { name: "Previous chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  expect(state.generations).toHaveLength(1);
  await page.getByText("Customize explanation", { exact: true }).click();
  await page.getByRole("combobox", { name: "Chapter depth" }).selectOption("overview");
  await expect(page.getByRole("button", { name: "Generate this chapter", exact: true })).toBeVisible();
  await page.getByRole("combobox", { name: "Chapter depth" }).selectOption("detailed");
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  await page.getByRole("navigation", { name: "Explanation chapters" }).getByRole("button", { name: /Key ideas/ }).click();
  const citation = page.getByRole("button", { name: "View source lines 2 through 3", exact: true });
  await citation.click();
  const evidence = page.getByRole("dialog", { name: "PR chapter source" });
  await expect(evidence.locator("mark code")).toHaveText(["+validate(request);", "+return success;"]);
  await expect(evidence.getByText(/not the original file/)).toBeVisible();
  await evidence.getByRole("button", { name: "Close source" }).click();
  await expect(citation).toBeFocused();
  expect(state.generations).toHaveLength(1);
});

test("cancels generation and ignores its late response after navigation", async ({ page }) => {
  const state = await workspace(page);
  state.holdGeneration = true;
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect.poll(() => state.generations.length).toBe(1);
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByRole("button", { name: "Generate this chapter", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  state.releaseGeneration();
  state.holdGeneration = false;
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 2", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toHaveCount(0);
});

test("keeps settings chapter-local and caches tone and focus variants", async ({ page }) => {
  const state = await workspace(page);
  await page.getByRole("button", { name: "Simplify", exact: true }).click();
  expect(state.generations).toHaveLength(0);
  await page.getByText("Customize explanation", { exact: true }).click();
  await page.getByRole("combobox", { name: "Chapter tone" }).selectOption("technical");
  await page.getByRole("combobox", { name: "Chapter focus" }).selectOption("behaviour");
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  expect(state.generations[0]).toMatchObject({ explanation_depth: "balanced", explanation_tone: "technical", explanation_focus: "behaviour" });
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Chapter depth" })).toHaveValue("detailed");
  await expect(page.getByRole("combobox", { name: "Chapter tone" })).toHaveValue("plain");
  await page.getByRole("button", { name: "Previous chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Chapter tone" })).toHaveValue("technical");
  await page.getByRole("combobox", { name: "Chapter focus" }).selectOption("example");
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 2", exact: true })).toBeVisible();
  await page.getByRole("combobox", { name: "Chapter focus" }).selectOption("behaviour");
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  expect(state.generations).toHaveLength(2);
  await expect(page.getByRole("complementary", { name: "PR coverage limits" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
});

test("loads excluded patches section by section and generates only requested evidence", async ({ page }) => {
  const state = await workspace(page);
  await page.getByText("File manifest and evidence coverage", { exact: true }).click();
  const load = page.getByRole("button", { name: "Load patch section for src/large.ts", exact: true });
  await load.click();
  await expect(page.getByRole("heading", { name: "Large file section 1", exact: true })).toBeVisible();
  expect(state.generations).toHaveLength(0);
  expect(state.evidence[0]).toMatchObject({ url: plan.url, headSha, filename: "src/large.ts", page: 1, offset: 0 });
  await expect(page.getByText("More sections remain", { exact: false })).toBeVisible();
  await load.click();
  await expect(page.getByRole("heading", { name: "Large file section 2", exact: true })).toBeVisible();
  expect(state.evidence[1].offset).toBe(12);
  await expect(load).toBeDisabled();
  await expect(page.getByText("All GitHub-returned patch sections retrieved", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  expect(state.generations[0].source_text).toContain("+part_12;");
  expect(state.generations[0].source_text).not.toContain("+part_0;");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
});

test("surfaces revision errors without adding or generating stale evidence", async ({ page }) => {
  const state = await workspace(page);
  state.evidenceError = "The PR revision changed. Reload the study plan.";
  await page.getByText("File manifest and evidence coverage", { exact: true }).click();
  await page.getByRole("button", { name: "Load patch section for src/large.ts", exact: true }).click();
  await expect(page.getByRole("alert")).toHaveText(state.evidenceError);
  await expect(page.getByRole("navigation", { name: "PR chapter outline" }).getByRole("button")).toHaveCount(2);
  expect(state.generations).toHaveLength(0);
});

test("cancels evidence loading and ignores late sections", async ({ page }) => {
  const state = await workspace(page);
  state.holdEvidence = true;
  await page.getByText("File manifest and evidence coverage", { exact: true }).click();
  await page.getByRole("button", { name: "Load patch section for src/large.ts", exact: true }).click();
  await expect.poll(() => state.evidence.length).toBe(1);
  await page.getByRole("button", { name: "Cancel patch loading", exact: true }).click();
  state.releaseEvidence();
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Implementation: src", exact: true })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "PR chapter outline" }).getByRole("button")).toHaveCount(2);
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("cover generation is opt-in, sends no code, and preserves behavioural explanation and citations", async ({ page }) => {
  const state = await workspace(page);
  const coverCaption = page.getByRole("region", { name: "Optional chapter cover" }).getByText("Conceptual illustration", { exact: true });
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  expect(state.covers).toHaveLength(0);
  await page.getByText("Optional chapter cover", { exact: true }).click();
  await expect(page.getByRole("button", { name: "Generate cover", exact: true })).toBeDisabled();
  await expect(page.getByText("Use the source-backed walkthrough", { exact: false })).toBeVisible();
  await page.getByRole("checkbox", { name: /I approve sending/ }).check();
  await page.getByRole("button", { name: "Generate cover", exact: true }).click();
  await expect(coverCaption).toBeVisible();
  expect(state.covers).toEqual([{ theme: "review", approved: true }]);
  expect(state.generations).toHaveLength(1);
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  await page.getByRole("navigation", { name: "Explanation chapters" }).getByRole("button", { name: /Key ideas/ }).click();
  await page.getByRole("button", { name: "View source lines 2 through 3", exact: true }).click();
  await page.getByRole("dialog", { name: "PR chapter source" }).getByRole("button", { name: "Close source" }).click();
  await page.getByRole("button", { name: "Previous chapter", exact: true }).click();
  await expect(coverCaption).toHaveCount(0);
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  await expect(coverCaption).toBeVisible();
  expect(state.covers).toHaveLength(1);
  await page.getByRole("button", { name: "Hide cover", exact: true }).click();
  await expect(page.getByRole("img", { name: /Decorative isometric illustration/ })).toHaveCount(0);
  await page.getByRole("button", { name: "Show cover", exact: true }).click();
  await expect(page.getByRole("img", { name: /Decorative isometric illustration/ })).toBeVisible();
});

test("cancels cover generation and ignores artwork arriving after chapter navigation", async ({ page }) => {
  const state = await workspace(page);
  state.holdCover = true;
  await page.getByText("Optional chapter cover", { exact: true }).click();
  await page.getByRole("checkbox", { name: /I approve sending/ }).check();
  await page.getByRole("button", { name: "Generate cover", exact: true }).click();
  await expect.poll(() => state.covers.length).toBe(1);
  await page.getByRole("button", { name: "Cancel cover generation", exact: true }).click();
  state.releaseCover();
  await page.getByRole("button", { name: "Next chapter", exact: true }).click();
  await expect(page.getByText("Conceptual illustration", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Previous chapter", exact: true }).click();
  await expect(page.getByText("Conceptual illustration", { exact: true })).toHaveCount(0);
  expect(state.generations).toHaveLength(0);
});
