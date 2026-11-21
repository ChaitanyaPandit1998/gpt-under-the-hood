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
    generations: [] as Record<string, unknown>[], evidence: [] as Record<string, unknown>[], imports: 0,
    covers: [] as Record<string, unknown>[], holdCover: false, releaseCover: () => {},
    holdGeneration: false, holdEvidence: false, evidenceError: "",
    releaseGeneration: () => {}, releaseEvidence: () => {},
  };
  // No GitHub or model requests leave the browser during these regression tests.
  await page.route("**/api/**", async (route) => {
    const endpoint = new URL(route.request().url()).pathname;
    if (endpoint === "/api/document-export") return route.fulfill({ json: { available: false } });
    if (endpoint === "/api/pull-request-plan") { state.imports += 1; return route.fulfill({ json: { plan } }); }
    if (endpoint === "/api/explain-progress") return route.fulfill({ json: { phase: "drafting" } });
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
      await route.fulfill({ json: {
        title: `Explanation ${number}`, summary: "This chapter explains only the supplied source.",
        concepts: [{ term: "Validation", explanation: "Checks the request.", source_ranges: [{ start_line: 2, end_line: 3 }] }],
        walkthrough: [], assumptions: [], visuals: [], visual_note: "",
        quality_review_status: "passed", quality_review_note: "Checked against chapter evidence.",
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

test("saves and resumes a pinned PR with cached variants and patch evidence", async ({ page }) => {
  const state = await workspace(page);
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 1", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Simplify", exact: true }).click();
  await page.getByRole("button", { name: "Generate this chapter", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Explanation 2", exact: true })).toBeVisible();
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
  await page.getByRole("button", { name: "Select source lines 2 through 3", exact: true }).click();
  const selection = await page.getByRole("textbox", { name: "Chapter source", exact: true }).evaluate((element) => {
    const textarea = element as HTMLTextAreaElement;
    return textarea.value.slice(textarea.selectionStart, textarea.selectionEnd);
  });
  expect(selection).toBe("+validate(request);\n+return success;");
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
  await page.getByRole("button", { name: "Select source lines 2 through 3", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Chapter source", exact: true })).toBeFocused();
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
