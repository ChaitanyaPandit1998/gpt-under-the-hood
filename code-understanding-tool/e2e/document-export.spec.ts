import { expect, test } from "@playwright/test";
import { openStudyActions } from "./support/study-actions";

test("exports an existing result with source consent, errors and cancellation without regeneration", async ({ page }, testInfo) => {
  const exports: Record<string, unknown>[] = [];
  let generations = 0;
  let mode = "success";
  let release = () => {};
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/document-export") {
      if (route.request().method() === "GET") return route.fulfill({ json: { available: true, diagramImages: true } });
      exports.push(route.request().postDataJSON());
      if (mode === "fail") return route.fulfill({ status: 500, json: { error: "Renderer unavailable. Retry manually." } });
      if (mode === "hold") await new Promise<void>((resolve) => { release = resolve; });
      return route.fulfill({ contentType: "application/pdf", body: "%PDF-test" }).catch(() => {});
    }
    if (path === "/api/explain-progress") return route.fulfill({ json: { phase: "drafting" } });
    generations += 1;
    return route.fulfill({ json: { title: "Current result", summary: "Source grounded summary", concepts: [], walkthrough: [], assumptions: [],
      visuals: [{ title: "Assignment map", kind: "concept_map", nodes: [{ id: "answer", label: "Assign 42", description: "Binds the value.", source_ranges: [] }], edges: [] }],
      visual_note: "Limited coverage", quality_review_status: "incomplete", quality_review_note: "Check branches" } });
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input", exact: true }).fill("const answer = 42;");
  await page.getByRole("button", { name: "Help me understand", exact: false }).click();
  await (await openStudyActions(page)).getByText("Export document", { exact: false }).click();
  await expect(page.getByText("For an illustrated PDF", { exact: false })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Include source text in this download" })).not.toBeChecked();
  await page.getByLabel("Document format").selectOption("pdf");
  const first = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download document", exact: true }).click();
  expect((await first).suggestedFilename()).toBe("study-guide.pdf");
  expect(exports[0]).toMatchObject({ includeSource: false, format: "pdf", explanation: { title: "Current result", quality_review_status: "incomplete" } });
  expect(exports[0]).not.toHaveProperty("diagramIndex");
  await page.getByRole("checkbox", { name: "Include source text in this download" }).check();
  await page.getByLabel("Diagram image").selectOption({ label: "Assignment map" });
  const second = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download document", exact: true }).click();
  await second;
  expect(exports[1]).toMatchObject({ includeSource: true, diagramIndex: 0 });
  mode = "fail";
  await page.getByRole("button", { name: "Download document", exact: true }).click();
  await expect(page.getByText("Renderer unavailable. Retry manually.", { exact: true })).toBeVisible();
  mode = "hold";
  await page.getByRole("button", { name: "Download document", exact: true }).click();
  await expect.poll(() => exports.length).toBe(4);
  const actions = await openStudyActions(page);
  await actions.locator(":scope > summary").click();
  await expect(actions.locator(":scope > summary .tool-active-indicator")).toBeVisible();
  await openStudyActions(page);
  await page.getByRole("button", { name: "Cancel export", exact: true }).click();
  release();
  await expect(page.getByText("Export cancelled.", { exact: true })).toBeVisible();
  await expect(actions.locator(":scope > summary .tool-active-indicator")).not.toBeVisible();
  expect(generations).toBe(1);
  await expect(page.getByRole("heading", { name: "Current result", exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
  await page.locator(".document-export:not(.diagram-export)").screenshot({ path: testInfo.outputPath("document-export.png") });
});

test("explains missing exporter setup without blocking study", async ({ page }) => {
  await page.route("**/api/document-export", (route) => route.fulfill({ json: { available: false } }));
  await page.route("**/api/explain", (route) => route.fulfill({ json: { title: "Readable", summary: "Available", concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "" } }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input", exact: true }).fill("x");
  await page.getByRole("button", { name: "Help me understand", exact: false }).click();
  await (await openStudyActions(page)).getByText("Export document", { exact: false }).click();
  await expect(page.getByText("This explanation has no diagram to embed.", { exact: false })).toBeVisible();
  await expect(page.getByText("Export is unavailable.", { exact: false })).toBeVisible();
  await expect(page.getByRole("button", { name: "Download document", exact: true })).toBeDisabled();
  await expect(page.getByText("Available", { exact: true })).toBeVisible();
});
