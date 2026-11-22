import { expect, test } from "@playwright/test";

test("opens the illustration designer for the selected diagram without generating automatically", async ({ page }) => {
  const briefs: { context: string; preferences: { style: string }; source?: string }[] = [];
  await page.route("**/api/explain", (route) => route.fulfill({ json: explanation }));
  await page.route("**/api/document-export", (route) => route.fulfill({ json: { available: false } }));
  await page.route("**/api/diagram-export", (route) => route.fulfill({ json: { available: false } }));
  await page.route("**/api/visual-brief", async (route) => {
    briefs.push(route.request().postDataJSON());
    await route.fulfill({ json: { brief: { disposition: "text-only", visualType: "none", takeaway: "Text is enough.", content: [],
      composition: "No image is needed.", guardrails: [], caption: "Read the explanation.", altText: "No diagram." } } });
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input", exact: true }).fill("check();");
  await page.getByRole("button", { name: "Help me understand", exact: false }).click();
  await page.getByRole("tab", { name: "Second view", exact: true }).click();
  await page.getByText("Export diagram image", { exact: false }).click();
  await page.getByRole("button", { name: "Create an illustration instead", exact: true }).click();
  await expect(page.locator(".visual-brief-panel > summary")).toBeFocused();
  await expect(page.getByText("Illustrate this diagram:", { exact: true })).toBeVisible();
  await page.getByText("Customize visual Optional", { exact: true }).click();
  await expect(page.getByLabel("Visual style", { exact: true })).toHaveValue("expressive");
  expect(briefs).toHaveLength(0);
  await page.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await expect(page.getByRole("region", { name: "Visual design brief" })).toBeVisible();
  expect(JSON.parse(briefs[0].context).selectedDiagram).toMatchObject({ title: "Second view" });
  expect(briefs[0].source).toBeUndefined();
  await page.getByRole("button", { name: "Use whole explanation", exact: true }).click();
  await page.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await expect(page.getByRole("region", { name: "Visual design brief" })).toBeVisible();
  expect(JSON.parse(briefs[1].context)).not.toHaveProperty("selectedDiagram");
});

const ranges = [{ start_line: 1, end_line: 1 }];
const node = { id: "a", label: "Decision", description: "Read the supplied condition.", source_ranges: ranges };
const explanation = {
  title: "Diagram study", summary: "The existing explanation stays available.", concepts: [], walkthrough: [], assumptions: [],
  quality_review_status: "incomplete", quality_review_note: "An unresolved concern remains.", visual_note: "Selective diagram, not full execution coverage.",
  visuals: [
    { title: "First view", kind: "control_flow", nodes: [node], edges: [{ from: "a", to: "a", label: "retry" }] },
    { title: "Second view", kind: "concept_map", nodes: [{ ...node, label: "Second concept" }], edges: [] },
  ],
};
const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=", "base64");

test("renders only the selected existing view on demand, previews/downloads PNG and cancels work", async ({ page }, testInfo) => {
  const requests: Record<string, unknown>[] = [];
  let modelCalls = 0, capabilityCalls = 0;
  let hold = false, fail = false;
  let release = () => {};
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/document-export") return route.fulfill({ json: { available: false } });
    if (path === "/api/diagram-export") {
      if (route.request().method() === "GET") { capabilityCalls += 1; return route.fulfill({ json: { available: true } }); }
      requests.push(route.request().postDataJSON());
      if (fail) return route.fulfill({ status: 500, json: { error: "Diagram renderer failed." } });
      if (hold) await new Promise<void>((resolve) => { release = resolve; });
      return route.fulfill({ contentType: "image/png", body: png }).catch(() => {});
    }
    if (path === "/api/explain-progress") return route.fulfill({ json: { phase: "drafting" } });
    modelCalls += 1;
    return route.fulfill({ json: explanation });
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input", exact: true }).fill("check();");
  await page.getByRole("button", { name: "Help me understand", exact: false }).click();
  await expect(page.getByRole("heading", { name: "Diagram study", exact: true })).toBeVisible();
  expect(requests).toHaveLength(0); expect(capabilityCalls).toBe(0);
  await page.getByRole("tab", { name: "Second view", exact: true }).click();
  await page.getByText("Export diagram image", { exact: false }).click();
  await page.getByRole("button", { name: "Generate diagram PNG", exact: true }).click();
  await expect(page.getByRole("img", { name: /Second view: exported diagram/ })).toBeVisible();
  expect(requests[0]).toMatchObject({ version: 1, layout: "detailed", visualIndex: 1, explanation: { quality_review_status: "incomplete", quality_review_note: explanation.quality_review_note } });
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download diagram PNG", exact: true }).click();
  expect((await download).suggestedFilename()).toBe("chapter-diagram.png");
  await page.getByLabel("Diagram layout").selectOption("automatic");
  await expect(page.getByRole("link", { name: "Download diagram PNG", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Generate diagram PNG", exact: true }).click();
  await expect(page.getByRole("link", { name: "Download diagram PNG", exact: true })).toBeVisible();
  expect(requests[1]).toMatchObject({ layout: "automatic", visualIndex: 1 });
  fail = true;
  await page.getByRole("button", { name: "Regenerate diagram PNG", exact: true }).click();
  await expect(page.getByText("Diagram renderer failed.", { exact: true })).toBeVisible();
  fail = false; hold = true;
  await page.getByRole("button", { name: "Regenerate diagram PNG", exact: true }).click();
  await expect.poll(() => requests.length).toBe(4);
  await page.getByRole("button", { name: "Cancel diagram export", exact: true }).click(); release();
  await expect(page.getByText("Diagram export cancelled.", { exact: true })).toBeVisible();
  expect(modelCalls).toBe(1);
  await expect(page.getByText(explanation.summary, { exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
  await page.locator(".diagram-export").screenshot({ path: testInfo.outputPath("diagram-export.png") });
});

test("keeps the interactive diagram available when VectorCraft is missing", async ({ page }) => {
  await page.route("**/api/explain", (route) => route.fulfill({ json: explanation }));
  await page.route("**/api/document-export", (route) => route.fulfill({ json: { available: false } }));
  await page.route("**/api/diagram-export", (route) => route.fulfill({ json: { available: false } }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input", exact: true }).fill("check();");
  await page.getByRole("button", { name: "Help me understand", exact: false }).click();
  await page.getByText("Export diagram image", { exact: false }).click();
  await expect(page.getByText("VectorCraft export is unavailable.", { exact: false })).toBeVisible();
  await expect(page.getByRole("button", { name: "Generate diagram PNG", exact: true })).toBeDisabled();
  await expect(page.getByRole("region", { name: "Interactive control flow diagram: First view", exact: true })).toBeVisible();
});
