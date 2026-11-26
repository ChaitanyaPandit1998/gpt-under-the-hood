import { createHash } from "node:crypto";
import { expect, test, type Page, type Route } from "@playwright/test";
import { openStudyActions } from "./support/study-actions";
import { visualCheckStages } from "../src/domain/visual-evaluation";
import { makeVisualPng } from "../src/test-fixtures/visual-png";

const png = makeVisualPng();
const brief = { disposition: "render", visualType: "concept-map", takeaway: "Show a fixed capacity.", composition: "One capacity label.",
  content: [{ label: "Capacity", role: "background", statement: "Capacity is fixed.", chapterLines: [] }],
  guardrails: ["Do not imply measured throughput."], caption: "A schematic capacity label, not observed measurements.", altText: "Capacity label." };
const hash = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
function reportFor(route: Route, blocking = false) {
  const body = route.request().postDataJSON();
  return { version: 1, contractId: body.contract.id, explanationSha256: body.contract.explanationSha256, repairAttempt: 0,
    artifacts: { briefSha256: hash(body.brief), planSha256: hash(body.plan), pngSha256: hash(Buffer.from(body.pngBase64, "base64")), captionSha256: hash(body.caption) },
    checks: visualCheckStages.map((stage) => ({ stage, status: blocking && stage === "alignment" ? "fail" : "pass", durationMs: 10,
      reviewer: { kind: "model", name: "Mocked browser reviewer", version: "test-1" }, summary: "Mocked stage result, not a live image judgment.",
      requirementIds: body.contract.requirements.map((item: { id: string }) => item.id),
      findings: blocking && stage === "alignment" ? [{ severity: "critical", message: "A visible relation contradicts the explanation.", evidence: "The arrow is reversed.", requirementIds: [body.contract.requirements[0].id] }] : [],
    })) };
}
async function openPanel(page: Page) {
  await page.route("**/api/explain", (route) => route.fulfill({ json: { title: "A bounded collection", summary: "The capacity is fixed.",
    concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "" } }));
  await page.route("**/api/visual-brief", (route) => route.fulfill({ json: { brief,
    generation: { artifactSha256: hash(JSON.stringify(brief)), instructions: [{ id: "visual-direction", version: 1, sha256: "a".repeat(64) }] },
  } }));
  await page.route("**/api/visual-plan", (route) => {
    const scene = { background: "#ffffff", items: [
    { kind: "text", text: "Capacity", x: 50, y: 80, size: 24, color: "#000000", align: "left" },
    ] };
    return route.fulfill({ json: { scene, generation: { artifactSha256: hash(JSON.stringify(scene)),
      instructions: [{ id: "visual-composition", version: 1, sha256: "b".repeat(64) }] } } });
  });
  await page.route("**/api/visual-render", (route) => route.fulfill({ contentType: "image/png", body: png }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input", exact: true }).fill("const capacity = 3;");
  await page.getByRole("button", { name: /^Help me understand/ }).click();
  await page.getByText("Design a visual", { exact: false }).click();
  await page.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await page.getByRole("button", { name: "Render image", exact: true }).click();
  const panel = page.locator("details.visual-review-panel");
  await panel.locator(":scope > summary").click();
  return panel;
}

test("keeps drawing labels readable on mobile and opens a scrollable actual-size preview", async ({ page }) => {
  await openPanel(page);
  const figure = page.getByRole("figure", { name: "GENERATED ILLUSTRATION", exact: true });
  const label = figure.locator("figcaption").getByRole("listitem").filter({ hasText: "Capacity" });
  if (page.viewportSize()!.width <= 600) {
    await expect(label).toBeVisible();
  } else {
    await expect(label).toBeHidden();
  }
  const enlarge = figure.getByRole("button", { name: "Enlarge illustration", exact: true });
  await enlarge.click();
  const preview = page.getByRole("dialog", { name: "Enlarged illustration", exact: true });
  await expect(preview).toBeVisible();
  if (page.viewportSize()!.width <= 600) {
    await expect(preview).toHaveCSS("width", `${page.viewportSize()!.width}px`);
  }
  await expect(preview.getByRole("listitem").filter({ hasText: "Capacity" })).toBeVisible();
  await preview.getByRole("button", { name: "Actual size", exact: true }).click();
  const scroller = preview.getByRole("region", { name: "Scrollable full-size illustration", exact: true });
  await expect(scroller.getByRole("img")).toHaveCSS("width", "1000px");
  await expect.poll(() => scroller.evaluate(el => el.scrollHeight > el.clientHeight)).toBe(true);
  await page.keyboard.press("Escape");
  await expect(preview).toBeHidden();
  await expect(enlarge).toBeFocused();
  await enlarge.click();
  await expect(preview.getByRole("button", { name: "Actual size", exact: true })).toBeVisible();
});

test("reviews only after checklist confirmation, displays findings and marks edited results outdated", async ({ page }, testInfo) => {
  const requests: ReturnType<Route["request"]>[] = [];
  await page.route("**/api/visual-review", (route) => {
    requests.push(route.request());
    return route.fulfill({ json: { report: reportFor(route, requests.length > 1) } });
  });
  const panel = await openPanel(page);
  const button = panel.getByRole("button", { name: "Review visual", exact: true });
  const consent = panel.getByRole("checkbox", { name: "I confirm this checklist", exact: false });
  const image = page.getByRole("img", { name: "Capacity label.", exact: true });
  const original = await image.getAttribute("src");
  expect(requests).toHaveLength(0);
  await expect(button).toBeDisabled();
  await panel.getByRole("radio", { name: "Intermediate", exact: true }).check();
  await panel.getByLabel("Where requirement 1 belongs", { exact: true }).selectOption("Caption");
  await panel.getByRole("checkbox", { name: "Include requirement 2", exact: true }).uncheck();
  await expect(button).toBeDisabled();
  await consent.check(); await button.click();
  const results = panel.getByRole("region", { name: "Visual review results" });
  await expect(results.getByRole("heading", { name: "Checks passed", exact: true })).toBeVisible();
  const body = requests[0].postDataJSON();
  expect(body).toMatchObject({ confirmed: true, caption: brief.caption, contract: { audience: "intermediate",
    requirements: [{ statement: "Caption: Capacity is fixed." }] } });
  expect(body).not.toHaveProperty("source");
  await expect(image).toHaveAttribute("src", original!);
  await panel.getByRole("textbox", { name: "What should this visual teach?", exact: true }).fill("Show why capacity is bounded.");
  await expect(results.getByRole("heading", { name: "Previous review (outdated)", exact: true })).toBeVisible();
  await expect(consent).not.toBeChecked(); await expect(button).toBeDisabled();
  expect(requests).toHaveLength(1);
  await consent.check(); await button.click();
  await expect(results.getByRole("heading", { name: "Blocking issue", exact: true })).toBeVisible();
  await results.locator("summary").filter({ hasText: "Factual alignment" }).click();
  await expect(results.getByText("critical: A visible relation contradicts the explanation.", { exact: true })).toBeVisible();
  await expect(image).toHaveAttribute("src", original!);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("visual-review-panel.png"), fullPage: true });
});

test("cancels review, ignores a late response and keeps the image after failure", async ({ page }) => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => { release = resolve; });
  let calls = 0;
  await page.route("**/api/visual-review", async (route) => {
    if (++calls === 1) { await held; await route.fulfill({ json: { report: reportFor(route) } }).catch(() => {}); }
    else await route.fulfill({ status: 502, json: { error: "Could not complete visual review. Your image is unchanged." } });
  });
  const panel = await openPanel(page);
  await panel.getByRole("radio", { name: "Beginner", exact: true }).check();
  await panel.getByRole("checkbox", { name: "I confirm this checklist", exact: false }).check();
  await panel.getByRole("button", { name: "Review visual", exact: true }).click();
  await expect.poll(() => calls).toBe(1);
  await panel.getByRole("button", { name: "Cancel visual review", exact: true }).click(); release();
  await expect(panel.getByText("Review cancelled. Your image is unchanged.", { exact: true })).toBeVisible();
  await expect(panel.getByRole("region", { name: "Visual review results" })).toHaveCount(0);
  await panel.getByRole("button", { name: "Review visual", exact: true }).click();
  await expect(panel.getByRole("alert")).toHaveText("Could not complete visual review. Your image is unchanged.");
  await expect(page.getByRole("img", { name: "Capacity label.", exact: true })).toBeVisible();
});

test("rejects a report for different PNG bytes instead of showing a pass", async ({ page }) => {
  await page.route("**/api/visual-review", (route) => {
    const report = reportFor(route); report.artifacts.pngSha256 = "b".repeat(64);
    return route.fulfill({ json: { report } });
  });
  const panel = await openPanel(page);
  await panel.getByRole("radio", { name: "Advanced", exact: true }).check();
  await panel.getByRole("checkbox", { name: "I confirm this checklist", exact: false }).check();
  await panel.getByRole("button", { name: "Review visual", exact: true }).click();
  await expect(panel.getByRole("alert")).toHaveText("The review does not match this image, caption, brief or drawing plan.");
  await expect(panel.getByRole("region", { name: "Visual review results" })).toHaveCount(0);
});

test("merges a late builder review with a newer inline review of the same image", async ({ page }) => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => { release = resolve; });
  let reviews = 0;
  await page.route("**/api/visual-review", async (route) => {
    if (++reviews === 2) await held;
    await route.fulfill({ json: { report: reportFor(route) } }).catch(() => {});
  });
  const builder = await openPanel(page);
  async function startReview(panel: typeof builder, audience: string) {
    await panel.getByRole("radio", { name: audience, exact: true }).check();
    await panel.getByRole("checkbox", { name: /I confirm this checklist/ }).check();
    await panel.getByRole("button", { name: "Review visual", exact: true }).click();
  }
  const history = page.locator("details.saved-visual-reviews");
  const summary = history.locator(":scope > summary");
  const image = page.getByRole("img", { name: "Capacity label.", exact: true });
  const original = await image.getAttribute("src");
  await startReview(builder, "Beginner");
  await expect(summary).toContainText("Review history (1)");
  try {
    await startReview(builder, "Intermediate");
    await expect.poll(() => reviews).toBe(2);
    await summary.click();
    await history.getByRole("button", { name: "Review this image again", exact: true }).click();
    const inline = history.locator("details.visual-review-panel");
    await inline.locator(":scope > summary").click();
    await startReview(inline, "Advanced");
    await expect(summary).toContainText("Review history (2)");
  } finally { release(); }
  await expect(summary).toContainText("Review history (3)");
  await expect(image).toHaveAttribute("src", original!);
});

test("retains accepted-image review history only through explicit saving and restores it without reviewing again", async ({ page }) => {
  let reviews = 0;
  await page.route("**/api/visual-review", (route) => {
    reviews++;
    return route.fulfill({ json: { report: reportFor(route) } });
  });
  const panel = await openPanel(page);
  await panel.getByRole("radio", { name: "Intermediate", exact: true }).check();
  await panel.getByRole("checkbox", { name: "I confirm this checklist", exact: false }).check();
  await panel.getByRole("button", { name: "Review visual", exact: true }).click();
  await expect(panel.getByRole("heading", { name: "Checks passed", exact: true })).toBeVisible();
  await expect(page.locator("details.saved-visual-reviews > summary")).toContainText("Review history (1)");
  await expect(page.getByRole("button", { name: "Resume saved study", exact: true })).toHaveCount(0);
  await (await openStudyActions(page)).getByRole("button", { name: "Save study locally", exact: true }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  await page.getByRole("checkbox", { name: "Keep source and completed explanation after refresh in this tab" }).uncheck();
  await page.reload();
  await page.getByRole("button", { name: "Resume saved study", exact: true }).click();
  const history = page.locator("details.saved-visual-reviews");
  await history.locator(":scope > summary").click();
  await expect(history.getByText(/Historical assessments, not approval/)).toBeVisible();
  await expect(history.locator("summary").filter({ hasText: "Historical result: pass" })).toBeVisible();
  expect(reviews).toBe(1);
  await history.getByText("Generation skills and profiles", { exact: true }).click();
  await expect(history.getByText(/visual-direction · version 1/)).toBeVisible();
  await expect(history.getByText(/visual-composition · version 1/)).toBeVisible();
  await history.getByRole("button", { name: "Review this image again", exact: true }).click();
  const restoredReview = history.locator("details.visual-review-panel");
  await restoredReview.locator(":scope > summary").click();
  const consent = restoredReview.getByRole("checkbox", { name: "I confirm this checklist", exact: false });
  await expect(consent).not.toBeChecked();
  await expect(restoredReview.getByRole("button", { name: "Review visual", exact: true })).toBeDisabled();
  expect(reviews).toBe(1);
  await restoredReview.getByRole("radio", { name: "Intermediate", exact: true }).check();
  await consent.check();
  await restoredReview.getByRole("button", { name: "Review visual", exact: true }).click();
  await expect(history.locator(":scope > summary")).toContainText("Review history (2)");
  expect(reviews).toBe(2);
  page.once("dialog", (dialog) => dialog.accept());
  await (await openStudyActions(page)).getByRole("button", { name: "Save study locally", exact: true }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Resume saved study", exact: true }).click();
  await expect(page.locator("details.saved-visual-reviews > summary")).toContainText("Review history (2)");
  expect(reviews).toBe(2);
});
