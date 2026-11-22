import { expect, test } from "@playwright/test";

for (const mode of ["code", "docs"] as const) {
test(`${mode}: adapts just one section, retains original warnings and restores cached variants`, async ({ page }, testInfo) => {
  const requests: Record<string, unknown>[] = [];
  const exports: Record<string, unknown>[] = [];
  const source = "const value = 42;\nconsole.log(value);";
  await page.route("**/api/**", async (route) => {
    if (route.request().url().includes("document-export")) {
      if (route.request().method() === "GET") return route.fulfill({ json: { available: true } });
      exports.push(route.request().postDataJSON());
      return route.fulfill({ contentType: "application/octet-stream", body: "fake-docx" });
    }
    if (route.request().url().includes("explain-progress")) return route.fulfill({ json: { phase: "drafting" } });
    if (route.request().url().endsWith("/visual-brief")) return route.fulfill({ json: { brief: {
      disposition: "render", visualType: "comparison", takeaway: "Compare two values.", composition: "Show two labels.",
      content: [{ label: "Value", role: "illustrative", statement: "Example label.", chapterLines: [] }],
      guardrails: [], caption: "Adapted example.", altText: "Adapted value illustration.",
    } } });
    if (route.request().url().endsWith("/visual-plan")) return route.fulfill({ json: { scene: {
      background: "#ffffff", items: [{ kind: "text", text: "Value", x: 50, y: 80, size: 24, color: "#202033", align: "left" }],
    } } });
    if (route.request().url().endsWith("/visual-render")) return route.fulfill({ contentType: "image/png",
      body: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=", "base64") });
    const body = route.request().postDataJSON() as Record<string, unknown>;
    requests.push(body);
    const original = requests.length === 1;
    return route.fulfill({ json: {
      title: original ? "Original explanation" : `Variant ${requests.length}`,
      summary: original ? "Original overview remains available." : "This is the adapted section.",
      concepts: [], walkthrough: [{ heading: "Assignment", explanation: "Assigns 42 to value.", source_ranges: [{ start_line: 1, end_line: 1 }] }],
      assumptions: [{ statement: "No broader context.", reason: "Only two source lines were supplied." }],
      visuals: [], visual_note: "", quality_review_status: "incomplete", quality_review_note: "A remaining concern could not be verified.",
    } });
  });
  await page.goto("/");
  if (mode === "docs") await page.getByRole("tab", { name: /Documentation/ }).click();
  const editor = page.getByRole("textbox", { name: mode === "code" ? "Code input" : "Documentation input", exact: true });
  await editor.fill(source);
  await page.getByRole("button", { name: "Help me understand", exact: false }).click();
  await expect(page.getByRole("heading", { name: "Original explanation", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Chapters", exact: true }).click();
  const overview = page.getByRole("region", { name: "Customize Overview", exact: true });
  await overview.getByRole("button", { name: "Go deeper", exact: true }).click();
  expect(requests).toHaveLength(1);
  await overview.getByText("Customize explanation", { exact: true }).click();
  await overview.getByRole("combobox", { name: "Chapter tone" }).selectOption("technical");
  await overview.getByRole("combobox", { name: "Chapter focus" }).selectOption("example");
  await overview.getByRole("button", { name: "Generate section variant", exact: true }).click();
  await expect(overview.getByRole("heading", { name: "Variant 2", exact: true })).toBeVisible();
  expect(requests[1]).toMatchObject({ source_text: source, material_kind: mode, explanation_depth: "detailed", explanation_tone: "technical", explanation_focus: "example" });
  expect(requests[1].learner_question).toContain("Explain ONLY");
  expect(requests[1].learner_question).toContain("Original overview remains available.");
  await overview.getByText("Export document", { exact: false }).click();
  const download = page.waitForEvent("download");
  await overview.getByRole("button", { name: "Download document", exact: true }).click();
  await download;
  expect(exports[0]).toMatchObject({ scope: "Adapted section: Overview", explanation: { title: "Variant 2" } });
  expect(exports[0].provenance).toContain('"tone":"technical"');
  await expect(page.getByText("Original overview remains available.", { exact: true })).toBeVisible();
  await expect(page.getByText("Second-pass review", { exact: false }).first()).toBeVisible();
  const citation = overview.getByRole("button", { name: "View source lines 1 through 1", exact: true });
  await citation.click();
  const evidence = page.getByRole("dialog", { name: "Source reference" });
  await expect(evidence.locator("mark code")).toHaveText(["const value = 42;"]);
  await expect(evidence.locator("div.source-evidence-line code")).toHaveText(["console.log(value);"]);
  await expect(evidence.getByRole("button", { name: "Close source" })).toBeFocused();
  const panelBox = (await evidence.boundingBox())!;
  const resultBox = (await page.locator(".result-card").boundingBox())!;
  if (testInfo.project.name === "mobile") {
    expect(panelBox.y + panelBox.height).toBeCloseTo(844, 0);
    expect(await evidence.evaluate((element) => element.matches(":modal"))).toBe(true);
  } else {
    expect(panelBox.x).toBeGreaterThan(resultBox.x + resultBox.width);
    expect(await evidence.evaluate((element) => element.matches(":modal"))).toBe(false);
  }
  await page.keyboard.press("Escape");
  await expect(evidence).toHaveCount(0);
  await expect(citation).toBeFocused();
  const outline = page.getByRole("navigation", { name: "Explanation chapters", exact: true });
  await outline.getByRole("button", { name: /Assignment/ }).click();
  const assignment = page.getByRole("region", { name: "Customize Assignment", exact: true });
  await assignment.getByText("Customize explanation", { exact: true }).click();
  await expect(assignment.getByRole("combobox", { name: "Chapter depth" })).toHaveValue("balanced");
  await expect(assignment.getByRole("combobox", { name: "Chapter tone" })).toHaveValue("plain");
  await outline.getByRole("button", { name: /Overview/ }).click();
  await expect(overview.getByRole("heading", { name: "Variant 2", exact: true })).toBeVisible();
  await overview.getByRole("combobox", { name: "Chapter focus" }).selectOption("behaviour");
  await expect(overview.getByRole("heading", { name: "Variant 2", exact: true })).toHaveCount(0);
  await overview.getByRole("combobox", { name: "Chapter focus" }).selectOption("example");
  await expect(overview.getByRole("heading", { name: "Variant 2", exact: true })).toBeVisible();
  expect(requests).toHaveLength(2);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
  if (mode === "code") await overview.screenshot({ path: testInfo.outputPath("chapter-controls.png") });
  await overview.getByText("Design a visual", { exact: false }).click();
  await overview.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await overview.getByRole("button", { name: "Render image", exact: true }).click();
  const illustration = overview.getByRole("img", { name: "Adapted value illustration.", exact: true });
  await expect(illustration).toBeVisible();
  await page.getByRole("button", { name: "Save study locally", exact: true }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Resume saved study", exact: true }).click();
  await expect(overview.getByRole("heading", { name: "Variant 2", exact: true })).toBeVisible();
  await expect(illustration).toBeVisible();
  await overview.getByText("Customize explanation", { exact: true }).click();
  await expect(overview.getByRole("combobox", { name: "Chapter focus" })).toHaveValue("example");
  await overview.getByRole("combobox", { name: "Chapter focus" }).selectOption("behaviour");
  await expect(overview.getByRole("heading", { name: "Variant 2", exact: true })).toHaveCount(0);
  await overview.getByRole("combobox", { name: "Chapter focus" }).selectOption("example");
  await expect(overview.getByRole("heading", { name: "Variant 2", exact: true })).toBeVisible();
  expect(requests).toHaveLength(2);
});
}

test("cancels section work when navigating and ignores its late response", async ({ page }) => {
  let count = 0;
  let release = () => {};
  await page.route("**/api/**", async (route) => {
    if (route.request().url().includes("document-export")) return route.fulfill({ json: { available: false } });
    if (route.request().url().includes("explain-progress")) return route.fulfill({ json: { phase: "drafting" } });
    count += 1;
    const original = count === 1;
    if (!original) await new Promise<void>((resolve) => { release = resolve; });
    await route.fulfill({ json: {
      title: original ? "Original" : "Late variant", summary: "Supplied source only.", concepts: [],
      walkthrough: [{ heading: "Assignment", explanation: "Assigns a value.", source_ranges: [{ start_line: 1, end_line: 1 }] }],
      assumptions: [], visuals: [], visual_note: "",
    } }).catch(() => {});
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input", exact: true }).fill("const value = 42;");
  await page.getByRole("button", { name: "Help me understand", exact: false }).click();
  await expect(page.getByRole("heading", { name: "Original", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Chapters", exact: true }).click();
  const overview = page.getByRole("region", { name: "Customize Overview", exact: true });
  await overview.getByRole("button", { name: "Generate section variant", exact: true }).click();
  await expect.poll(() => count).toBe(2);
  const outline = page.getByRole("navigation", { name: "Explanation chapters", exact: true });
  await outline.getByRole("button", { name: /Assignment/ }).click();
  release();
  await outline.getByRole("button", { name: /Overview/ }).click();
  await expect(overview.getByRole("button", { name: "Generate section variant", exact: true })).toBeEnabled();
  await expect(page.getByRole("heading", { name: "Late variant", exact: true })).toHaveCount(0);
});
