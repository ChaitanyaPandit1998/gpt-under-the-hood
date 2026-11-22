import { expect, test } from "@playwright/test";

test("compares a new illustration without publishing it until selected", async ({ page }, testInfo) => {
  let briefs = 0;
  let plans = 0;
  const resolutions: string[] = [];
  await page.route("**/api/explain", (route) => route.fulfill({ json: {
    title: "Grouping items", summary: "Group items by color.", concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "",
  } }));
  await page.route("**/api/visual-brief", (route) => {
    const caption = ++briefs === 1 ? "Items in two baskets" : "Items on a color wheel";
    return route.fulfill({ json: { brief: {
      disposition: "render", visualType: "comparison", takeaway: "Both forms show the same groups.", composition: caption,
      content: [{ label: "Items", role: "illustrative", statement: "Color distinguishes each group.", chapterLines: [] }],
      guardrails: [], caption, altText: caption,
    } } });
  });
  await page.route("**/api/visual-plan", (route) => {
    plans += 1;
    return route.fulfill({ json: { scene: { background: "#ffffff", items: [
      { kind: "text", text: route.request().postDataJSON().caption, x: 50, y: 80, size: 24, color: "#202033", align: "left" },
    ] } } });
  });
  await page.route("**/api/visual-render", (route) => {
    resolutions.push(route.request().postDataJSON().resolution);
    return route.fulfill({ contentType: "image/png", body: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=", "base64") });
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input", exact: true }).fill("groupBy(items, color);");
  await page.getByRole("button", { name: "Help me understand" }).click();
  await page.getByText("Design a visual", { exact: false }).click();
  await page.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await page.getByRole("radio", { name: "HD", exact: true }).check();
  const render = page.getByRole("button", { name: "Render image", exact: true });
  await render.click();
  const inline = page.locator("figure.inline-illustration");
  await expect(inline.getByRole("img")).toHaveAccessibleName("Items in two baskets");

  await page.getByRole("button", { name: "Regenerate visual brief", exact: true }).click();
  const step = page.getByRole("region", { name: "Render the image", exact: true });
  await expect(step.getByRole("radio", { name: "HD", exact: true })).toBeChecked();
  await render.click();
  const comparison = page.getByRole("group", { name: "Compare illustrations" });
  const current = comparison.getByRole("figure", { name: "Current illustration" }).getByRole("img");
  const candidate = comparison.getByRole("figure", { name: "New illustration" }).getByRole("img");
  await expect(current).toHaveAccessibleName("Items in two baskets");
  await expect(candidate).toHaveAccessibleName("Items on a color wheel");
  const currentBox = (await current.boundingBox())!;
  const candidateBox = (await candidate.boundingBox())!;
  expect(candidateBox.width).toBeCloseTo(currentBox.width, 0);
  if (testInfo.project.name === "mobile") expect(candidateBox.y).toBeGreaterThan(currentBox.y + currentBox.height);
  else expect(candidateBox.x).toBeGreaterThan(currentBox.x + currentBox.width);
  await expect(inline.getByRole("img")).toHaveAccessibleName("Items in two baskets");
  await page.getByRole("button", { name: "PDF preview", exact: true }).click();
  const pdf = page.getByRole("dialog", { name: "PDF preview" });
  const pdfImage = pdf.getByRole("article", { name: "Selected PDF content" }).getByRole("img");
  await expect(pdfImage).toHaveAccessibleName("Items in two baskets");
  await page.keyboard.press("Escape");

  await page.getByRole("button", { name: "Keep current illustration" }).click();
  await expect(comparison).toHaveCount(0);
  await expect(render).toBeFocused();
  await expect(inline.getByRole("img")).toHaveAccessibleName("Items in two baskets");
  await render.click();
  await expect(candidate).toBeVisible();
  await page.getByRole("button", { name: "Use new illustration" }).click();
  await expect(comparison).toHaveCount(0);
  await expect(inline.getByRole("img")).toHaveAccessibleName("Items on a color wheel");
  await page.getByRole("button", { name: "PDF preview", exact: true }).click();
  await expect(pdfImage).toHaveAccessibleName("Items on a color wheel");
  expect(plans).toBe(2);
  expect(resolutions).toEqual(["hd", "hd", "hd"]);
});

test("reuses the explanation, sends only selected preferences and opts in to source", async ({ page }, testInfo) => {
  const requests: Record<string, unknown>[] = [];
  const renders: Record<string, unknown>[] = [];
  let plans = 0;
  const scene = { background: "#ffffff", items: [{ kind: "text", text: "Retry", x: 50, y: 80, size: 24, color: "#202033", align: "left" }] };
  let failRender = false;
  await page.route("**/api/explain", (route) => route.fulfill({ json: {
    title: "A retry example", summary: "Retry after rejection.", concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "",
  } }));
  await page.route("**/api/document-export", (route) => route.fulfill({ json: { available: false } }));
  await page.route("**/api/visual-plan", (route) => { plans += 1; return route.fulfill({ json: { scene } }); });
  await page.route("**/api/visual-render", async (route) => {
    renders.push(route.request().postDataJSON());
    if (failRender) return route.fulfill({ status: 502, json: { error: "Temporary PNG failure." } });
    await route.fulfill({ contentType: "image/png", body: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=", "base64") });
  });
  await page.route("**/api/visual-brief", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({ json: { brief: {
      disposition: "render", visualType: "decision-flow", takeaway: "Where retry branches.", composition: "Show a warm-colored decision.",
      content: [{ label: "Retry", role: "illustrative", statement: "Repeat after rejection.", chapterLines: [] }],
      guardrails: ["Not a runtime trace."], caption: "Retry branch.", altText: "An operation branches to retry.",
    } } });
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input", exact: true }).fill("retry(operation);");
  await page.getByRole("button", { name: "Help me understand", exact: false }).click();
  await page.getByText("Design a visual", { exact: false }).click();
  await expect(page.getByText("Original source: not attached.", { exact: true })).toBeVisible();
  const renderStep = page.getByRole("region", { name: "Render the image", exact: true });
  await expect(renderStep.getByText("Generate a brief to unlock image rendering.")).toBeVisible();
  await expect(renderStep.getByRole("button", { name: "Render image", exact: true })).toHaveCount(0);
  await expect(page.getByLabel("Visual style", { exact: true })).toBeHidden();
  await page.getByText("Customize visual Optional", { exact: true }).click();
  await page.getByLabel("Visual style", { exact: true }).selectOption("expressive");
  await page.getByLabel("Visual audience").selectOption("intermediate");
  await page.getByLabel("Visual form").selectOption("decision-flow");
  await page.getByLabel("Output target").selectOption("print");
  await page.getByLabel("What should the visual explain?", { exact: false }).fill("When do we retry?");
  const warm = page.getByRole("checkbox", { name: "Warm palette", exact: true });
  await warm.focus();
  await warm.press("Space");
  await expect(warm).toBeChecked();
  await page.getByRole("checkbox", { name: "Print-friendly", exact: true }).check();
  await page.getByRole("checkbox", { name: "Step by step", exact: true }).check();
  await page.getByLabel("Art direction or constraints", { exact: false }).fill("Leave space around labels.");
  await page.getByLabel("Additional context or data", { exact: false }).fill("A teaching example waits 2 seconds before attempting again.");
  expect(requests).toHaveLength(0);
  await page.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await expect(page.getByRole("region", { name: "Visual design brief" })).toBeVisible();
  await page.getByRole("link", { name: "Review brief and render", exact: true }).click();
  await expect(renderStep).toBeFocused();
  expect(requests[0]).not.toHaveProperty("source");
  expect(requests[0]).toMatchObject({ preferences: { audience: "intermediate", form: "decision-flow", style: "expressive", target: "print", question: "When do we retry?", direction: "Use a warm accent palette with readable contrast.\nUse clear labels, strong contrast, and cues that work without color.\nLeave space around labels.", additionalContext: "When the material has an ordered progression, guide readers with short numbered callouts.\nA teaching example waits 2 seconds before attempting again.", includeSource: false } });
  await warm.uncheck();
  await expect(page.getByLabel("Art direction or constraints", { exact: false })).toHaveValue("Leave space around labels.");
  await page.getByRole("checkbox", { name: "Include original source for line references" }).check();
  await expect(page.getByText("Original source: included with numbered lines.", { exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "Visual design brief" })).toHaveCount(0);
  await page.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await expect(page.getByRole("region", { name: "Visual design brief" })).toBeVisible();
  expect(requests[1]).toMatchObject({ source: "retry(operation);", preferences: { includeSource: true, direction: "Use clear labels, strong contrast, and cues that work without color.\nLeave space around labels." } });
  expect(renders).toHaveLength(0);
  await page.getByRole("radio", { name: "Full HD", exact: true }).check();
  expect(renders).toHaveLength(0);
  await page.getByRole("button", { name: "Render image", exact: true }).click();
  await expect(page.getByRole("img", { name: "An operation branches to retry." })).toBeVisible();
  await page.getByRole("link", { name: "View inline illustration" }).click();
  await expect(page.locator("figure.inline-illustration")).toBeFocused();
  await page.getByText("Design a visual", { exact: false }).click();
  await expect(page.getByRole("img", { name: "An operation branches to retry." })).toBeVisible();
  await page.getByRole("button", { name: "Enlarge illustration" }).click();
  await expect(page.getByRole("dialog", { name: "Enlarged illustration" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "Enlarged illustration" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Enlarge illustration" })).toBeFocused();
  expect(renders[0]).toEqual({ resolution: "full-hd", scene });
  expect(renders[0]).not.toHaveProperty("source");
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download image", exact: true }).click();
  expect((await download).suggestedFilename()).toBe("explanation-visual.png");
  await page.getByText("Design a visual", { exact: false }).click();
  await page.getByRole("radio", { name: "HD", exact: true }).check();
  failRender = true;
  await page.getByRole("button", { name: "Render image again" }).click();
  await expect(page.getByRole("alert")).toHaveText("Temporary PNG failure.");
  await expect(page.getByRole("img", { name: "An operation branches to retry." })).toHaveAttribute("width", "1920");
  failRender = false;
  await page.getByRole("button", { name: "Retry PNG render" }).click();
  await expect(page.getByRole("img", { name: "An operation branches to retry." })).toHaveAttribute("width", "1280");
  expect(plans).toBe(1);
  expect(renders[1]).toEqual({ resolution: "hd", scene });
  expect(renders[2]).toEqual(renders[1]);
  await expect(page.getByRole("heading", { name: "A retry example", exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(warm).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);

  await page.getByRole("button", { name: "Preview prompt", exact: true }).click();
  const prompt = page.getByRole("textbox", { name: "Visual design prompt" });
  await expect(prompt).toHaveValue(/When do we retry\?[\s\S]*Visual style: Expressive illustration[\s\S]*numbered callouts/);
  expect(await prompt.inputValue()).not.toContain("retry(operation);");
  expect(requests).toHaveLength(2);
  await page.evaluate(() => {
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: {
      writeText: async (value: string) => { document.body.dataset.copiedVisualPrompt = value; },
    } });
  });
  await prompt.fill("Show a single annotated comparison.");
  await expect(page.getByText("Your edited prompt, instead of the current design selections and notes.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Copy prompt", exact: true }).click();
  await expect(page.getByText("Prompt copied.", { exact: true })).toBeVisible();
  await expect(page.locator("body")).toHaveAttribute("data-copied-visual-prompt", "Show a single annotated comparison.");
  await page.getByRole("checkbox", { name: "Include original source for line references" }).uncheck();
  await page.getByRole("button", { name: "Hide prompt", exact: true }).click();
  await page.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await expect(page.getByRole("region", { name: "Visual design brief" })).toBeVisible();
  expect(requests[2]).toMatchObject({ designPrompt: "Show a single annotated comparison.", preferences: { includeSource: false } });
  expect(requests[2]).not.toHaveProperty("source");
  await page.getByRole("button", { name: "Preview prompt", exact: true }).click();
  await expect(prompt).toHaveValue("Show a single annotated comparison.");
  await page.getByRole("button", { name: "Reset to selections", exact: true }).click();
  await expect(prompt).toHaveValue(/numbered callouts/);
  expect(requests).toHaveLength(3);

  await page.getByRole("button", { name: "Chapters", exact: true }).click();
  const reader = page.getByRole("region", { name: "Overview", exact: true });
  await reader.getByRole("link", { name: "Refine chapter", exact: true }).click();
  await expect(reader.getByRole("region", { name: "Customize Overview", exact: true })).toBeFocused();
  await reader.getByText("Illustrate chapter", { exact: false }).click();
  await reader.getByRole("button", { name: "Generate visual brief", exact: true }).click();
  await reader.getByRole("button", { name: "Render image", exact: true }).click();
  await expect(reader.getByRole("img", { name: "An operation branches to retry." })).toBeVisible();
  expect(JSON.parse(requests[3].context as string)).toMatchObject({ scope: "Chapter: Overview", summary: "Retry after rejection." });
  await reader.getByRole("button", { name: "Chapter PDF preview", exact: true }).click();
  const singleChapter = page.getByRole("dialog", { name: "PDF preview" });
  const singleChapterPaper = singleChapter.getByRole("article", { name: "Selected PDF content" });
  await expect(singleChapterPaper.getByText("Chapter: Overview", { exact: false })).toBeVisible();
  await expect(singleChapterPaper.getByRole("heading", { name: "Overview", exact: true })).toBeVisible();
  await expect(singleChapterPaper.getByRole("heading", { name: "Assumptions and limits", exact: true })).toHaveCount(0);
  await expect(singleChapterPaper.getByRole("img")).toHaveCount(1);
  await expect(singleChapter.getByRole("checkbox", { name: "Whole-explanation illustration", exact: true })).toHaveCount(0);
  await expect(singleChapter.getByRole("checkbox", { name: "Include original source appendix", exact: true })).toHaveCount(0);
  await singleChapter.getByRole("button", { name: "Close PDF preview" }).click();
  await reader.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.getByRole("region", { name: "Limits and understanding check" }).getByRole("img", { name: "An operation branches to retry." })).toHaveCount(0);
  await page.getByRole("button", { name: "Previous", exact: true }).click();
  await expect(reader.getByRole("img", { name: "An operation branches to retry." })).toBeVisible();
  await page.getByRole("button", { name: "Full explanation", exact: true }).click();
  await expect(page.getByRole("img", { name: "An operation branches to retry." })).toHaveCount(2);
  await page.getByRole("button", { name: "PDF preview", exact: true }).click();
  const pdf = page.getByRole("dialog", { name: "PDF preview" });
  const paper = pdf.getByRole("article", { name: "Selected PDF content" });
  await expect(paper.getByRole("img")).toHaveCount(2);
  await expect(paper.getByRole("heading", { name: "Original source appendix" })).toHaveCount(0);
  await pdf.getByRole("checkbox", { name: "Whole-explanation illustration", exact: true }).uncheck();
  await expect(paper.getByRole("img")).toHaveCount(1);
  await pdf.getByRole("checkbox", { name: "Overview", exact: true }).uncheck();
  await expect(pdf.getByRole("checkbox", { name: "Overview illustration", exact: true })).toBeDisabled();
  await expect(paper.getByRole("img")).toHaveCount(0);
  await expect(paper.getByRole("heading", { name: "Overview", exact: true })).toHaveCount(0);
  await pdf.getByRole("checkbox", { name: "Assumptions and limits", exact: true }).uncheck();
  await expect(pdf.getByRole("button", { name: "Print / Save as PDF" })).toBeDisabled();
  await pdf.getByRole("checkbox", { name: "Overview", exact: true }).check();
  await expect(paper.getByRole("img")).toHaveCount(1);
  await pdf.getByRole("checkbox", { name: "Include original source appendix", exact: true }).check();
  await expect(paper.getByText("1 | retry(operation);", { exact: true })).toBeVisible();
  await pdf.getByRole("checkbox", { name: "Include original source appendix", exact: true }).uncheck();
  await page.evaluate(() => { window.print = () => { document.body.dataset.pilotPrinted = "yes"; }; });
  await pdf.getByRole("button", { name: "Print / Save as PDF" }).click();
  await expect(page.locator("body")).toHaveAttribute("data-pilot-printed", "yes");
  if (testInfo.project.name === "desktop") {
    const path = testInfo.outputPath("selected-study.pdf");
    await page.pdf({ path, preferCSSPageSize: true });
    await testInfo.attach("selected study PDF", { path, contentType: "application/pdf" });
  }
  await page.keyboard.press("Escape");
  await expect(pdf).toHaveCount(0);
  expect(renders).toHaveLength(4);
  expect(plans).toBe(2);
  await page.getByRole("button", { name: "Chapters", exact: true }).click();
  await reader.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.getByRole("button", { name: "Resume saved study" })).toHaveCount(0);
  await page.getByRole("button", { name: "Save study locally" }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  await page.getByRole("checkbox", { name: "Keep source and completed explanation after refresh in this tab" }).uncheck();
  await page.reload();
  await expect(page.getByRole("heading", { name: "A retry example", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Resume saved study" }).click();
  await expect(page.getByRole("region", { name: "Limits and understanding check", exact: true })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Code input", exact: true })).toHaveValue("retry(operation);");
  await page.getByRole("button", { name: "Previous", exact: true }).click();
  await expect(reader.getByRole("img", { name: "An operation branches to retry." })).toBeVisible();
  expect(plans).toBe(2);
  await page.getByRole("button", { name: "Delete saved study" }).click();
  await expect(page.getByText("Saved study deleted from this browser.", { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("button", { name: "Resume saved study" })).toHaveCount(0);
});
