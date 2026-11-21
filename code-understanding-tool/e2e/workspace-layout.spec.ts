import { expect, test } from "@playwright/test";

test("gives the desktop input card more room while keeping mobile compact", async ({ page }, testInfo) => {
  await page.goto("/");
  const workspace = page.getByRole("region", { name: "Learning workspace", exact: true });
  const editor = page.getByRole("textbox", { name: "Code input", exact: true });
  await expect(workspace).toBeVisible();
  const cardBounds = await workspace.boundingBox();
  const editorBounds = await editor.boundingBox();
  expect(cardBounds).not.toBeNull();
  expect(editorBounds).not.toBeNull();
  if (testInfo.project.name === "desktop") {
    expect(cardBounds!.width).toBeGreaterThan(1000);
    expect(editorBounds!.height).toBeGreaterThanOrEqual(320);
  } else {
    expect(cardBounds!.width).toBeLessThanOrEqual(390);
    expect(editorBounds!.height).toBeGreaterThanOrEqual(200);
    expect(editorBounds!.height).toBeLessThan(320);
  }
  await editor.fill("const value = 42;");
  await expect(editor).toHaveValue("const value = 42;");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
});
