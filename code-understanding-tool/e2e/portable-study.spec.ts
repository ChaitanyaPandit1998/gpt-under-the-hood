import { expect, test } from "@playwright/test";

const explanation = { title: "A portable concept", summary: "A generic lesson.", concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "" };

test("exports a saved study and imports it without model requests", async ({ page }) => {
  await page.route("**/api/explain", route => route.fulfill({ json: explanation }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Code input" }).fill("const count = 3;");
  await page.getByRole("button", { name: "Help me understand" }).click();
  await page.getByRole("button", { name: "Save study locally", exact: true }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  page.on("dialog", dialog => dialog.accept());
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export saved study", exact: true }).click();
  const download = await downloadEvent;
  const path = await download.path();
  expect(path).toBeTruthy();
  await page.getByRole("button", { name: "Delete saved study", exact: true }).click();
  await expect(page.getByRole("button", { name: "Resume saved study" })).toHaveCount(0);
  await page.getByLabel("Study file", { exact: true }).setInputFiles(path!);
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  let calls = 0;
  await page.route("**/api/**", route => { if (route.request().method() === "POST") calls++; return route.abort(); });
  await page.getByRole("button", { name: "Resume saved study", exact: true }).click();
  await expect(page.getByRole("heading", { name: explanation.title, exact: true })).toBeVisible();
  expect(calls).toBe(0);
  await page.getByLabel("Study file", { exact: true }).setInputFiles({ name: "invalid.json", mimeType: "application/json", buffer: Buffer.from('{"format":"invalid"}') });
  await expect(page.getByText("Invalid or unsupported study file.", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Resume saved study", exact: true })).toBeVisible();
});
