import { expect, test } from "@playwright/test";

test("starter cards prepare each input mode without fetching or generating", async ({ page }, testInfo) => {
  const requests: { endpoint: string; body: unknown }[] = [];
  await page.route("**/api/**", (route) => {
    requests.push({ endpoint: new URL(route.request().url()).pathname, body: route.request().postDataJSON() });
    return route.fulfill({ status: 503, json: { error: "Import test response." } });
  });
  await page.goto("/");
  const cards = page.getByRole("region", { name: "Try an example" });
  await expect(cards.getByRole("button")).toHaveCount(3);
  const codeCard = cards.getByRole("button", { name: /Retry a failed request/ });
  const conceptCard = cards.getByRole("button", { name: /Understand cache-aside/ });
  const prCard = cards.getByRole("button", { name: /Explore a FastAPI migration/ });
  if (testInfo.project.name === "mobile") {
    const codeBox = (await codeCard.boundingBox())!;
    const prBox = (await prCard.boundingBox())!;
    expect(prBox.y).toBeGreaterThan(codeBox.y + codeBox.height);
  }
  await codeCard.focus();
  await codeCard.press("Enter");
  const code = page.getByRole("textbox", { name: "Code input", exact: true });
  await expect(code).toBeFocused();
  await expect(code).toHaveValue(/async function retry/);

  await code.fill("Keep my draft");
  page.once("dialog", (dialog) => dialog.dismiss());
  await conceptCard.click();
  await expect(code).toHaveValue("Keep my draft");
  page.once("dialog", (dialog) => dialog.accept());
  await conceptCard.click();
  const docs = page.getByRole("textbox", { name: "Documentation input", exact: true });
  await expect(docs).toBeFocused();
  await expect(docs).toHaveValue(/^# Cache-aside/);

  page.once("dialog", (dialog) => dialog.accept());
  await prCard.click();
  const url = page.getByRole("textbox", { name: "GitHub pull request URL" });
  await expect(url).toBeFocused();
  await expect(url).toHaveValue("https://github.com/fastapi/fastapi/pull/9816");
  await expect(page.getByRole("textbox", { name: "Pull request description and diff" })).toBeEmpty();
  await expect(page.getByRole("button", { name: "Help me understand" })).toBeDisabled();
  expect(requests).toEqual([]);

  await page.getByRole("button", { name: "Load PR", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Import test response.");
  expect(requests).toEqual([{ endpoint: "/api/pull-request-plan", body: { url: await url.inputValue() } }]);
});
