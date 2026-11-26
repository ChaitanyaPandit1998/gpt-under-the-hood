import type { Page } from "@playwright/test";

export function studyActions(page: Page) {
  return page.getByRole("group", { name: "Explanation tools", exact: true }).locator(":scope > details.study-actions");
}

export async function openStudyActions(page: Page) {
  const actions = studyActions(page);
  const trigger = actions.locator(":scope > summary");
  await trigger.waitFor();
  if (await actions.getAttribute("open") === null) await trigger.click();
  return actions;
}
