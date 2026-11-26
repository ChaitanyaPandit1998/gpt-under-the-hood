import { expect, test } from "@playwright/test";
import { openStudyActions } from "./support/study-actions";
import { makeVisualPng } from "../src/test-fixtures/visual-png";

test("restores a legacy full-view study in chapters without losing illustrations", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async (bytes) => {
    const png = new Blob([new Uint8Array(bytes)], { type: "image/png" });
    const image = { png, caption: "An illustration for the whole study.", alt: "Whole-study illustration", resolution: "standard" };
    const study = {
      version: 1, mode: "code", source: "const limit = 3;", depth: "balanced", level: "intermediate",
      explanation: {
        title: "A configured limit", summary: "The limit is fixed.", concepts: [],
        walkthrough: [{ heading: "Set the limit", explanation: "Assign three to the limit.", source_ranges: [] }],
        assumptions: [], visuals: [], visual_note: "",
      },
      reading: { illustration: image, chapter: {
        chapterMode: false, chapterId: "step-0", images: {
          "step-0": { ...image, caption: "An illustration for the selected chapter.", alt: "Chapter illustration" },
        },
      } },
    };
    await new Promise<void>((resolve, reject) => {
      const open = indexedDB.open("understand-local-study", 1);
      open.onupgradeneeded = () => open.result.createObjectStore("snapshots");
      open.onerror = () => reject(open.error);
      open.onsuccess = () => {
        const db = open.result;
        const tx = db.transaction("snapshots", "readwrite");
        tx.objectStore("snapshots").put(study, "current");
        tx.oncomplete = () => { db.close(); resolve(); };
        tx.onabort = () => { db.close(); reject(tx.error); };
      };
    });
  }, Array.from(makeVisualPng()));
  await page.reload();
  await page.getByRole("button", { name: "Resume saved study", exact: true }).click();
  const selected = page.getByRole("region", { name: "Set the limit", exact: true });
  await expect(selected.getByText("Assign three to the limit.", { exact: true })).toBeVisible();
  await expect(selected.getByRole("img", { name: "Chapter illustration", exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: "Whole-study illustration", exact: true })).toBeVisible();
  await expect(page.locator(".illustration-labels")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Chapters", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Full explanation", exact: true })).toHaveCount(0);
  await page.getByRole("navigation", { name: "Explanation chapters", exact: true })
    .getByRole("button", { name: /Overview/ }).click();
  await expect(page.getByText("The limit is fixed.", { exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: "Chapter illustration", exact: true })).not.toBeVisible();
  await expect(page.getByRole("img", { name: "Whole-study illustration", exact: true })).toBeVisible();
  page.once("dialog", (dialog) => dialog.accept());
  await (await openStudyActions(page)).getByRole("button", { name: "Save study locally", exact: true }).click();
  await expect(page.getByText("Study saved in this browser.", { exact: true })).toBeVisible();
  const reading = await page.evaluate(() => new Promise<{ chapterMode: boolean; chapterId: string }>((resolve, reject) => {
    const open = indexedDB.open("understand-local-study", 1);
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const db = open.result;
      const tx = db.transaction("snapshots", "readonly");
      const result = tx.objectStore("snapshots").get("current");
      tx.oncomplete = () => { db.close(); resolve(result.result.reading.chapter); };
      tx.onabort = () => { db.close(); reject(tx.error); };
    };
  }));
  expect(reading).toMatchObject({ chapterMode: true, chapterId: "overview" });
});
