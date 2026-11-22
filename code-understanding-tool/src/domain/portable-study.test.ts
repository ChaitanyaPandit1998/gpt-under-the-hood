import { describe, expect, it } from "vitest";
import { exportPortableStudy, importPortableStudy, portableStudyLimit } from "./portable-study";
import type { SavedStudy } from "./saved-study";

const png = new Blob([Uint8Array.from(atob("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="), c => c.charCodeAt(0))], { type: "image/png" });
const illustration = { png, alt: "Example", caption: "Fictional analogy", resolution: "standard" as const };
const study: SavedStudy = {
  version: 1, mode: "docs", source: "A generic concept.", depth: "balanced", level: "beginner",
  explanation: { title: "Concept", summary: "An overview.", concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "" },
  reading: { chapter: { chapterMode: true, chapterId: "overview", images: {}, activeStoryKey: "story",
    stories: { story: { story: { title: "A journey", setup: "A fictional guide.", takeaway: "A generic lesson.", scenes: [{ heading: "Start", narrative: "A guide arrives.", meaning: "An introduction.", source_ranges: [] }] }, images: { 0: { "story-illustration": illustration } } } } } },
};

describe("portable studies", () => {
  it("round trips a story and native PNG blobs", async () => {
    const restored = await importPortableStudy(await exportPortableStudy(study));
    expect(restored).toEqual(study);
    if (restored.mode === "pull_request") throw new Error("Wrong mode");
    const image = restored.reading.chapter?.stories?.story.images?.[0]["story-illustration"];
    expect(await image?.png.arrayBuffer()).toEqual(await png.arrayBuffer());
  });
  it.each([
    { format: "other", version: 1, study },
    { format: "understand-study", version: 2, study },
    { format: "understand-study", version: 1, study: { ...study, mode: "invalid" } },
    { format: "understand-study", version: 1, study: { ...study, reading: { chapter: { chapterMode: true, chapterId: "x", images: [] } } } },
    { format: "understand-study", version: 1, study: { ...study, reading: { illustration: { ...illustration, png: { $png: btoa("not a png") } } } } },
  ])("rejects invalid envelopes or content", async envelope => {
    await expect(importPortableStudy(new Blob([JSON.stringify(envelope)]))).rejects.toThrow();
  });
  it("rejects dangerous object keys", async () => {
    await expect(importPortableStudy(new Blob(['{"format":"understand-study","version":1,"study":{"__proto__":{}}}']))).rejects.toThrow();
  });
  it("rejects excessive files before reading", async () => {
    const text = () => { throw new Error("must not read"); };
    await expect(importPortableStudy({ size: portableStudyLimit + 1, text } as unknown as Blob)).rejects.toThrow("40 MB");
  });
  it("rejects PR studies explicitly", async () => {
    await expect(importPortableStudy(new Blob([JSON.stringify({ format: "understand-study", version: 1, study: { version: 1, mode: "pull_request" } })]))).rejects.toThrow("PR studies");
  });
});
