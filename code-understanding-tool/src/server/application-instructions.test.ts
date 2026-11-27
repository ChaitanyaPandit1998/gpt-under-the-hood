import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { loadApplicationInstruction, loadStoryInstructions } from "./application-instructions";

const root = fileURLToPath(new URL("../../", import.meta.url));

describe("application instruction loader", () => {
  it("loads only the selected profiles and binds exact instruction text", async () => {
    const bundle = await loadStoryInstructions(root, "advanced", "overview");
    expect(bundle.resources.map(({ id }) => id)).toEqual([
      "story-planning", "story-writing", "story-review", "audience-advanced", "depth-overview",
    ]);
    for (const resource of bundle.resources) {
      expect(resource.version).toBe(1);
      expect(resource.sha256).toBe(createHash("sha256").update(resource.text).digest("hex"));
    }
    expect(bundle.generation).not.toContain(bundle.resources[2].text);
    expect(bundle.review).not.toContain(bundle.resources[0].text);
    expect(bundle.review).not.toContain(bundle.resources[1].text);
  });

  it.each(["../../package.json", "toString", "constructor"])("rejects unregistered instruction ID %s", async (id) => {
    await expect(loadApplicationInstruction(root, id as Parameters<typeof loadApplicationInstruction>[1]))
      .rejects.toThrow("Unknown application instruction");
  });

  it("does not load instructions after cancellation", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(loadStoryInstructions(root, "beginner", "balanced", controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  });
});
