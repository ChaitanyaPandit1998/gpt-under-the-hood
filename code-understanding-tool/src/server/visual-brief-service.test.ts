import { describe, expect, it, vi } from "vitest";
import { fileURLToPath } from "node:url";
import { defaultVisualPreferences, type VisualBrief } from "../domain/visual-brief";
import { createVisualBrief, createVisualScene } from "./visual-brief-service";
import { generationStage, type ApplicationInstruction } from "./application-instructions";
import { validateGenerationStage } from "../domain/visual-provenance";

const root = fileURLToPath(new URL("../../", import.meta.url));

const brief: VisualBrief = {
  disposition: "render", visualType: "conceptual-illustration", takeaway: "A generic teaching claim.",
  composition: "Keep uncertainty visible.", content: [], guardrails: ["Do not invent evidence."],
  caption: "An illustration.", altText: "A teaching illustration.",
};
const scene = { background: "#ffffff", items: [{ kind: "text", text: "Illustrative", x: 50, y: 80, size: 20, color: "#202020", align: "left" }] };

describe("drawing plan recovery", () => {
  it("keeps captions outside PNGs and limits labels without topic-specific instructions", async () => {
    const run = vi.fn().mockResolvedValue(scene);
    await createVisualScene(brief, root, run, new AbortController().signal);
    expect(run.mock.calls[0][0]).toContain("Never copy the caption");
    expect(run.mock.calls[0][0]).toContain("40 words total");
  });
  it.each([false, true])("repairs invalid output once (compact: %s)", async (compact) => {
    const run = vi.fn().mockResolvedValueOnce({ items: [] }).mockResolvedValueOnce(scene);
    const signal = new AbortController().signal;
    await expect(createVisualScene(brief, root, run, signal, compact)).resolves.toEqual(scene);
    expect(run).toHaveBeenCalledTimes(2);
    expect(run.mock.calls[1][0]).toContain("Do not invent evidence.");
    expect(run.mock.calls[1][0]).toContain("failed structural validation");
    expect(run.mock.calls[1][5]).toBe(signal);
  });

  it("stops after one unsuccessful repair", async () => {
    const run = vi.fn().mockResolvedValue({});
    await expect(createVisualScene(brief, root, run, new AbortController().signal)).rejects.toThrow("invalid drawing");
    expect(run).toHaveBeenCalledTimes(2);
  });

  it("repairs a missing required qualifier and checks labels after compact expansion", async () => {
    const packed = { ...scene, items: [...scene.items, { kind: "text-run", labels: ["Unproved", "x < 1"], x: 60, y: 120,
      dx: 0, dy: 40, size: 20, color: "#202020", align: "left" }] };
    const run = vi.fn().mockResolvedValueOnce(scene).mockResolvedValueOnce(packed);
    const result = await createVisualScene({ ...brief, requiredLabels: ["Unproved", "x < 1"] }, root, run, new AbortController().signal, true);
    expect(run).toHaveBeenCalledTimes(2);
    expect(run.mock.calls[1][0]).toContain('Missing labels (literal data): ["Unproved","x < 1"]');
    expect(result.items.slice(1)).toMatchObject([{ text: "Unproved" }, { text: "x < 1" }]);
  });

  it("does not retry provider failures", async () => {
    const run = vi.fn().mockRejectedValue(new Error("Provider unavailable"));
    await expect(createVisualScene(brief, root, run, new AbortController().signal)).rejects.toThrow("Provider unavailable");
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("does not repair after cancellation", async () => {
    const controller = new AbortController();
    const run = vi.fn().mockImplementation(async () => { controller.abort(); return {}; });
    await expect(createVisualScene(brief, root, run, controller.signal)).rejects.toThrow();
    expect(run).toHaveBeenCalledTimes(1);
  });
});

describe("skill-backed visual planning", () => {
  it("returns lineage for the exact resources used by a repaired drawing", async () => {
    let resources: ApplicationInstruction[] = [];
    const run = vi.fn().mockResolvedValueOnce({}).mockResolvedValueOnce(scene);
    const result = await createVisualScene(brief, root, run, new AbortController().signal, true, loaded => { resources = loaded; });
    expect(resources.map(({ id }) => id)).toEqual(["visual-composition", "visual-composition-compact"]);
    const lineage = generationStage(result, resources);
    await expect(validateGenerationStage(lineage, JSON.stringify(result))).resolves.toEqual(lineage);
    for (const resource of resources) {
      for (const [prompt] of run.mock.calls) expect(prompt).toContain(`sha256=${resource.sha256}`);
    }
    expect(lineage.instructions[0]).not.toHaveProperty("text");
  });
  it.each([false, true])("loads compact instructions only for the selected schema (compact: %s)", async (compact) => {
    const run = vi.fn().mockResolvedValue(scene);
    await createVisualScene(brief, root, run, new AbortController().signal, compact);
    expect(run).toHaveBeenCalledTimes(1);
    const prompt = run.mock.calls[0][0] as string;
    expect(prompt).toContain("Application instruction: visual-composition;");
    expect(prompt.includes("Application instruction: visual-composition-compact;")).toBe(compact);
    expect(prompt).toContain(JSON.stringify(brief));
    expect(prompt).not.toContain("Application instruction: story-writing;");
  });

  it.each(["auto", "advanced"] as const)("reuses visual direction with the %s audience", async (audience) => {
    const output: VisualBrief = { ...brief, content: [{ label: "Input", role: "illustrative", statement: "A supplied teaching input.", chapterLines: [] }] };
    const run = vi.fn().mockResolvedValue(output);
    const result = await createVisualBrief({ context: "Explain the supplied relationship.",
      preferences: { ...defaultVisualPreferences, audience } }, root, run, new AbortController().signal);
    expect(result).toEqual(output);
    expect(run).toHaveBeenCalledTimes(1);
    const prompt = run.mock.calls[0][0] as string;
    expect(prompt).toContain("Application instruction: visual-direction;");
    expect(prompt.includes("Application instruction: audience-advanced;")).toBe(audience === "advanced");
    expect(prompt).not.toContain("Application instruction: audience-beginner;");
    expect(prompt).not.toContain("Application instruction: visual-composition;");
  });

  it("fails without a provider call when composition guidance is unavailable", async () => {
    const run = vi.fn();
    await expect(createVisualScene(brief, `${root}/missing-application-root`, run, new AbortController().signal)).rejects.toThrow();
    expect(run).not.toHaveBeenCalled();
  });
});
