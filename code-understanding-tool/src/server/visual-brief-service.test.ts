import { describe, expect, it, vi } from "vitest";
import type { VisualBrief } from "../domain/visual-brief";
import { createVisualScene } from "./visual-brief-service";

const brief: VisualBrief = {
  disposition: "render", visualType: "conceptual-illustration", takeaway: "A generic teaching claim.",
  composition: "Keep uncertainty visible.", content: [], guardrails: ["Do not invent evidence."],
  caption: "An illustration.", altText: "A teaching illustration.",
};
const scene = { background: "#ffffff", items: [{ kind: "text", text: "Illustrative", x: 50, y: 80, size: 20, color: "#202020", align: "left" }] };

describe("drawing plan recovery", () => {
  it.each([false, true])("repairs invalid output once (compact: %s)", async (compact) => {
    const run = vi.fn().mockResolvedValueOnce({ items: [] }).mockResolvedValueOnce(scene);
    const signal = new AbortController().signal;
    await expect(createVisualScene(brief, run, signal, compact)).resolves.toEqual(scene);
    expect(run).toHaveBeenCalledTimes(2);
    expect(run.mock.calls[1][0]).toContain("Do not invent evidence.");
    expect(run.mock.calls[1][0]).toContain("failed structural validation");
    expect(run.mock.calls[1][5]).toBe(signal);
  });

  it("stops after one unsuccessful repair", async () => {
    const run = vi.fn().mockResolvedValue({});
    await expect(createVisualScene(brief, run, new AbortController().signal)).rejects.toThrow("invalid drawing");
    expect(run).toHaveBeenCalledTimes(2);
  });

  it("does not retry provider failures", async () => {
    const run = vi.fn().mockRejectedValue(new Error("Provider unavailable"));
    await expect(createVisualScene(brief, run, new AbortController().signal)).rejects.toThrow("Provider unavailable");
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("does not repair after cancellation", async () => {
    const controller = new AbortController();
    const run = vi.fn().mockImplementation(async () => { controller.abort(); return {}; });
    await expect(createVisualScene(brief, run, controller.signal)).rejects.toThrow();
    expect(run).toHaveBeenCalledTimes(1);
  });
});
