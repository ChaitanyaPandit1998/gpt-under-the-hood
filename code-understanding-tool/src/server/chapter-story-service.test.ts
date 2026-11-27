import { randomUUID } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { createReviewedChapterStory } from "./chapter-story-service";
import type { CodexRunner } from "./explanation-service";

const root = fileURLToPath(new URL("../../", import.meta.url));
const input = { source: "const capacity = 3;", context: "Explain the fixed capacity.", mode: "code" as const,
  learnerLevel: "intermediate" as const, depth: "detailed" as const };
const story = { title: "The ticket counter", setup: "A clerk counts the supply.",
  scenes: [{ heading: "Three tickets", narrative: "The clerk counts three tickets.", meaning: "The capacity is set to three.",
    source_ranges: [{ start_line: 1, end_line: 1 }] }], takeaway: "The supply is fixed." };
const pass = { status: "pass", findings: [] };

describe("skill-backed chapter story generation", () => {
  it("keeps two calls, independent review instructions, and the original response schema", async () => {
    const run = vi.fn<CodexRunner>().mockResolvedValueOnce(story).mockResolvedValueOnce(pass);
    const phases: string[] = [];
    const result = await createReviewedChapterStory(input, run, new AbortController().signal, (phase) => phases.push(phase), root);
    expect(result).toEqual(story);
    expect(run).toHaveBeenCalledTimes(2);
    expect(phases).toEqual(["drafting_story", "reviewing_story"]);
    const generation = run.mock.calls[0][0];
    const review = run.mock.calls[1][0];
    expect(generation).toContain("Application instruction: story-planning;");
    expect(generation).toContain("Application instruction: story-writing;");
    expect(generation).not.toContain("Application instruction: story-review;");
    expect(review).toContain("Application instruction: story-review;");
    expect(review).not.toContain("Application instruction: story-writing;");
    for (const [prompt] of run.mock.calls) {
      expect(prompt).toContain("Application instruction: audience-intermediate;");
      expect(prompt).toContain("Application instruction: depth-detailed;");
      expect(prompt).toContain("1 | const capacity = 3;");
    }
  });

  it("retains one repair and a final independent review", async () => {
    const run = vi.fn<CodexRunner>().mockResolvedValueOnce(story)
      .mockResolvedValueOnce({ status: "revise", findings: ["Clarify the mapping limit."] })
      .mockResolvedValueOnce(story).mockResolvedValueOnce(pass);
    const phases: string[] = [];
    await createReviewedChapterStory(input, run, new AbortController().signal, (phase) => phases.push(phase), root);
    expect(run).toHaveBeenCalledTimes(4);
    expect(phases).toEqual(["drafting_story", "reviewing_story", "repairing_story", "checking_story"]);
    expect(run.mock.calls[2][0]).toContain("Application instruction: story-writing;");
    expect(run.mock.calls[3][0]).not.toContain("Application instruction: story-writing;");
  });

  it("fails without a provider call when required skill files are missing", async () => {
    const run = vi.fn<CodexRunner>();
    await expect(createReviewedChapterStory(input, run, new AbortController().signal, () => {},
      join(tmpdir(), `missing-story-skills-${randomUUID()}`))).rejects.toThrow();
    expect(run).not.toHaveBeenCalled();
  });
});
