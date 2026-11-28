import { describe, expect, it } from "vitest";
import { defaultVisualPreferences, parseVisualBriefRequest, validateVisualBrief } from "./visual-brief";

const brief = { disposition: "render", visualType: "heatmap", takeaway: "Read one row.", composition: "Draw a 3 by 3 grid.",
  caption: "Toy weights.", altText: "A grid of toy weights.", guardrails: [],
  content: [{ label: "Weights", role: "illustrative", statement: "Toy data.", chapterLines: [] }] };

describe("visual brief contract", () => {
  it("accepts valid illustration and text-only briefs", () => {
    expect(validateVisualBrief(brief)).toEqual(brief);
    expect(validateVisualBrief({ ...brief, disposition: "text-only", visualType: "none", content: [] }).disposition).toBe("text-only");
  });
  it("accepts bounded exact labels without changing legacy briefs", () => {
    const requiredLabels = ["Unproved", "x < 1", "Elapsed time (ms)"];
    const extended = { ...brief, requiredLabels };
    expect(validateVisualBrief(extended)).toEqual(extended);
    for (const invalid of [null, [" "], [" line"], ["line\nbreak"], ["x".repeat(61)], ["Caf\u00e9", "Cafe\u0301"], Array(9).fill("x")]) {
      expect(() => validateVisualBrief({ ...brief, requiredLabels: invalid })).toThrow("Invalid required visual labels");
    }
    expect(() => validateVisualBrief({ ...extended, disposition: "text-only", visualType: "none" })).toThrow("text-only brief cannot require");
  });
  it("rejects invalid shapes, source coordinates and executable markup", () => {
    expect(() => validateVisualBrief({ ...brief, operations: [] })).toThrow();
    expect(() => validateVisualBrief({ ...brief, disposition: "text-only" })).toThrow();
    expect(() => validateVisualBrief({ ...brief, composition: "<svg onload='x'>" })).toThrow();
    const sourced = { ...brief, content: [{ ...brief.content[0], role: "source", chapterLines: [5] }] };
    expect(() => validateVisualBrief(sourced, 4)).toThrow();
    expect(() => validateVisualBrief(sourced, 5, new Set([4]))).toThrow();
    expect(validateVisualBrief(sourced, 5)).toEqual(sourced);
  });
  it("rejects unsupported or oversized preferences and drops extra fields", () => {
    const body = { context: "Explanation", preferences: defaultVisualPreferences };
    expect(() => parseVisualBriefRequest({ ...body, source: "Private snippet" })).toThrow();
    expect(() => parseVisualBriefRequest({ ...body, preferences: { ...defaultVisualPreferences, style: "other" } })).toThrow();
    expect(() => parseVisualBriefRequest({ ...body, preferences: { ...defaultVisualPreferences, direction: "x".repeat(801) } })).toThrow();
    expect(() => parseVisualBriefRequest({ ...body, preferences: { ...defaultVisualPreferences, additionalContext: "x".repeat(4001) } })).toThrow();
    expect(parseVisualBriefRequest({ ...body, preferences: { ...defaultVisualPreferences, ignored: "x" } }).preferences).toEqual(defaultVisualPreferences);
  });
  it("preserves arbitrary learner data without requiring a source or a preset form", () => {
    const additionalContext = "Measured rise: 2 cm at 5 min; 7 cm at 20 min.";
    const preferences = { ...defaultVisualPreferences, additionalContext, direction: "Use an annotated comparison, not a chart." };
    expect(parseVisualBriefRequest({ context: "Unfamiliar experiment", preferences })).toEqual({
      context: "Unfamiliar experiment", preferences, source: undefined,
    });
  });
});
