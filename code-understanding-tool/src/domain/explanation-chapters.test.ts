import { describe, expect, it } from "vitest";
import type { Explanation } from "./explanation-types";
import { explanationChapters, prefersChapters } from "./explanation-chapters";

const shortExplanation: Explanation = {
  title: "A small change", summary: "Validates a request.",
  concepts: [], walkthrough: [], visuals: [], visual_note: "", assumptions: [],
};

describe("explanation chapters", () => {
  it("keeps short explanations in full view by default", () => {
    expect(prefersChapters(shortExplanation)).toBe(false);
    expect(explanationChapters(shortExplanation).map((chapter) => chapter.id)).toEqual(["overview", "recap"]);
  });

  it("builds a deterministic outline without empty optional chapters", () => {
    const explanation: Explanation = { ...shortExplanation,
      concepts: [{ term: "Validation", explanation: "Checks the input", source_ranges: [] }],
      walkthrough: [{ heading: "Validate the request", explanation: "Reject bad requests", source_ranges: [] },
        { heading: "Return the response", explanation: "Return success", source_ranges: [] }],
      visual_note: "No map was generated.",
    };
    expect(explanationChapters(explanation)).toEqual([
      { id: "overview", title: "Overview" }, { id: "concepts", title: "Key ideas" },
      { id: "step-0", title: "Validate the request" }, { id: "step-1", title: "Return the response" },
      { id: "visuals", title: "Maps and coverage" }, { id: "recap", title: "Limits and understanding check" },
    ]);
  });

  it("defaults detailed walkthroughs and long prose to chapters", () => {
    expect(prefersChapters({ ...shortExplanation, walkthrough: Array.from({ length: 6 }, () => ({ heading: "Step", explanation: "Details", source_ranges: [] })) })).toBe(true);
    expect(prefersChapters({ ...shortExplanation, summary: "word ".repeat(900) })).toBe(true);
  });

  it("gives duplicate step headings unique chapter IDs", () => {
    const chapters = explanationChapters({ ...shortExplanation, walkthrough: Array.from({ length: 2 }, () => ({ heading: "Same", explanation: "Details", source_ranges: [] })) });
    expect(new Set(chapters.map((chapter) => chapter.id)).size).toBe(chapters.length);
  });
});
