import type { Explanation } from "./explanation-types";

export type ExplanationChapter = { id: string; title: string };

export function explanationChapters(explanation: Explanation): ExplanationChapter[] {
  return [
    { id: "overview", title: "Overview" },
    ...(explanation.concepts.length ? [{ id: "concepts", title: "Key ideas" }] : []),
    ...explanation.walkthrough.map((step, index) => ({ id: `step-${index}`, title: step.heading })),
    ...(explanation.visuals.length || explanation.visual_note ? [{ id: "visuals", title: "Maps and coverage" }] : []),
    { id: "recap", title: "Limits and understanding check" },
  ];
}

export function prefersChapters(explanation: Explanation): boolean {
  const text = [explanation.summary, ...explanation.concepts.map((item) => item.explanation),
    ...explanation.walkthrough.map((item) => item.explanation)].join(" ");
  return explanation.walkthrough.length >= 6 || text.trim().split(/\s+/).length >= 900;
}
