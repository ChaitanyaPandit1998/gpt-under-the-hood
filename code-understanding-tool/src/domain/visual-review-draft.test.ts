import { expect, it } from "vitest";
import { createVisualReviewDraft, contractFromVisualDraft } from "./visual-review-draft";
import { validateVisualContract } from "./visual-evaluation";
import type { VisualBrief } from "./visual-brief";

const brief: VisualBrief = { disposition: "render", visualType: "concept-map", takeaway: "Show the limit.", composition: "One label.",
  caption: "An illustrative limit.", altText: "A limit label.", guardrails: ["Not a runtime trace."],
  content: [{ label: "Limit", role: "background", statement: "The limit is fixed.", chapterLines: [] },
    { label: "Ribbon", role: "decorative", statement: "A decorative ribbon.", chapterLines: [] }] };
it("creates an editable checklist without silently assigning an unknown audience", () => {
  const draft = createVisualReviewDraft(brief, "auto");
  expect(draft.audience).toBe("");
  expect(draft.requirements.map((item) => item.enabled)).toEqual([true, false, true]);
  expect(draft.requirements.every((item) => item.evidenceReference.includes("brief"))).toBe(true);
  expect(() => contractFromVisualDraft(draft, "id", "a".repeat(64))).toThrow("audience");
});
it("preserves explicit audience/purpose and content placement", () => {
  const draft = createVisualReviewDraft(brief, "advanced", "story-illustration");
  draft.requirements[0].location = "Caption";
  const contract = validateVisualContract(contractFromVisualDraft(draft, "id", "a".repeat(64)));
  expect(contract).toMatchObject({ audience: "advanced", purpose: "story-illustration" });
  expect(contract.requirements).toHaveLength(2);
  expect(contract.requirements[0].statement).toBe("Caption: The limit is fixed.");
  expect(contract.requirements[0]).not.toHaveProperty("enabled");
});
it("does not mutate the brief when the user edits the checklist", () => {
  const draft = createVisualReviewDraft(brief, "beginner");
  draft.requirements[0].statement = "Another requirement.";
  expect(brief.content[0].statement).toBe("The limit is fixed.");
});
it("does not hide an empty requirement behind its placement prefix", () => {
  const draft = createVisualReviewDraft(brief, "beginner");
  draft.requirements[0].statement = " ";
  expect(() => contractFromVisualDraft(draft, "id", "a".repeat(64))).toThrow("Write or deselect");
});
