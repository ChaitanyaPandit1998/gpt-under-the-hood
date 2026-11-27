import type { SceneVisualPurpose, VisualBrief, VisualPreferences } from "./visual-brief";
import type { VisualContract } from "./visual-evaluation";

export const reviewAudienceChoices = [
  { value: "beginner", title: "Beginner", description: "Understand the idea without losing important qualifiers." },
  { value: "intermediate", title: "Intermediate", description: "Explain relationships, mechanisms and assumptions." },
  { value: "advanced", title: "Advanced", description: "Use precise notation, domains and limitations where relevant." },
] as const;
export const reviewLocations = ["Image", "Caption", "Image and caption", "Surrounding explanation"] as const;
export type VisualReviewDraft = {
  audience: VisualContract["audience"] | "";
  purpose: SceneVisualPurpose;
  teachingGoal: string;
  audienceObjective: string;
  requirements: (VisualContract["requirements"][number] & { enabled: boolean; location: typeof reviewLocations[number] })[];
};

export function createVisualReviewDraft(brief: VisualBrief, audience: VisualPreferences["audience"], purpose?: SceneVisualPurpose): VisualReviewDraft {
  return {
    audience: audience === "auto" ? "" : audience, purpose: purpose ?? "factual-diagram",
    teachingGoal: brief.takeaway,
    audienceObjective: "Understand this visual's teaching goal at the selected level, with its caption, without mistaking an analogy or schematic for observed evidence.",
    requirements: [
      ...brief.content.map((item, index) => ({ id: `content-${index + 1}`, kind: "claim" as const, statement: item.statement,
        evidence: item.role === "source" ? "source" as const : item.role === "illustrative" || item.role === "decorative" ? "illustrative" as const : "background" as const,
        evidenceReference: `Drafted from visual brief item ${index + 1}; independently compare with the supplied explanation.`,
        enabled: item.role !== "decorative", location: "Image and caption" as const })),
      ...brief.guardrails.map((statement, index) => ({ id: `guardrail-${index + 1}`, kind: "qualifier" as const, statement,
        evidence: "background" as const, evidenceReference: `Drafted from visual brief guardrail ${index + 1}; not independently verified.`,
        enabled: true, location: "Image and caption" as const })),
    ],
  };
}

export function contractFromVisualDraft(draft: VisualReviewDraft, id: string, explanationSha256: string): VisualContract {
  if (!draft.audience) throw new Error("Select a review audience.");
  if (draft.requirements.some((item) => item.enabled && !item.statement.trim())) throw new Error("Write or deselect each included requirement.");
  return { version: 1, id, explanationSha256, audience: draft.audience, purpose: draft.purpose,
    teachingGoal: draft.teachingGoal, audienceObjectives: [draft.audienceObjective],
    requirements: draft.requirements.filter((item) => item.enabled).map((item) => ({
      id: item.id, kind: item.kind, evidence: item.evidence, evidenceReference: item.evidenceReference,
      statement: `${item.location}: ${item.statement}`,
    })),
  };
}
