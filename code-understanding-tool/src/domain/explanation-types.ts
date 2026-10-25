import type { SourceRange } from "./source-ranges";

export const MAX_FOLLOW_UP_HISTORY_MESSAGES = 20;

export type Mode = "code" | "docs";
export type ExplanationDepth = "overview" | "balanced" | "detailed";
export type LearnerLevel = "beginner" | "intermediate" | "advanced";

export type Explanation = {
  title: string;
  summary: string;
  concepts: { term: string; explanation: string; source_ranges: SourceRange[] }[];
  walkthrough: { heading: string; explanation: string; source_ranges: SourceRange[] }[];
  assumptions: { statement: string; reason: string }[];
  visual_note: string;
  quality_review_status?: "passed" | "revised" | "incomplete";
  quality_review_note?: string;
  visuals: {
    title: string;
    kind: "control_flow" | "concept_map";
    nodes: { id: string; label: string; description: string; source_ranges: SourceRange[] }[];
    edges: { from: string; to: string; label: string }[];
  }[];
};

export type Transcription = { transcription: string; uncertainties: { text: string; reason: string }[] };
export type FollowUpMessage =
  | { role: "learner"; text: string; selected_text?: string }
  | { role: "assistant"; text: string; source_ranges: SourceRange[]; assumptions: { statement: string; reason: string }[] };
export type FollowUpAnswer = { answer: string; source_ranges: SourceRange[]; assumptions: { statement: string; reason: string }[] };
