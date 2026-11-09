import type { Explanation, FollowUpAnswer, Transcription } from "../domain/explanation-types.ts";

type Quiz = { questions: { prompt: string; options: string[]; correct_index: number; explanation: string }[] };
export type ExplanationReview = { status: "pass" | "revise" | "uncertain"; findings: string[] };
type RecordValue = Record<string, unknown>;

function isRecord(value: unknown): value is RecordValue {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isText(value: unknown, limit: number, allowEmpty = true): value is string {
  return typeof value === "string" && value.length <= limit && (allowEmpty || value.trim().length > 0);
}

function isArray(value: unknown, limit: number, validate: (item: unknown) => boolean): value is unknown[] {
  return Array.isArray(value) && value.length <= limit && value.every(validate);
}

function isSourceRanges(value: unknown, lineCount: number): boolean {
  return isArray(value, 10, (range) => isRecord(range)
    && Number.isInteger(range.start_line) && Number.isInteger(range.end_line)
    && Number(range.start_line) >= 1 && Number(range.end_line) >= Number(range.start_line)
    && Number(range.end_line) <= lineCount);
}

function sourceRangeFailure(value: unknown, lineCount: number): string | undefined {
  if (!Array.isArray(value) || value.length > 10) return "source_ranges_shape";
  for (const range of value) {
    if (!isRecord(range) || !Number.isInteger(range.start_line) || !Number.isInteger(range.end_line)
      || Number(range.start_line) < 1 || Number(range.end_line) < Number(range.start_line)) return "source_range_invalid";
    if (Number(range.end_line) > lineCount) return "source_range_out_of_bounds";
  }
  return undefined;
}

function isAssumptions(value: unknown): boolean {
  return isArray(value, 20, (item) => isRecord(item)
    && isText(item.statement, 2_000) && isText(item.reason, 2_000));
}

function isExplanation(value: unknown, lineCount: number): value is Explanation {
  if (!isRecord(value) || !isText(value.title, 300, false) || !isText(value.summary, 8_000)
    || !isText(value.visual_note, 2_000) || !isAssumptions(value.assumptions)) return false;

  const conceptsValid = isArray(value.concepts, 20, (item) => isRecord(item)
    && isText(item.term, 300, false) && isText(item.explanation, 8_000)
    && isSourceRanges(item.source_ranges, lineCount));
  const walkthroughValid = isArray(value.walkthrough, 40, (item) => isRecord(item)
    && isText(item.heading, 300, false) && isText(item.explanation, 8_000)
    && isSourceRanges(item.source_ranges, lineCount));
  const visualsValid = isArray(value.visuals, 5, (visual) => {
    if (!isRecord(visual) || !isText(visual.title, 300, false)
      || !["control_flow", "concept_map"].includes(String(visual.kind))) return false;
    if (!isArray(visual.nodes, 30, (node) => isRecord(node)
      && isText(node.id, 120, false) && isText(node.label, 300, false)
      && isText(node.description, 2_000) && isSourceRanges(node.source_ranges, lineCount))) return false;
    if (!isArray(visual.edges, 60, (edge) => isRecord(edge)
      && isText(edge.from, 120, false) && isText(edge.to, 120, false) && isText(edge.label, 300))) return false;
    const nodeIds = new Set(visual.nodes.map((node) => (node as RecordValue).id));
    return nodeIds.size === visual.nodes.length && visual.edges.every((edge) => {
      const item = edge as RecordValue;
      return nodeIds.has(item.from) && nodeIds.has(item.to);
    });
  });
  return conceptsValid && walkthroughValid && visualsValid;
}

export function validateExplanation(value: unknown, sourceText: string): Explanation {
  const lineCount = sourceText.split(/\r\n|\r|\n/).length;
  if (!isExplanation(value, lineCount)) {
    throw new ExplanationValidationError(explanationValidationCategory(value, lineCount));
  }
  return value;
}

/** Safe failure labels for logs: never include model text, prompts, or source content. */
export class ExplanationValidationError extends Error {
  readonly failureCategory: string;

  constructor(failureCategory: string) {
    super("The generated explanation was incomplete or invalid. Please retry the request.");
    this.name = "ExplanationValidationError";
    this.failureCategory = failureCategory;
  }
}

function explanationValidationCategory(value: unknown, lineCount: number): string {
  if (!isRecord(value)) return "explanation_root_invalid";
  if (!isText(value.title, 300, false) || !isText(value.summary, 8_000)
    || !isText(value.visual_note, 2_000) || !isAssumptions(value.assumptions)) return "explanation_top_level_fields_invalid";
  if (!Array.isArray(value.concepts) || value.concepts.length > 20) return "explanation_concepts_invalid";
  for (const concept of value.concepts) {
    if (!isRecord(concept) || !isText(concept.term, 300, false) || !isText(concept.explanation, 8_000)) return "explanation_concept_invalid";
    const rangesFailure = sourceRangeFailure(concept.source_ranges, lineCount);
    if (rangesFailure) return `explanation_concept_${rangesFailure}`;
  }
  if (!Array.isArray(value.walkthrough) || value.walkthrough.length > 40) return "explanation_walkthrough_invalid";
  for (const step of value.walkthrough) {
    if (!isRecord(step) || !isText(step.heading, 300, false) || !isText(step.explanation, 8_000)) return "explanation_walkthrough_step_invalid";
    const rangesFailure = sourceRangeFailure(step.source_ranges, lineCount);
    if (rangesFailure) return `explanation_walkthrough_${rangesFailure}`;
  }
  if (!Array.isArray(value.visuals) || value.visuals.length > 5) return "explanation_visuals_invalid";
  for (const visual of value.visuals) {
    if (!isRecord(visual) || !isText(visual.title, 300, false)
      || !["control_flow", "concept_map"].includes(String(visual.kind))) return "explanation_visual_invalid";
    if (!Array.isArray(visual.nodes) || visual.nodes.length > 30) return "explanation_visual_nodes_invalid";
    for (const node of visual.nodes) {
      if (!isRecord(node) || !isText(node.id, 120, false) || !isText(node.label, 300, false)
        || !isText(node.description, 2_000)) return "explanation_visual_node_invalid";
      const rangesFailure = sourceRangeFailure(node.source_ranges, lineCount);
      if (rangesFailure) return `explanation_visual_node_${rangesFailure}`;
    }
    if (!Array.isArray(visual.edges) || visual.edges.length > 60) return "explanation_visual_edges_invalid";
    const nodeIds = new Set(visual.nodes.map((node) => (node as RecordValue).id));
    if (nodeIds.size !== visual.nodes.length) return "explanation_visual_node_ids_duplicate";
    for (const edge of visual.edges) {
      if (!isRecord(edge) || !isText(edge.from, 120, false) || !isText(edge.to, 120, false) || !isText(edge.label, 300)) return "explanation_visual_edge_invalid";
      if (!nodeIds.has(edge.from) || !nodeIds.has(edge.to)) return "explanation_visual_edge_reference_invalid";
    }
  }
  return "explanation_schema_invalid";
}

export function validateExplanationReview(value: unknown): ExplanationReview {
  if (!isRecord(value) || !["pass", "revise", "uncertain"].includes(String(value.status))
    || !isArray(value.findings, 8, (finding) => isText(finding, 600, false))) {
    throw new Error("The second-pass review returned an incomplete result.");
  }
  if (value.status === "pass" && value.findings.length > 0) {
    throw new Error("The second-pass review returned inconsistent findings.");
  }
  if (value.status === "revise" && value.findings.length === 0) {
    throw new Error("The second-pass review requested a revision without identifying an issue.");
  }
  return value as ExplanationReview;
}

export function validateTranscription(value: unknown): Transcription {
  if (!isRecord(value) || !isText(value.transcription, 200_000)
    || !isArray(value.uncertainties, 100, (item) => isRecord(item)
      && isText(item.text, 2_000) && isText(item.reason, 2_000))) {
    throw new Error("The screenshot transcription was incomplete or invalid. Please retry.");
  }
  return value as Transcription;
}

export function validateFollowUp(value: unknown, sourceText: string): FollowUpAnswer {
  const lineCount = sourceText.split(/\r\n|\r|\n/).length;
  if (!isRecord(value) || !isText(value.answer, 20_000, false)
    || !isSourceRanges(value.source_ranges, lineCount) || !isAssumptions(value.assumptions)) {
    throw new Error("The generated follow-up was incomplete or invalid. Please retry.");
  }
  return value as FollowUpAnswer;
}

export function validateQuiz(value: unknown): Quiz {
  if (!isRecord(value) || !Array.isArray(value.questions) || value.questions.length !== 3
    || !value.questions.every((question) => isRecord(question)
      && isText(question.prompt, 2_000, false)
      && Array.isArray(question.options) && question.options.length === 4
      && question.options.every((option) => isText(option, 1_000, false))
      && Number.isInteger(question.correct_index) && Number(question.correct_index) >= 0 && Number(question.correct_index) <= 3
      && isText(question.explanation, 4_000, false))) {
    throw new Error("The generated knowledge check was incomplete or invalid. Please retry.");
  }
  return value as Quiz;
}
