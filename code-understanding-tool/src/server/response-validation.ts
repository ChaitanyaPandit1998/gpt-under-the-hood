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
    throw new Error("The generated explanation was incomplete or invalid. Please retry the request.");
  }
  return value;
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
