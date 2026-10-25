const sourceRanges = {
  type: "array",
  maxItems: 10,
  items: {
    type: "object",
    additionalProperties: false,
    required: ["start_line", "end_line"],
    properties: {
      start_line: { type: "integer", minimum: 1 },
      end_line: { type: "integer", minimum: 1 },
    },
  },
};

const assumptions = {
  type: "array",
  maxItems: 20,
  items: {
    type: "object",
    additionalProperties: false,
    required: ["statement", "reason"],
    properties: { statement: { type: "string", maxLength: 2_000 }, reason: { type: "string", maxLength: 2_000 } },
  },
};

export const explanationSchema = {
  type: "object",
  additionalProperties: false,
  required: ["title", "summary", "concepts", "walkthrough", "assumptions", "visual_note", "visuals"],
  properties: {
    title: { type: "string", maxLength: 300 },
    summary: { type: "string", maxLength: 8_000 },
    visual_note: { type: "string", maxLength: 2_000 },
    concepts: {
      type: "array", maxItems: 20,
      items: {
        type: "object", additionalProperties: false,
        required: ["term", "explanation", "source_ranges"],
        properties: { term: { type: "string", maxLength: 300 }, explanation: { type: "string", maxLength: 8_000 }, source_ranges: sourceRanges },
      },
    },
    walkthrough: {
      type: "array", maxItems: 40,
      items: {
        type: "object", additionalProperties: false,
        required: ["heading", "explanation", "source_ranges"],
        properties: { heading: { type: "string", maxLength: 300 }, explanation: { type: "string", maxLength: 8_000 }, source_ranges: sourceRanges },
      },
    },
    assumptions,
    visuals: {
      type: "array", maxItems: 5,
      items: {
        type: "object", additionalProperties: false,
        required: ["title", "kind", "nodes", "edges"],
        properties: {
          title: { type: "string", maxLength: 300 },
          kind: { type: "string", enum: ["control_flow", "concept_map"] },
          nodes: {
            type: "array", maxItems: 30,
            items: {
              type: "object", additionalProperties: false,
              required: ["id", "label", "description", "source_ranges"],
              properties: {
                id: { type: "string", maxLength: 120 }, label: { type: "string", maxLength: 300 },
                description: { type: "string", maxLength: 2_000 }, source_ranges: sourceRanges,
              },
            },
          },
          edges: {
            type: "array", maxItems: 60,
            items: {
              type: "object", additionalProperties: false,
              required: ["from", "to", "label"],
              properties: { from: { type: "string", maxLength: 120 }, to: { type: "string", maxLength: 120 }, label: { type: "string", maxLength: 300 } },
            },
          },
        },
      },
    },
  },
  $defs: { sourceRanges },
};

export const explanationReviewSchema = {
  type: "object",
  additionalProperties: false,
  required: ["status", "findings"],
  properties: {
    status: { type: "string", enum: ["pass", "revise", "uncertain"] },
    findings: { type: "array", maxItems: 8, items: { type: "string", maxLength: 600 } },
  },
};

export const transcriptionSchema = {
  type: "object", additionalProperties: false, required: ["transcription", "uncertainties"],
  properties: {
    transcription: { type: "string", maxLength: 200_000 },
    uncertainties: {
      type: "array", maxItems: 100,
      items: { type: "object", additionalProperties: false, required: ["text", "reason"], properties: { text: { type: "string", maxLength: 2_000 }, reason: { type: "string", maxLength: 2_000 } } },
    },
  },
};

export const followUpSchema = {
  type: "object", additionalProperties: false, required: ["answer", "source_ranges", "assumptions"],
  properties: { answer: { type: "string", maxLength: 20_000 }, source_ranges: sourceRanges, assumptions },
  $defs: { sourceRanges },
};

export const quizSchema = {
  type: "object", additionalProperties: false, required: ["questions"],
  properties: {
    questions: {
      type: "array", minItems: 3, maxItems: 3,
      items: {
        type: "object", additionalProperties: false,
        required: ["prompt", "options", "correct_index", "explanation"],
        properties: {
          prompt: { type: "string", maxLength: 2_000 },
          options: { type: "array", minItems: 4, maxItems: 4, items: { type: "string", maxLength: 1_000 } },
          correct_index: { type: "integer", minimum: 0, maximum: 3 },
          explanation: { type: "string", maxLength: 4_000 },
        },
      },
    },
  },
};
