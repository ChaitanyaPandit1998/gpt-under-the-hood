export const visualForms = [
  ["coordinate-figure", "Coordinate figure"], ["decision-flow", "Decision flow"],
  ["heatmap", "Heatmap"], ["timeline", "Timeline"], ["comparison", "Side-by-side comparison"],
  ["cutaway", "Cutaway"], ["concept-map", "Concept map"], ["conceptual-illustration", "Conceptual illustration"],
] as const;
export const visualPreferenceOptions = {
  form: [["auto", "Choose for me"], ...visualForms],
  audience: [["auto", "Match explanation"], ["beginner", "Beginner"], ["intermediate", "Intermediate"], ["advanced", "Advanced"]],
  style: [["editorial", "Clean editorial"], ["technical", "Technical schematic"], ["expressive", "Expressive illustration"]],
  target: [["screen", "Screen"], ["print", "PDF page"], ["both", "Screen and PDF"]],
} as const;

export type VisualPreferences = {
  [K in keyof typeof visualPreferenceOptions]: typeof visualPreferenceOptions[K][number][0];
} & { question: string; direction: string; additionalContext: string; includeSource: boolean };
export const visualInputLimits = { question: 500, direction: 800, additionalContext: 4_000, prompt: 8_000, source: 200_000, context: 100_000 } as const;
export const defaultVisualPreferences: VisualPreferences = {
  form: "auto", audience: "auto", style: "editorial", target: "both", question: "", direction: "", additionalContext: "", includeSource: false,
};

export type SceneVisualPurpose = "story-illustration" | "factual-diagram";
export const sceneVisualGuidance: Record<SceneVisualPurpose, string> = {
  "story-illustration": "Depict an explicitly fictional moment from the narrative. Label it as an illustrative analogy, not data, proof, observed behaviour, or a runtime trace. Keep any factual qualifiers in the scene's mapping.",
  "factual-diagram": "Use only the scene's factual mapping and any explicitly attached source. Do not draw fictional story elements. Preserve uncertainty, exact values, and constraints; never invent data, relationships, or outcomes. Explain hypothetical outcomes in text rather than drawing point markers, positions, or values unless explicitly supplied as a schematic example. Prefer text only if the evidence cannot support a useful picture.",
};

export function buildVisualPrompt(preferences: VisualPreferences) {
  const label = (key: keyof typeof visualPreferenceOptions) =>
    visualPreferenceOptions[key].find(([id]) => id === preferences[key])?.[1];
  return [
    `Create a teaching visual brief. Focus: ${preferences.question.trim() || "Explain the main idea of the supplied material."}`,
    `Visual form: ${label("form")}`,
    `Audience: ${label("audience")}`,
    `Visual style: ${label("style")}`,
    `Output target: ${label("target")}`,
    preferences.direction.trim() && `Art direction and constraints:\n${preferences.direction.trim()}`,
    preferences.additionalContext.trim() && `Teaching notes and supplied data:\n${preferences.additionalContext.trim()}`,
  ].filter(Boolean).join("\n\n");
}

const evidenceRoles = ["source", "background", "illustrative", "decorative"] as const;
const resultForms = [...visualForms.map(([id]) => id), "none"] as const;
type Content = { label: string; role: typeof evidenceRoles[number]; statement: string; chapterLines: number[] };
export type VisualBrief = {
  disposition: "render" | "text-only"; visualType: typeof resultForms[number];
  takeaway: string; content: Content[]; composition: string; guardrails: string[]; caption: string; altText: string;
};
const text = (maxLength: number) => ({ type: "string", minLength: 1, maxLength });
export const visualBriefSchema = {
  type: "object", additionalProperties: false,
  required: ["disposition", "visualType", "takeaway", "content", "composition", "guardrails", "caption", "altText"],
  properties: {
    disposition: { type: "string", enum: ["render", "text-only"] },
    visualType: { type: "string", enum: resultForms },
    takeaway: text(600),
    content: { type: "array", maxItems: 16, items: {
      type: "object", additionalProperties: false, required: ["label", "role", "statement", "chapterLines"],
      properties: {
        label: text(180), role: { type: "string", enum: evidenceRoles }, statement: text(700),
        chapterLines: { type: "array", maxItems: 10, items: { type: "integer", minimum: 1 } },
      },
    } },
    composition: text(1_800), guardrails: { type: "array", maxItems: 10, items: text(600) }, caption: text(600), altText: text(1_200),
  },
};

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

export function parseVisualBriefRequest(value: unknown): {
  context: string; preferences: VisualPreferences; source?: string; designPrompt?: string; purpose?: SceneVisualPurpose;
} {
  const body = record(value);
  const preferences = record(body?.preferences);
  const additionalContext = preferences?.additionalContext === undefined ? "" : preferences.additionalContext;
  const designPrompt = body?.designPrompt;
  const purpose = body?.purpose;
  if (!body || !preferences || typeof body.context !== "string" || !body.context.trim() || body.context.length > visualInputLimits.context ||
      typeof preferences.includeSource !== "boolean" || !["question", "direction"].every((key) =>
        typeof preferences[key] === "string" && preferences[key].length <= visualInputLimits[key as "question" | "direction"]) ||
      typeof additionalContext !== "string" || additionalContext.length > visualInputLimits.additionalContext ||
      (designPrompt !== undefined && (typeof designPrompt !== "string" || !designPrompt.trim() || designPrompt.length > visualInputLimits.prompt)) ||
      (purpose !== undefined && purpose !== "story-illustration" && purpose !== "factual-diagram") ||
      !Object.entries(visualPreferenceOptions).every(([key, options]) => options.some(([id]) => id === preferences[key])) ||
      (preferences.includeSource
        ? typeof body.source !== "string" || !body.source.trim() || body.source.length > visualInputLimits.source
        : body.source !== undefined)) throw new Error("Invalid visual brief preferences or context.");
  const { form, audience, style, target, question, direction, includeSource } = preferences as VisualPreferences;
  return { context: body.context, preferences: { form, audience, style, target, question, direction, additionalContext, includeSource },
    source: body.source as string | undefined, ...(designPrompt !== undefined ? { designPrompt } : {}),
    ...(purpose !== undefined ? { purpose } : {}) };
}

export function validateVisualBrief(value: unknown, lineCount = 0, visibleLines?: ReadonlySet<number>): VisualBrief {
  const brief = record(value);
  const plain = (value: unknown, limit: number) => typeof value === "string" && value.trim().length > 0 &&
    value.length <= limit && !/```|<\/?(?:svg|script)\b/i.test(value);
  const fields = visualBriefSchema.properties;
  const validContent = (value: unknown) => {
    const item = record(value);
    return item && Object.keys(item).length === 4 && plain(item.label, 180) && plain(item.statement, 700) &&
      evidenceRoles.some((role) => role === item.role) && Array.isArray(item.chapterLines) && item.chapterLines.length <= 10 &&
      (item.role === "source"
        ? item.chapterLines.length > 0 && item.chapterLines.every((line) => Number.isInteger(line) && line >= 1 && line <= lineCount && (!visibleLines || visibleLines.has(line)))
        : item.chapterLines.length === 0);
  };
  if (!brief || Object.keys(brief).length !== visualBriefSchema.required.length ||
    !resultForms.some((form) => form === brief.visualType) ||
    (brief.disposition === "render" ? brief.visualType === "none" : brief.disposition !== "text-only" || brief.visualType !== "none") ||
    !plain(brief.takeaway, fields.takeaway.maxLength) || !plain(brief.composition, fields.composition.maxLength) ||
    !plain(brief.caption, fields.caption.maxLength) || !plain(brief.altText, fields.altText.maxLength) ||
    !Array.isArray(brief.guardrails) || brief.guardrails.length > 10 || !brief.guardrails.every((item) => plain(item, 600)) ||
    !Array.isArray(brief.content) || brief.content.length > 16 || (brief.disposition === "render" && !brief.content.length) ||
    !brief.content.every(validContent)) throw new Error("The model returned an invalid visual brief. Try again with a narrower question.");
  return brief as VisualBrief;
}
