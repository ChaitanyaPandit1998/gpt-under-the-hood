import { isValidRange, type SourceRange } from "./source-ranges.ts";
import type { ExplanationDepth, LearnerLevel, Mode } from "./explanation-types.ts";

export type StoryPreferences = { depth: ExplanationDepth; learnerLevel: LearnerLevel };
export const defaultStoryPreferences: StoryPreferences = { depth: "balanced", learnerLevel: "beginner" };
export type ChapterStory = {
  title: string;
  setup: string;
  scenes: { heading: string; narrative: string; meaning: string; source_ranges: SourceRange[] }[];
  takeaway: string;
};

export const storyTextLimits = { title: 180, setup: 1_000, heading: 180, narrative: 2_000, meaning: 1_500, takeaway: 1_000 };
const text = (field: keyof typeof storyTextLimits) => ({ type: "string", minLength: 1, maxLength: storyTextLimits[field] });
export const chapterStorySchema = {
  type: "object", additionalProperties: false, required: ["title", "setup", "scenes", "takeaway"],
  properties: {
    title: text("title"), setup: text("setup"), takeaway: text("takeaway"),
    scenes: {
      type: "array", minItems: 1, maxItems: 5,
      items: {
        type: "object", additionalProperties: false, required: ["heading", "narrative", "meaning", "source_ranges"],
        properties: {
          heading: text("heading"), narrative: text("narrative"), meaning: text("meaning"),
          source_ranges: {
            type: "array", maxItems: 4,
            items: {
              type: "object", additionalProperties: false, required: ["start_line", "end_line"],
              properties: { start_line: { type: "integer", minimum: 1 }, end_line: { type: "integer", minimum: 1 } },
            },
          },
        },
      },
    },
  },
};

function record(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function validText(value: unknown, max: number): value is string {
  return typeof value === "string" && !!value.trim() && value.length <= max;
}

export function validateChapterStory(value: unknown, lineCount: number): ChapterStory {
  const valid = record(value) && Object.keys(value).length === 4
    && validText(value.title, storyTextLimits.title) && validText(value.setup, storyTextLimits.setup)
    && validText(value.takeaway, storyTextLimits.takeaway)
    && Array.isArray(value.scenes) && value.scenes.length >= 1 && value.scenes.length <= 5
    && value.scenes.every((scene: unknown) => record(scene) && Object.keys(scene).length === 4
      && validText(scene.heading, storyTextLimits.heading) && validText(scene.narrative, storyTextLimits.narrative)
      && validText(scene.meaning, storyTextLimits.meaning)
      && Array.isArray(scene.source_ranges) && scene.source_ranges.length <= 4
      && scene.source_ranges.every((range: unknown) => record(range) && Object.keys(range).length === 2
        && typeof range.start_line === "number" && typeof range.end_line === "number"
        && isValidRange({ start_line: range.start_line, end_line: range.end_line }, lineCount)));
  if (!valid) throw new Error("The story format or source citations were invalid.");
  return value as ChapterStory;
}

export const chapterStoryContextLimit = 24_000;
export function parseChapterStoryRequest(value: unknown, maxSourceCharacters: number) {
  if (!record(value)
    || !validText(value.source, maxSourceCharacters)
    || !validText(value.context, chapterStoryContextLimit)
    || (value.mode !== "code" && value.mode !== "docs" && value.mode !== "pull_request")
    || (value.depth !== "overview" && value.depth !== "balanced" && value.depth !== "detailed")
    || (value.learnerLevel !== "beginner" && value.learnerLevel !== "intermediate" && value.learnerLevel !== "advanced")) {
    throw new Error("A valid chapter, source, and reading preferences are required.");
  }
  return { source: value.source, context: value.context, mode: value.mode as Mode, depth: value.depth, learnerLevel: value.learnerLevel };
}
