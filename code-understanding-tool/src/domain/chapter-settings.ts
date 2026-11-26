import type { ExplanationDepth, LearnerLevel } from "./explanation-types";

export type ChapterSettings = {
  depth: ExplanationDepth;
  learnerLevel: LearnerLevel;
  tone: "plain" | "technical";
  focus: "auto" | "concepts" | "behaviour" | "example";
  visualPreference?: "auto" | "no";
};

export const defaultChapterSettings: ChapterSettings = {
  depth: "balanced", learnerLevel: "beginner", tone: "plain", focus: "auto",
};

export function chapterVariantKey(settings: ChapterSettings): string {
  const parts: string[] = [settings.depth, settings.learnerLevel, settings.tone, settings.focus];
  if (settings.visualPreference === "no") parts.push("no");
  return JSON.stringify(parts);
}

export function adjustChapterDepth(settings: ChapterSettings, direction: -1 | 1): ChapterSettings {
  const depths: ExplanationDepth[] = ["overview", "balanced", "detailed"];
  return { ...settings, depth: depths[Math.max(0, Math.min(2, depths.indexOf(settings.depth) + direction))] };
}
