import { validateExplanation } from "../server/response-validation";
import type { Explanation, ExplanationDepth, LearnerLevel } from "./explanation-types";

const key = "understand-temporary-explanation";
const disabled = "off";
const maxLength = 1_000_000;

export type TemporaryStudy = {
  version: 1;
  mode: "code" | "docs";
  source: string;
  depth: ExplanationDepth;
  level: LearnerLevel;
  explanation: Explanation;
  chapterId?: string;
};

export function readTemporaryStudy(): { enabled: boolean; study?: TemporaryStudy } {
  try {
    const raw = sessionStorage.getItem(key);
    if (raw === disabled) return { enabled: false };
    if (!raw || raw.length > maxLength) return { enabled: true };
    const item = JSON.parse(raw) as Partial<TemporaryStudy> | null;
    if (!item || item.version !== 1 || (item.mode !== "code" && item.mode !== "docs") ||
      typeof item.source !== "string" || !item.source.trim() ||
      !["overview", "balanced", "detailed"].includes(item.depth ?? "") ||
      !["beginner", "intermediate", "advanced"].includes(item.level ?? "") ||
      (item.chapterId !== undefined && (typeof item.chapterId !== "string" || item.chapterId.length > 120))) return { enabled: true };
    validateExplanation(item.explanation, item.source);
    return { enabled: true, study: item as TemporaryStudy };
  } catch {
    return { enabled: true };
  }
}

export function writeTemporaryStudy(enabled: boolean, study?: TemporaryStudy): boolean {
  try {
    if (!enabled) {
      sessionStorage.removeItem(key);
      sessionStorage.setItem(key, disabled);
    } else if (!study) sessionStorage.removeItem(key);
    else {
      const payload = JSON.stringify(study);
      if (payload.length > maxLength) {
        sessionStorage.removeItem(key);
        return false;
      }
      try {
        sessionStorage.setItem(key, payload);
      } catch {
        // Do not recover a previous explanation if writing the current one failed.
        sessionStorage.removeItem(key);
        return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}
