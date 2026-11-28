import type { Explanation, ExplanationDepth, LearnerLevel } from "./explanation-types";
import type { VisualResolution } from "./visual-resolution";
import type { ChapterSettings } from "./chapter-settings";
import type { PullRequestChapter, PullRequestPlan } from "./pull-request-plan";
import type { ChapterArtwork } from "./chapter-illustrations";
import type { ChapterStory } from "./chapter-story";
import type { SceneVisualPurpose } from "./visual-brief";
import { validateSavedVisualArtifacts, type VisualProvenance } from "./visual-provenance";
import { parseStoryCast, type StoryCast } from "./story-cast";

export type RenderedIllustration = { png: Blob; alt: string; caption: string; resolution: VisualResolution; provenance?: VisualProvenance };
export type StorySceneImages = Partial<Record<SceneVisualPurpose, RenderedIllustration>>;
export type SavedChapterStory = { story: ChapterStory; images?: Record<number, StorySceneImages> };
export type AdaptedStudy = {
  settings: ChapterSettings; visible: boolean; variants: Record<string, Explanation>;
  images?: Record<string, { explanation: Explanation; illustration: RenderedIllustration }>;
};
export type ReadingState = {
  chapterMode: boolean; chapterId: string; images: Record<string, RenderedIllustration>;
  adapted?: Record<string, AdaptedStudy>; covers?: Record<string, ChapterArtwork>;
  stories?: Record<string, SavedChapterStory>; activeStoryKey?: string; storyCast?: StoryCast;
};
export type ResultReadingState = { chapter?: ReadingState; illustration?: RenderedIllustration };
type SavedSourceStudy = {
  version: 1; source: string; mode: "code" | "docs"; depth: ExplanationDepth;
  level: LearnerLevel; explanation: Explanation; reading: ResultReadingState;
};
export type CachedPrResult = { explanation: Explanation; reading?: ResultReadingState };
export type PrReadingState = {
  chapterId: string; settings: Record<number, ChapterSettings>; extraChapters: PullRequestChapter[];
  evidenceProgress: Record<string, { nextOffset: number | null; sections: number; totalCharacters: number; loadedCharacters: number }>;
  results: Record<string, CachedPrResult>; covers: Record<string, ChapterArtwork>;
  storyCast?: StoryCast;
};
export type SavedPrStudy = { version: 1; mode: "pull_request"; plan: PullRequestPlan; reading: PrReadingState };
export type SavedStudy = SavedSourceStudy | SavedPrStudy;

type Action = { type: "read" } | { type: "save"; study: SavedStudy } | { type: "delete" };

async function validateStoredStudy(study: SavedStudy) {
  parseStoryCast(study.mode === "pull_request" ? study.reading.storyCast : study.reading.chapter?.storyCast);
  await validateSavedVisualArtifacts(study);
}

export async function accessSavedStudy(action: Action): Promise<SavedStudy | undefined> {
  if (action.type === "save") await validateStoredStudy(action.study);
  return new Promise((resolve, reject) => {
    const open = indexedDB.open("understand-local-study", 1);
    open.onupgradeneeded = () => open.result.createObjectStore("snapshots");
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const db = open.result;
      const tx = db.transaction("snapshots", action.type === "read" ? "readonly" : "readwrite");
      const store = tx.objectStore("snapshots");
      const operation = action.type === "read" ? store.get("current") :
        action.type === "save" ? store.put(action.study, "current") : store.delete("current");
      tx.oncomplete = () => {
        db.close();
        const study = action.type === "read" ? operation.result as SavedStudy | undefined : undefined;
        if (study?.version === 1) void validateStoredStudy(study).then(() => resolve(study), reject);
        else resolve(undefined);
      };
      tx.onabort = () => { db.close(); reject(tx.error); };
    };
  });
}
