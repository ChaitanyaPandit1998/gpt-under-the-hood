import type { Explanation, ExplanationDepth, LearnerLevel } from "./explanation-types";
import type { VisualResolution } from "./visual-resolution";
import type { ChapterSettings } from "./chapter-settings";
import type { PullRequestChapter, PullRequestPlan } from "./pull-request-plan";
import type { ChapterArtwork } from "./chapter-illustrations";

export type RenderedIllustration = { png: Blob; alt: string; caption: string; resolution: VisualResolution };
export type AdaptedStudy = {
  settings: ChapterSettings; visible: boolean; variants: Record<string, Explanation>;
  images?: Record<string, { explanation: Explanation; illustration: RenderedIllustration }>;
};
export type ReadingState = {
  chapterMode: boolean; chapterId: string; images: Record<string, RenderedIllustration>;
  adapted?: Record<string, AdaptedStudy>; covers?: Record<string, ChapterArtwork>;
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
};
export type SavedPrStudy = { version: 1; mode: "pull_request"; plan: PullRequestPlan; reading: PrReadingState };
export type SavedStudy = SavedSourceStudy | SavedPrStudy;

type Action = { type: "read" } | { type: "save"; study: SavedStudy } | { type: "delete" };

export function accessSavedStudy(action: Action): Promise<SavedStudy | undefined> {
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
        resolve(study?.version === 1 ? study : undefined);
      };
      tx.onabort = () => { db.close(); reject(tx.error); };
    };
  });
}
