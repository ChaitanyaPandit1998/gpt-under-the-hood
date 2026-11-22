import type { SavedStudy, ResultReadingState } from "./saved-study";
import { validateChapterStory } from "./chapter-story";
import { validateExplanation } from "../server/response-validation";

export const portableStudyLimit = 40 * 1024 * 1024;
const bad = () => { throw new Error("Invalid or unsupported study file."); };
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return bad();
  return value as Record<string, unknown>;
}
function text(value: unknown): asserts value is string { if (typeof value !== "string") bad(); }
function choices(value: unknown, options: string[]) { if (!options.includes(String(value))) bad(); }
function image(value: unknown) {
  const item = record(value);
  if (!(item.png instanceof Blob) || item.png.type !== "image/png") bad();
  text(item.alt); text(item.caption);
  choices(item.resolution, ["standard", "hd", "full-hd"]);
}
function settings(value: unknown) {
  const item = record(value);
  choices(item.depth, ["overview", "balanced", "detailed"]);
  choices(item.learnerLevel, ["beginner", "intermediate", "advanced"]);
  choices(item.tone, ["plain", "technical"]);
  choices(item.focus, ["auto", "concepts", "behaviour", "example"]);
}
function explanation(value: unknown, source: string) {
  validateExplanation(value, source);
  const item = record(value);
  if (item.quality_review_status !== undefined) choices(item.quality_review_status, ["passed", "revised", "incomplete"]);
  if (item.quality_review_note !== undefined) text(item.quality_review_note);
  if (item.incomplete_items !== undefined) {
    if (!Array.isArray(item.incomplete_items) || item.incomplete_items.length > 100) bad();
    for (const entry of item.incomplete_items as unknown[]) {
      const incomplete = record(entry);
      choices(incomplete.kind, ["unfinished_marker", "empty_branch"]);
      text(incomplete.title); text(incomplete.explanation);
      validateExplanation({ title: "Check", summary: "", concepts: [{ term: "Check", explanation: "", source_ranges: incomplete.source_ranges }], walkthrough: [], assumptions: [], visuals: [], visual_note: "" }, source);
    }
  }
}
function reading(value: unknown, source: string): asserts value is ResultReadingState {
  const item = record(value);
  if (item.illustration !== undefined) image(item.illustration);
  if (item.chapter === undefined) return;
  const chapter = record(item.chapter);
  text(chapter.chapterId);
  if (typeof chapter.chapterMode !== "boolean") bad();
  Object.values(record(chapter.images)).forEach(image);
  for (const adapted of Object.values(record(chapter.adapted ?? {}))) {
    const variant = record(adapted);
    settings(variant.settings);
    if (typeof variant.visible !== "boolean") bad();
    Object.values(record(variant.variants)).forEach(v => explanation(v, source));
    for (const entry of Object.values(record(variant.images ?? {}))) {
      const v = record(entry); explanation(v.explanation, source); image(v.illustration);
    }
  }
  // Cover data URLs are not portable yet; reject rather than accept external URLs.
  if (Object.keys(record(chapter.covers ?? {})).length) throw new Error("Remove decorative covers before transferring this study. Scene images are supported.");
  const stories = record(chapter.stories ?? {});
  for (const entry of Object.values(stories)) {
    const saved = record(entry);
    const story = validateChapterStory(saved.story, source.split(/\r\n|\r|\n/).length);
    for (const [index, images] of Object.entries(record(saved.images ?? {}))) {
      if (!/^\d+$/.test(index) || Number(index) >= story.scenes.length) bad();
      for (const [purpose, png] of Object.entries(record(images))) {
        choices(purpose, ["story-illustration", "factual-diagram"]); image(png);
      }
    }
  }
  if (chapter.activeStoryKey !== undefined) {
    text(chapter.activeStoryKey);
    if (!Object.hasOwn(stories, chapter.activeStoryKey)) bad();
  }
}
function validate(value: unknown): SavedStudy {
  const study = record(value);
  if (study.version !== 1) return bad();
  if (study.mode === "pull_request") throw new Error("Portable PR studies are not supported yet.");
  choices(study.mode, ["code", "docs"]);
  choices(study.depth, ["overview", "balanced", "detailed"]);
  choices(study.level, ["beginner", "intermediate", "advanced"]);
  text(study.source);
  if (study.source.length > 200_000) bad();
  explanation(study.explanation, study.source);
  reading(study.reading, study.source);
  return study as unknown as SavedStudy;
}
async function encode(value: unknown): Promise<unknown> {
  if (value instanceof Blob) {
    const bytes = new Uint8Array(await value.arrayBuffer());
    let binary = "";
    for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    return { $png: btoa(binary) };
  }
  if (Array.isArray(value)) return Promise.all(value.map(encode));
  if (value && typeof value === "object") {
    const entries = [];
    for (const [key, v] of Object.entries(value)) if (v !== undefined) entries.push([key, await encode(v)]);
    return Object.fromEntries(entries);
  }
  return value;
}
function decode(value: unknown, depth = 0): unknown {
  if (depth > 30) bad();
  if (Array.isArray(value)) return value.map(v => decode(v, depth + 1));
  if (value && typeof value === "object") {
    const item = record(value);
    if (Object.hasOwn(item, "$png")) {
      if (Object.keys(item).length !== 1 || typeof item.$png !== "string") bad();
      const bytes = Uint8Array.from(atob(item.$png as string), c => c.charCodeAt(0));
      if (bytes.length < 24 || ![137, 80, 78, 71, 13, 10, 26, 10].every((n, i) => bytes[i] === n)) bad();
      const view = new DataView(bytes.buffer);
      if (view.getUint32(16) < 1 || view.getUint32(20) < 1 || view.getUint32(16) > 4096 || view.getUint32(20) > 4096) bad();
      return new Blob([bytes], { type: "image/png" });
    }
    for (const key of Object.keys(item)) {
      if (["__proto__", "constructor", "prototype"].includes(key)) bad();
      item[key] = decode(item[key], depth + 1);
    }
  }
  return value;
}
export async function exportPortableStudy(study: SavedStudy): Promise<Blob> {
  validate(study);
  const json = JSON.stringify({ format: "understand-study", version: 1, study: await encode(study) });
  const blob = new Blob([json], { type: "application/json" });
  if (blob.size > portableStudyLimit) throw new Error("Study file exceeds the 40 MB transfer limit.");
  return blob;
}
export async function importPortableStudy(file: Blob): Promise<SavedStudy> {
  if (file.size > portableStudyLimit) throw new Error("Study file exceeds the 40 MB transfer limit.");
  const envelope = record(JSON.parse(await file.text()));
  if (envelope.format !== "understand-study" || envelope.version !== 1) return bad();
  return validate(decode(envelope.study));
}
