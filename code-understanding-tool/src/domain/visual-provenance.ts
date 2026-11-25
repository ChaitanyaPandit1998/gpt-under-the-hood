import { validateVisualBrief, type VisualBrief, type VisualPreferences, type SceneVisualPurpose } from "./visual-brief";
import { validateVisualScene, type VisualScene } from "./visual-scene";
import { validateVisualContract, validateVisualEvaluationReport, type VisualContract, type VisualEvaluationReport } from "./visual-evaluation";
import type { RenderedIllustration } from "./saved-study";

export type SavedVisualReview = { reviewedAt: string; contract: VisualContract; report: VisualEvaluationReport };
export type InstructionStamp = { id: string; version: 1; sha256: string };
export type GenerationStage = { artifactSha256: string; instructions: InstructionStamp[] };
export type VisualGeneration = { brief?: GenerationStage; plan?: GenerationStage };
export type VisualProvenance = {
  version: 1; brief: VisualBrief; scene: VisualScene;
  context: { explanation: string; scope: string; audience: VisualPreferences["audience"]; purpose?: SceneVisualPurpose };
  reviews: SavedVisualReview[];
  generation?: VisualGeneration;
};
export const visualReviewHistoryLimit = 5;
const provenanceSizeLimit = 2_000_000;

export function appendVisualReview(provenance: VisualProvenance, review: SavedVisualReview): VisualProvenance {
  const next = { ...provenance, reviews: [...provenance.reviews, review].slice(-visualReviewHistoryLimit) };
  while (next.reviews.length > 1 && JSON.stringify(next).length > provenanceSizeLimit) next.reviews.shift();
  return next;
}

export function mergeVisualReviewHistory(current: RenderedIllustration | undefined, incoming: RenderedIllustration): RenderedIllustration | undefined {
  if (!current?.provenance || !incoming.provenance || current.png !== incoming.png || current.caption !== incoming.caption ||
    JSON.stringify([current.provenance.brief, current.provenance.scene, current.provenance.context]) !==
    JSON.stringify([incoming.provenance.brief, incoming.provenance.scene, incoming.provenance.context])) return current;
  const additions = incoming.provenance.reviews.filter(review => !current.provenance!.reviews.some(entry =>
    entry.contract.id === review.contract.id && entry.reviewedAt === review.reviewedAt));
  if (!additions.length) return current;
  const ordered = [...current.provenance.reviews, ...additions].sort((a, b) => Date.parse(a.reviewedAt) - Date.parse(b.reviewedAt));
  const provenance = ordered.reduce(appendVisualReview, { ...current.provenance, reviews: [] });
  return { ...current, provenance };
}
const invalid = () => { throw new Error("Invalid or mismatched visual provenance."); };
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return invalid();
  return value as Record<string, unknown>;
}
function exact(item: Record<string, unknown>, keys: string[]) {
  if (Object.keys(item).length !== keys.length || keys.some((key) => !Object.hasOwn(item, key))) invalid();
}
async function hash(value: string | Blob) {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : await value.arrayBuffer();
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), byte => byte.toString(16).padStart(2, "0")).join("");
}

export async function validateGenerationStage(value: unknown, artifact: string): Promise<GenerationStage> {
  const stage = record(value);
  exact(stage, ["artifactSha256", "instructions"]);
  if (stage.artifactSha256 !== await hash(artifact) || !Array.isArray(stage.instructions) ||
    stage.instructions.length < 1 || stage.instructions.length > 10) return invalid();
  const seen = new Set<string>();
  for (const value of stage.instructions) {
    const stamp = record(value);
    exact(stamp, ["id", "version", "sha256"]);
    if (typeof stamp.id !== "string" || !/^[a-z][a-z0-9-]{0,79}$/.test(stamp.id) || seen.has(stamp.id) ||
      stamp.version !== 1 || typeof stamp.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(stamp.sha256)) return invalid();
    seen.add(stamp.id);
  }
  return stage as GenerationStage;
}

export async function validateVisualProvenance(value: unknown, image: { png: Blob; caption: string }): Promise<VisualProvenance> {
  const item = record(value);
  exact(item, ["version", "brief", "scene", "context", "reviews", ...(Object.hasOwn(item, "generation") ? ["generation"] : [])]);
  if (item.version !== 1 || JSON.stringify(item).length > provenanceSizeLimit) invalid();
  const brief = validateVisualBrief(item.brief, 200_001);
  validateVisualScene(item.scene);
  if (brief.caption !== image.caption || image.png.size > 10 * 1024 * 1024) invalid();
  const context = record(item.context);
  const hasPurpose = Object.hasOwn(context, "purpose");
  exact(context, ["explanation", "scope", "audience", ...(hasPurpose ? ["purpose"] : [])]);
  for (const [key, limit] of [["explanation", 100_000], ["scope", 2000]] as const) {
    if (typeof context[key] !== "string" || !context[key].trim() || context[key].length > limit) invalid();
  }
  if (!["auto", "beginner", "intermediate", "advanced"].includes(String(context.audience)) ||
    (hasPurpose && !["story-illustration", "factual-diagram"].includes(String(context.purpose)))) invalid();
  const reviews = item.reviews;
  if (!Array.isArray(reviews) || reviews.length > visualReviewHistoryLimit) return invalid();
  const explanationHash = await hash(context.explanation as string);
  const artifacts = { briefSha256: await hash(JSON.stringify(item.brief)), planSha256: await hash(JSON.stringify(item.scene)),
    pngSha256: await hash(image.png), captionSha256: await hash(image.caption) };
  if (item.generation !== undefined) {
    const generation = record(item.generation);
    const keys = Object.keys(generation);
    if (!keys.length || keys.some(key => key !== "brief" && key !== "plan")) invalid();
    if (Object.hasOwn(generation, "brief")) await validateGenerationStage(generation.brief, JSON.stringify(item.brief));
    if (Object.hasOwn(generation, "plan")) await validateGenerationStage(generation.plan, JSON.stringify(item.scene));
  }
  for (const value of reviews) {
    const entry = record(value);
    exact(entry, ["reviewedAt", "contract", "report"]);
    if (typeof entry.reviewedAt !== "string" || entry.reviewedAt.length > 40 || !Number.isFinite(Date.parse(entry.reviewedAt))) invalid();
    const contract = validateVisualContract(entry.contract);
    const report = validateVisualEvaluationReport(entry.report, contract);
    if (contract.explanationSha256 !== explanationHash || Object.entries(artifacts).some(([key, digest]) =>
      report.artifacts[key as keyof typeof artifacts] !== digest)) invalid();
  }
  return item as unknown as VisualProvenance;
}

// Also used for IndexedDB restores, not just portable JSON imports.
export async function validateSavedVisualArtifacts(value: unknown): Promise<void> {
  const ancestors = new WeakSet<object>();
  let visited = 0;
  async function visit(value: unknown, depth: number): Promise<void> {
    if (depth > 30 || ++visited > 100_000) return invalid();
    if (!value || typeof value !== "object" || value instanceof Blob) return;
    if (ancestors.has(value)) return invalid();
    const item = value as Record<string, unknown>;
    if (item.png instanceof Blob && item.provenance !== undefined) {
      const caption = item.caption;
      if (typeof caption !== "string") return invalid();
      await validateVisualProvenance(item.provenance, { png: item.png, caption });
      return;
    }
    ancestors.add(value);
    for (const child of Object.values(item)) await visit(child, depth + 1);
    ancestors.delete(value);
  }
  await visit(value, 0);
}
