import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { ExplanationDepth, LearnerLevel } from "../domain/explanation-types.ts";
import type { GenerationStage } from "../domain/visual-provenance.ts";

// Paths are application-owned; request data selects IDs, never filesystem paths.
const instructionFiles = {
  "story-planning": "skills/plan-chapter-story/SKILL.md",
  "story-writing": "skills/write-chapter-story/SKILL.md",
  "story-review": "skills/review-chapter-story/SKILL.md",
  "visual-direction": "skills/design-explanation-visual/SKILL.md",
  "visual-composition": "skills/compose-vectorcraft-visual/SKILL.md",
  "visual-composition-compact": "skills/compose-vectorcraft-visual/references/compact-scenes.md",
  "visual-review": "skills/review-explanation-visual/SKILL.md",
  "audience-beginner": "skills/learning-profiles/audience/beginner.md",
  "audience-intermediate": "skills/learning-profiles/audience/intermediate.md",
  "audience-advanced": "skills/learning-profiles/audience/advanced.md",
  "depth-overview": "skills/learning-profiles/depth/overview.md",
  "depth-balanced": "skills/learning-profiles/depth/balanced.md",
  "depth-detailed": "skills/learning-profiles/depth/detailed.md",
} as const;

type InstructionId = keyof typeof instructionFiles;
export type ApplicationInstruction = { id: InstructionId; version: 1; sha256: string; text: string };

export function formatApplicationInstruction(resource: ApplicationInstruction) {
  return `Application instruction: ${resource.id}; version=${resource.version}; sha256=${resource.sha256}\n${resource.text}`;
}

export function generationStage(artifact: unknown, instructions: ApplicationInstruction[]): GenerationStage {
  return { artifactSha256: createHash("sha256").update(JSON.stringify(artifact)).digest("hex"),
    instructions: instructions.map(({ id, version, sha256 }) => ({ id, version, sha256 })) };
}

export async function loadApplicationInstruction(root: string, id: InstructionId, signal?: AbortSignal): Promise<ApplicationInstruction> {
  signal?.throwIfAborted();
  if (!Object.hasOwn(instructionFiles, id)) throw new Error("Unknown application instruction.");
  const text = await readFile(resolve(root, instructionFiles[id]), { encoding: "utf8", signal });
  if (!text.trim() || Buffer.byteLength(text, "utf8") > 20_000) throw new Error("Invalid application instruction.");
  signal?.throwIfAborted();
  return { id, version: 1, sha256: createHash("sha256").update(text).digest("hex"), text };
}

export async function loadStoryInstructions(root: string, audience: LearnerLevel, depth: ExplanationDepth, signal?: AbortSignal) {
  const [planning, writing, review, audienceProfile, depthProfile] = await Promise.all([
    loadApplicationInstruction(root, "story-planning", signal),
    loadApplicationInstruction(root, "story-writing", signal),
    loadApplicationInstruction(root, "story-review", signal),
    loadApplicationInstruction(root, `audience-${audience}`, signal),
    loadApplicationInstruction(root, `depth-${depth}`, signal),
  ]);
  return {
    generation: [planning, writing].map(formatApplicationInstruction).join("\n\n"),
    review: formatApplicationInstruction(review),
    profiles: [audienceProfile, depthProfile].map(formatApplicationInstruction).join("\n\n"),
    resources: [planning, writing, review, audienceProfile, depthProfile],
  };
}
