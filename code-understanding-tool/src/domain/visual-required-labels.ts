import type { SceneVisualPurpose } from "./visual-brief.ts";
import type { VisualScene } from "./visual-scene.ts";

export const storyQualifier = "Fictional analogy";
const normalizeLabel = (label: string) => label.trim().normalize("NFC");

export function requiredVisualLabels(labels: readonly string[] = [], purpose?: SceneVisualPurpose): string[] {
  const required = purpose === "story-illustration" ? [storyQualifier, ...labels] : labels;
  return [...new Set(required.map(normalizeLabel))];
}

export class MissingVisualLabelsError extends Error {
  constructor(readonly labels: readonly string[]) {
    super(`Drawing is missing required labels: ${labels.map(label => JSON.stringify(label)).join(", ")}. Retry the drawing plan or revise the brief.`);
    this.name = "MissingVisualLabelsError";
  }
}

export function assertRequiredVisualLabels(scene: VisualScene, required: readonly string[]) {
  if (!required.length) return;
  const present = new Set(scene.items.flatMap(item => item.kind === "text" ? [normalizeLabel(item.text)] : []));
  const missing = required.filter(label => !present.has(normalizeLabel(label)));
  if (missing.length) throw new MissingVisualLabelsError(missing);
}
