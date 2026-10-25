export const generationPhases = [
  "preparing",
  "drafting",
  "reviewing",
  "repairing",
  "final_review",
  "finalizing",
  "complete",
  "failed",
] as const;

export type GenerationPhase = typeof generationPhases[number];

export function isGenerationPhase(value: unknown): value is GenerationPhase {
  return typeof value === "string" && generationPhases.includes(value as GenerationPhase);
}

export const generationPhaseCopy: Record<GenerationPhase, { button: string; detail: string }> = {
  preparing: { button: "Preparing your source…", detail: "Getting your material ready for explanation." },
  drafting: { button: "Drafting explanation…", detail: "Building the first explanation and visuals." },
  reviewing: { button: "Reviewing the explanation…", detail: "Checking the answer against your input." },
  repairing: { button: "Addressing a review finding…", detail: "Making a focused correction to the draft." },
  final_review: { button: "Checking the revised answer…", detail: "Verifying the correction and checking for new issues." },
  finalizing: { button: "Finishing your explanation…", detail: "Preparing the reviewed result for display." },
  complete: { button: "Explanation ready", detail: "Your explanation is ready." },
  failed: { button: "Could not finish", detail: "The request did not complete." },
};
