import { generateCloudflareImage, CLOUDFLARE_IMAGE_MODEL } from "./cloudflare-image-client.ts";

export const IMAGE_MODEL = CLOUDFLARE_IMAGE_MODEL;
export const IMAGE_STEPS = 4;
// The live REST schema rejects seed, despite older examples showing it.
export const IMAGE_SEED = null;
export type ImageCase = { id: string; title: string; summary: string; visualBrief: string; requiredElements: string[]; failureModes: string[] };
export type ImageTrial = { id: string; caseId: string; style: "concept-card" | "storyboard" | "editorial" | "isometric"; prompt: string; steps: number; seed: null };

export function imageTrials(cases: ImageCase[], suite: "technical" | "decorative" = "technical"): ImageTrial[] {
  if (!Array.isArray(cases) || cases.length < 1 || cases.length > 3) throw new Error("This pilot permits one to three curated cases only.");
  const seen = new Set<string>();
  return cases.flatMap((item) => {
    if (!item || !/^[a-z0-9-]{1,60}$/.test(item.id) || seen.has(item.id)) throw new Error("Each image case needs a unique safe ID.");
    seen.add(item.id);
    if (![item.title, item.summary, item.visualBrief].every((value) => typeof value === "string" && value.trim())) throw new Error("Each case needs a title, summary, and visual brief.");
    const text = `${item.title}\n${item.summary}\n${item.visualBrief}`;
    if (/https?:\/\/|[\w.+-]+@[\w.-]+\.[a-z]{2,}|\b(?:sk-|ghp_|github_pat_)[a-z0-9_-]+/i.test(text)) throw new Error("Potential identifying or secret content detected. Review the curated case manually.");
    const styles = suite === "decorative" ? ["editorial", "isometric"] as const : ["concept-card", "storyboard"] as const;
    return styles.map((style) => {
      const directions = {
        "concept-card": "Create a clean editorial concept illustration on a warm light background with large simple shapes, restrained teal and amber accents, and clear visual relationships. One unified composition.",
        storyboard: "Create a three-panel educational storyboard on a warm light background with large simple shapes, restrained teal and amber accents, and a clear left-to-right reading order. Keep the same objects consistent across panels.",
        editorial: "Decorative chapter cover artwork. Flat vector-like editorial illustration, crisp silhouettes, ivory background, teal and amber accents, subtle paper texture. One centered still-life composition with generous empty space.",
        isometric: "Decorative chapter cover artwork. Minimal isometric 3D still life, softly rounded objects, matte materials, ivory background, teal and amber accents, soft natural shadows. One centered composition with generous empty space.",
      };
      const prompt = suite === "decorative"
        ? `${directions[style]}\nArtwork brief: ${item.visualBrief}\nNo text, letters, numbers, logos, watermarks, arrows, connections, diagrams, panels, or flowcharts. Decorative metaphor only, not a technical explanation or a promise of security.`
        : `${directions[style]}\nConcept: ${item.summary}\nVisual brief: ${item.visualBrief}\nNo words, letters, code, logos, watermarks, or decorative circuitry. Use shapes and symbols to communicate. This is a conceptual teaching illustration, not a verified architecture or execution diagram.`;
      if (prompt.length > 2_048) throw new Error("The curated prompt exceeds the model's 2048-character limit.");
      return { id: `${item.id}-${style}`, caseId: item.id, style, prompt, steps: IMAGE_STEPS, seed: IMAGE_SEED };
    });
  });
}

export async function generateBenchmarkImage(trial: ImageTrial, credentials: { accountId: string; apiToken: string }, fetcher: typeof fetch = fetch) {
  if (!trial.prompt || trial.prompt.length > 2_048 || trial.steps !== IMAGE_STEPS || trial.seed !== IMAGE_SEED) throw new Error("Invalid benchmark generation settings.");
  return generateCloudflareImage(trial.prompt, credentials, fetcher);
}
