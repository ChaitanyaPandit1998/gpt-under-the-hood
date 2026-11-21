export const artworkThemes = {
  review: { label: "Reviewing requests", brief: "One blank ivory document with a folded corner and one large teal magnifying glass examining its surface. A small amber geometric accent balances the composition. Keep the document blank. Make both objects recognizable at thumbnail size." },
  compatibility: { label: "Adapting versions", brief: "Two different simple geometric plugs beside one smooth teal adapter block on an ivory surface, with a small amber accent. A static still life of three separate objects, not an assembly instruction or a picture of connected systems." },
  retries: { label: "Retrying work", brief: "A small teal toolbox and one simple amber wrench beside a blank ivory task card. One quiet static still life representing working on a task, with no attempt counts or outcome claims." },
} as const;
export type ArtworkTheme = keyof typeof artworkThemes;
export type ChapterArtwork = { theme: ArtworkTheme; dataUrl: string; model: string; latencyMs: number };

export function isArtworkTheme(value: unknown): value is ArtworkTheme {
  return typeof value === "string" && Object.hasOwn(artworkThemes, value);
}

export function chapterArtworkPrompt(theme: ArtworkTheme): string {
  return `Decorative chapter cover artwork. Minimal isometric 3D still life, softly rounded objects, matte materials, ivory background, teal and amber accents, soft natural shadows. One centered composition with generous empty space.\nArtwork brief: ${artworkThemes[theme].brief}\nNo text, letters, numbers, logos, watermarks, arrows, connections, diagrams, panels, or flowcharts. Decorative metaphor only, not a technical explanation or a promise of security.`;
}

export function parseArtworkRequest(input: unknown): ArtworkTheme {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("Choose a reviewed artwork theme and approve the provider request.");
  const payload = input as Record<string, unknown>;
  if (Object.keys(payload).some((key) => key !== "theme" && key !== "approved")) throw new Error("Only a generic theme and approval may be submitted. Chapter text, source, and custom prompts are not accepted.");
  if (payload.approved !== true || !isArtworkTheme(payload.theme)) throw new Error("Choose a reviewed artwork theme and approve the provider request.");
  return payload.theme;
}
