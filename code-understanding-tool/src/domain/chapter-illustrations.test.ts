import { describe, expect, it } from "vitest";
import { artworkThemes, chapterArtworkPrompt, parseArtworkRequest, type ArtworkTheme } from "./chapter-illustrations";

describe("privacy-preserving chapter artwork", () => {
  it("accepts only reviewed themes with explicit approval", () => {
    expect(parseArtworkRequest({ theme: "review", approved: true })).toBe("review");
    for (const payload of [null, [], {}, { theme: "__proto__", approved: true }, { theme: "review", approved: "true" }, { theme: "review", approved: false }, { theme: "review", approved: true, summary: "private" }, { theme: "review", approved: true, prompt: "custom" }]) {
      expect(() => parseArtworkRequest(payload)).toThrow();
    }
  });

  it("builds fixed decorative prompts with no chapter input or execution-flow claims", () => {
    for (const theme of Object.keys(artworkThemes) as ArtworkTheme[]) {
      const prompt = chapterArtworkPrompt(theme);
      expect(prompt.length).toBeLessThanOrEqual(2048);
      expect(prompt).toContain("isometric");
      expect(prompt).toContain("not a technical explanation");
      expect(prompt).not.toMatch(/https?:\/\/|github\.com|source_text/);
    }
  });
});
