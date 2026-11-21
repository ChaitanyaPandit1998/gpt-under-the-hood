import { describe, expect, it } from "vitest";
import { adjustChapterDepth, chapterVariantKey, defaultChapterSettings } from "./chapter-settings";

describe("chapter preferences", () => {
  it("changes depth without mutating other preferences and clamps at either end", () => {
    const settings = { ...defaultChapterSettings, tone: "technical" as const, focus: "behaviour" as const };
    expect(adjustChapterDepth(settings, -1)).toEqual({ ...settings, depth: "overview" });
    expect(adjustChapterDepth(adjustChapterDepth(settings, -1), -1).depth).toBe("overview");
    expect(adjustChapterDepth(adjustChapterDepth(settings, 1), 1).depth).toBe("detailed");
    expect(settings.depth).toBe("balanced");
  });
  it("keys variants by every preference regardless of object property order", () => {
    const original = chapterVariantKey(defaultChapterSettings);
    expect(chapterVariantKey({ focus: "auto", tone: "plain", learnerLevel: "beginner", depth: "balanced" })).toBe(original);
    expect(chapterVariantKey({ ...defaultChapterSettings, tone: "technical" })).not.toBe(original);
    expect(chapterVariantKey({ ...defaultChapterSettings, focus: "example" })).not.toBe(original);
    expect(chapterVariantKey({ ...defaultChapterSettings, depth: "detailed" })).not.toBe(original);
    expect(chapterVariantKey({ ...defaultChapterSettings, learnerLevel: "advanced" })).not.toBe(original);
  });
});
