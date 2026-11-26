import { describe, expect, it } from "vitest";
import { splitReadingParagraphs } from "./reading-paragraphs";

describe("splitReadingParagraphs", () => {
  it("leaves short text and intentional line breaks intact", () => {
    expect(splitReadingParagraphs("A short explanation.")).toEqual(["A short explanation."]);
    expect(splitReadingParagraphs("First thought.\r\n\r\n  Second thought.\r\nA continuation."))
      .toEqual(["First thought.", "Second thought.\nA continuation."]);
    expect(splitReadingParagraphs(" \n ")).toEqual([]);
  });

  it("groups long uninterrupted text without losing or rewriting sentences", () => {
    const sentence = "Checking each item once makes the result predictable and the cost easy to explain.";
    const text = Array(20).fill(sentence).join(" ");
    const paragraphs = splitReadingParagraphs(text);
    expect(paragraphs.length).toBeGreaterThan(2);
    expect(paragraphs.join(" ")).toBe(text);
    expect(paragraphs.every(p => p.startsWith("Checking") && p.endsWith("explain."))).toBe(true);
  });

  it("does not break a single sentence or split inside code fences", () => {
    const oneSentence = "uninterrupted ".repeat(60).trim();
    const code = "Explanation\n\n```js\nconst value = 1;\n\nlog(value);\n```";
    expect(splitReadingParagraphs(oneSentence)).toEqual([oneSentence]);
    expect(splitReadingParagraphs(code)).toEqual([code]);
  });
});
