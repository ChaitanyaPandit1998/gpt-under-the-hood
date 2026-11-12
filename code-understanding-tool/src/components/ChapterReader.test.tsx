import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { Explanation } from "../domain/explanation-types";
import { ChapterPage, ChapterReader } from "./ChapterReader";

const explanation: Explanation = { title: "Example", summary: "Summary", concepts: [], walkthrough: [], visuals: [], visual_note: "", assumptions: [] };

describe("chapter reader", () => {
  it("shows every section for short explanations", () => {
    const html = renderToStaticMarkup(<ChapterReader explanation={explanation}><ChapterPage id="overview">Overview content</ChapterPage><ChapterPage id="recap">Recap content</ChapterPage></ChapterReader>);
    expect(html).not.toContain("hidden=");
    expect(html).toContain('aria-pressed="false">Chapters');
    expect(html).not.toContain('aria-label="Explanation chapters"');
    expect(html).toContain("Recap content");
  });

  it("opens long explanations at the overview and retains hidden content for full view and print", () => {
    const html = renderToStaticMarkup(<ChapterReader explanation={{ ...explanation, summary: "word ".repeat(900) }}><ChapterPage id="overview">Overview content</ChapterPage><ChapterPage id="recap">Recap content</ChapterPage></ChapterReader>);
    expect(html).toContain('aria-current="step"');
    expect(html).toContain("Chapter 1 of 2");
    expect(html).toContain('class="chapter-page" hidden="">Recap content');
    expect(html).toContain('disabled="">Previous');
    expect(html).toContain('aria-label="Chapter navigation"');
  });

  it("prints all chapter pages and hides reading controls", () => {
    const styles = readFileSync(new URL("../styles.css", import.meta.url), "utf8");
    expect(styles).toMatch(/@media print\s*\{\s*\.chapter-toolbar,\.chapter-sidebar,\.chapter-heading,\.chapter-pagination\s*\{display:none!important\}/);
    expect(styles).toMatch(/\.chapter-page\[hidden\]\s*\{display:block!important\}/);
  });
});
