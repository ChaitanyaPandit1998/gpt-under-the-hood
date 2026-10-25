import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const stylesheet = readFileSync(new URL("../styles.css", import.meta.url), "utf8");

describe("diagram print and narrow-screen styling", () => {
  it("keeps the print heading with its diagram title at page breaks", () => {
    const headingRule = stylesheet.match(/\.diagram-print-heading\s*\{([^}]*)\}/)?.[1] ?? "";

    expect(headingRule).toMatch(/break-inside\s*:\s*avoid-page/);
    expect(headingRule).toMatch(/page-break-inside\s*:\s*avoid/);
    expect(headingRule).toMatch(/break-after\s*:\s*avoid-page/);
    expect(stylesheet).toMatch(/\.diagram-section-caption\s*\{\s*display\s*:\s*none\s*!important\s*\}/);
  });

  it("gives the narrow-screen fit control a persistent, touch-sized target", () => {
    const mobileRule = stylesheet.match(/@media\s*\(max-width:\s*700px\)\s*\{\.flow-canvas\s*\{[^}]*\}\.diagram-fit-control\s*\{([^}]*)\}/)?.[1] ?? "";

    expect(stylesheet).toMatch(/\.flow-canvas\s*\{\s*position\s*:\s*relative/);
    expect(mobileRule).toMatch(/top\s*:\s*10px/);
    expect(mobileRule).toMatch(/bottom\s*:\s*auto/);
    expect(mobileRule).toMatch(/min-height\s*:\s*44px/);
    expect(stylesheet).toMatch(/\.flow-canvas\.compact-layout\s+\.flow-card-node\s*\{/);
  });
});
