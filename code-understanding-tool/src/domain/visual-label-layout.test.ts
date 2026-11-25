import { expect, it } from "vitest";
import { checkVisualLabels } from "./visual-label-layout";
import type { VisualScene } from "./visual-scene";

const label = { kind: "text", text: "A label", x: 50, y: 50, size: 20, color: "#000000", align: "left" } as const;
const measure = () => ({ left: 0, right: 80, ascent: 16, descent: 4 });
const scene = (...items: VisualScene["items"]): VisualScene => ({ background: "#ffffff", items });

it("accepts separated labels", () => {
  expect(checkVisualLabels(scene(label, { ...label, y: 90 }), measure)).toEqual([]);
});
it("detects glyph bounds outside the canvas", () => {
  expect(checkVisualLabels(scene({ ...label, x: 950 }), measure)).toContain("Label 1 may be clipped.");
  expect(checkVisualLabels(scene({ ...label, y: 5 }), measure)).toContain("Label 1 may be clipped.");
});
it("uses alignment-aware measured bounds", () => {
  expect(checkVisualLabels(scene({ ...label, x: 20, align: "right" }),
    () => ({ ...measure(), left: 80, right: 0 }))).toContain("Label 1 may be clipped.");
});
it("detects label overlap but allows touching bounds", () => {
  expect(checkVisualLabels(scene(label, { ...label, x: 100 }), measure)).toContain("Labels 1 and 2 overlap.");
  expect(checkVisualLabels(scene(label, { ...label, x: 130 }), measure)).toEqual([]);
});
it("rejects paragraphs and invalid measurements", () => {
  expect(checkVisualLabels(scene({ ...label, text: "x".repeat(61) }), measure)[0]).toContain("too long");
  expect(checkVisualLabels(scene(label), () => ({ ...measure(), right: NaN }))).toContain("Label 1 may be clipped.");
});
it("rejects a paragraph split across short labels", () => {
  const labels = Array.from({ length: 9 }, (_, i) => ({ ...label, text: "five words in this label", y: 30 + i * 50 }));
  expect(checkVisualLabels(scene(...labels), measure)[0]).toContain("too much text");
});
