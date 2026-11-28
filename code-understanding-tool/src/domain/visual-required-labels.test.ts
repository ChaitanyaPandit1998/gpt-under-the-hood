import { describe, expect, it } from "vitest";
import { assertRequiredVisualLabels, MissingVisualLabelsError, requiredVisualLabels } from "./visual-required-labels";
import type { VisualScene } from "./visual-scene";

const scene = (...labels: string[]): VisualScene => ({ background: "#ffffff", items: labels.map((text) => ({
  kind: "text", text, x: 50, y: 60, size: 20, color: "#202020", align: "left",
})) });

describe("required visual labels", () => {
  it("adds the fictional qualifier only for Story illustrations and deduplicates it", () => {
    expect(requiredVisualLabels(["Fictional analogy", "7"], "story-illustration")).toEqual(["Fictional analogy", "7"]);
    expect(requiredVisualLabels(undefined, "factual-diagram")).toEqual([]);
  });
  it("requires whole labels and preserves meaningful capitalization and notation", () => {
    const required = ["Unproved", "7", "x < 1", "R"];
    expect(() => assertRequiredVisualLabels(scene("This is Unproved", "17", "x > 1", "r"), required))
      .toThrow(new MissingVisualLabelsError(required));
    expect(() => assertRequiredVisualLabels(scene(...required), required)).not.toThrow();
    expect(() => assertRequiredVisualLabels(scene("  Cafe\u0301 "), ["Caf\u00e9"])).not.toThrow();
  });
});
