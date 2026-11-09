import { describe, expect, it } from "vitest";
import { ExplanationValidationError, validateExplanation } from "./response-validation";

const validExplanation = {
  title: "Example",
  summary: "A short explanation.",
  concepts: [{ term: "Concept", explanation: "Supported by the input.", source_ranges: [{ start_line: 1, end_line: 1 }] }],
  walkthrough: [],
  assumptions: [],
  visual_note: "",
  visuals: [{ title: "Map", kind: "control_flow", nodes: [{ id: "start", label: "Start", description: "Begin.", source_ranges: [] }], edges: [] }],
};

describe("validateExplanation diagnostics", () => {
  it("classifies an out-of-range source link without revealing generated content", () => {
    const explanation = {
      ...validExplanation,
      visuals: [{ ...validExplanation.visuals[0], nodes: [{ ...validExplanation.visuals[0].nodes[0], source_ranges: [{ start_line: 1, end_line: 3 }] }] }],
    };

    try {
      validateExplanation(explanation, "one line only");
      throw new Error("Expected validation to fail.");
    } catch (error) {
      expect(error).toBeInstanceOf(ExplanationValidationError);
      expect((error as ExplanationValidationError).failureCategory).toBe("explanation_visual_node_source_range_out_of_bounds");
      expect((error as Error).message).toBe("The generated explanation was incomplete or invalid. Please retry the request.");
      expect((error as Error).message).not.toContain("one line only");
    }
  });

  it("still accepts a structurally valid, source-linked explanation", () => {
    expect(validateExplanation(validExplanation, "one line only")).toEqual(validExplanation);
  });
});
