import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { diagramCoveragePlan } from "./diagram-coverage-plan";
import { createReviewedExplanation, type CodexRunner } from "./explanation-service";

const cases = JSON.parse(readFileSync(new URL("../../evals/cases.json", import.meta.url), "utf8")) as { cases: { id: string; source_text: string }[] };
const source = cases.cases.find((item) => item.id === "python-field-options-independent-decisions")!.source_text;
const draft = { title: "Field options", summary: "Prepare supplied settings.", concepts: [], walkthrough: [], assumptions: [], visual_note: "", visuals: [] };

describe("Python diagram planning aid", () => {
  it("includes independent guards, both version fallbacks, and the final filter", () => {
    const plan = diagramCoveragePlan(source, "auto");
    for (const fragment of ["if example is not unset:", "if examples is not None:", "if regex is not None:", "selected = json_schema_extra or extra", "if v2:", "kwargs['pattern'] = pattern or regex", "kwargs['regex'] = pattern or regex", "filtered = {k: v for k, v in kwargs.items() if v is not unset}"]) expect(plan).toContain(fragment);
    expect(plan).toContain("complementary outgoing paths");
    expect(plan).toContain("NOT instructions or verified coverage");
  });

  it("ignores removed diff decisions and preserves original line numbers", () => {
    const plan = diagramCoveragePlan("--- a/example.py\n+++ b/example.py\n@@ -1,2 +1,2 @@\n-if removed:\n+if current:\n+    kwargs['rule'] = pattern or regex", "auto");
    expect(plan).not.toContain("if removed:");
    expect(plan).toContain('"lines":[5]');
    expect(plan).toContain('"lines":[6]');
  });

  it("ignores strings, comments, docstrings and diff operators inside literals", () => {
    const input = "# if comment:\ntext = 'if string:'\nmessage = \"x or y\"\nnotes = '''\nif documentation:\nx = a or b\n'''\nvalue = pattern or regex # explanatory comment\nif ready:\n    pass";
    const plan = diagramCoveragePlan(input, "auto");
    expect(plan).not.toContain("if documentation:");
    expect(plan).not.toContain("if comment:");
    expect(plan).not.toContain("if string:");
    expect(plan).not.toContain("x or y");
    expect(plan).toContain("value = pattern or regex");
    expect(plan).toContain('"lines":[9]');
  });

  it("keeps repeated candidate coordinates and bounds the plan", () => {
    expect(diagramCoveragePlan("if ready:\n    pass\nif ready:", "auto")).toContain('"lines":[1,3]');
    const large = diagramCoveragePlan(Array.from({ length: 1000 }, (_, i) => `if condition_${i}:`).join("\n"), "auto");
    expect(large).toContain("Candidate list truncated");
    expect(large.length).toBeLessThan(15000);
    expect(large).not.toContain("condition_60:");
  });

  it("respects disabled visuals and makes no plan for ordinary prose", () => {
    expect(diagramCoveragePlan(source, "no")).toBe("");
    expect(diagramCoveragePlan("A version adapter passes options to its parent.", "auto")).toBe("");
  });

  it("supports assertion and conditional-expression candidates", () => {
    const plan = diagramCoveragePlan("assert default is ...\nvalue = left if enabled else right", "auto");
    expect(plan).toContain("failure raises when assertions enabled");
    expect(plan).toContain("conditional expression: show both selected values");
  });

  it("carries the complete plan through draft, review, repair, and final audit", async () => {
    const prompts: string[] = [];
    const runner: CodexRunner = async (prompt) => {
      prompts.push(prompt);
      if (prompt.includes("You are the second-pass quality reviewer")) return { status: "revise", findings: ["Show every independent guard and fallback."] };
      if (prompt.includes("You are performing the final quality check after a repair")) return { status: "pass", findings: [] };
      return draft;
    };
    const result = await createReviewedExplanation({ sourceText: source, materialKind: "code", teachingMode: "control_flow", explanationDepth: "balanced", learnerLevel: "beginner", visualPreference: "auto" }, runner);
    expect(prompts).toHaveLength(4);
    for (const prompt of prompts) {
      expect(prompt).toContain("Diagram planning aid");
      expect(prompt).toContain("comprehension filter");
      expect(prompt).toContain("WHOLE candidate list");
    }
    expect(result.quality_review_status).toBe("revised");
  });

  it("does not upgrade a remaining missing-branch finding to passed", async () => {
    const runner: CodexRunner = async (prompt) => prompt.includes("quality reviewer") || prompt.includes("final quality check")
      ? { status: "revise", findings: ["The final unset filter still lacks the omit branch."] } : draft;
    const result = await createReviewedExplanation({ sourceText: source, materialKind: "code", teachingMode: "control_flow", explanationDepth: "balanced", learnerLevel: "beginner", visualPreference: "auto" }, runner);
    expect(result.quality_review_status).toBe("incomplete");
    expect(result.quality_review_note).toContain("final unset filter still lacks the omit branch");
  });
});
