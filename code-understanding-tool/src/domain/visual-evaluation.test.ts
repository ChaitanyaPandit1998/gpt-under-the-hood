import { describe, expect, it } from "vitest";
import { validateVisualContract, validateVisualEvaluationReport, visualCheckStages, visualEvaluationVerdict,
  type VisualContract, type VisualEvaluationReport } from "./visual-evaluation";

const digest = "a".repeat(64);
const contract: VisualContract = {
  version: 1, id: "contract-1", explanationSha256: digest, purpose: "factual-diagram", audience: "intermediate",
  teachingGoal: "Explain how a bounded queue handles capacity.",
  audienceObjectives: ["Distinguish an accepted item from a rejected item."],
  requirements: [
    { id: "capacity", kind: "claim", statement: "Items beyond capacity are rejected.", evidence: "source", evidenceReference: "Explanation: capacity rule" },
    { id: "no-measurements", kind: "prohibition", statement: "Do not imply observed throughput.", evidence: "background", evidenceReference: "No runtime measurements supplied" },
  ],
};
function report(): VisualEvaluationReport {
  return {
    version: 1, contractId: contract.id, explanationSha256: digest,
    artifacts: { briefSha256: digest, planSha256: digest, pngSha256: digest }, repairAttempt: 0,
    checks: visualCheckStages.map((stage) => ({ stage, status: "pass", durationMs: 10,
      reviewer: { kind: "human", name: "Test reviewer", version: "1" }, summary: "Checked against the supplied artifacts.",
      requirementIds: contract.requirements.map(({ id }) => id), findings: [] })),
  };
}

describe("visual evaluation contracts", () => {
  it("accepts generic requirements and explicit audience objectives", () => {
    expect(validateVisualContract(contract)).toEqual(contract);
  });
  it.each(["beginner", "intermediate", "advanced"] as const)("supports %s without topic-specific rules", (audience) => {
    expect(validateVisualContract({ ...contract, audience }).audience).toBe(audience);
  });
  it.each([
    { version: 2 }, { unknown: true }, { audience: "expert" }, { explanationSha256: "stale" },
    { teachingGoal: " " }, { requirements: [] }, { audienceObjectives: [] },
    { requirements: [contract.requirements[0], contract.requirements[0]] },
    { requirements: [{ ...contract.requirements[0], evidenceReference: "" }] },
  ])("rejects invalid contract fields %j", (change) => {
    expect(() => validateVisualContract({ ...contract, ...change })).toThrow();
  });
});

describe("visual evaluation reports", () => {
  it("passes a complete, covered report", () => {
    expect(validateVisualEvaluationReport(report(), contract)).toEqual(report());
    expect(visualEvaluationVerdict(report(), contract)).toBe("pass");
  });
  it.each(["contractId", "explanationSha256"] as const)("rejects a stale %s", (field) => {
    expect(() => validateVisualEvaluationReport({ ...report(), [field]: "b".repeat(64) }, contract)).toThrow();
  });
  it("rejects malformed artifact digests and extra fields", () => {
    const value = report();
    value.artifacts.pngSha256 = "not-a-digest";
    expect(() => validateVisualEvaluationReport(value, contract)).toThrow();
    expect(() => validateVisualEvaluationReport({ ...report(), verdict: "pass" }, contract)).toThrow();
  });
  it("accepts a bound caption digest but rejects malformed caption bindings", () => {
    const value = report(); value.artifacts.captionSha256 = digest;
    expect(validateVisualEvaluationReport(value, contract).artifacts.captionSha256).toBe(digest);
    value.artifacts.captionSha256 = "invalid";
    expect(() => validateVisualEvaluationReport(value, contract)).toThrow();
  });
  it("rejects duplicate stages and unknown requirement references", () => {
    const value = report();
    value.checks[1] = value.checks[0];
    expect(() => validateVisualEvaluationReport(value, contract)).toThrow();
    value.checks = report().checks;
    value.checks[0].requirementIds = ["invented"];
    expect(() => validateVisualEvaluationReport(value, contract)).toThrow();
  });
  it("rejects unbounded repairs, invalid timings and incomplete reviewers", () => {
    expect(() => validateVisualEvaluationReport({ ...report(), repairAttempt: 2 }, contract)).toThrow();
    for (const durationMs of [-1, NaN, Infinity]) {
      const value = report();
      value.checks[0].durationMs = durationMs;
      expect(() => validateVisualEvaluationReport(value, contract)).toThrow();
    }
    const value = report();
    value.checks[0].reviewer.version = "";
    expect(() => validateVisualEvaluationReport(value, contract)).toThrow();
  });
  it.each(["not-run", "uncertain"] as const)("does not approve a %s review", (status) => {
    const value = report();
    value.checks[2].status = status;
    expect(visualEvaluationVerdict(value, contract)).toBe("uncertain");
  });
  it("does not approve missing stages or uncovered requirements", () => {
    const value = report();
    value.checks.pop();
    expect(visualEvaluationVerdict(value, contract)).toBe("uncertain");
    value.checks = report().checks.map((check) => ({ ...check, requirementIds: [] }));
    expect(visualEvaluationVerdict(value, contract)).toBe("uncertain");
  });
  it("blocks a factual contradiction even when every stage claims to pass", () => {
    const value = report();
    value.checks[3].findings = [{ severity: "critical", message: "The arrow reverses the stated relationship.",
      evidence: "The image points from rejected to accepted.", requirementIds: ["capacity"] }];
    expect(visualEvaluationVerdict(value, contract)).toBe("fail");
    value.checks.pop();
    expect(visualEvaluationVerdict(value, contract)).toBe("fail");
  });
  it("requires revision for major findings but permits minor notes", () => {
    const value = report();
    const finding = { severity: "major" as const, message: "A label is hard to read.", evidence: "Small text in PNG.", requirementIds: [] };
    value.checks[2].findings = [finding];
    expect(visualEvaluationVerdict(value, contract)).toBe("revise");
    value.checks[2].findings = [{ ...finding, severity: "minor" }];
    expect(visualEvaluationVerdict(value, contract)).toBe("pass");
    value.checks[2].status = "revise";
    expect(visualEvaluationVerdict(value, contract)).toBe("revise");
    value.checks[2].status = "fail";
    expect(visualEvaluationVerdict(value, contract)).toBe("fail");
  });
  it("validates finding evidence and references too", () => {
    const value = report();
    value.checks[0].findings = [{ severity: "minor", message: "Check a label.", evidence: "", requirementIds: [] }];
    expect(() => validateVisualEvaluationReport(value, contract)).toThrow();
    value.checks[0].findings[0].evidence = "Observed in PNG";
    value.checks[0].findings[0].requirementIds = ["unknown"];
    expect(() => validateVisualEvaluationReport(value, contract)).toThrow();
  });
});
