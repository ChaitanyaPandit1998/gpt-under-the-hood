import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { expect, it, vi } from "vitest";
import { visualCheckStages, visualEvaluationVerdict, type VisualContract } from "../domain/visual-evaluation";
import { makeVisualPng } from "../test-fixtures/visual-png";
import type { CodexRunner } from "./explanation-service";
import { reviewVisualImage } from "./visual-review-service";

const explanation = "Items beyond queue capacity are rejected.";
const contract: VisualContract = { version: 1, id: "capacity", explanationSha256: createHash("sha256").update(explanation).digest("hex"),
  purpose: "factual-diagram", audience: "intermediate", teachingGoal: "Show the capacity boundary.", audienceObjectives: ["Explain rejection at capacity."],
  requirements: [{ id: "boundary", kind: "claim", statement: explanation, evidence: "source", evidenceReference: "Explanation" }] };
const png = makeVisualPng();
const input = () => ({ contract, explanation, caption: "Illustrative capacity boundary, not measured throughput.", brief: "Show the boundary.",
  plan: JSON.stringify({ background: "#ffffff", items: [{ kind: "text", text: "Capacity", x: 80, y: 100, size: 24, color: "#000000", align: "left" }] }),
  png, width: 1000, height: 700 });
const reviewer = { name: "mock-vision-reviewer", version: "test-1" };
const response = () => ({ imageInspected: true, checks: visualCheckStages.map((stage) => ({ stage, status: "pass", summary: "Mocked review result, not a live quality assessment.",
  requirementIds: ["boundary"], findings: [] as { severity: string; message: string; evidence: string; requirementIds: string[] }[] })) });

it("attaches the exact PNG and caption, records reviewer identity and cleans temporary files", async () => {
  let directory = "";
  const run = vi.fn<CodexRunner>(async (prompt, schema, _output, cwd, image) => {
    directory = cwd;
    expect(image).toBeTruthy();
    expect(await readFile(image!)).toEqual(png);
    expect((await stat(image!)).mode & 0o777).toBe(0o600);
    expect(prompt).toContain(input().caption);
    expect(prompt).toContain("Inspect the actual attachment");
    expect(prompt).toContain("narrative transformations");
    expect(prompt).toContain("untrusted reference data");
    expect(JSON.parse(await readFile(schema, "utf8")).required).toContain("imageInspected");
    return response();
  });
  const report = await reviewVisualImage(input(), { run, reviewer });
  expect(report.artifacts.captionSha256).toBe(createHash("sha256").update(input().caption).digest("hex"));
  expect(report.checks.every((check) => check.reviewer.name === reviewer.name)).toBe(true);
  expect(visualEvaluationVerdict(report, contract)).toBe("pass");
  expect(run).toHaveBeenCalledTimes(1);
  await expect(stat(directory)).rejects.toThrow();
});
it("does not call the reviewer for undecodable PNGs", async () => {
  const run = vi.fn<CodexRunner>();
  const report = await reviewVisualImage({ ...input(), png: png.subarray(0, 33) }, { run, reviewer });
  expect(run).not.toHaveBeenCalled();
  expect(visualEvaluationVerdict(report, contract)).toBe("fail");
});
it("retains deterministic blockers when the model says pass", async () => {
  const run = vi.fn<CodexRunner>().mockResolvedValue(response());
  const report = await reviewVisualImage({ ...input(), png: makeVisualPng(1000, 700, "uniform") }, { run, reviewer });
  expect(visualEvaluationVerdict(report, contract)).toBe("revise");
  expect(report.checks.find((check) => check.stage === "raster")?.findings[0].message).toContain("uniform");
});
it("keeps critical factual findings separate from aesthetics", async () => {
  const output = response();
  output.checks[3].findings = [{ severity: "critical", message: "The arrow reverses the stated relation.", evidence: "Arrow beside the capacity label.", requirementIds: ["boundary"] }];
  const report = await reviewVisualImage(input(), { run: vi.fn<CodexRunner>().mockResolvedValue(output), reviewer });
  expect(visualEvaluationVerdict(report, contract)).toBe("fail");
});
it("requires complete alignment coverage even if another stage covers every requirement", async () => {
  const output = response(); output.checks[3].requirementIds = [];
  const report = await reviewVisualImage(input(), { run: vi.fn<CodexRunner>().mockResolvedValue(output), reviewer });
  expect(visualEvaluationVerdict(report, contract)).toBe("uncertain");
});
it.each(["unseen", "missing-stage", "duplicate-stage", "unknown-reference", "extra-field"])("does not approve invalid/incomplete review: %s", async (kind) => {
  const output = response();
  if (kind === "unseen") output.imageInspected = false;
  if (kind === "missing-stage") output.checks.pop();
  if (kind === "duplicate-stage") output.checks[1] = output.checks[0];
  if (kind === "unknown-reference") output.checks[0].requirementIds = ["invented"];
  const value = kind === "extra-field" ? { ...output, verdict: "pass" } : output;
  const report = await reviewVisualImage(input(), { run: vi.fn<CodexRunner>().mockResolvedValue(value), reviewer });
  expect(visualEvaluationVerdict(report, contract)).toBe("uncertain");
});
it("handles provider failure conservatively without leaking diagnostics", async () => {
  const run = vi.fn<CodexRunner>().mockRejectedValue(new Error("private provider diagnostic"));
  const report = await reviewVisualImage(input(), { run, reviewer });
  expect(visualEvaluationVerdict(report, contract)).toBe("uncertain");
  expect(JSON.stringify(report)).not.toContain("private provider diagnostic");
  expect(run).toHaveBeenCalledTimes(1);
});
it("propagates cancellation and removes the attachment", async () => {
  const controller = new AbortController();
  let directory = "";
  const run = vi.fn<CodexRunner>(async (_prompt, _schema, _output, cwd) => {
    directory = cwd; controller.abort(new Error("cancelled")); return response();
  });
  await expect(reviewVisualImage(input(), { run, reviewer, signal: controller.signal })).rejects.toThrow("cancelled");
  await expect(stat(directory)).rejects.toThrow();
});
it("rejects oversized or missing captions before provider calls", async () => {
  const run = vi.fn<CodexRunner>();
  for (const caption of [" ", "x".repeat(4001)]) await expect(reviewVisualImage({ ...input(), caption }, { run, reviewer })).rejects.toThrow();
  expect(run).not.toHaveBeenCalled();
});
