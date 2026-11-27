import { createHash } from "node:crypto";
import { expect, it } from "vitest";
import { visualEvaluationVerdict, type VisualContract } from "../domain/visual-evaluation";
import type { VisualScene } from "../domain/visual-scene";
import { evaluateVisualDeterministically, type DeterministicVisualInput } from "./visual-evaluation-runner";
import { makeVisualPng } from "../test-fixtures/visual-png";

const explanation = "A bounded queue rejects items beyond its capacity.";
const hash = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");
const contract: VisualContract = { version: 1, id: "queue", explanationSha256: hash(explanation),
  purpose: "factual-diagram", audience: "intermediate", teachingGoal: "Show the capacity boundary.",
  audienceObjectives: ["Distinguish accepted and rejected items."], requirements: [{ id: "capacity", kind: "claim",
    statement: explanation, evidence: "source", evidenceReference: "Explanation sentence 1" }] };
const label = { kind: "text", x: 100, y: 100, size: 24, text: "Capacity", color: "#000000", align: "left" } as const;
const scene = (...items: VisualScene["items"]): VisualScene => ({ background: "#ffffff", items });
const encodedPng = makeVisualPng(1920, 1080);
function input(drawing = scene(label)): DeterministicVisualInput {
  return { contract, explanation, brief: "Show a capacity boundary.", plan: JSON.stringify(drawing), png: encodedPng,
    width: 1920, height: 1080, measure: () => ({ left: 0, right: 100, ascent: 20, descent: 4 }),
    measurementVersion: "test-metrics-1" };
}
const messages = (value: DeterministicVisualInput) => evaluateVisualDeterministically(value).checks.flatMap((check) => check.findings.map((f) => f.message));

it("hashes exact artifacts and never treats successful decoding as semantic approval", () => {
  const value = input();
  const report = evaluateVisualDeterministically(value);
  expect(report.artifacts).toEqual({ briefSha256: hash(value.brief), planSha256: hash(value.plan), pngSha256: hash(value.png) });
  expect(visualEvaluationVerdict(report, contract)).toBe("uncertain");
  expect(report.checks.find((check) => check.stage === "raster")?.summary).toContain("decoded with CRC");
  expect(report.checks.filter((check) => check.status === "not-run").map((check) => check.stage)).toEqual(["brief", "alignment", "audience"]);
  expect(report.checks.every((check) => check.requirementIds.length === 0)).toBe(true);
});
it("rejects a matching header without pixel data", () => {
  const value = input();
  value.png = value.png.subarray(0, 33);
  expect(visualEvaluationVerdict(evaluateVisualDeterministically(value), contract)).toBe("fail");
});
it("rejects an explanation changed after contract creation", () => {
  expect(() => evaluateVisualDeterministically({ ...input(), explanation: "A different explanation" })).toThrow("does not match");
});
it.each([0, -1, 1.5, NaN, 8193])("rejects invalid expected dimensions %s", (width) => {
  expect(() => evaluateVisualDeterministically({ ...input(), width })).toThrow("dimensions");
});
it("requires provenance for supplied measurements", () => {
  expect(() => evaluateVisualDeterministically({ ...input(), measurementVersion: undefined })).toThrow("measurement version");
});
it("keeps missing measurement uncertain", () => {
  const report = evaluateVisualDeterministically({ ...input(), measure: undefined });
  expect(report.checks[1].summary).toContain("measurement unavailable");
  expect(visualEvaluationVerdict(report, contract)).toBe("uncertain");
});
it.each(["not-json", "{}"])("reports malformed plans without crashing: %s", (plan) => {
  const report = evaluateVisualDeterministically({ ...input(), plan });
  expect(report.checks[1].status).toBe("fail");
  expect(visualEvaluationVerdict(report, contract)).toBe("fail");
});
it("detects clipped and overlapping labels using shared geometry checks", () => {
  expect(messages(input(scene({ ...label, x: 980 })))).toContain("Label 1 may be clipped.");
  expect(messages(input(scene(label, { ...label, x: 120 })))).toContain("Labels 1 and 2 overlap.");
});
it("detects low contrast and uses a fully covering background rectangle", () => {
  expect(messages(input(scene({ ...label, color: "#eeeeee" })))).toContain("Item 1: low label contrast.");
  const background = { kind: "rectangle", x: 80, y: 60, width: 180, height: 100, color: "#000000" } as const;
  expect(messages(input(scene(background, { ...label, color: "#ffffff" })))).toEqual([]);
  expect(messages(input(scene(background, label)))).toContain("Item 2: low label contrast.");
});
it("does not invent a single contrast background for partially covered text", () => {
  const partial = { kind: "rectangle", x: 150, y: 60, width: 100, height: 100, color: "#000000" } as const;
  expect(messages(input(scene(partial, label)))).toEqual([]);
  expect(evaluateVisualDeterministically(input(scene(partial, label))).checks[1].status).toBe("uncertain");
});
it("flags possible occlusion by later rectangles and strokes", () => {
  const rectangle = { kind: "rectangle", x: 80, y: 60, width: 180, height: 100, color: "#000000" } as const;
  expect(messages(input(scene(label, rectangle)))).toContain("Item 1: a later shape may obscure this label.");
  const line: VisualScene["items"][number] = { kind: "path", points: [{ x: 90, y: 90 }, { x: 220, y: 90 }], color: "#000000", fill: "none", closed: false, arrow: false, weight: 2 };
  expect(messages(input(scene(label, line)))).toContain("Item 1: a later shape may obscure this label.");
});
it("detects small text after export scaling, not just source font size", () => {
  const value = input(scene({ ...label, size: 14 }));
  value.width = 500; value.height = 350;
  expect(messages(value)).toContain("Item 1: exported text is small.");
});
it("detects paragraphs split into many short labels", () => {
  const value = input(scene(...Array.from({ length: 9 }, (_, i) => ({ ...label, text: "five words in this label", y: 40 + i * 60 }))));
  expect(messages(value)).toContain("The drawing has too much text; keep longer explanations in the caption.");
});
it.each(["truncated", "signature", "dimensions", "ihdr"])("fails invalid PNG metadata: %s", (defect) => {
  const value = input();
  const png = Buffer.from(value.png);
  if (defect === "signature") png[0] = 0;
  if (defect === "dimensions") png.writeUInt32BE(1280, 16);
  if (defect === "ihdr") png.writeUInt32BE(12, 8);
  value.png = defect === "truncated" ? png.subarray(0, 20) : png;
  expect(visualEvaluationVerdict(evaluateVisualDeterministically(value), contract)).toBe("fail");
});
it("caps findings without losing the need for revision", () => {
  const value = input(scene(...Array.from({ length: 96 }, () => ({ ...label, color: "#eeeeee" }))));
  const report = evaluateVisualDeterministically(value);
  expect(report.checks[1].findings).toHaveLength(40);
  expect(report.checks[1].summary).toContain("truncated");
  expect(report.checks[1].status).toBe("revise");
});
