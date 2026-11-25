import { createHash } from "node:crypto";
import { checkVisualLabels, type LabelMetrics } from "../domain/visual-label-layout.ts";
import { validateVisualContract, validateVisualEvaluationReport, visualCheckStages,
  type VisualContract, type VisualEvaluationReport } from "../domain/visual-evaluation.ts";
import { validateVisualScene, visualCanvas, type VisualScene } from "../domain/visual-scene.ts";
import { decodeVisualPng } from "./visual-png.ts";

type TextItem = Extract<VisualScene["items"][number], { kind: "text" }>;
type Finding = VisualEvaluationReport["checks"][number]["findings"][number];
type Box = { left: number; right: number; top: number; bottom: number };
const intersects = (a: Box, b: Box) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
const contains = (a: Box, b: Box) => a.left <= b.left && a.right >= b.right && a.top <= b.top && a.bottom >= b.bottom;
const digest = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");
function luminance(color: string) {
  const [r, g, b] = [1, 3, 5].map((offset) => {
    const channel = parseInt(color.slice(offset, offset + 2), 16) / 255;
    return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export type DeterministicVisualInput = {
  contract: VisualContract;
  explanation: string;
  brief: string;
  plan: string;
  png: Uint8Array;
  width: number;
  height: number;
  measure?: (item: TextItem) => LabelMetrics;
  measurementVersion?: string;
  repairAttempt?: 0 | 1;
};

// Hash exact supplied artifacts. Decode and plan checks are not semantic approval.
export function evaluateVisualDeterministically(input: DeterministicVisualInput): VisualEvaluationReport {
  const contract = validateVisualContract(input.contract);
  if (digest(input.explanation) !== contract.explanationSha256) throw new Error("The explanation does not match the visual contract.");
  if (![input.width, input.height].every((size) => Number.isInteger(size) && size >= 1 && size <= 8192)) {
    throw new Error("Invalid expected visual dimensions.");
  }
  if (input.measure && !input.measurementVersion?.trim()) throw new Error("Label measurements require a font/measurement version.");
  const checks: VisualEvaluationReport["checks"] = visualCheckStages.map((stage) => ({
    stage, status: "not-run", reviewer: { kind: "deterministic", name: "visual-preflight", version: "1" },
    durationMs: 0, summary: "No reviewer ran for this stage.", requirementIds: [], findings: [],
  }));
  const planCheck = checks.find((check) => check.stage === "plan")!;
  const started = performance.now();
  const issues: Finding[] = [];
  const issue = (message: string, evidence: string) => issues.push({ severity: "major", message, evidence, requirementIds: [] });
  let scene: VisualScene | undefined;
  try { scene = validateVisualScene(JSON.parse(input.plan)); }
  catch {
    planCheck.status = "fail";
    issue("Invalid drawing plan.", "The supplied JSON does not satisfy the scene schema.");
  }
  if (scene) {
    const scale = Math.min(input.width / visualCanvas.width, input.height / visualCanvas.height);
    if (input.measure) {
      // Measure once so repeated checks use the same font snapshot.
      const metrics = new Map(scene.items.filter((item) => item.kind === "text").map((item) => [item, input.measure!(item)]));
      for (const message of checkVisualLabels(scene, (item) => metrics.get(item)!)) issue(message, "Measured scene label bounds.");
      for (const [index, item] of scene.items.entries()) {
        if (item.kind !== "text") continue;
        const m = metrics.get(item)!;
        const box = { left: item.x - m.left, right: item.x + m.right, top: item.y - m.ascent, bottom: item.y + m.descent };
        let background = scene.background;
        let knownBackground = true;
        for (const [shapeIndex, shape] of scene.items.entries()) {
          if (shape.kind === "text") continue;
          const shapeBox = shape.kind === "rectangle"
            ? { left: shape.x, right: shape.x + shape.width, top: shape.y, bottom: shape.y + shape.height }
            : { left: Math.min(...shape.points.map((p) => p.x)) - shape.weight / 2,
              right: Math.max(...shape.points.map((p) => p.x)) + shape.weight / 2,
              top: Math.min(...shape.points.map((p) => p.y)) - shape.weight / 2,
              bottom: Math.max(...shape.points.map((p) => p.y)) + shape.weight / 2 };
          if (!intersects(box, shapeBox)) continue;
          if (shapeIndex > index) issue(`Item ${index + 1}: a later shape may obscure this label.`, `Shape item ${shapeIndex + 1} intersects measured label bounds; path checks use conservative bounding boxes.`);
          if (shapeIndex < index) {
            if (shape.kind === "rectangle" && contains(shapeBox, box)) { background = shape.color; knownBackground = true; }
            else knownBackground = false;
          }
        }
        if (knownBackground) {
          const a = luminance(item.color), b = luminance(background);
          const ratio = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
          if (ratio < 4.5) issue(`Item ${index + 1}: low label contrast.`, `Plan colors produce ${ratio.toFixed(2)}:1 contrast against ${background}; preflight minimum is 4.5:1.`);
        }
      }
      planCheck.reviewer.version = `1; ${input.measurementVersion}`;
    }
    for (const [index, item] of scene.items.entries()) {
      if (item.kind === "text" && item.size * scale < 14) issue(`Item ${index + 1}: exported text is small.`, `Nominal font size ${(item.size * scale).toFixed(2)} output pixels; preflight minimum is 14. This is not mobile legibility validation.`);
    }
    planCheck.status = issues.length ? "revise" : "uncertain";
  }
  planCheck.findings = issues.slice(0, 40);
  planCheck.summary = `Plan-only checks${input.measure ? " with supplied label metrics" : "; label measurement unavailable"}. Semantic fidelity and actual raster appearance remain unreviewed.${issues.length > 40 ? " Findings truncated to 40." : ""}`;
  planCheck.durationMs = performance.now() - started;
  const raster = checks.find((check) => check.stage === "raster")!;
  const rasterStarted = performance.now();
  try {
    const decoded = decodeVisualPng(input.png, input.width, input.height);
    raster.status = "uncertain";
    raster.summary = "PNG decoded with CRC and dimension checks. OCR, legibility and semantic visual review were not run.";
    if (decoded.visiblePixels === 0 || decoded.uniform) {
      raster.status = "revise";
      raster.findings.push({ severity: "major", message: "The raster is transparent or a single uniform color.",
        evidence: `${decoded.visiblePixels} nontransparent pixels; uniform=${decoded.uniform}. No visible teaching structure was detected.`, requirementIds: [] });
    }
  } catch {
    raster.status = "fail";
    raster.summary = "PNG decoding, CRC, format, size or dimension checks failed.";
    raster.findings.push({ severity: "critical", message: "PNG could not be safely decoded at the requested dimensions.",
      evidence: `Expected ${input.width} x ${input.height}; static non-interlaced 8-bit PNG within evaluation limits.`, requirementIds: [] });
  }
  raster.durationMs = performance.now() - rasterStarted;
  return validateVisualEvaluationReport({ version: 1, contractId: contract.id, explanationSha256: contract.explanationSha256,
    artifacts: { briefSha256: digest(input.brief), planSha256: digest(input.plan), pngSha256: digest(input.png) },
    repairAttempt: input.repairAttempt ?? 0, checks }, contract);
}
