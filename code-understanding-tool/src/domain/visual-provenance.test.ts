import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { appendVisualReview, mergeVisualReviewHistory, validateGenerationStage, validateSavedVisualArtifacts, validateVisualProvenance, type VisualProvenance } from "./visual-provenance";
import { visualCheckStages, type VisualContract } from "./visual-evaluation";
import { exportPortableStudy, importPortableStudy } from "./portable-study";
import type { SavedStudy } from "./saved-study";
import { makeVisualPng } from "../test-fixtures/visual-png";

const sha = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");
const bytes = makeVisualPng();
const png = new Blob([new Uint8Array(bytes)], { type: "image/png" });
function fixture() {
  const brief = { disposition: "render" as const, visualType: "comparison" as const, takeaway: "The capacity is fixed.",
    composition: "Show the supplied capacity label.", content: [{ label: "Capacity", role: "illustrative" as const,
      statement: "Capacity is three.", chapterLines: [] }], guardrails: [], caption: "An illustrative capacity.", altText: "A capacity label." };
  const scene = { background: "#ffffff", items: [{ kind: "text" as const, text: "Capacity", x: 50, y: 80, size: 24, color: "#000000", align: "left" as const }] };
  const context = { explanation: "The capacity is three.", scope: "Chapter: Overview", audience: "intermediate" as const };
  const contract: VisualContract = { version: 1, id: "capacity", explanationSha256: sha(context.explanation), audience: "intermediate",
    purpose: "factual-diagram", teachingGoal: brief.takeaway, audienceObjectives: ["Understand the fixed capacity."],
    requirements: [{ id: "capacity", kind: "claim", statement: "Capacity is three.", evidence: "illustrative", evidenceReference: "Supplied example" }] };
  const provenance: VisualProvenance = { version: 1, brief, scene, context, reviews: [{ reviewedAt: "2026-11-22T12:00:00.000Z", contract,
    report: { version: 1, contractId: contract.id, explanationSha256: contract.explanationSha256, repairAttempt: 0,
      artifacts: { briefSha256: sha(JSON.stringify(brief)), planSha256: sha(JSON.stringify(scene)), pngSha256: sha(bytes), captionSha256: sha(brief.caption) },
      checks: visualCheckStages.map(stage => ({ stage, status: "pass", reviewer: { kind: "model", name: "Mock reviewer", version: "test-1" },
        summary: "Mock assessment, not a live evaluation.", durationMs: 1, requirementIds: ["capacity"], findings: [] })),
    } }] };
  return { png, alt: brief.altText, caption: brief.caption, resolution: "standard" as const, provenance };
}

describe("saved visual provenance", () => {
  it("accepts optional artifact-bound lineage and rejects altered generation artifacts", async () => {
    const image = fixture();
    const stage = { artifactSha256: sha(JSON.stringify(image.provenance.brief)),
      instructions: [{ id: "visual-direction", version: 1 as const, sha256: "a".repeat(64) }] };
    image.provenance.generation = { brief: stage };
    await expect(validateVisualProvenance(image.provenance, image)).resolves.toEqual(image.provenance);
    await expect(validateGenerationStage(stage, "different artifact")).rejects.toThrow();
    stage.instructions.push({ ...stage.instructions[0] });
    await expect(validateVisualProvenance(image.provenance, image)).rejects.toThrow();
  });

  it("merges concurrent review metadata but never replaces a newer image", () => {
    const image = fixture();
    const review = { ...image.provenance.reviews[0], contract: { ...image.provenance.reviews[0].contract, id: "new-checklist" },
      report: { ...image.provenance.reviews[0].report, contractId: "new-checklist" } };
    const incoming = { ...image, provenance: appendVisualReview(image.provenance, review) };
    const merged = mergeVisualReviewHistory(image, incoming);
    expect(merged?.provenance?.reviews).toHaveLength(2);
    expect(mergeVisualReviewHistory(merged, incoming)).toBe(merged);
    const replacement = { ...image, png: new Blob(["new PNG identity"], { type: "image/png" }) };
    expect(mergeVisualReviewHistory(replacement, incoming)).toBe(replacement);
  });
  it("accepts exact artifact bindings and older images without metadata", async () => {
    const image = fixture();
    await expect(validateVisualProvenance(image.provenance, image)).resolves.toEqual(image.provenance);
    await expect(validateSavedVisualArtifacts({ png, caption: "An older image" })).resolves.toBeUndefined();
  });

  it.each(["caption", "png", "explanation", "brief", "scene", "digest", "extra", "too-many"])("rejects mismatched or invalid metadata: %s", async (change) => {
    const image = fixture();
    if (change === "caption") image.caption = "Changed caption";
    if (change === "png") image.png = new Blob([new Uint8Array(makeVisualPng(1000, 700, "uniform"))], { type: "image/png" });
    if (change === "explanation") image.provenance.context.explanation = "Different explanation";
    if (change === "brief") image.provenance.brief.takeaway = "Different takeaway";
    if (change === "scene") image.provenance.scene.items[0].color = "#ff0000";
    if (change === "digest") image.provenance.reviews[0].report.artifacts.pngSha256 = "a".repeat(64);
    if (change === "extra") Object.assign(image.provenance, { approved: true });
    if (change === "too-many") image.provenance.reviews = Array.from({ length: 6 }, () => image.provenance.reviews[0]);
    await expect(validateSavedVisualArtifacts({ illustration: image })).rejects.toThrow();
  });

  it("rejects cyclic saved objects", async () => {
    const item: Record<string, unknown> = {};
    item.self = item;
    await expect(validateSavedVisualArtifacts(item)).rejects.toThrow();
  });

  it("keeps at most five reports without mutating prior history", () => {
    const { provenance } = fixture();
    let updated = provenance;
    for (let i = 0; i < 6; i++) updated = appendVisualReview(updated, { ...provenance.reviews[0], reviewedAt: `2026-11-22T12:00:0${i}.000Z` });
    expect(provenance.reviews).toHaveLength(1);
    expect(updated.reviews).toHaveLength(5);
    expect(updated.reviews[0].reviewedAt).toBe("2026-11-22T12:00:01.000Z");
  });

  it("round trips review history through portable studies and rejects altered imports", async () => {
    const image = fixture();
    const study: SavedStudy = { version: 1, mode: "docs", source: "Capacity is three.", depth: "balanced", level: "intermediate",
      explanation: { title: "Capacity", summary: "Capacity is fixed.", concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "" },
      reading: { illustration: image } };
    const file = await exportPortableStudy(study);
    const restored = await importPortableStudy(file);
    expect(restored).toEqual(study);
    const envelope = JSON.parse(await file.text());
    envelope.study.reading.illustration.provenance.context.explanation = "An altered context.";
    await expect(importPortableStudy(new Blob([JSON.stringify(envelope)]))).rejects.toThrow();
  });
});
