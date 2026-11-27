import { contractFromVisualDraft, type VisualReviewDraft } from "../domain/visual-review-draft";
import { validateVisualContract, validateVisualEvaluationReport } from "../domain/visual-evaluation";
import type { RenderedIllustration } from "../domain/saved-study";
import type { VisualBrief } from "../domain/visual-brief";
import type { VisualScene } from "../domain/visual-scene";

async function sha256(bytes: Uint8Array<ArrayBuffer>) {
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
const hashText = (value: string) => sha256(new TextEncoder().encode(value));

export async function requestVisualReview(input: { draft: VisualReviewDraft; explanation: string; brief: VisualBrief; scene: VisualScene; image: RenderedIllustration }, signal: AbortSignal) {
  const { explanation, image } = input;
  const contract = validateVisualContract(contractFromVisualDraft(input.draft, crypto.randomUUID(), await hashText(explanation)));
  if (image.png.size > 10 * 1024 * 1024) throw new Error("The image exceeds the 10 MiB review limit.");
  const bytes = new Uint8Array(await image.png.arrayBuffer());
  const brief = JSON.stringify(input.brief), plan = JSON.stringify(input.scene);
  const artifacts = { briefSha256: await hashText(brief), planSha256: await hashText(plan), pngSha256: await sha256(bytes), captionSha256: await hashText(image.caption) };
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 8192) binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
  signal.throwIfAborted();
  const response = await fetch("/api/visual-review", { method: "POST", headers: { "Content-Type": "application/json" }, signal,
    body: JSON.stringify({ contract, explanation, brief, plan, caption: image.caption, pngBase64: btoa(binary), resolution: image.resolution, confirmed: true }) });
  const data = await response.json() as { report?: unknown; error?: string };
  if (!response.ok) throw new Error(data.error || "Could not review the image. Your image is unchanged.");
  const report = validateVisualEvaluationReport(data.report, contract);
  if (Object.entries(artifacts).some(([key, value]) => report.artifacts[key as keyof typeof artifacts] !== value)) {
    throw new Error("The review does not match this image, caption, brief or drawing plan.");
  }
  return { report, contract };
}
