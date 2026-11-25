import { createHash } from "node:crypto";
import { validateVisualContract } from "../domain/visual-evaluation.ts";
import { parseVisualResolution, visualResolutions } from "../domain/visual-resolution.ts";
import { visualPngLimits } from "./visual-png.ts";

export const visualReviewRequestLimit = 14_500_000;
export function parseVisualReviewRequest(value: unknown) {
  const invalid = () => { throw new Error("Confirm a valid checklist, explanation, caption and PNG before reviewing."); };
  if (!value || typeof value !== "object" || Array.isArray(value)) return invalid();
  const data = value as Record<string, unknown>;
  const keys = ["contract", "explanation", "brief", "plan", "caption", "pngBase64", "resolution", "confirmed"];
  if (Object.keys(data).length !== keys.length || keys.some((key) => !Object.hasOwn(data, key)) || data.confirmed !== true) return invalid();
  for (const [key, max] of [["explanation", 100_000], ["brief", 30_000], ["plan", 100_000], ["caption", 4_000]] as const) {
    if (typeof data[key] !== "string" || !data[key].trim() || data[key].length > max) return invalid();
  }
  const contract = validateVisualContract(data.contract);
  const explanation = data.explanation as string;
  if (createHash("sha256").update(explanation).digest("hex") !== contract.explanationSha256) return invalid();
  const encoded = data.pngBase64;
  if (typeof encoded !== "string" || !encoded.length || encoded.length > Math.ceil(visualPngLimits.bytes / 3) * 4 ||
      !/^[A-Za-z0-9+/]+={0,2}$/.test(encoded)) return invalid();
  const png = Buffer.from(encoded, "base64");
  if (!png.length || png.length > visualPngLimits.bytes || png.toString("base64") !== encoded) return invalid();
  const resolution = parseVisualResolution(data.resolution);
  return { contract, explanation, brief: data.brief as string, plan: data.plan as string, caption: data.caption as string,
    png, ...visualResolutions[resolution] };
}
