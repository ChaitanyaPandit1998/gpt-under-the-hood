import { createHash } from "node:crypto";
import { expect, it } from "vitest";
import { parseVisualReviewRequest } from "./visual-review-request";
import type { VisualContract } from "../domain/visual-evaluation";

const explanation = "A fixed limit.";
const contract: VisualContract = { version: 1, id: "test", explanationSha256: createHash("sha256").update(explanation).digest("hex"),
  audience: "beginner", purpose: "factual-diagram", teachingGoal: "Show the limit.", audienceObjectives: ["Understand the limit."],
  requirements: [{ id: "limit", kind: "claim", statement: "The limit is fixed.", evidence: "background", evidenceReference: "Explanation" }] };
const request = () => ({ contract, explanation, brief: "Brief", plan: "Plan", caption: "Caption", pngBase64: "YWJj", resolution: "standard", confirmed: true });
it("parses a confirmed bounded request, leaving actual decoding to preflight", () => {
  const parsed = parseVisualReviewRequest(request());
  expect(parsed.png.toString()).toBe("abc");
  expect(parsed).toMatchObject({ width: 1000, height: 700, explanation });
});
it.each([
  { confirmed: false }, { confirmed: undefined }, { extra: "data" }, { explanation: "changed" },
  { caption: " " }, { caption: "x".repeat(4001) }, { resolution: "4k" },
  { pngBase64: "a" }, { pngBase64: "YWJj\n" }, { pngBase64: "data:image/png;base64,YWJj" },
])("rejects invalid transport data %j", (change) => {
  expect(() => parseVisualReviewRequest({ ...request(), ...change })).toThrow();
});
