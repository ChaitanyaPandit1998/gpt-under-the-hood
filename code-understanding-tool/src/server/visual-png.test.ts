import { expect, it } from "vitest";
import { makeVisualPng } from "../test-fixtures/visual-png";
import { decodeVisualPng, visualPngLimits } from "./visual-png";

it("decodes real pixel data", () => {
  expect(decodeVisualPng(makeVisualPng(32, 24), 32, 24)).toEqual({ width: 32, height: 24, visiblePixels: 768, uniform: false });
});
it.each(["uniform", "transparent"] as const)("detects a %s image", (mode) => {
  const decoded = decodeVisualPng(makeVisualPng(32, 24, mode), 32, 24);
  expect(decoded.uniform).toBe(true);
  expect(decoded.visiblePixels).toBe(mode === "transparent" ? 0 : 768);
});
it("rejects corrupt checksums", () => {
  const png = makeVisualPng(32, 24);
  png[29] ^= 255;
  expect(() => decodeVisualPng(png, 32, 24)).toThrow();
});
it("rejects truncated data, trailing bytes and mismatched dimensions", () => {
  const png = makeVisualPng(32, 24);
  expect(() => decodeVisualPng(png.subarray(0, png.length - 3), 32, 24)).toThrow();
  expect(() => decodeVisualPng(Buffer.concat([png, Buffer.from([0])]), 32, 24)).toThrow();
  expect(() => decodeVisualPng(png, 33, 24)).toThrow();
});
it("enforces byte and pixel limits before decoding", () => {
  expect(() => decodeVisualPng(Buffer.alloc(visualPngLimits.bytes + 1), 32, 24)).toThrow("byte limit");
  const png = makeVisualPng(32, 24);
  png.writeUInt32BE(8192, 16); png.writeUInt32BE(8192, 20);
  expect(() => decodeVisualPng(png, 8192, 8192)).toThrow("oversized");
});
it("rejects a duplicate header before it can change decoder allocation limits", () => {
  const png = makeVisualPng(32, 24);
  const duplicate = Buffer.from(png.subarray(8, 33));
  duplicate.writeUInt32BE(8192, 8);
  expect(() => decodeVisualPng(Buffer.concat([png.subarray(0, 33), duplicate, png.subarray(33)]), 32, 24)).toThrow("structure");
});
it("rejects unsupported interlacing and animation", () => {
  const png = makeVisualPng(32, 24);
  const interlaced = Buffer.from(png); interlaced[28] = 1;
  expect(() => decodeVisualPng(interlaced, 32, 24)).toThrow("Unsupported");
  const animation = Buffer.alloc(12); animation.write("acTL", 4);
  expect(() => decodeVisualPng(Buffer.concat([png.subarray(0, 33), animation, png.subarray(33)]), 32, 24)).toThrow("Animated");
});
