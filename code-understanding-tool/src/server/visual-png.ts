import { inflateSync } from "node:zlib";
import { PNG } from "pngjs";

export const visualPngLimits = { bytes: 10 * 1024 * 1024, pixels: 16_000_000 } as const;

// Restrict evaluation to the renderer's static, non-interlaced 8-bit PNG output.
export function decodeVisualPng(value: Uint8Array, width: number, height: number) {
  if (value.byteLength > visualPngLimits.bytes) throw new Error("PNG exceeds the evaluation byte limit.");
  const bytes = Buffer.from(value);
  if (bytes.length < 33 || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) ||
      bytes.readUInt32BE(8) !== 13 || bytes.toString("ascii", 12, 16) !== "IHDR") throw new Error("Invalid PNG header.");
  const w = bytes.readUInt32BE(16), h = bytes.readUInt32BE(20);
  if (!w || !h || w !== width || h !== height || w * h > visualPngLimits.pixels) throw new Error("Unexpected or oversized PNG dimensions.");
  const channels = new Map([[0, 1], [2, 3], [3, 1], [4, 2], [6, 4]]).get(bytes[25]);
  if (bytes[24] !== 8 || !channels || bytes[26] !== 0 || bytes[27] !== 0 || bytes[28] !== 0) {
    throw new Error("Unsupported PNG format for evaluation.");
  }
  const data: Buffer[] = [];
  let ended = false, chunkCount = 0;
  for (let offset = 8; offset < bytes.length;) {
    if (offset + 12 > bytes.length) throw new Error("Truncated PNG chunk.");
    const size = bytes.readUInt32BE(offset), end = offset + size + 12;
    if (end > bytes.length) throw new Error("Truncated PNG chunk.");
    const type = bytes.toString("ascii", offset + 4, offset + 8);
    if (++chunkCount > 2048 || (type === "IHDR" && offset !== 8)) throw new Error("Unexpected PNG chunk structure.");
    if (["acTL", "fcTL", "fdAT"].includes(type)) throw new Error("Animated PNG is unsupported.");
    if (type === "IDAT") data.push(bytes.subarray(offset + 8, end - 4));
    if (type === "IEND") {
      if (size !== 0 || end !== bytes.length) throw new Error("Unexpected PNG trailing data.");
      ended = true;
    }
    offset = end;
  }
  if (!ended || !data.length) throw new Error("Incomplete PNG.");
  // Bound decompression before the decoder allocates its bitmap; reject excess scanlines.
  const expected = (w * channels + 1) * h;
  if (inflateSync(Buffer.concat(data), { maxOutputLength: expected }).length !== expected) throw new Error("Invalid PNG scanlines.");
  const decoded = PNG.sync.read(bytes, { checkCRC: true });
  if (decoded.width !== width || decoded.height !== height || decoded.data.length !== w * h * 4) throw new Error("Invalid decoded PNG.");
  let visiblePixels = 0, firstColor: number | undefined, uniform = true;
  for (let offset = 0; offset < decoded.data.length; offset += 4) {
    if (decoded.data[offset + 3] !== 0) visiblePixels++;
    const color = decoded.data.readUInt32BE(offset);
    if (firstColor === undefined) firstColor = color;
    else if (firstColor !== color) uniform = false;
  }
  return { width: w, height: h, visiblePixels, uniform };
}
