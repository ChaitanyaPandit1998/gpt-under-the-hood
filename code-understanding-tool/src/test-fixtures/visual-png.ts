import { PNG } from "pngjs";

export function makeVisualPng(width = 1000, height = 700, mode: "marked" | "uniform" | "transparent" = "marked"): Buffer {
  const image = new PNG({ width, height });
  image.data.fill(mode === "transparent" ? 0 : 255);
  if (mode === "marked") {
    for (let y = 0; y < Math.min(8, height); y++) {
      for (let x = 0; x < Math.min(8, width); x++) {
        const offset = (y * width + x) * 4;
        image.data[offset] = image.data[offset + 1] = image.data[offset + 2] = 0;
      }
    }
  }
  return PNG.sync.write(image);
}
