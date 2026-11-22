export const visualResolutions = {
  standard: { title: "Standard", width: 1000, height: 700 },
  hd: { title: "HD", width: 1280, height: 720 },
  "full-hd": { title: "Full HD", width: 1920, height: 1080 },
} as const;

export type VisualResolution = keyof typeof visualResolutions;

export function parseVisualResolution(value: unknown): VisualResolution {
  if (typeof value === "string" && Object.hasOwn(visualResolutions, value)) return value as VisualResolution;
  throw new Error("Unsupported image resolution.");
}
