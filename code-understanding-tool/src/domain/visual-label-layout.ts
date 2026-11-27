import { visualCanvas, type VisualScene } from "./visual-scene.ts";

type TextItem = Extract<VisualScene["items"][number], { kind: "text" }>;
export type LabelMetrics = { left: number; right: number; ascent: number; descent: number };

// This checks text geometry, not factual accuracy or the final raster output.
export function checkVisualLabels(scene: VisualScene, measure: (item: TextItem) => LabelMetrics): string[] {
  const issues: string[] = [];
  const wordCount = scene.items.filter((item) => item.kind === "text")
    .reduce((count, item) => count + item.text.trim().split(/\s+/).length, 0);
  if (wordCount > 40) issues.push("The drawing has too much text; keep longer explanations in the caption.");
  const boxes = scene.items.filter((item) => item.kind === "text").map((item, index) => {
    const metrics = measure(item);
    const box = { left: item.x - metrics.left, right: item.x + metrics.right,
      top: item.y - metrics.ascent, bottom: item.y + metrics.descent };
    if (!Object.values(box).every(Number.isFinite) || box.left < 0 || box.top < 0
      || box.right > visualCanvas.width || box.bottom > visualCanvas.height) {
      issues.push(`Label ${index + 1} may be clipped.`);
    }
    if (item.text.length > 60) issues.push(`Label ${index + 1} is too long; move explanations into the caption.`);
    return box;
  });
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      if (a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom) {
        issues.push(`Labels ${i + 1} and ${j + 1} overlap.`);
      }
    }
  }
  return issues;
}
