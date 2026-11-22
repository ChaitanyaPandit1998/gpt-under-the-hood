export const visualCanvas = { width: 1000, height: 700 } as const;
type Point = { x: number; y: number };
type SceneItem =
  | ({ kind: "text"; text: string; size: number; color: string; align: "left" | "center" | "right" } & Point)
  | ({ kind: "rectangle"; width: number; height: number; color: string } & Point)
  | { kind: "path"; points: Point[]; color: string; fill: string; closed: boolean; arrow: boolean; weight: number };
export type VisualScene = { background: string; items: SceneItem[] };

const maxSceneItems = 96;
const maxCopies = 24;
const number = { type: "number" };
const color = { type: "string", pattern: "^#[0-9a-fA-F]{6}$" };
function object(properties: Record<string, unknown>) {
  return { type: "object", additionalProperties: false, required: Object.keys(properties), properties };
}
const point = { x: number, y: number };
const itemSchemas = [
  object({ kind: { type: "string", enum: ["text"] }, ...point, text: { type: "string", minLength: 1, maxLength: 120 }, size: number, color, align: { type: "string", enum: ["left", "center", "right"] } }),
  object({ kind: { type: "string", enum: ["rectangle"] }, ...point, width: number, height: number, color }),
  object({ kind: { type: "string", enum: ["path"] }, points: { type: "array", minItems: 2, maxItems: 48, items: object(point) }, color,
    fill: { type: "string", pattern: "^(none|#[0-9a-fA-F]{6})$" }, closed: { type: "boolean" }, arrow: { type: "boolean" }, weight: number }),
];
const sceneSchema = (items: object[]) =>
  object({ background: color, items: { type: "array", minItems: 1, maxItems: maxSceneItems, items: { anyOf: items } } });
export const visualSceneSchema = sceneSchema(itemSchemas);

const offsets = { dx: number, dy: number };
const repeatSchema = object({
  kind: { type: "string", enum: ["repeat"] }, ...offsets,
  count: { type: "integer", minimum: 2, maximum: maxCopies },
  item: { anyOf: [itemSchemas[1], itemSchemas[2]] },
});
const textRunSchema = object({
  kind: { type: "string", enum: ["text-run"] }, ...point, ...offsets, size: number, color,
  align: { type: "string", enum: ["left", "center", "right"] },
  labels: { type: "array", minItems: 2, maxItems: maxCopies, items: { type: "string", minLength: 1, maxLength: 120 } },
});
export const compactVisualSceneSchema = sceneSchema([...itemSchemas, repeatSchema, textRunSchema]);

const objectValue = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);
const within = (value: unknown, min: number, max: number): value is number => typeof value === "number" && Number.isFinite(value) && value >= min && value <= max;
const isColor = (value: unknown): value is string => typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value);
function isPoint(value: Record<string, unknown>): value is Record<string, unknown> & Point {
  return within(value.x, 0, visualCanvas.width) && within(value.y, 0, visualCanvas.height);
}
function exact(value: Record<string, unknown>, schema: ReturnType<typeof object>) {
  return Object.keys(value).length === schema.required.length && schema.required.every((key) => Object.hasOwn(value, key));
}

function isItem(item: unknown): item is SceneItem {
  if (!objectValue(item) || !isColor(item.color)) return false;
  if (item.kind === "text") return exact(item, itemSchemas[0]) && isPoint(item) &&
    within(item.size, 14, 64) && ["left", "center", "right"].some((align) => align === item.align) &&
    typeof item.text === "string" && Boolean(item.text.trim()) && item.text.length <= 120 &&
    !/[\x00-\x1f\x7f]/.test(item.text);
  if (item.kind === "rectangle") return exact(item, itemSchemas[1]) && isPoint(item) &&
    within(item.width, 1, visualCanvas.width - item.x) && within(item.height, 1, visualCanvas.height - item.y);
  return item.kind === "path" && exact(item, itemSchemas[2]) && typeof item.closed === "boolean" &&
    typeof item.arrow === "boolean" && !(item.closed && item.arrow) && within(item.weight, 0.5, 16) &&
    (item.fill === "none" || (item.closed && isColor(item.fill))) &&
    Array.isArray(item.points) && item.points.length >= (item.closed ? 3 : 2) && item.points.length <= 48 &&
    item.points.every((p) => objectValue(p) && Object.keys(p).length === 2 && isPoint(p));
}

function invalidDrawing(): never {
  throw new Error("The model returned an invalid drawing. Try again with a simpler brief.");
}

function readScene(value: unknown) {
  if (!objectValue(value) || !exact(value, visualSceneSchema) || !isColor(value.background) ||
    !Array.isArray(value.items) || !value.items.length || value.items.length > maxSceneItems) invalidDrawing();
  const items: unknown[] = value.items;
  return { background: value.background, items };
}

export function validateVisualScene(value: unknown): VisualScene {
  const { background, items } = readScene(value);
  if (!items.every(isItem) || !items.some((item) => item.kind === "text")) invalidDrawing();
  return { background, items };
}

function expandItem(item: unknown): unknown[] {
  if (!objectValue(item) || (item.kind !== "repeat" && item.kind !== "text-run")) return [item];
  const { dx, dy } = item;
  if (!within(dx, -visualCanvas.width, visualCanvas.width) || !within(dy, -visualCanvas.height, visualCanvas.height) ||
    (dx === 0 && dy === 0)) invalidDrawing();
  const shift = (p: Point, index: number) => ({ x: p.x + dx * index, y: p.y + dy * index });
  if (item.kind === "repeat") {
    const { count, item: template } = item;
    if (!exact(item, repeatSchema) || !within(count, 2, maxCopies) || !Number.isInteger(count) ||
      !isItem(template) || template.kind === "text") invalidDrawing();
    return Array.from({ length: count }, (_, i) => template.kind === "path"
      ? { ...template, points: template.points.map((p) => shift(p, i)) }
      : { ...template, ...shift(template, i) });
  }
  if (!exact(item, textRunSchema) || !isPoint(item) || !Array.isArray(item.labels) ||
    item.labels.length < 2 || item.labels.length > maxCopies) invalidDrawing();
  return item.labels.map((text: unknown, i: number) => ({
    kind: "text", ...shift(item, i), text, size: item.size, color: item.color, align: item.align,
  }));
}

export function expandCompactVisualScene(value: unknown): VisualScene {
  const { background, items } = readScene(value);
  const expanded: unknown[] = [];
  for (const item of items) {
    const batch = expandItem(item);
    if (expanded.length + batch.length > maxSceneItems) invalidDrawing();
    expanded.push(...batch);
  }
  return validateVisualScene({ background, items: expanded });
}
