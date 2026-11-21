import { describe, expect, it } from "vitest";
import { expandCompactVisualScene, validateVisualScene } from "./visual-scene";
import { sceneCommands } from "../server/vectorcraft-export";

const label = { kind: "text", x: 50, y: 60, text: "Literal label", size: 28, color: "#115577", align: "left" };
const figure = { background: "#fefefd", items: [label] };
describe("visual scene boundary", () => {
  it("maps valid shapes to fixed VectorCraft operations", () => {
    const path = { kind: "path", points: [{ x: 50, y: 90 }, { x: 150, y: 90 }], color: "#445544", fill: "none", closed: false, arrow: true, weight: 2 };
    const scene = { ...figure, items: [label, path] };
    expect(validateVisualScene(scene)).toEqual(scene);
    expect(sceneCommands(scene)).toContainEqual({ command: "path.create", params: { anchors: path.points, closed: false } });
    expect(sceneCommands(scene).filter((item) => item.command === "text.create")).toHaveLength(1);
  });
  it.each([["hd", 1280, 720], ["full-hd", 1920, 1080]] as const)("renders %s dimensions without stretching", (resolution, width, height) => {
    const item = { kind: "rectangle", x: 250, y: 100, width: 100, height: 100, color: "#112233" };
    const commands = sceneCommands({ ...figure, items: [label, item] }, resolution);
    expect(commands[0].params).toMatchObject({ width, height });
    const shape = commands.filter((entry) => entry.command === "shape.rectangle").at(-1)?.params;
    const scale = height / 700;
    expect(shape?.width).toBeCloseTo(100 * scale);
    expect(shape?.width).toBe(shape?.height);
    expect(shape?.x).toBeCloseTo(width / 2 - 250 * scale);
  });
  it("rejects arbitrary operations, extra fields, bad colors and out-of-bounds geometry", () => {
    const invalid = [
      { ...label, kind: "file.open" },
      { ...label, command: "file.open" },
      { ...label, color: "url(http://example.test/image)" },
      { ...label, x: Infinity },
      { kind: "rectangle", x: 990, y: 0, width: 20, height: 10, color: "#000000" },
      { kind: "path", points: [{ x: 40, y: 50 }, { x: -1, y: 50 }], color: "#112233", fill: "none", closed: false, arrow: false, weight: 1 },
    ];
    for (const item of invalid) expect(() => sceneCommands({ ...figure, items: [label, item] })).toThrow();
  });
});

describe("compact scene expansion", () => {
  const rect = { kind: "rectangle", x: 100, y: 100, width: 40, height: 30, color: "#00aabb" };
  const repeat = { kind: "repeat", count: 3, dx: 45, dy: 0, item: rect };
  const textRun = { kind: "text-run", labels: ["One", "Two"], x: 120, y: 120, dx: 45, dy: 0, size: 18, color: "#000000", align: "center" };
  const path = { kind: "path", points: [{ x: 150, y: 200 }, { x: 300, y: 200 }], color: "#003311", fill: "none", closed: false, arrow: false, weight: 2 };

  it("preserves back-to-front order, label formatting and shifted path points", () => {
    const scene = expandCompactVisualScene({ ...figure, items: [label, repeat, textRun, { kind: "repeat", count: 2, dx: -20, dy: 10, item: path }] });
    expect(scene.items).toEqual([
      label, rect, { ...rect, x: 145 }, { ...rect, x: 190 },
      { kind: "text", text: "One", x: 120, y: 120, size: 18, color: "#000000", align: "center" },
      { kind: "text", text: "Two", x: 165, y: 120, size: 18, color: "#000000", align: "center" },
      path, { ...path, points: [{ x: 130, y: 210 }, { x: 280, y: 210 }] },
    ]);
    expect(sceneCommands(scene).filter((entry) => entry.command === "text.create")).toHaveLength(3);
    expect(expandCompactVisualScene(figure)).toEqual(figure);
  });

  it("rejects compact objects at the ordinary scene boundary", () => {
    const compact = { ...figure, items: [label, repeat, textRun] };
    expect(() => validateVisualScene(compact)).toThrow();
    expect(() => sceneCommands(compact)).toThrow();
  });

  it("rejects invalid counts, formats, deltas, nesting, and out-of-bounds copies", () => {
    const invalid = [
      { ...repeat, count: 25 }, { ...repeat, count: 2.5 }, { ...repeat, dx: 0 }, { ...repeat, dx: Infinity },
      { ...repeat, dx: 480 }, { ...repeat, item: repeat }, { ...repeat, item: { ...rect, command: "file.open" } },
      { ...repeat, dx: -200, item: path },
      { ...textRun, labels: ["Good", "Bad\nlabel"] }, { ...textRun, labels: ["Good", 4] },
      { ...textRun, labels: [" ", "Good"] }, { ...textRun, x: 990 }, { ...textRun, extra: true },
    ];
    for (const item of invalid) expect(() => expandCompactVisualScene({ ...figure, items: [label, item] })).toThrow();
  });

  it("caps the expanded scene at 96 primitives", () => {
    const run = { ...textRun, dx: 20, labels: Array.from({ length: 24 }, (_, i) => String(i + 1)) };
    expect(expandCompactVisualScene({ ...figure, items: [run, run, run, run] }).items).toHaveLength(96);
    expect(() => expandCompactVisualScene({ ...figure, items: [label, run, run, run, run] })).toThrow();
  });
});
