import { describe, expect, it } from "vitest";
import { expandCompactVisualScene, expandStoryVisualScene, validateVisualScene } from "./visual-scene";
import { sceneCommands } from "../server/vectorcraft-export";

const label = { kind: "text", x: 50, y: 60, text: "Literal label", size: 28, color: "#115577", align: "left" };
const figure = { background: "#fefefd", items: [label] };
const torso = { kind: "rectangle", x: 190, y: 190, width: 100, height: 150, color: "#445544" };
const head = { kind: "story-head", asset: "short-hair", torsoIndex: 0, collarX: 240, scale: 1, mirror: false };

describe("Story character placement", () => {
  it("reuses the same geometry at a different anchor, direction and scale", () => {
    const placed = expandStoryVisualScene({ ...figure, items: [torso, head, label] });
    const mirrored = expandStoryVisualScene({ ...figure, items: [
      { ...torso, x: 550, y: 170 }, { ...head, collarX: 620, scale: 2, mirror: true }, label,
    ] });
    expect(validateVisualScene(placed)).toEqual(placed);
    expect(placed.items.slice(1, -1)).toHaveLength(9);
    expect(mirrored.items.at(-1)).toEqual(label);
    placed.items.slice(1, -1).forEach((path, i) => {
      if (path.kind !== "path") throw new Error("Expected a head path");
      expect(mirrored.items[i + 1]).toEqual({ ...path, weight: path.weight * 2,
        points: path.points.map(({ x, y }) => ({
          x: expect.closeTo(620 - (x - 240) * 2, 8), y: expect.closeTo(174 + (y - 192) * 2, 8),
        })) });
    });
  });

  it.each([false, true])("joins the neck to a sloped shirt opening (mirrored: %s)", (mirror) => {
    const shirt = { kind: "path", color: "#445544", fill: "#445544", closed: true, arrow: false, weight: 1,
      points: [{ x: 200, y: 200 }, { x: 300, y: 250 }, { x: 300, y: 370 }, { x: 200, y: 370 }] };
    const [,,neck] = expandStoryVisualScene({ ...figure, items: [shirt, { ...head, asset: "long-hair", scale: 2, mirror }, label] }).items;
    if (neck.kind !== "path") throw new Error("Expected a neck path");
    expect(Math.max(...neck.points.map(point => point.y))).toBe(237);
    expect(neck.points.some(({ x, y }) => Math.abs(x - 240) <= 6 && y > 200 + (x - 200) / 2)).toBe(true);
  });

  it("allows complete scenes with no heads and composes with the compact mode", () => {
    expect(expandStoryVisualScene(figure)).toEqual(figure);
    const repeat = { kind: "repeat", count: 2, dx: 40, dy: 0, item: { kind: "rectangle", x: 100, y: 180, width: 20, height: 20, color: "#fafafa" } };
    const body = { ...figure, items: [label, repeat, torso, { ...head, torsoIndex: 2, asset: "long-hair" }] };
    expect(() => expandStoryVisualScene(body)).toThrow();
    expect(expandStoryVisualScene(body, true).items.map(item => item.kind)).toEqual([
      "text", "rectangle", "rectangle", "rectangle", ...Array(10).fill("path"),
    ]);
  });

  it("resolves only locked slots to their assigned face and does not require any people", () => {
    const cast = { "long-hair": "A guide" };
    const raw = (asset: string) => ({ ...figure, items: [torso, { ...head, asset }, label] });
    const locked = expandStoryVisualScene(raw("cast-2"), false, cast);
    expect(locked).toEqual(expandStoryVisualScene(raw("long-hair")));
    for (const asset of ["cast-1", "long-hair", "short-hair"]) {
      expect(() => expandStoryVisualScene(raw(asset), false, cast)).toThrow();
    }
    expect(expandStoryVisualScene(figure, false, cast)).toEqual(figure);
  });

  it("rejects external or invalid assets, clipping, excessive placements and over-budget expansion", () => {
    for (const badHead of [
      { ...head, asset: "https://example.test/head" }, { ...head, asset: "__proto__" },
      { ...head, scale: Infinity }, { ...head, scale: 0.5 }, { ...head, mirror: "false" },
      { ...head, collarX: 10 }, { ...head, torsoIndex: 2 }, { ...head, torsoIndex: 0.5 }, { ...head, url: "https://example.test/head" },
    ]) expect(() => expandStoryVisualScene({ ...figure, items: [torso, badHead, label] })).toThrow("invalid drawing");
    for (const unsupported of [label, { ...torso, y: 30 }, { ...torso, height: 3 }]) {
      expect(() => expandStoryVisualScene({ ...figure, items: [unsupported, head, label] })).toThrow("invalid drawing");
    }
    expect(() => expandStoryVisualScene({ ...figure, items: [torso, label, ...Array(5).fill(head)] })).toThrow("invalid drawing");
    expect(() => expandStoryVisualScene({ ...figure, items: [torso, ...Array(87).fill(label), head] })).toThrow("invalid drawing");
    const raw = { ...figure, items: [torso, head, label] };
    expect(() => validateVisualScene(raw)).toThrow();
    expect(() => expandCompactVisualScene(raw)).toThrow();
  });
});

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
