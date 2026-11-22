import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { access, lstat, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { isAbsolute, join } from "node:path";
import { parseStudyExport, type StudyExportSnapshot } from "./study-export.ts";
import { runCraft, type CraftRunner } from "./wordcraft-export.ts";
import { diagramPlan } from "./vectorcraft-lanes.mjs";
import { diagramExportLimits, type DiagramExportLayout } from "../domain/diagram-export-layout.ts";
import { validateVisualScene, visualCanvas, type VisualScene } from "../domain/visual-scene.ts";
import { visualResolutions, type VisualResolution } from "../domain/visual-resolution.ts";

export const VECTORCRAFT_SHA256 = "d40e68ac59768c3957abe23c674484ad46a3abbe606e92a747001c8e4082edc4";
export type DiagramSnapshot = { study: StudyExportSnapshot; visualIndex: number; layout?: DiagramExportLayout };
export type DiagramExporter = { available: () => Promise<boolean>; export: (snapshot: DiagramSnapshot, signal: AbortSignal) => Promise<Buffer>; renderScene?: (scene: VisualScene, signal: AbortSignal, resolution?: VisualResolution) => Promise<Buffer> };

export function parseDiagramExport(value: unknown): DiagramSnapshot {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid diagram request.");
  const body = value as Record<string, unknown>;
  const { snapshot: study } = parseStudyExport({ ...body, format: "docx", includeSource: false });
  const layout = body.layout;
  if (layout !== undefined && layout !== "detailed" && layout !== "automatic") throw new Error("Unknown diagram layout.");
  return { study, visualIndex: validateDiagramSelection(study, body.visualIndex, layout), layout };
}

export function validateDiagramSelection(study: StudyExportSnapshot, index: unknown, layout: DiagramExportLayout = "detailed") {
  if (typeof index !== "number" || !Number.isInteger(index) || index < 0 || index >= study.explanation.visuals.length) throw new Error("Select an existing diagram.");
  const visualIndex = index;
  const visual = study.explanation.visuals[visualIndex];
  const limit = diagramExportLimits[layout];
  if (visual.nodes.length < 1 || visual.nodes.length > limit.nodes || visual.edges.length > limit.edges) throw new Error(`Diagram PNG supports 1-${limit.nodes} nodes and up to ${limit.edges} connections.`);
  return visualIndex;
}

type Command = { command: string; params: Record<string, unknown> };

function automaticCommands({ study, visualIndex }: DiagramSnapshot): Command[] {
  const e = study.explanation;
  const visual = e.visuals[visualIndex];
  const { commands, height, mode } = diagramPlan(visual, "CHAPTER DIAGRAM  /  AUTOMATIC LAYOUT", "chapter source");
  let y = height + 15;
  function note(value: string) {
    for (const line of value.split(/\r\n|\r|\n/)) {
      const chars = Array.from(line);
      for (let offset = 0; offset < chars.length; offset += 54) {
        commands.push({ command: "text.create", params: { x: 48, y, text: chars.slice(offset, offset + 54).join(""),
          font: "Source Sans 3", size: 14, color: "#45655d" } });
        commands.push({ command: "paint.setStroke", params: { none: true } });
        y += 20;
      }
    }
    y += 8;
  }
  if (mode === "lanes") {
    const outgoing = new Map<string, number>();
    for (const edge of visual.edges) outgoing.set(edge.from, (outgoing.get(edge.from) ?? 0) + 1);
    const labels = new Map(visual.nodes.map((node) => [node.id, node.label]));
    for (const edge of visual.edges) if (edge.label.trim() && outgoing.get(edge.from) === 1) {
      note(`Connection: ${labels.get(edge.from)} -> ${labels.get(edge.to)}: ${edge.label}`);
    }
  }
  note(`Scope: ${study.scope}`);
  if (study.provenance) note(study.provenance);
  note(`Review: ${e.quality_review_status ?? "not provided"}. ${e.quality_review_note ?? "No review note provided."}`);
  note(`Coverage: ${e.visual_note || "No coverage note provided."}`);
  for (const item of e.incomplete_items ?? []) note(`Incomplete code: ${item.title}. ${item.explanation} Chapter lines: ${item.source_ranges.map((r) => `${r.start_line}-${r.end_line}`).join(", ")}`);
  if (y + 20 > 8000) throw new Error("Diagram is too tall for a readable PNG. Export the document instead.");
  commands[0].params.height = y + 20;
  return commands;
}

/** Bounded, literal text only. All connections, including loops, are retained. */
export function diagramCommands(snapshot: DiagramSnapshot) {
  if (snapshot.layout === "automatic") return automaticCommands(snapshot);
  const { study, visualIndex } = snapshot;
  const e = study.explanation;
  const visual = e.visuals[visualIndex];
  const commands: Command[] = [];
  const add = (command: string, params: Record<string, unknown>) => commands.push({ command, params });
  const lines = (value: string, columns: number) => value.split(/\r\n|\r|\n/).flatMap((line) => {
    const chars = Array.from(line.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, "").replaceAll("\t", "    "));
    const parts: string[] = [];
    while (chars.length) {
      let end = Math.min(columns, chars.length);
      if (end < chars.length) {
        const boundary = chars.slice(0, end + 1).lastIndexOf(" ");
        if (boundary > columns / 3) end = boundary;
      }
      parts.push(chars.splice(0, end).join(""));
      if (chars[0] === " ") chars.shift();
    }
    return parts.length ? parts : [""];
  });
  const text = (value: string, x: number, y: number, size = 16) => {
    add("text.create", { x, y, text: value, font: "Source Sans 3", size, color: "#193e38" });
    add("paint.setStroke", { none: true });
  };
  const paragraph = (value: string, y: number, columns = 78, x = 40, size = 16) => {
    // Conservative glyph budget also handles wide labels without shrinking text.
    const wrapped = lines(value, Math.min(columns, Math.floor((960 - x) / (size * 1.25))));
    wrapped.forEach((line, i) => text(line, x, y + i * (size + 7), size));
    return y + wrapped.length * (size + 7) + 10;
  };
  let y = paragraph(visual.title, 45, 52, 40, 24);
  y = paragraph("Selected diagram only. Model review is not execution proof. Node order is a layout choice, not an implied execution sequence.", y);
  y = paragraph(`Scope: ${study.scope}`, y);
  if (study.provenance) y = paragraph(study.provenance, y);
  y = paragraph(`Review: ${e.quality_review_status ?? "not provided"}. ${e.quality_review_note ?? "No review note provided."}`, y);
  y = paragraph(`Coverage: ${e.visual_note || "No coverage note provided."}`, y);
  for (const item of e.incomplete_items ?? []) y = paragraph(`Incomplete code: ${item.title}. ${item.explanation} Chapter lines ${item.source_ranges.map((r) => `${r.start_line}-${r.end_line}`).join(", ")}`, y);
  y = paragraph("Source appendix omitted; explanation labels may quote code. Citations use original chapter coordinates. Consult the full explanation for assumptions.", y);
  y = paragraph(`Source SHA256: ${createHash("sha256").update(study.source).digest("hex")}`, y);
  const headerCommands = commands.splice(0);
  const positions = visual.nodes.map((node) => {
    const label = lines(node.label, 23);
    const description = lines(node.description, 27);
    const citations = lines(`Chapter lines: ${node.source_ranges.map((r) => `${r.start_line}-${r.end_line}`).join(", ") || "not provided"}`, 27);
    const height = 48 + label.length * 26 + (description.length + citations.length) * 23;
    const position = { y, height, label, description, citations };
    y += height + 35;
    return position;
  });
  const indexById = new Map(visual.nodes.map((node, index) => [node.id, index]));
  for (const [index, edge] of visual.edges.entries()) {
    const from = positions[indexById.get(edge.from)!];
    const to = positions[indexById.get(edge.to)!];
    const start = from.y + from.height / 2 - 7;
    const end = to.y + to.height / 2 + 7;
    const lane = 650 + index * 22;
    add("path.create", { anchors: [{ x: 620, y: start }, { x: lane, y: start }, { x: lane, y: end }, { x: 620, y: end }], closed: false });
    add("paint.setFill", { none: true }); add("paint.setStroke", { color: "#49675e" });
    add("stroke.set", { weight: 1.6, startArrow: null, endArrow: "Triangle", arrowAlign: "tip" });
    text(`E${index + 1}`, lane + 2, start - 4, 12);
  }
  for (const [index, node] of visual.nodes.entries()) {
    const p = positions[index];
    add("shape.rectangle", { x: 40, y: p.y, width: 580, height: p.height });
    add("paint.setFill", { color: "#e9f2ed" }); add("paint.setStroke", { color: "#0a7768" });
    add("stroke.set", { weight: 1.3, startArrow: null, endArrow: null });
    let baseline = p.y + 28;
    text(`Node ${index + 1}`, 56, baseline, 14); baseline += 26;
    for (const line of p.label) { text(line, 56, baseline, 19); baseline += 26; }
    for (const line of [...p.description, ...p.citations]) { text(line, 56, baseline, 16); baseline += 23; }
    add("object.setProps", { name: `source-node-${index}`, data: { "source-id": node.id, "source-lines": JSON.stringify(node.source_ranges) } });
  }
  y = paragraph("Connections (exact supplied relationships)", y, 70, 40, 20);
  for (const [index, edge] of visual.edges.entries()) y = paragraph(`E${index + 1}: Node ${indexById.get(edge.from)! + 1} -> Node ${indexById.get(edge.to)! + 1}${edge.label ? `: ${edge.label}` : " (unlabelled)"}`, y);
  const width = 1000, height = Math.ceil(y + 30);
  if (height > 8000) throw new Error("Diagram is too tall for a readable PNG. Export the document instead.");
  return [{ command: "file.new", params: { width, height, name: "Chapter diagram", backgroundContents: "white", units: "Points" } }, ...headerCommands, ...commands];
}

export async function renderVectorcraftDiagram(cli: string, snapshot: DiagramSnapshot, signal: AbortSignal, run: CraftRunner = runCraft) {
  return renderVectorcraftCommands(cli, diagramCommands(snapshot), signal, run);
}

export function sceneCommands(value: unknown, resolution: VisualResolution = "standard"): Command[] {
  const scene = validateVisualScene(value);
  const { width, height } = visualResolutions[resolution];
  const scale = Math.min(width / visualCanvas.width, height / visualCanvas.height);
  const offsetX = (width - visualCanvas.width * scale) / 2;
  const offsetY = (height - visualCanvas.height * scale) / 2;
  const transform = ({ x, y }: { x: number; y: number }) => ({ x: x * scale + offsetX, y: y * scale + offsetY });
  const commands: Command[] = [];
  const add = (command: string, params: Record<string, unknown>) => commands.push({ command, params });
  add("file.new", { width, height, name: "Explanation visual", backgroundContents: "white", units: "Points" });
  add("shape.rectangle", { x: 0, y: 0, width, height });
  add("paint.setFill", { color: scene.background });
  add("paint.setStroke", { none: true });
  for (const item of scene.items) {
    if (item.kind === "text") {
      add("text.create", { ...transform(item), text: item.text, size: item.size * scale, color: item.color, font: "Source Sans 3", justify: item.align });
      add("paint.setStroke", { none: true });
    } else if (item.kind === "rectangle") {
      add("shape.rectangle", { ...transform(item), width: item.width * scale, height: item.height * scale });
      add("paint.setFill", { color: item.color });
      add("paint.setStroke", { none: true });
    } else {
      add("path.create", { anchors: item.points.map(transform), closed: item.closed });
      add("paint.setFill", item.fill === "none" ? { none: true } : { color: item.fill });
      add("paint.setStroke", { color: item.color });
      add("stroke.set", { weight: item.weight * scale, startArrow: null, endArrow: item.arrow ? "Triangle" : null, arrowAlign: "tip" });
    }
  }
  return commands;
}

async function renderVectorcraftCommands(cli: string, commands: Command[], signal: AbortSignal, run: CraftRunner = runCraft) {
  const directory = await mkdtemp(join(tmpdir(), "understand-diagram-"));
  try {
    if (!/\b0\.24\.1\b/.test(await run(cli, ["--safe-mode", "--version"], directory, signal))) throw new Error("Unsupported VectorCraft version.");
    const script = join(directory, "commands.json"), output = join(directory, "diagram.png");
    await writeFile(script, JSON.stringify(commands), { mode: 0o600 });
    const result = await run(cli, ["--safe-mode", "run", "--commands", script, "--export", output, "--scale", "1"], directory, signal);
    const height = Number(commands[0].params.height), width = Number(commands[0].params.width);
    for (const line of result.trim().split("\n").filter(Boolean)) {
      const step = JSON.parse(line) as { error?: unknown; result?: { warning?: unknown; warnings?: unknown[]; overflow?: unknown[]; bounds?: { x: number; y: number; width: number; height: number } } };
      if (step.error || step.result?.warning || step.result?.warnings?.length || step.result?.overflow?.length) throw new Error("VectorCraft reported an incomplete render.");
      const b = step.result?.bounds;
      if (b && (![b.x, b.y, b.width, b.height].every(Number.isFinite) || b.x < 0 || b.y < 0 || b.x + b.width > width || b.y + b.height > height)) throw new Error("Diagram content exceeds the artboard.");
    }
    const stats = await lstat(output);
    if (!stats.isFile() || stats.size > 25 * 1024 * 1024) throw new Error("Invalid diagram artifact.");
    const bytes = await readFile(output);
    if (signal.aborted || bytes.length < 24 || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
      || bytes.readUInt32BE(16) !== width || bytes.readUInt32BE(20) !== height) throw new Error("Invalid diagram PNG.");
    return bytes;
  } finally { await rm(directory, { recursive: true, force: true }); }
}

export function createVectorcraftExporter(options: { enabled?: boolean; cli?: string } = {}): DiagramExporter {
  async function available() {
    if (!options.enabled || !options.cli || !isAbsolute(options.cli)) return false;
    try {
      await access(options.cli, constants.X_OK);
      return createHash("sha256").update(await readFile(options.cli)).digest("hex") === VECTORCRAFT_SHA256;
    } catch { return false; }
  }
  async function approvedPath() {
    if (!await available()) throw new Error("The approved VectorCraft executable is unavailable.");
    return options.cli!;
  }
  return { available, async export(snapshot, signal) {
    return renderVectorcraftDiagram(await approvedPath(), snapshot, signal);
  }, async renderScene(scene, signal, resolution) {
    return renderVectorcraftCommands(await approvedPath(), sceneCommands(scene, resolution), signal);
  } };
}
