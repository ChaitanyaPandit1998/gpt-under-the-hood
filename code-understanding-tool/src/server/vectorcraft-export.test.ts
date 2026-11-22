import { access, readFile, symlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createVectorcraftExporter, diagramCommands, parseDiagramExport, renderVectorcraftDiagram } from "./vectorcraft-export.ts";
import type { CraftRunner } from "./wordcraft-export.ts";

const request = {
  version: 1, source: "first\nsecond", scope: "Selected diagram", provenance: "Public PR revision abc",
  visualIndex: 0, explanation: { title: "Explanation", summary: "Summary", concepts: [], walkthrough: [], assumptions: [],
    visual_note: "Selective coverage.", quality_review_status: "incomplete", quality_review_note: "Not verified.",
    visuals: [{ title: "Branch diagram", kind: "control_flow", nodes: [
      { id: "a", label: "Decision", description: "Check condition.", source_ranges: [{ start_line: 1, end_line: 1 }] },
      { id: "b", label: "Next", description: "Continue.", source_ranges: [{ start_line: 2, end_line: 2 }] },
    ], edges: [{ from: "a", to: "b", label: "yes" }, { from: "b", to: "a", label: "retry" }, { from: "b", to: "b", label: "loop" }] }],
  },
};

describe("VectorCraft diagram export", () => {
  it("is disabled by default and rejects unapproved binaries", async () => {
    expect(await createVectorcraftExporter().available()).toBe(false);
    expect(await createVectorcraftExporter({ enabled: true, cli: process.execPath }).available()).toBe(false);
  });
  it("retains every relationship, warnings and citations, including cycles and self-loops", () => {
    const commands = diagramCommands(parseDiagramExport(request));
    expect(commands.filter((step) => step.command === "path.create")).toHaveLength(3);
    const script = JSON.stringify(commands);
    for (const expected of ["Selective coverage.", "Not verified.", "Chapter lines: 1-1", "Chapter lines: 2-2", "E2: Node 2 -> Node 1: retry", "E3: Node 2 -> Node 2: loop", "Public PR revision abc"]) expect(script).toContain(expected);
    expect(script).not.toContain("first\\nsecond");
  });
  it("supports a larger explicit automatic layout and keeps fallback warnings and connections", () => {
    expect(() => parseDiagramExport({ ...request, layout: "unknown" })).toThrow("Unknown diagram layout");
    const larger = structuredClone(request);
    larger.explanation.visuals[0].edges = Array.from({ length: 13 }, (_, i) => ({ from: "a", to: "b", label: `route ${i + 1}` }));
    expect(() => parseDiagramExport(larger)).toThrow("12 connections");
    const commands = diagramCommands(parseDiagramExport({ ...larger, layout: "automatic" }));
    expect(commands[0].params.width).toBe(1100);
    const text = commands.filter((c) => c.command === "text.create").map((c) => c.params.text);
    expect(text).toContain("E13 Node 1 -> Node 2");
    expect(text).toContain("route 13");
    expect(text).toContain("Review: incomplete. Not verified.");
    expect(text).toContain("Coverage: Selective coverage.");
  });
  it("retains non-decision edge labels below the lanes", () => {
    const linear = structuredClone(request);
    linear.explanation.visuals[0].edges = [{ from: "a", to: "b", label: "continued" }];
    const commands = diagramCommands(parseDiagramExport({ ...linear, layout: "automatic" }));
    expect(commands[0].params.width).toBe(960);
    expect(commands.some((c) => c.params.text === "Connection: Decision -> Next: continued")).toBe(true);
  });
  it("does not treat labels as markup, paths or commands", () => {
    const value = structuredClone(request);
    value.explanation.visuals[0].nodes[0].label = '<script>x</script> file.open={"path":"/etc/passwd"}';
    const commands = diagramCommands(parseDiagramExport(value));
    expect(commands.some((c) => c.command === "file.open")).toBe(false);
    expect(commands.filter((c) => c.command === "text.create").map((c) => c.params.text).join("")).toContain("<script>x</script>");
  });
  it("rejects missing diagrams, invalid references, dense views and excessive height", () => {
    expect(() => parseDiagramExport({ ...request, visualIndex: 9 })).toThrow();
    const broken = structuredClone(request); broken.explanation.visuals[0].edges[0].to = "missing";
    expect(() => parseDiagramExport(broken)).toThrow();
    const dense = structuredClone(request); dense.explanation.visuals[0].edges = Array.from({ length: 13 }, () => ({ from: "a", to: "b", label: "edge" }));
    expect(() => parseDiagramExport(dense)).toThrow("12 connections");
    const long = structuredClone(request); long.explanation.quality_review_note = "x\n".repeat(4000);
    expect(() => diagramCommands(parseDiagramExport(long))).toThrow("too tall");
  });
  it("uses safe mode, fixed paths and bounded PNG output, then cleans up", async () => {
    let directory = "";
    const runner: CraftRunner = async (_cli, args, cwd) => {
      directory = cwd;
      expect(args[0]).toBe("--safe-mode");
      if (args[1] === "--version") return "vectorcraft-cli 0.24.1";
      expect(args).toEqual(["--safe-mode", "run", "--commands", join(cwd, "commands.json"), "--export", join(cwd, "diagram.png"), "--scale", "1"]);
      const script = JSON.parse(await readFile(args[3], "utf8")); expect(script[0].command).toBe("file.new");
      const png = Buffer.alloc(24); Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(png); png.writeUInt32BE(1000, 16); png.writeUInt32BE(script[0].params.height, 20);
      await writeFile(args[5], png);
      return '{"command":"export","result":{}}';
    };
    expect((await renderVectorcraftDiagram("/approved/cli", parseDiagramExport(request), new AbortController().signal, runner)).length).toBe(24);
    await expect(access(directory)).rejects.toThrow();
  });
  it.each(["warnings", "overflow", "out of bounds", "error", "invalid PNG", "symlink", "failure", "version"])("fails closed and cleans up on %s", async (reason) => {
    let directory = "";
    const runner: CraftRunner = async (_cli, args, cwd) => {
      directory = cwd;
      if (args[1] === "--version") return reason === "version" ? "0.25.0" : "0.24.1";
      if (reason === "failure") throw new Error("private diagnostic");
      if (reason === "symlink") await symlink(join(cwd, "commands.json"), join(cwd, "diagram.png"));
      else await writeFile(join(cwd, "diagram.png"), "invalid png");
      if (reason === "warnings") return '{"result":{"warnings":["missing font"]}}';
      if (reason === "overflow") return '{"result":{"overflow":["text-id"]}}';
      if (reason === "out of bounds") return '{"result":{"bounds":{"x":950,"y":0,"width":200,"height":20}}}';
      if (reason === "error") return '{"error":"failed"}';
      return "";
    };
    await expect(renderVectorcraftDiagram("/approved/cli", parseDiagramExport(request), new AbortController().signal, runner)).rejects.toThrow();
    await expect(access(directory)).rejects.toThrow();
  });
});
