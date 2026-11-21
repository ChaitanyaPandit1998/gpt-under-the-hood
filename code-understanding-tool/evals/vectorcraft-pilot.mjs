import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const args = process.argv.slice(2);
const option = (name) => args[args.indexOf(name) + 1];
for (const key of ["--cli", "--input", "--output"]) assert.ok(args.includes(key) && option(key), `Supply ${key}`);
const cli = resolve(option("--cli"));
const output = resolve(option("--output"));
const input = JSON.parse(readFileSync(resolve(option("--input")), "utf8"));
assert.equal(input.url, "https://github.com/fastapi/fastapi/pull/9816");
assert.equal(input.headSha, "e676874f76ff728af2fe60eebf50246d630e6caa");
const visual = input.explanation.visuals.find((item) => item.title === "Schema extras selection");
assert.ok(visual);
assert.deepEqual(visual.nodes.map((node) => node.id).sort(), ["check", "explicit", "fallback", "next"]);
assert.equal(visual.edges.length, 4);
const expectedEdges = new Set(["check:explicit", "check:fallback", "explicit:next", "fallback:next"]);
for (const edge of visual.edges) assert.ok(expectedEdges.has(`${edge.from}:${edge.to}`));
mkdirSync(output, { recursive: true });
const timings = [], commands = [];
const version = execFileSync(cli, ["--version"], { encoding: "utf8", timeout: 10_000 }).trim();
const add = (command, params = {}) => commands.push({ command, params });
function run(commandArgs) {
  const start = performance.now();
  const result = execFileSync(cli, ["--safe-mode", ...commandArgs], { encoding: "utf8", timeout: 120_000, maxBuffer: 20_000_000, env: { ...process.env, VECTORCRAFT_NO_PREFS: "1" } });
  timings.push({ command: commandArgs[0], milliseconds: Math.round(performance.now() - start) });
  const results = result.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
  for (const step of results) {
    assert.ok(!step.error, "VectorCraft command failed");
    assert.ok(!step.result?.warning, `VectorCraft warning: ${step.result?.warning}`);
    assert.ok(!(step.result?.warnings?.length), `VectorCraft export warnings: ${JSON.stringify(step.result?.warnings)}`);
  }
  return results;
}
function text(name, value, x, y, size = 22, center = false) {
  add("text.create", { x, y, text: value, font: "Source Sans 3", size, color: "#193e38", justify: center ? "center" : "left" });
  add("paint.setStroke", { none: true });
  add("object.setProps", { name });
}
const positions = { check: [300, 130, 360, 70], explicit: [80, 285, 320, 70], fallback: [560, 285, 320, 70], next: [280, 430, 400, 60] };
const routes = { "check:explicit": [[390, 200], [240, 200], [240, 285]], "check:fallback": [[570, 200], [720, 200], [720, 285]], "explicit:next": [[240, 355], [240, 460], [280, 460]], "fallback:next": [[720, 355], [720, 460], [680, 460]] };
add("file.new", { width: 960, height: 620, name: "FastAPI schema extras", backgroundContents: "white", units: "Points" });
for (const edge of visual.edges) {
  const key = `${edge.from}:${edge.to}`;
  add("path.create", { anchors: routes[key].map(([x, y]) => ({ x, y })), closed: false });
  add("paint.setFill", { none: true }); add("paint.setStroke", { color: "#49675e" });
  add("stroke.set", { weight: 1.8, startArrow: null, endArrow: "Triangle", arrowAlign: "tip" });
  add("object.setProps", { name: `edge:${key}`, data: { from: edge.from, to: edge.to, outcome: edge.label } });
}
for (const node of visual.nodes) {
  const [x, y, width, height] = positions[node.id];
  add("shape.rectangle", { x, y, width, height });
  add("paint.setFill", { color: "#e9f2ed" }); add("paint.setStroke", { color: "#0a7768" });
  add("stroke.set", { weight: 1.3, startArrow: null, endArrow: null });
  add("object.setProps", { name: `node:${node.id}`, data: { "source-lines": JSON.stringify(node.source_ranges), "source-pr": input.url, "source-head": input.headSha } });
  text(`label:${node.id}`, node.label, x + width / 2, y + height / 2 + 7, node.id === "next" ? 21 : 22, true);
}
text("heading", "Schema extras selection", 50, 60, 34);
text("scope", "A source-linked view, not complete control-flow coverage", 50, 97, 18);
text("truthy", "truthy", 125, 250, 18);
text("falsy", "falsy", 755, 250, 18);
text("citation", `Chapter lines 153, 583 | PR #9816 | Snapshot ${input.headSha.slice(0, 12)}`, 50, 545, 15);
text("warning", "Partial patch. Model review is not execution proof.", 50, 573, 15);
text("source-url", input.url, 50, 601, 14);
add("document.inspect", { depth: 5 });
const script = resolve(output, "commands.json");
writeFileSync(script, JSON.stringify(commands, null, 2));
const native = resolve(output, "schema-extras.vectorcraft"), svg = resolve(output, "schema-extras.svg"), pdf = resolve(output, "schema-extras.pdf"), png = resolve(output, "schema-extras.png");
const results = run(["run", "--commands", script, "--export", native, "--export", svg, "--export", pdf, "--export", png, "--scale", "1.5"]);
const inspection = results.find((step) => step.command === "document.inspect").result;
writeFileSync(resolve(output, "inspection.json"), JSON.stringify(inspection, null, 2));
const nodes = inspection.layers.flatMap((layer) => layer.children ?? []);
assert.equal(nodes.filter((node) => node.name.startsWith("node:")).length, 4);
assert.equal(nodes.filter((node) => node.name.startsWith("edge:")).length, 4);
assert.equal(nodes.filter((node) => node.name.startsWith("label:") && node.kind === "Type").length, 4);
for (const node of nodes) {
  const b = node.bounds;
  assert.ok(b.x >= -1 && b.y >= -1 && b.x + b.width <= 961 && b.y + b.height <= 621, `Out of bounds: ${node.name}`);
}
const svgText = readFileSync(svg, "utf8");
assert.ok(svgText.includes("<text"), "SVG lost live text");
assert.ok(svgText.includes(input.headSha), "SVG lost source metadata");
assert.ok(svgText.includes("data-source-lines"), "SVG lost citation metadata");
assert.ok(!/<(?:script|foreignObject|image)\b|\bonload\s*=|javascript:|\bhref\s*=/i.test(svgText), "Unexpected active or external SVG content");
run(["convert", svg, resolve(output, "roundtrip.svg")]);
const roundtrip = readFileSync(resolve(output, "roundtrip.svg"), "utf8");
for (const node of visual.nodes) assert.ok(roundtrip.includes(node.label), "Round-trip lost label");
assert.ok(roundtrip.includes("data-source-lines"));
const editable = nodes.find((node) => node.name === "label:fallback");
run(["run", "--in", native, "--cmd", "text.setText", "--params", JSON.stringify({ id: editable.id, text: "Use extra" }), "--export", resolve(output, "edit-smoke.svg")]);
assert.ok(readFileSync(resolve(output, "edit-smoke.svg"), "utf8").includes("Use extra"));
const pdfText = execFileSync("pdftotext", ["-layout", pdf, "-"], { encoding: "utf8" });
for (const required of ["Chapter lines", "not execution proof", input.url]) assert.ok(pdfText.includes(required), `PDF lost ${required}`);
writeFileSync(resolve(output, "source-visual.json"), JSON.stringify({ url: input.url, headSha: input.headSha, visual, review_status: input.explanation.quality_review_status }, null, 2));
const report = { version, nodes: 4, edges: 4, liveText: true, nativeTextEdit: true, svgRoundTrip: true, sourceMetadata: true, pdfCitations: true, noActiveSvgContent: true, allBoundsInsideArtboard: true, warnings: [], timings, visualReview: "pending", note: "One curated public-case layout, not a generic diagram layout engine or semantic verification. Safe mode and no-prefs enabled; no model/image calls. Not integrated into the app." };
writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ output, ...report }, null, 2));
