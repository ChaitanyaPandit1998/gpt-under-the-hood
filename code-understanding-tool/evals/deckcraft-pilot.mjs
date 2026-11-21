import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

// A curated export experiment for one public case, not a generic export adapter.
const args = process.argv.slice(2);
const option = (name) => args[args.indexOf(name) + 1];
for (const name of ["--cli", "--input", "--output"]) assert.ok(args.includes(name) && option(name), `Supply ${name}`);
const cli = resolve(option("--cli"));
const output = resolve(option("--output"));
const input = JSON.parse(readFileSync(resolve(option("--input")), "utf8"));
assert.equal(input.url, "https://github.com/fastapi/fastapi/pull/9816", "This pilot only supports the reviewed public fixture.");
assert.deepEqual(input.chapter.files, ["fastapi/params.py"]);
assert.equal(input.headSha, "e676874f76ff728af2fe60eebf50246d630e6caa", "The curated pilot requires the reviewed snapshot.");
assert.ok(["passed", "revised"].includes(input.explanation.quality_review_status));
assert.match(input.headSha, /^[a-f0-9]{40}$/i);
const sourceLines = input.chapter.sourceText.split(/\r\n|\r|\n/);
for (const [line, fragment] of [[113, "if example is not _Unset:"], [145, "if examples is not None:"], [153, "current_json_schema_extra = json_schema_extra or extra"], [222, "assert default is ..."], [600, "use_kwargs = {k: v for k, v in kwargs.items() if v is not _Unset}"], [678, "[Remaining patch omitted:"]]) {
  assert.ok(sourceLines[line - 1]?.includes(fragment), `Reviewed source coordinates changed at line ${line}`);
}
mkdirSync(output, { recursive: true });
const timings = [];
function run(commandArgs) {
  const started = performance.now();
  const result = execFileSync(cli, commandArgs, { encoding: "utf8", timeout: 120_000, maxBuffer: 16_000_000 });
  timings.push({ command: commandArgs[0], milliseconds: Math.round(performance.now() - started) });
  return result;
}

const commands = [];
const add = (command, params) => commands.push(params === undefined ? command : `${command}=${JSON.stringify(params)}`);
const ink = "#193E38", teal = "#0A7768", paper = "#FAF8F1", pale = "#E9F2ED", muted = "#49675E";
function typography(size, color = ink, bold = false) {
  add("format.autofit", { mode: "none" });
  add("format.font", { family: "Source Sans 3" });
  add("format.size", { size });
  add("format.color", { color });
  add("format.bold", { on: bold });
  add("format.margins", { left: 8, right: 8, top: 5, bottom: 5 });
}
function text(name, value, rect, size = 20, color = ink, bold = false) {
  add("insert.textBox", { rect, text: value });
  typography(size, color, bold);
  add("shape.setBounds", { x: rect[0], y: rect[1], w: rect[2], h: rect[3] });
  add("shape.rename", { name });
}
function node(name, value, rect, size = 18) {
  add("shape.insert", { preset: "rect", rect, text: value });
  add("shape.fill", { color: pale });
  add("shape.line", { color: teal, width: 1.2 });
  typography(size);
  add("format.align", { align: "center" });
  add("format.anchor", { anchor: "middle" });
  add("shape.setBounds", { x: rect[0], y: rect[1], w: rect[2], h: rect[3] });
  add("shape.rename", { name });
  add("shape.altText", { text: value });
}
function edge(name, points) {
  add("shape.freeform", { points, closed: false });
  add("shape.fill", { none: true });
  add("shape.line", { color: muted, width: 1.5, head: "none", tail: "triangle" });
  add("shape.rename", { name });
}
function slide(title, citations) {
  add("slide.new", { layout: "blank" });
  add("slide.rename", { name: title });
  add("design.background", { color: paper });
  add("slide.notes", { text: [
    title, `Source: ${input.url}`, `Head revision: ${input.headSha}`,
    `Citations: chapter-source.txt, ${citations}. These are chapter/diff coordinates, not original-file line numbers.`,
    `Review status: ${input.explanation.quality_review_status}. ${input.explanation.quality_review_note}`,
    "Scope: partial patch, not the complete repository. Model review is not runtime verification. This is a curated five-slide selection, not complete control-flow coverage. See explanation.json for the original explanation and all source ranges.",
  ].join("\n\n") });
  text("source-warning", `Chapter lines ${citations} | PR #9816 | Snapshot ${input.headSha.slice(0, 12)}\nPartial patch | Model review is not execution proof | Full source: chapter-source.txt`, [52, 496, 856, 36], 12, muted);
  add("format.margins", { top: 0, bottom: 0 });
}
function heading(value) { text("heading", value, [52, 30, 856, 62], 36, ink, true); }
function label(value, rect) { text(`branch-${value}`, value, rect, 16, teal, true); }

add("file.new", { blank: true, size: [960, 540] });
add("file.properties", { title: "FastAPI field options - public PR study", subject: "DeckCraft export feasibility pilot", author: "Code Understanding Tool" });

slide("FastAPI field options", "53-172, 539-602");
text("cover-title", "FastAPI\nfield options", [52, 90, 856, 145], 48, ink, true);
text("cover-summary", "Collect settings. Adapt them for Pydantic v1 or v2.\nRemove unset entries before calling the parent.", [52, 272, 856, 90], 24);
text("cover-context", `Public PR #9816 | fastapi/params.py\nSnapshot ${input.headSha.slice(0, 12)} | Revised; final model check passed`, [52, 390, 856, 68], 18, muted);

slide("Three independent checks", "113-153, 543-583");
heading("Three independent checks");
text("warning-scope", "Checks run in order. Warning paths below assume the warning call returns.", [52, 90, 856, 35], 18, muted);
for (const [index, condition, action] of [[0, "example is not _Unset", "Warn about example"], [1, "examples is not None", "Add examples"], [2, "regex is not None", "Warn about regex"]]) {
  const y = 135 + index * 100;
  edge(`true-${index}`, [[380, y + 25], [590, y + 25]]);
  label("true", [450, y - 5, 90, 28]);
  if (index < 2) {
    edge(`false-${index}`, [[220, y + 50], [220, y + 100]]);
    edge(`rejoin-${index}`, [[720, y + 50], [720, y + 75], [260, y + 75], [260, y + 100]]);
    label("false", [125, y + 51, 90, 28]);
  } else {
    edge("false-final", [[220, y + 50], [220, 455], [350, 455]]);
    edge("rejoin-final", [[720, y + 50], [720, 455], [630, 455]]);
    label("false", [125, y + 55, 90, 28]);
  }
  node(`condition-${index}`, condition, [60, y, 320, 50], 19);
  node(`action-${index}`, action, [590, y, 260, 50], 19);
}
node("warnings-exit", "Select schema extras next", [350, 435, 280, 40], 18);

slide("Schema extras selection", "153-154, 583-584");
heading("Schema extras selection");
text("truthiness-scope", "An empty dictionary takes the fallback. This is not an is-not-None check.", [52, 90, 856, 35], 18, muted);
edge("extras-truthy", [[390, 200], [250, 200], [250, 285]]);
edge("extras-falsy", [[570, 200], [710, 200], [710, 285]]);
edge("extras-left-rejoin", [[250, 355], [250, 435], [300, 435]]);
edge("extras-right-rejoin", [[710, 355], [710, 435], [660, 435]]);
node("extras-check", "Is json_schema_extra truthy?", [300, 130, 360, 70], 22);
node("extras-explicit", "Select json_schema_extra", [90, 285, 320, 70], 21);
node("extras-fallback", "Select extra", [550, 285, 320, 70], 21);
node("extras-exit", "Check the Pydantic version", [300, 415, 360, 40], 19);
label("truthy", [145, 230, 120, 30]); label("falsy", [710, 230, 110, 30]);

slide("Version-specific options", "154-169, 584-600");
heading("Version-specific options");
edge("version-true", [[360, 125], [180, 125], [180, 180]]);
edge("version-false", [[600, 125], [730, 125], [730, 205]]);
edge("v2-after-options", [[180, 220], [180, 265]]);
edge("v2-pattern", [[120, 315], [120, 360]]);
edge("v2-regex", [[260, 315], [320, 315], [320, 360]]);
edge("v1-pattern", [[660, 255], [585, 255], [585, 315]]);
edge("v1-regex", [[800, 255], [815, 255], [815, 315]]);
edge("v1-pattern-merge", [[585, 370], [585, 420], [600, 420]]);
edge("v1-regex-merge", [[815, 370], [815, 400]]);
edge("v2-pattern-exit", [[120, 415], [120, 470], [360, 470]]);
edge("v2-regex-exit", [[320, 415], [320, 470], [360, 470]]);
edge("v1-exit", [[730, 445], [730, 470], [600, 470]]);
node("version-check", "Is PYDANTIC_V2 true?", [360, 100, 240, 50], 20);
node("v2-options", "Add v2 options", [70, 180, 220, 40], 19);
node("v2-check", "Is pattern truthy?", [70, 265, 220, 50], 19);
node("v2-pattern", "pattern key\nuses pattern", [30, 360, 180, 55], 18);
node("v2-regex", "pattern key\nuses regex", [230, 360, 180, 55], 18);
node("v1-check", "Is pattern truthy?", [600, 205, 260, 50], 19);
node("v1-pattern", "regex key\nuses pattern", [490, 315, 190, 55], 18);
node("v1-regex", "regex key\nuses regex", [720, 315, 190, 55], 18);
node("v1-merge", "Merge extras (may overwrite regex)", [600, 400, 260, 45], 17);
node("version-exit", "Filter final arguments", [360, 450, 240, 40], 18);
label("v2", [250, 145, 70, 28]); label("v1", [680, 155, 70, 28]);
label("truthy", [30, 325, 100, 28]); label("falsy", [315, 325, 100, 28]);
label("truthy", [490, 275, 100, 28]); label("falsy", [815, 275, 100, 28]);

slide("Filtering and limits", "169-171, 600-602, 678");
heading("Filtering and limits");
text("scope-limits", "Only a partial patch was supplied.\nThe form constructor ends mid-call.\nParent internals are outside the excerpt.\nThis condensed deck is not a full study.", [52, 132, 450, 190], 22);
text("source-url", input.url, [52, 345, 450, 65], 17, teal);
add("insert.hyperlink", { url: input.url });
text("source-revision", `Snapshot ${input.headSha.slice(0, 12)}\nSee the full source and explanation beside this deck.`, [52, 416, 450, 65], 17, muted);
edge("filter-keep", [[650, 195], [607, 195], [607, 270]]);
edge("filter-omit", [[830, 195], [817, 195], [817, 270]]);
edge("keep-exit", [[607, 325], [607, 375], [650, 375], [650, 405]]);
edge("omit-exit", [[817, 325], [817, 375], [830, 375], [830, 405]]);
node("filter-check", "For each entry: v is not _Unset?", [570, 145, 340, 50], 20);
node("filter-keep", "Keep entry", [520, 270, 175, 55], 20);
node("filter-omit", "Omit entry", [730, 270, 175, 55], 20);
node("filter-exit", "After all entries: call parent", [570, 405, 340, 45], 19);
label("true", [545, 225, 90, 30]); label("false", [820, 225, 90, 30]);

const native = resolve(output, "fastapi-study.deckcraft");
const pptx = resolve(output, "fastapi-study.pptx");
const pdf = resolve(output, "fastapi-study.pdf");
writeFileSync(resolve(output, "commands.txt"), commands.join("\n") + "\n");
writeFileSync(resolve(output, "chapter-source.txt"), `Source: ${input.url}\nHead: ${input.headSha}\nCoordinates below refer to the original chapter text.\n\n` + input.chapter.sourceText.split(/\r\n|\r|\n/).map((line, i) => `${i + 1} | ${line}`).join("\n"));
writeFileSync(resolve(output, "explanation.json"), JSON.stringify(input.explanation, null, 2));
const version = run(["--version"]).trim();
run(["run", "--cmd-file", resolve(output, "commands.txt"), "--save", native]);
run(["convert", native, pptx]);
run(["convert", native, pdf]);
run(["render", native, "--all", "--scale", "1.5", resolve(output, "native-preview")]);
run(["render", pptx, "--all", "--scale", "1.5", resolve(output, "pptx-preview")]);

const checks = [];
for (const [format, file] of [["native", native], ["pptx", pptx]]) {
  const info = JSON.parse(run(["info", file]));
  writeFileSync(resolve(output, `${format}-info.json`), JSON.stringify(info, null, 2));
  assert.equal(info.slides.length, 5, `${format}: expected exactly five slides`);
  const inspectArgs = ["run", "--in", file, "--print"];
  for (let i = 0; i < 5; i += 1) inspectArgs.push("--cmd", `slide.inspect={"index":${i}}`);
  const inspected = run(inspectArgs).trim().split("\n").filter((line) => line.startsWith("slide.inspect")).map((line) => JSON.parse(line.slice(line.indexOf("{"))));
  assert.equal(inspected.length, 5);
  for (const slide of inspected) {
    assert.ok(slide.notes.includes(input.url), `${format}: missing notes citation`);
    assert.ok(slide.notes.includes(input.headSha), `${format}: missing revision`);
    assert.ok(slide.notes.includes(input.explanation.quality_review_note), `${format}: review note lost`);
    assert.ok(slide.notes.includes("not complete control-flow coverage"), `${format}: scope warning lost`);
    assert.ok(slide.shapes.some((shape) => shape.text?.includes("Model review is not execution proof")), `${format}: visible warning lost`);
    for (const shape of slide.shapes) {
      const [x, y, w, h] = shape.box;
      assert.ok(x >= -1 && y >= -1 && x + w <= 961 && y + h <= 541, `${format}: shape outside slide`);
    }
  }
  writeFileSync(resolve(output, `${format}-inspection.json`), JSON.stringify(inspected, null, 2));
  checks.push({ format, slides: 5, notesCitations: true, revisionRetained: true, reviewNoteRetained: true, visibleWarnings: true, shapesWithinBounds: true });
}
const previews = readdirSync(resolve(output, "native-preview")).filter((name) => name.endsWith(".png"));
assert.equal(previews.length, 5);
for (const name of previews) assert.ok(readFileSync(resolve(output, "native-preview", name)).equals(readFileSync(resolve(output, "pptx-preview", name))), "PPTX round-trip rendering differs; inspect before accepting");
checks.push({ format: "pptx-round-trip", previewImagesIdentical: true, renderer: "DeckCraft only; not Microsoft PowerPoint" });
const pdfText = execFileSync("pdftotext", ["-layout", pdf, "-"], { encoding: "utf8", timeout: 30_000 });
const pages = pdfText.split("\f").filter((page) => page.trim());
assert.equal(pages.length, 5);
for (const page of pages) {
  assert.ok(page.includes("Chapter lines"));
  assert.ok(page.includes(input.headSha.slice(0, 12)));
  assert.ok(page.includes("Model review is not execution proof"));
}
checks.push({ format: "pdf", pages: 5, visibleCitations: true, visibleWarnings: true, textExtraction: "Poppler" });
const report = { version, input: { url: input.url, headSha: input.headSha }, checks, timings,
  manualVisualReview: "pending", pdfVisualReview: "pending", note: "Curated public-case feasibility pilot, not generic conversion or proof of semantic completeness. No model/image calls. No system-wide install." };
writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ output, version, checks, timings }, null, 2));
