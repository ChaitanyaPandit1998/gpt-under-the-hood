import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

// Deliberately case-specific: this is a local renderer test, not an app exporter.
const args = process.argv.slice(2);
const option = (name) => args[args.indexOf(name) + 1];
for (const key of ["--cli", "--input", "--output"]) assert.ok(args.includes(key) && option(key), `Supply ${key}`);
const cli = resolve(option("--cli"));
const output = resolve(option("--output"));
const input = JSON.parse(readFileSync(resolve(option("--input")), "utf8"));
assert.equal(input.url, "https://github.com/fastapi/fastapi/pull/9816");
assert.equal(input.headSha, "e676874f76ff728af2fe60eebf50246d630e6caa");
assert.deepEqual(input.chapter.files, ["fastapi/params.py"]);
const explanation = input.explanation;
assert.ok(["passed", "revised"].includes(explanation.quality_review_status));
assert.equal(explanation.walkthrough.length, 4);
mkdirSync(output, { recursive: true });
const timings = [];
function run(cliArgs) {
  const started = performance.now();
  const value = execFileSync(cli, cliArgs, { encoding: "utf8", timeout: 120_000, maxBuffer: 20_000_000, env: { ...process.env, WORDCRAFT_REGION: "en-GB" } });
  timings.push({ command: cliArgs[0], milliseconds: Math.round(performance.now() - started) });
  return value;
}
const commands = [];
const add = (command, params = {}) => commands.push({ command, params });
const clean = (text) => text.replaceAll("`", "");
let first = true;
function paragraph(text, style = "Normal", pageBreak = false) {
  if (!first) add("text.newParagraph");
  first = false;
  add("styles.apply", { style });
  add("para.pageBreakBefore", { value: pageBreak });
  add("para.keepNext", { value: ["Title", "Subtitle", "Heading1", "Heading2"].includes(style) });
  add("para.borders", { kind: "none" });
  add("text.insert", { text: clean(text) });
}
const ranges = (items) => items.map((range) => range.start_line === range.end_line ? `${range.start_line}` : `${range.start_line}-${range.end_line}`).join(", ");
const cite = (items) => paragraph(`Source: chapter lines ${ranges(items)} in chapter-source.txt.`, "SourceCitation");
function chapter(title, pageBreak = true) {
  paragraph(title, "Heading1", pageBreak);
  paragraph("Scope: partial PR patch. Model review is not runtime verification.", "SourceCitation");
}
add("file.properties", { title: "FastAPI field options study guide", author: "Code Understanding Tool", subject: "Source grounded PR study", custom: { source_pr: input.url, source_head: input.headSha, review_status: explanation.quality_review_status, original_review_note: explanation.quality_review_note } });
add("layout.size", { name: "A4" });
add("layout.margins", { top: 56, bottom: 56, left: 56, right: 56 });
for (const [style, size, bold] of [["Normal", 11.5, false], ["Title", 28, true], ["Subtitle", 14, false], ["Heading1", 18, true], ["Heading2", 13, true]]) {
  add("styles.modify", { style, chr: { font: "Liberation Sans", size, bold, color: { Rgb: [0, 0, 0] } }, para: { spaceBefore: style === "Heading2" ? 10 : 0, spaceAfter: 8, lineSpacing: { rule: "multiple", value: 1.12 }, widowControl: true, keepNext: style !== "Normal" } });
}
add("styles.create", { name: "SourceCitation", basedOn: "Normal" });
add("styles.modify", { style: "SourceCitation", chr: { font: "Liberation Sans", size: 9.5, color: { Rgb: [55, 55, 55] } }, para: { spaceAfter: 6, keepNext: false } });

paragraph("FastAPI field options study guide", "Title");
paragraph("A chapter based explanation of public PR 9816", "Subtitle");
paragraph("This guide explains how the visible parameter and body constructors prepare options for Pydantic v1 and v2. Read it alongside the supplied patch: it does not establish the behaviour of the whole repository or execute the changed code.");
paragraph(`PR: ${input.url}`, "SourceCitation");
paragraph(`Snapshot: ${input.headSha}`, "SourceCitation");
paragraph("Chapter coordinates below refer to the numbered companion source, not the original repository file.", "SourceCitation");
chapter("1 Overview", false);
paragraph(explanation.summary);
for (const concept of explanation.concepts) {
  paragraph(concept.term.replaceAll(/[^A-Za-z0-9 ]/g, ""), "Heading2");
  paragraph(concept.explanation); cite(concept.source_ranges);
}
chapter("2 Entry and independent warnings");
for (const step of explanation.walkthrough.slice(0, 2)) {
  paragraph(step.heading, "Heading2"); paragraph(step.explanation); cite(step.source_ranges);
}
paragraph("The warning paths describe what happens when the warning call returns. They do not claim that every warnings configuration continues execution.");

chapter("3 Schema extras and version selection");
paragraph(explanation.walkthrough[2].explanation); cite(explanation.walkthrough[2].source_ranges);
if (args.includes("--figure")) {
  paragraph("Schema extras selection", "Heading2");
  add("insert.picture", { path: resolve(option("--figure")), width: 465, alt: "Source linked decision view: truthy json_schema_extra selects that dictionary; otherwise extra is selected. Both outcomes proceed to the version check." });
  add("caret.docEnd");
  paragraph("Figure 1  Source linked view reused from the DeckCraft pilot. Chapter lines 153 and 583; the drawing is explanatory, not an execution trace.", "SourceCitation");
}

chapter("4 Filtering and forwarding");
paragraph(explanation.walkthrough[3].explanation); cite(explanation.walkthrough[3].source_ranges);
paragraph("The check is identity based rather than truthiness based. A value of None or False is retained when it is not the unset marker; the marker itself is omitted.");
cite([{ start_line: 169, end_line: 171 }, { start_line: 600, end_line: 602 }]);
chapter("5 Limits and review", false);
for (const assumption of explanation.assumptions) paragraph(`${assumption.statement} ${assumption.reason}`);
paragraph("The supplied patch ends during the form constructor. Parent validation, schema generation, and the version flag setup are outside this excerpt. The complete original explanation and source ranges are preserved in explanation.json.");
paragraph(`Review result: ${explanation.quality_review_status}. The explanation was revised and passed its final model review. This is not an independent proof of correctness or a runtime test.`);
paragraph("The original model review note is preserved in the DOCX custom properties and the companion explanation. This guide is a selected learning view, not complete control flow coverage.");
paragraph(`Source: ${input.url}`, "SourceCitation");
paragraph(`Revision: ${input.headSha}`, "SourceCitation");

const docx = resolve(output, "fastapi-study-guide.docx");
const pdf = resolve(output, "fastapi-study-guide.pdf");
const script = resolve(output, "commands.json");
writeFileSync(script, JSON.stringify({ commands, save: [docx] }, null, 2));
writeFileSync(resolve(output, "explanation.json"), JSON.stringify(explanation, null, 2));
writeFileSync(resolve(output, "chapter-source.txt"), `PR: ${input.url}\nRevision: ${input.headSha}\n\n` + input.chapter.sourceText.split(/\r\n|\r|\n/).map((line, index) => `${index + 1} | ${line}`).join("\n"));
const version = run(["--version"]).trim();
run(["run", "--script", script, "--author", "Code Understanding Tool"]);
run(["convert", docx, pdf, "--strict"]);
const inspection = JSON.parse(run(["inspect", docx]));
writeFileSync(resolve(output, "inspection.json"), JSON.stringify(inspection, null, 2));
const extracted = run(["text", docx]);
for (const required of [input.url, input.headSha, "partial PR patch", "not runtime verification", "1 Overview", "2 Entry and independent warnings", "3 Schema extras and version selection", "4 Filtering and forwarding", "5 Limits and review"]) assert.ok(extracted.includes(required), `Missing: ${required}`);
assert.equal(inspection.blocks.filter((block) => block.style === "Heading1").length, 5);
assert.equal(inspection.blocks[0].style, "Title");
run(["convert", docx, resolve(output, "roundtrip.docx")]);
assert.equal(run(["text", resolve(output, "roundtrip.docx")]), extracted, "DOCX round-trip text changed");
const metadata = execFileSync("unzip", ["-p", docx, "docProps/custom.xml"], { encoding: "utf8" });
assert.ok(metadata.includes(input.headSha));
assert.ok(metadata.includes("original_review_note"));
assert.ok(metadata.includes(explanation.quality_review_status));
run(["convert", docx, resolve(output, "document-model.json")]);
const model = JSON.parse(readFileSync(resolve(output, "document-model.json"), "utf8"));
assert.equal(model.customProps.find((property) => property.name === "original_review_note")?.value, explanation.quality_review_note);
const pdfText = execFileSync("pdftotext", ["-layout", pdf, "-"], { encoding: "utf8" });
for (const required of ["1 Overview", "5 Limits and review", "chapter lines", "runtime verification", input.headSha]) assert.ok(pdfText.includes(required), `PDF lost ${required}`);
const pages = pdfText.split("\f").filter((page) => page.trim()).length;
for (let page = 1; page <= pages; page += 1) run(["render", docx, resolve(output, `page-${page}.png`), "--page", String(page), "--scale", "1.5", "--strict"]);
const report = { version, pages, headings: 5, titleStyle: true, sourceCitations: true, originalReviewProperty: true, textRoundTrip: true, figureSource: args.includes("--figure") ? "Existing DeckCraft pilot image; not editable as a vector in this DOCX" : null, timings, visualReview: "pending", independentDocxRender: "pending", note: "Curated public-case feasibility test. No model/image calls. Not integrated into the app or tested in Microsoft Word." };
writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ output, ...report }, null, 2));
