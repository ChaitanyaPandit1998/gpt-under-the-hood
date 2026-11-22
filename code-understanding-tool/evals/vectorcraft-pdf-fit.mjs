import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { documentCommands, parseStudyExport } from "../src/server/study-export.ts";

// Keep the app's PNG acceptance rules unchanged: this offline pilot uses the
// same command builder and page-fit rule with prototype-sized local images.
const [input, title, pngPath, output, cliPath] = process.argv.slice(2);
assert(input && title && pngPath && output?.endsWith(".pdf") && cliPath,
  "Usage: node evals/vectorcraft-pdf-fit.mjs INPUT VISUAL_TITLE PNG OUTPUT.pdf WORDCRAFT_CLI");
const saved = JSON.parse(await readFile(input, "utf8"));
const { snapshot } = parseStudyExport({
  version: 1, format: "pdf", source: saved.chapter.sourceText, explanation: saved.explanation,
  includeSource: false, scope: saved.chapter.title, provenance: "PR: " + saved.url + "\nRevision: " + saved.headSha,
});
const visualIndex = snapshot.explanation.visuals.findIndex((view) => view.title === title);
assert(visualIndex >= 0, "No matching diagram.");
const bytes = await readFile(pngPath);
assert(bytes.length >= 24 && bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])), "Expected a local PNG.");
const width = bytes.readUInt32BE(16), height = bytes.readUInt32BE(20);
assert(width > 0 && width <= 1800 && height > 0 && height <= 8000, "Invalid prototype dimensions.");
const figureWidth = Math.min(480, 620 * width / height);
const directory = await mkdtemp(join(tmpdir(), "vectorcraft-pdf-fit-"));
try {
  const image = join(directory, "diagram.png");
  const docx = join(directory, "study.docx"), pdf = join(directory, "study.pdf");
  const script = join(directory, "commands.json");
  await writeFile(image, bytes, { mode: 0o600 });
  await writeFile(script, JSON.stringify(documentCommands(snapshot, docx, { path: image, visualIndex, width: figureWidth })), { mode: 0o600 });
  const cli = resolve(cliPath);
  const run = (args) => execFileSync(cli, args, {
    cwd: directory, encoding: "utf8", timeout: 15000, maxBuffer: 1_000_000,
    env: { HOME: directory, TMPDIR: directory, WORDCRAFT_REGION: "en-GB" },
  });
  assert.match(run(["--version"]), /\b0\.18\.1\b/);
  run(["run", "--script", script, "--author", "Code Understanding Tool"]);
  run(["convert", docx, pdf, "--strict"]);
  const artifact = await readFile(pdf);
  assert.equal(artifact.subarray(0, 5).toString(), "%PDF-");
  await writeFile(output, artifact, { mode: 0o600 });
  console.log(JSON.stringify({ output, width, height, figureWidthPt: figureWidth, scale: figureWidth / width }));
} finally {
  await rm(directory, { recursive: true, force: true });
}
