import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

// Hand-authored applications of the visual design skill, not model outputs.
const [cliPath, input, outputPath] = process.argv.slice(2);
assert(cliPath && input && outputPath, "Usage: node evals/vectorcraft-visual-skill.mjs VECTORCRAFT_CLI SAVED_PR.json OUTPUT_DIR");
const cli = resolve(cliPath), out = resolve(outputPath), width = 842, height = 595;
const saved = JSON.parse(await readFile(input, "utf8"));
assert.equal(saved.url, "https://github.com/fastapi/fastapi/pull/9816");
const extras = saved.explanation.visuals.find((view) => view.title === "Schema extras selection");
assert(extras);
const expected = ["check:explicit", "check:fallback", "explicit:next", "fallback:next"].sort();
assert.deepEqual(extras.edges.map(({ from, to }) => from + ":" + to).sort(), expected);
const source = saved.chapter.sourceText.split(/\r\n|\r|\n/);
for (const line of [153, 583]) assert(source[line - 1].includes("json_schema_extra or extra"));
const refs = (id) => extras.nodes.find((node) => node.id === id).source_ranges.map(({ start_line: start, end_line: end }) => start === end ? start : start + "-" + end).join(", ");

function page({ background, foreground, muted, title, kicker, subtitle }) {
  const commands = [];
  const add = (command, params) => commands.push({ command, params });
  const paint = (color, border, weight = 1) => {
    add("paint.setFill", color ? { color } : { none: true });
    add("paint.setStroke", border ? { color: border } : { none: true });
    if (border) add("stroke.set", { weight, startArrow: null, endArrow: null });
  };
  const text = (value, x, y, size = 17, color = foreground, justify = "left") => {
    add("text.create", { text: value, x, y, size, color, justify, font: "Source Sans 3" });
    add("paint.setStroke", { none: true });
  };
  const box = (x, y, w, h, color, border, weight) => {
    add("shape.rectangle", { x, y, width: w, height: h });
    paint(color, border, weight);
  };
  const path = (points, ink, { weight = 2, arrow = false, fill } = {}) => {
    add("path.create", { anchors: points.map(([x,y]) => ({ x,y })), closed: Boolean(fill) });
    paint(fill, ink, weight);
    add("stroke.set", { weight, startArrow: null, endArrow: arrow ? "Triangle" : null, arrowAlign: "tip" });
  };
  add("file.new", { width, height, name: title, backgroundContents: "white", units: "Points" });
  box(0, 0, width, height, background);
  text(kicker, 43, 42, 14, muted);
  text(title, 42, 92, 40);
  text(subtitle, 43, 125, 18, muted);
  return { commands, text, box, path };
}

const math = page({ background: "#101c2a", foreground: "#fff4df", muted: "#b5c7d6",
  title: "A line through the critical strip", kicker: "01  /  MATHEMATICAL GEOMETRY",
  subtitle: "The Riemann Hypothesis concerns the real part of every nontrivial zero." });
{
  const { text, box, path } = math;
  box(52, 159, 505, 335, "#172939");
  box(159, 189, 289, 242, "#234452");
  path([[448,189],[448,431]], "#83a9ab", { weight: 1 });
  path([[159,431],[159,180]], "#c4dbe2", { arrow: true, weight: 1.3 });
  path([[84,315],[526,315]], "#c4dbe2", { arrow: true, weight: 1.3 });
  path([[303.5,189],[303.5,431]], "#ffbd73", { weight: 3 });
  text("Imaginary part", 167, 180, 15);
  text("Real part", 529, 302, 15, "#c4dbe2", "right");
  text("Critical strip", 226, 231, 19, "#c4dbe2", "center");
  text("0", 159, 451, 15, "#c4dbe2", "center");
  text("1/2", 303.5, 451, 17, "#ffbd73", "center");
  text("1", 448, 451, 15, "#c4dbe2", "center");
  text("Schematic only. No zeros are plotted.", 304, 476, 15, "#c4dbe2", "center");
  text("The claim", 588, 183, 25);
  text("All nontrivial zeros lie", 588, 219);
  text("on the highlighted line.", 588, 242);
  text("Still unproved.", 588, 283, 18, "#ffbd73");
  // A ribbon in the margin, intentionally outside the coordinate plot.
  for (let row = 0; row < 24; row++) {
    const color = ["#327f89", "#40939c", "#69abb0", "#dfa76d", "#e8b985", "#efd1a3"][Math.floor(row / 4)];
    const points = Array.from({ length: 42 }, (_, i) => [581 + i * 5.25, 355 + row * 3.8 + 22 * Math.sin(i / 7 + row / 12)]);
    path(points, color, { weight: 1.1 });
  }
  text("Ribbon motif, not data", 588, 482, 14, "#b5c7d6");
  text("Read the geometry: the line at Re(s) = 1/2 bisects the strip 0 < Re(s) < 1.", 43, 550, 17);
}

const branch = page({ background: "#fbf5e8", foreground: "#263932", muted: "#52655c",
  title: "Which schema extras does FastAPI use?", kicker: "02  /  SOURCE-LINKED CODE BEHAVIOR",
  subtitle: "Truthiness selects the extras before the version-specific code runs." });
{
  const { text, box, path } = branch;
  const accent = "#39786a", option = "#a15e33";
  path([[421,159],[571,215],[421,271],[271,215]], accent, { fill: "#dbece2" });
  text("json_schema_extra truthy?", 421, 220, 20, "#263932", "center");
  text("Lines " + refs("check"), 421, 239, 14, "#52655c", "center");
  path([[271,215],[198,215],[198,305]], accent, { arrow: true });
  path([[571,215],[644,215],[644,305]], option, { arrow: true });
  text("Truthy", 185, 274, 17, accent, "right");
  text("Falsy", 658, 274, 17, option);
  box(55, 305, 286, 68, "#eff6ed", accent);
  box(501, 305, 286, 68, "#fff1dc", option);
  text("Use json_schema_extra", 198, 333, 21, "#263932", "center");
  text("Lines " + refs("explicit"), 198, 355, 14, "#52655c", "center");
  text("Use extra as supplied", 644, 333, 21, "#263932", "center");
  text("Lines " + refs("fallback"), 644, 355, 14, "#52655c", "center");
  path([[198,373],[198,416],[338,416],[338,440]], accent, { arrow: true });
  path([[644,373],[644,416],[504,416],[504,440]], option, { arrow: true });
  box(254, 440, 334, 63, "#e0ece4", accent);
  text("Check the Pydantic version", 421, 467, 21, "#263932", "center");
  text("Lines " + refs("next"), 421, 490, 14, "#52655c", "center");
  text("FastAPI PR #9816  |  " + saved.headSha.slice(0, 12) + "  |  Chapter patch coordinates", 43, 543, 15);
  text("Selected behavior only. Not a runtime trace; version-specific handling is outside this view.", 43, 565, 14, "#52655c");
}

const heatmap = page({ background: "#fff8eb", foreground: "#3a2d22", muted: "#755c46",
  title: "Read attention one row at a time", kicker: "03  /  ILLUSTRATIVE ATTENTION WEIGHTS",
  subtitle: "Each query distributes one unit of attention across the keys." });
{
  const { text, box, path } = heatmap;
  const tokens = ["The", "cat", "sat"], weights = [[.10,.70,.20],[.20,.20,.60],[.55,.30,.15]];
  for (const row of weights) assert(Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) < 1e-9);
  const palette = ["#f2dfba", "#ebcc92", "#d49d5e", "#ac6438", "#813f27"];
  text("KEYS", 307, 171, 14, "#755c46", "center");
  text("QUERY", 44, 214, 14, "#755c46");
  text("SUM", 532, 214, 14, "#755c46", "center");
  tokens.forEach((word, i) => {
    text(word, 143 + i * 122 + 57, 214, 22, "#3a2d22", "center");
    text(word, 47, 267 + i * 78, 22);
  });
  weights.forEach((row, i) => {
    row.forEach((weight, j) => {
      const x = 143 + j * 122, y = 229 + i * 78, best = weight === Math.max(...row);
      const color = palette[Math.min(4, Math.floor(weight * 7))];
      box(x, y, 114, 68, color, best ? "#37291d" : undefined, 2.2);
      text(weight.toFixed(2), x + 57, y + 42, 24, weight >= .55 ? "#fff9ed" : "#3a2d22", "center");
    });
    text("1.00", 532, 272 + i * 78, 20, "#755c46", "center");
  });
  path([[566,229],[566,453]], "#d9b98a", { weight: 1 });
  text("Start with cat", 598, 261, 25);
  text("Its largest weight", 598, 296, 18);
  text("goes to sat (0.60).", 598, 320, 18);
  text("Outlined cells mark", 598, 377, 16, "#755c46");
  text("each row maximum.", 598, 399, 16, "#755c46");
  text("Three words, one head. These weights are made up for teaching, not learned or measured.", 43, 547, 16);
}

await mkdir(out, { recursive: true });
const temp = await mkdtemp(join(tmpdir(), "visual-skill-"));
const results = [];
try {
  for (const [name, figure, keywords] of [
    ["critical-strip", math, ["No zeros are plotted", "Still unproved"]],
    ["fastapi-branches", branch, ["Truthy", "Falsy", "153, 583", "Not a runtime trace"]],
    ["attention-heatmap", heatmap, ["0.60", "1.00", "made up for teaching"]],
  ]) {
    const script = join(temp, name + ".json"), png = join(out, name + ".png"), pdf = join(out, name + ".pdf");
    await writeFile(script, JSON.stringify(figure.commands), { mode: 0o600 });
    const start = performance.now();
    const stdout = execFileSync(cli, ["--safe-mode", "run", "--commands", script, "--export", png, "--export", pdf, "--scale", "1.5"], {
      cwd: temp, encoding: "utf8", timeout: 15000, maxBuffer: 1_000_000, env: { HOME: temp, VECTORCRAFT_NO_PREFS: "1" },
    });
    const milliseconds = Math.round(performance.now() - start);
    for (const line of stdout.trim().split("\n")) {
      const result = JSON.parse(line);
      assert(!result.error && !result.result?.warning && !result.result?.warnings?.length && !result.result?.overflow?.length, line);
      const b = result.result?.bounds;
      if (b) assert(b.x >= 0 && b.y >= 0 && b.x + b.width <= width && b.y + b.height <= height, "Object outside artboard: " + line);
    }
    const rendered = await readFile(png);
    assert.equal(rendered.readUInt32BE(16), 1263);
    const extracted = execFileSync("pdftotext", [pdf, "-"], { encoding: "utf8" });
    for (const keyword of keywords) assert(extracted.includes(keyword), "Missing PDF text: " + keyword);
    results.push({ name, milliseconds, png, pdf });
  }
} finally {
  await rm(temp, { recursive: true, force: true });
}
console.log(JSON.stringify(results, null, 2));
