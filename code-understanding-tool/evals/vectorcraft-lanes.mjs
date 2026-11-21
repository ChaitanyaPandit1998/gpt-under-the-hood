import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { diagramPlan } from "../src/server/vectorcraft-lanes.mjs";
export { diagramPlan, laneCommands, laneLayout } from "../src/server/vectorcraft-lanes.mjs";

async function main() {
  const [input, title, output, cliPath] = process.argv.slice(2);
  assert(input && title && output?.endsWith(".png") && cliPath, "Usage: node evals/vectorcraft-lanes.mjs INPUT VISUAL_TITLE OUTPUT.png CLI");
  const saved = JSON.parse(await readFile(input, "utf8"));
  const visual = saved.explanation.visuals.find((view) => view.title === title);
  assert(visual, "No matching visual.");
  const match = /^https:\/\/github\.com\/([\w.-]+\/[\w.-]+)\/pull\/(\d+)$/.exec(saved.url);
  const sourceLabel = match ? match[1] + "  /  PR #" + match[2] : "Saved chapter fixture";
  const { commands, layout, width, height, mode } = diagramPlan(visual, sourceLabel, saved.lineLabel);
  const directory = await mkdtemp(join(tmpdir(), "vectorcraft-lanes-"));
  try {
    const script = join(directory, "commands.json"), png = join(directory, "diagram.png");
    await writeFile(script, JSON.stringify(commands), { mode: 0o600 });
    const start = performance.now();
    const stdout = execFileSync(resolve(cliPath), ["--safe-mode", "run", "--commands", script, "--export", png, "--scale", "1"], {
      cwd: directory, encoding: "utf8", timeout: 15000, maxBuffer: 1_000_000, env: { HOME: directory, VECTORCRAFT_NO_PREFS: "1" },
    });
    for (const line of stdout.trim().split("\n")) {
      const step = JSON.parse(line);
      assert(!step.error && !step.result?.warning && !step.result?.warnings?.length, "VectorCraft reported a warning or error.");
      const b = step.result?.bounds;
      if (b) assert(b.x >= 0 && b.y >= 0 && b.x + b.width <= width && b.y + b.height <= height, "Drawing exceeds artboard.");
    }
    const data = await readFile(png);
    assert.equal(data.readUInt32BE(16), width);
    assert.equal(data.readUInt32BE(20), height);
    await mkdir(dirname(resolve(output)), { recursive: true });
    await writeFile(output, data, { mode: 0o600 });
    console.log(JSON.stringify({ output, mode, nodes: layout.positions.size, connections: layout.edges.length, renderMs: Math.round(performance.now()-start) }));
  } finally { await rm(directory, { recursive: true, force: true }); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
