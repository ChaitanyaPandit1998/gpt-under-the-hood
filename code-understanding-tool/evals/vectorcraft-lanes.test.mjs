import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { diagramPlan, laneCommands, laneLayout } from "./vectorcraft-lanes.mjs";

const node = (id) => ({ id, label: id, source_ranges: [{ start_line: 1, end_line: 2 }] });
const edge = (from, to, label = "") => ({ from, to, label });
const jobFixture = JSON.parse(readFileSync(new URL("./fixtures/vectorcraft-job-processing.json", import.meta.url), "utf8"));
const visual = {
  title: "Nested branches",
  nodes: ["Start", "Left", "Right", "A", "B", "Join"].map(node),
  edges: [edge("Start", "Left", "Yes"), edge("Start", "Right", "No"), edge("Left", "A", "Truthy"),
    edge("Left", "B", "Falsy"), edge("A", "Join"), edge("B", "Join"), edge("Right", "Join")],
};

test("draws each node and edge, with source lines and decision labels", () => {
  const { layout, commands } = laneCommands(visual, "Example");
  assert.equal(layout.positions.size, visual.nodes.length);
  assert.deepEqual(layout.edges, visual.edges);
  assert.equal(commands.filter((c) => c.command === "stroke.set" && c.params.endArrow === "Triangle").length, visual.edges.length);
  const labels = commands.filter((c) => c.command === "text.create").map((c) => c.params.text);
  for (const label of ["Yes", "No", "Truthy", "Falsy"]) assert(labels.includes(label));
  assert.equal(labels.filter((line) => line === "Lines 1-2").length, visual.nodes.length);
  assert(layout.positions.get("Left").x < layout.positions.get("Right").x);
  for (const { from, to } of visual.edges) assert(layout.positions.get(from).y < layout.positions.get(to).y);
});

test("rejects cycles, parallel edges, missing endpoints, and branches without a join", () => {
  for (const extra of [edge("Join", "Left"), edge("Start", "Left"), edge("Join", "Unknown")]) {
    assert.throws(() => laneLayout({ ...visual, edges: [...visual.edges, extra] }));
  }
  assert.throws(() => laneLayout({ ...visual, edges: visual.edges.filter((e) => e.from !== "Right") }), /shared join/);
});

test("connection-index fallback preserves cycles, self-loops, and parallel labels", () => {
  const edges = [...visual.edges, edge("Join", "Start", "Retry"), edge("Join", "Join", "Self-loop"), edge("Start", "Left", "Second reason")];
  const plan = diagramPlan({ ...visual, edges }, "Example");
  assert.equal(plan.mode, "connection-index");
  assert.deepEqual(plan.layout.edges, edges);
  const labels = plan.commands.filter((c) => c.command === "text.create").map((c) => c.params.text);
  assert.equal(labels.filter((line) => /^E\d+ Node \d+ -> Node \d+$/.test(line)).length, edges.length);
  for (const label of ["Retry", "Self-loop", "Second reason"]) assert(labels.includes(label));
  assert.equal(labels.filter((line) => line === "Lines 1-2").length, visual.nodes.length);
  assert.throws(() => diagramPlan({ ...visual, edges: [...edges, edge("A", "missing")] }, "Example"), /endpoint/);
});

for (const [file, mode] of [["jsonl-buffering", "lanes"], ["reset-content", "connection-index"]]) {
  test("selects " + mode + " for the " + file + " PR fixture", () => {
    const fixture = JSON.parse(readFileSync(new URL("./fixtures/vectorcraft-" + file + ".json", import.meta.url), "utf8"));
    const view = fixture.explanation.visuals[0];
    const plan = diagramPlan(view, "FastAPI", fixture.lineLabel);
    assert.equal(plan.mode, mode);
    assert.equal(plan.layout.positions.size, view.nodes.length);
    assert.deepEqual(plan.layout.edges, view.edges);
  });
}

test("lays out a 16-node job and uses the fallback when a retry points backward", () => {
  const [view] = jobFixture.explanation.visuals;
  const lanes = diagramPlan(view, "Job processing", jobFixture.lineLabel);
  assert.equal(lanes.mode, "lanes");
  assert.equal(lanes.layout.positions.size, 16);
  assert.equal(lanes.commands.filter((command) => command.params?.endArrow === "Triangle").length, 18);
  const retry = edge("backup", "config", "Retryable failure with budget remaining");
  const result = diagramPlan({ ...view, edges: [...view.edges, retry] }, "Job processing", jobFixture.lineLabel);
  assert.equal(result.mode, "connection-index");
  assert.equal(result.layout.edges.length, 19);
  assert.deepEqual(result.layout.edges.at(-1), retry);
});
