import { describe, expect, it } from "vitest";
import { createGraphLayout, graphNodesOverlap, type DiagramVisual } from "./graph-layout";

function visual(nodeIds: string[], edges: DiagramVisual["edges"]): DiagramVisual {
  return {
    kind: "control_flow",
    nodes: nodeIds.map((id) => ({ id, label: id, description: `Details for ${id}`, source_ranges: [] })),
    edges,
  };
}

function positionsById(layout: ReturnType<typeof createGraphLayout>) {
  return new Map(layout.nodes.map((node) => [node.id, node.position]));
}

function countAdjacentLayerCrossings(layout: ReturnType<typeof createGraphLayout>): number {
  const positions = positionsById(layout);
  const edges = layout.edges.filter((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    return source && target && target.x - source.x === 330;
  });
  let crossings = 0;
  for (let leftIndex = 0; leftIndex < edges.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < edges.length; rightIndex += 1) {
      const left = edges[leftIndex];
      const right = edges[rightIndex];
      const leftSource = positions.get(left.source)!;
      const rightSource = positions.get(right.source)!;
      const leftTarget = positions.get(left.target)!;
      const rightTarget = positions.get(right.target)!;
      if (leftSource.x === rightSource.x && leftTarget.x === rightTarget.x &&
          (leftSource.y - rightSource.y) * (leftTarget.y - rightTarget.y) < 0) crossings += 1;
    }
  }
  return crossings;
}

describe("createGraphLayout", () => {
  it("lays out branching and converging paths without overlapping cards", () => {
    const graph = visual(["start", "resolved", "rejected", "retry", "exhausted", "finish"], [
      { from: "start", to: "resolved", label: "resolves" },
      { from: "start", to: "rejected", label: "rejects" },
      { from: "rejected", to: "retry", label: "attempts > 1" },
      { from: "rejected", to: "exhausted", label: "attempts ≤ 1" },
      { from: "resolved", to: "finish", label: "returns" },
      { from: "retry", to: "finish", label: "runs again" },
      { from: "exhausted", to: "finish", label: "throws" },
    ]);

    const layout = createGraphLayout(graph, "light");
    const positions = positionsById(layout);

    expect(layout.nodes).toHaveLength(graph.nodes.length);
    expect(layout.edges).toHaveLength(graph.edges.length);
    expect(graphNodesOverlap(layout)).toBe(false);
    for (const edge of layout.edges) {
      expect(positions.get(edge.target)!.x).toBeGreaterThan(positions.get(edge.source)!.x);
    }
    expect(positions.get("finish")!.x).toBeGreaterThan(positions.get("retry")!.x);
    expect(positions.get("finish")!.x).toBeGreaterThan(positions.get("exhausted")!.x);
  });

  it("reorders a reversed branch merge to remove the crossing", () => {
    const graph = visual(["start", "left", "right", "join-left", "join-right"], [
      { from: "start", to: "left", label: "left" },
      { from: "start", to: "right", label: "right" },
      { from: "left", to: "join-right", label: "crosses input order" },
      { from: "right", to: "join-left", label: "crosses input order" },
    ]);

    const layout = createGraphLayout(graph, "light");
    const positions = positionsById(layout);

    expect(positions.get("join-right")!.y).toBeLessThan(positions.get("join-left")!.y);
    expect(countAdjacentLayerCrossings(layout)).toBe(0);
    expect(graphNodesOverlap(layout)).toBe(false);
  });

  it("keeps a converging node beyond every parent on an uneven path", () => {
    const graph = visual(["start", "short", "long-1", "long-2", "merge"], [
      { from: "start", to: "short", label: "short path" },
      { from: "start", to: "long-1", label: "long path" },
      { from: "long-1", to: "long-2", label: "continues" },
      { from: "short", to: "merge", label: "joins" },
      { from: "long-2", to: "merge", label: "joins" },
    ]);

    const layout = createGraphLayout(graph, "dark");
    const positions = positionsById(layout);

    expect(positions.get("merge")!.x).toBeGreaterThan(positions.get("long-2")!.x);
    expect(graphNodesOverlap(layout)).toBe(false);
  });

  it("reserves enough space between cards for relationship labels", () => {
    const graph = visual(["initial value", "reset()"], [
      { from: "initial value", to: "reset()", label: "value restored by" },
    ]);

    const layout = createGraphLayout(graph, "light");
    const positions = positionsById(layout);
    const availableLabelSpace = positions.get("reset()")!.x - positions.get("initial value")!.x - 250;

    expect(availableLabelSpace).toBeGreaterThan("value restored by".length * 6 + 14);
    expect(graphNodesOverlap(layout)).toBe(false);
  });

  it("omits duplicate, cyclic, self-referencing, and missing-node connections", () => {
    const graph = visual(["start", "middle", "end", ""], [
      { from: "start", to: "middle", label: "first" },
      { from: "start", to: "middle", label: "duplicate" },
      { from: "middle", to: "end", label: "forward" },
      { from: "end", to: "start", label: "cycle" },
      { from: "end", to: "end", label: "self" },
      { from: "missing", to: "end", label: "unknown source" },
    ]);

    const layout = createGraphLayout(graph, "light");

    expect(layout.nodes.map((node) => node.id)).toEqual(["start", "middle", "end"]);
    expect(layout.edges.map((edge) => edge.data?.label)).toEqual(["first", "forward"]);
  });
});
