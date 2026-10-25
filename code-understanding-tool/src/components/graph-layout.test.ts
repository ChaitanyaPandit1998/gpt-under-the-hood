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

  it("uses a vertical, full-width layout for compact branching diagrams", () => {
    const graph = visual(["start", "accepted", "rejected", "finish"], [
      { from: "start", to: "accepted", label: "accepted" },
      { from: "start", to: "rejected", label: "rejected" },
      { from: "accepted", to: "finish", label: "continue" },
      { from: "rejected", to: "finish", label: "continue" },
    ]);

    const layout = createGraphLayout(graph, "light", true);
    const positions = positionsById(layout);

    expect(layout.nodes.every((node) => node.data.orientation === "vertical")).toBe(true);
    expect(layout.edges).toHaveLength(graph.edges.length);
    for (const edge of layout.edges) {
      expect(positions.get(edge.target)!.y).toBeGreaterThan(positions.get(edge.source)!.y);
    }
    expect(positions.get("accepted")!.x).not.toBe(positions.get("rejected")!.x);
    expect(graphNodesOverlap(layout)).toBe(false);
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

  it("preserves every outcome in a branch-heavy pricing flow", () => {
    const graph = visual(
      [
        "check-address", "address-error", "check-total", "free-base", "standard-base",
        "check-expedited", "add-expedited-fee", "keep-base", "check-member",
        "apply-discount", "keep-undiscounted", "return-cost",
      ],
      [
        { from: "check-address", to: "address-error", label: "missing: return error" },
        { from: "check-address", to: "check-total", label: "present" },
        { from: "check-total", to: "free-base", label: "total > 100" },
        { from: "check-total", to: "standard-base", label: "total ≤ 100" },
        { from: "free-base", to: "check-expedited", label: "continue" },
        { from: "standard-base", to: "check-expedited", label: "continue" },
        { from: "check-expedited", to: "add-expedited-fee", label: "expedited" },
        { from: "check-expedited", to: "keep-base", label: "not expedited" },
        { from: "add-expedited-fee", to: "check-member", label: "continue" },
        { from: "keep-base", to: "check-member", label: "continue" },
        { from: "check-member", to: "apply-discount", label: "member" },
        { from: "check-member", to: "keep-undiscounted", label: "not a member" },
        { from: "apply-discount", to: "return-cost", label: "return amount" },
        { from: "keep-undiscounted", to: "return-cost", label: "return amount" },
      ],
    );

    const layout = createGraphLayout(graph, "light");
    const actualBranches = new Set(layout.edges.map((edge) => `${edge.source} -> ${edge.target}: ${edge.data?.label}`));

    expect(layout.nodes.map((node) => node.id)).toHaveLength(12);
    expect(actualBranches).toEqual(new Set(graph.edges.map((edge) => `${edge.from} -> ${edge.to}: ${edge.label}`)));
    const expectedOutcomeLabels: Record<string, string[]> = {
      "check-address": ["missing: return error", "present"],
      "check-total": ["total > 100", "total ≤ 100"],
      "check-expedited": ["expedited", "not expedited"],
      "check-member": ["member", "not a member"],
    };
    for (const [decision, expectedLabels] of Object.entries(expectedOutcomeLabels)) {
      const labels = layout.edges.filter((edge) => edge.source === decision).map((edge) => edge.data?.label);
      expect(labels.sort()).toEqual(expectedLabels.sort());
      expect(labels).not.toContain("either outcome");
      expect(labels).not.toContain("continue");
    }
    expect(layout.nodes.find((node) => node.id === "address-error")!.position.x).toBeLessThan(
      layout.nodes.find((node) => node.id === "return-cost")!.position.x,
    );
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
