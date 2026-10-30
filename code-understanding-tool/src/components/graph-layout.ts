import { MarkerType, type Edge, type Node } from "@xyflow/react";
import type { SourceRange } from "../domain/source-ranges";
import type { IncompleteCodeItem } from "../domain/incomplete-code.ts";
import { themeAccents, type ThemeMode } from "../domain/theme";

export type DiagramVisual = {
  kind: "control_flow" | "concept_map";
  nodes: { id: string; label: string; description: string; source_ranges: SourceRange[] }[];
  edges: { from: string; to: string; label: string }[];
};

type FlowNodeData = { label: string; description: string; caption: string; source_ranges: SourceRange[]; orientation: "horizontal" | "vertical"; incomplete: boolean };
export type FlowNode = Node<FlowNodeData, "explanation">;
export type FlowEdge = Edge<{ label: string }, "readable">;
export type GraphLayout = { nodes: FlowNode[]; edges: FlowEdge[] };

const horizontalGap = 420;
const verticalGap = 190;
const estimatedCardWidth = 250;
const estimatedCardHeight = 130;

export function createGraphLayout(visual: DiagramVisual, theme: ThemeMode, compact = false, incompleteItems: IncompleteCodeItem[] = []): GraphLayout {
  const seenNodeIds = new Set<string>();
  const validNodes = visual.nodes.filter(({ id }) => {
    if (!id.trim() || seenNodeIds.has(id)) return false;
    seenNodeIds.add(id);
    return true;
  });
  const nodeIds = new Set(validNodes.map(({ id }) => id));
  const adjacency = new Map(validNodes.map(({ id }) => [id, [] as string[]]));
  const edgeKeys = new Set<string>();
  const validEdges: FlowEdge[] = [];

  for (const edge of visual.edges) {
    const key = `${edge.from}\u0000${edge.to}`;
    if (!nodeIds.has(edge.from) || !nodeIds.has(edge.to) || edge.from === edge.to || edgeKeys.has(key)) continue;
    if (isReachable(adjacency, edge.to, edge.from)) continue;
    adjacency.get(edge.from)?.push(edge.to);
    edgeKeys.add(key);
    validEdges.push({
      id: `edge-${validEdges.length}-${edge.from}-${edge.to}`,
      type: "readable",
      source: edge.from,
      target: edge.to,
      data: { label: edge.label },
      animated: false,
      markerEnd: { type: MarkerType.ArrowClosed, color: themeAccents[theme].marker },
      style: { stroke: themeAccents[theme].edge, strokeWidth: 1.7 },
    });
  }

  const rankById = rankNodes(validNodes, validEdges);
  const layers = groupByRank(validNodes, rankById);
  reduceLayerCrossings(layers, validEdges);
  const flowOrder = layers.flat();
  const sequence = new Map(flowOrder.map((node, index) => [node.id, index + 1]));
  const positionById = new Map<string, { x: number; y: number }>();
  layers.forEach((layer, rank) => layer.forEach((node, row) => {
    positionById.set(node.id, compact
      ? { x: row * 300, y: rank * verticalGap }
      : { x: rank * horizontalGap, y: row * verticalGap });
  }));

  const nodes = flowOrder.map((node): FlowNode => ({
    id: node.id,
    type: "explanation",
    position: positionById.get(node.id) ?? { x: 0, y: 0 },
    data: {
      label: node.label,
      description: node.description,
      caption: visual.kind === "control_flow" ? `STEP ${String(sequence.get(node.id) ?? 1).padStart(2, "0")}` : "CONCEPT",
      source_ranges: node.source_ranges,
      orientation: compact ? "vertical" : "horizontal",
      incomplete: incompleteItems.some((item) => item.source_ranges.some((range) => node.source_ranges.some((nodeRange) => nodeRange.start_line <= range.end_line && nodeRange.end_line >= range.start_line))),
    },
  }));

  return { nodes, edges: validEdges };
}

function isReachable(adjacency: Map<string, string[]>, from: string, target: string): boolean {
  const pending = [from];
  const visited = new Set<string>();
  while (pending.length) {
    const current = pending.pop()!;
    if (current === target) return true;
    if (visited.has(current)) continue;
    visited.add(current);
    pending.push(...(adjacency.get(current) ?? []));
  }
  return false;
}

function rankNodes(nodes: DiagramVisual["nodes"], edges: FlowEdge[]): Map<string, number> {
  const incomingCounts = new Map(nodes.map(({ id }) => [id, 0]));
  const outgoing = new Map(nodes.map(({ id }) => [id, [] as FlowEdge[]]));
  for (const edge of edges) {
    incomingCounts.set(edge.target, (incomingCounts.get(edge.target) ?? 0) + 1);
    outgoing.get(edge.source)?.push(edge);
  }

  const ranks = new Map(nodes.map(({ id }) => [id, 0]));
  const pending = nodes.filter(({ id }) => incomingCounts.get(id) === 0).map(({ id }) => id);
  for (let cursor = 0; cursor < pending.length; cursor += 1) {
    const source = pending[cursor];
    for (const edge of outgoing.get(source) ?? []) {
      ranks.set(edge.target, Math.max(ranks.get(edge.target) ?? 0, (ranks.get(source) ?? 0) + 1));
      const remainingIncoming = (incomingCounts.get(edge.target) ?? 1) - 1;
      incomingCounts.set(edge.target, remainingIncoming);
      if (remainingIncoming === 0) pending.push(edge.target);
    }
  }
  return ranks;
}

function groupByRank(nodes: DiagramVisual["nodes"], ranks: Map<string, number>): DiagramVisual["nodes"][] {
  const layers: DiagramVisual["nodes"][] = [];
  for (const node of nodes) {
    const rank = ranks.get(node.id) ?? 0;
    (layers[rank] ??= []).push(node);
  }
  return layers;
}

function reduceLayerCrossings(layers: DiagramVisual["nodes"][], edges: FlowEdge[]) {
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  for (const edge of edges) {
    incoming.set(edge.target, [...(incoming.get(edge.target) ?? []), edge.source]);
    outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge.target]);
  }

  for (let pass = 0; pass < 4; pass += 1) {
    for (let rank = 1; rank < layers.length; rank += 1) {
      sortByNeighborBarycenter(layers[rank], incoming, layers.slice(0, rank));
    }
    for (let rank = layers.length - 2; rank >= 0; rank -= 1) {
      sortByNeighborBarycenter(layers[rank], outgoing, layers.slice(rank + 1));
    }
  }
}

function sortByNeighborBarycenter(
  layer: DiagramVisual["nodes"],
  neighborsByNode: Map<string, string[]>,
  referenceLayers: DiagramVisual["nodes"][],
) {
  const order = new Map<string, number>();
  referenceLayers.flat().forEach((node, index) => order.set(node.id, index));
  const originalOrder = new Map(layer.map((node, index) => [node.id, index]));
  const barycenter = (id: string) => {
    const neighbors = neighborsByNode.get(id) ?? [];
    const positions = neighbors.map((neighbor) => order.get(neighbor)).filter((position): position is number => position !== undefined);
    return positions.length ? positions.reduce((sum, position) => sum + position, 0) / positions.length : originalOrder.get(id) ?? 0;
  };
  layer.sort((left, right) => barycenter(left.id) - barycenter(right.id) || (originalOrder.get(left.id) ?? 0) - (originalOrder.get(right.id) ?? 0));
}

export function graphNodesOverlap(layout: GraphLayout): boolean {
  return layout.nodes.some((left, index) => layout.nodes.slice(index + 1).some((right) => {
    const horizontalOverlap = Math.abs(left.position.x - right.position.x) < estimatedCardWidth;
    const verticalOverlap = Math.abs(left.position.y - right.position.y) < estimatedCardHeight;
    return horizontalOverlap && verticalOverlap;
  }));
}
