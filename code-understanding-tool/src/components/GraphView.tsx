import { useMemo, useState } from "react";
import { ReactFlow, Background, Handle, MarkerType, Position, BaseEdge, EdgeLabelRenderer, getSmoothStepPath, type Edge, type EdgeProps, type Node, type NodeProps } from "@xyflow/react";
import { isValidRange, type SourceRange } from "../domain/source-ranges";

type DiagramVisual = {
  kind: "control_flow" | "concept_map";
  nodes: { id: string; label: string; description: string; source_ranges: SourceRange[] }[];
  edges: { from: string; to: string; label: string }[];
};
type FlowNodeData = { label: string; description: string; caption: string; source_ranges: SourceRange[] };
type FlowNode = Node<FlowNodeData, "explanation">;
type ReadableEdge = Edge<{ label: string }, "readable">;

export function GraphView({ visual, theme, lineCount, source, onSelectRange }: { visual: DiagramVisual; theme: "light" | "dark"; lineCount: number; source: string; onSelectRange: (range: SourceRange) => void }) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const layout = useMemo(() => layoutGraph(visual, theme), [visual, theme]);
  const nodeTypes = useMemo(() => ({ explanation: ExplanationFlowNode }), []);
  const edgeTypes = useMemo(() => ({ readable: ReadableFlowEdge }), []);
  const selectedNode = visual.nodes.find((node) => node.id === selectedNodeId);
  const sourceLines = useMemo(() => source.split("\n"), [source]);
  const invalidEdgeCount = visual.edges.length - layout.edges.length;

  return <div className="graph-panel" aria-label={visual.kind === "control_flow" ? "Interactive control flow diagram" : "Interactive concept map"}>
    {visual.nodes.length > 0 ? <div className={`flow-canvas ${theme === "dark" ? "theme-dark" : ""}`}>
      <ReactFlow
        nodes={layout.nodes}
        edges={layout.edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        colorMode={theme}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.35}
        maxZoom={1.5}
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable
        onNodeClick={(_, node) => setSelectedNodeId(node.id)}
        aria-label={visual.kind === "control_flow" ? "Control flow diagram" : "Concept map"}
      >
        <Background color={theme === "dark" ? "#343747" : "#e8e6f2"} gap={22} size={1} />
      </ReactFlow>
    </div> : <p className="empty-graph">No visual relationships were generated.</p>}
    {visual.nodes.length > 0 && <details className="diagram-text-alternative">
      <summary>Read diagram as text</summary>
      <ol>{visual.nodes.map((node) => <li key={node.id}><strong>{node.label}.</strong> {node.description}</li>)}</ol>
      {visual.edges.length > 0 && <ul aria-label="Diagram connections">{visual.edges.map((edge, index) => <li key={`${edge.from}-${edge.to}-${index}`}>{visual.nodes.find((node) => node.id === edge.from)?.label ?? edge.from} {edge.label ? `(${edge.label})` : "connects to"} {visual.nodes.find((node) => node.id === edge.to)?.label ?? edge.to}.</li>)}</ul>}
    </details>}
    {selectedNode ? <div className="selected-node-detail" aria-live="polite">
      <div><span className="node-caption">SELECTED {visual.kind === "control_flow" ? "STEP" : "CONCEPT"}</span><h3>{selectedNode.label}</h3><p>{selectedNode.description}</p></div>
      {selectedNode.source_ranges.some((range) => isValidRange(range, lineCount)) ? <div className="selected-source"><span className="node-caption">RELATED SOURCE</span>{selectedNode.source_ranges.filter((range) => isValidRange(range, lineCount)).map((range, index) => <div className="selected-source-range" key={`${range.start_line}-${range.end_line}-${index}`}><button className="source-range" type="button" onClick={() => onSelectRange(range)} aria-label={`Select source lines ${range.start_line} through ${range.end_line}`}>Lines {range.start_line}{range.end_line !== range.start_line ? `–${range.end_line}` : ""} · Jump to source</button><pre>{sourceLines.slice(range.start_line - 1, range.end_line).map((text, lineIndex) => `${range.start_line + lineIndex}  ${text}`).join("\n")}</pre></div>)}</div> : <p className="empty-graph">No valid source lines are linked to this item.</p>}
    </div> : <p className="graph-hint">Select a node to inspect its explanation and related source lines. Drag to rearrange; use the controls to zoom or fit the diagram.</p>}
    {invalidEdgeCount > 0 && <p className="empty-graph">{invalidEdgeCount} connection{invalidEdgeCount === 1 ? " was" : "s were"} omitted because they referenced missing nodes.</p>}
  </div>;
}

function ExplanationFlowNode({ data, selected }: NodeProps<FlowNode>) {
  return <div className={`flow-card-node ${selected ? "selected" : ""}`}>
    <Handle type="target" position={Position.Left} />
    <span className="node-caption">{data.caption}</span>
    <strong>{data.label}</strong>
    <p>{data.description}</p>
    <Handle type="source" position={Position.Right} />
  </div>;
}

function ReadableFlowEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, style, data }: EdgeProps<ReadableEdge>) {
  const [path, labelX, labelY] = getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  const label = data?.label;
  return <>
    <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
    {label && <EdgeLabelRenderer><div className="flow-edge-label" style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}>{label}</div></EdgeLabelRenderer>}
  </>;
}

function layoutGraph(visual: DiagramVisual, theme: "light" | "dark"): { nodes: FlowNode[]; edges: Edge[] } {
  const validNodes = visual.nodes.filter((node) => node.id.trim().length > 0);
  const ids = new Set(validNodes.map((node) => node.id));
  const validEdges = visual.edges.filter((edge) => ids.has(edge.from) && ids.has(edge.to));
  const outgoing = new Map<string, string[]>();
  const incomingCounts = new Map(validNodes.map((node) => [node.id, 0]));
  for (const edge of validEdges) {
    outgoing.set(edge.from, [...(outgoing.get(edge.from) ?? []), edge.to]);
    incomingCounts.set(edge.to, (incomingCounts.get(edge.to) ?? 0) + 1);
  }

  const ranks = new Map<string, number>();
  const queue = validNodes.filter((node) => (incomingCounts.get(node.id) ?? 0) === 0).map((node) => node.id);
  queue.forEach((id) => ranks.set(id, 0));
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const id = queue[cursor];
    for (const target of outgoing.get(id) ?? []) {
      if (ranks.has(target)) continue;
      ranks.set(target, (ranks.get(id) ?? 0) + 1);
      queue.push(target);
    }
  }
  for (const node of validNodes) if (!ranks.has(node.id)) ranks.set(node.id, 0);

  const rowByRank = new Map<number, number>();
  const nodes = validNodes.map((node, index): FlowNode => {
    const rank = ranks.get(node.id) ?? 0;
    const row = rowByRank.get(rank) ?? 0;
    rowByRank.set(rank, row + 1);
    return {
      id: node.id,
      type: "explanation",
      position: { x: rank * 330, y: row * 175 },
      data: {
        label: node.label,
        description: node.description,
        caption: visual.kind === "control_flow" ? `STEP ${String(index + 1).padStart(2, "0")}` : "CONCEPT",
        source_ranges: node.source_ranges,
      },
    };
  });
  const edges: Edge[] = validEdges.map((edge, index) => ({
    id: `edge-${index}-${edge.from}-${edge.to}`,
    type: "readable",
    source: edge.from,
    target: edge.to,
    data: { label: edge.label },
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: "#8176df" },
    style: { stroke: theme === "dark" ? "#a79af7" : "#968ce5", strokeWidth: 1.7 },
  }));
  return { nodes, edges };
}
