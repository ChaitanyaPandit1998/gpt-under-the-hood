import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ReactFlow, Background, Handle, Position, BaseEdge, EdgeLabelRenderer, getSmoothStepPath, type EdgeProps, type NodeProps, type ReactFlowInstance } from "@xyflow/react";
import { isValidRange, type SourceRange } from "../domain/source-ranges";
import type { IncompleteCodeItem } from "../domain/incomplete-code.ts";
import { themeAccents, type ThemeMode } from "../domain/theme";
import { createGraphLayout, type DiagramVisual, type FlowEdge, type FlowNode } from "./graph-layout";

export function GraphView({ visual, theme, lineCount, source, onSelectRange, incompleteItems = [] }: { visual: DiagramVisual & { title: string }; theme: ThemeMode; lineCount: number; source: string; onSelectRange: (range: SourceRange) => void; incompleteItems?: IncompleteCodeItem[] }) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [showFitControl, setShowFitControl] = useState(false);
  const [compactLayout, setCompactLayout] = useState(false);
  const flowCanvas = useRef<HTMLDivElement>(null);
  const flowInstance = useRef<ReactFlowInstance<FlowNode, FlowEdge> | null>(null);
  const fitFrame = useRef<number | null>(null);
  const layout = useMemo(() => createGraphLayout(visual, theme, compactLayout, incompleteItems), [visual, theme, compactLayout, incompleteItems]);
  const compactCanvasHeight = compactLayout
    ? Math.max(410, ...layout.nodes.map((node) => node.position.y + 170))
    : undefined;
  const nodeTypes = useMemo(() => ({ explanation: ExplanationFlowNode }), []);
  const edgeTypes = useMemo(() => ({ readable: ReadableFlowEdge }), []);
  const selectedNode = visual.nodes.find((node) => node.id === selectedNodeId);
  const sourceLines = useMemo(() => source.split("\n"), [source]);
  const invalidEdgeCount = visual.edges.length - layout.edges.length;
  const fitGraph = useCallback(() => {
    if (fitFrame.current !== null) cancelAnimationFrame(fitFrame.current);
    fitFrame.current = requestAnimationFrame(() => {
      const instance = flowInstance.current;
      if (!instance) return;
      void instance.fitView({ padding: 0.2, minZoom: 0.12, maxZoom: 1.5, duration: 0 }).then(() => {
        const isNarrow = (flowCanvas.current?.clientWidth ?? window.innerWidth) < 640;
        setShowFitControl(isNarrow || instance.getViewport().zoom < 0.42);
      });
    });
  }, []);

  useEffect(() => {
    const canvas = flowCanvas.current;
    if (!canvas || visual.nodes.length === 0) return;

    const observer = new ResizeObserver(([entry]) => {
      const isNarrow = entry.contentRect.width < 640;
      setCompactLayout((current) => current === isNarrow ? current : isNarrow);
      fitGraph();
    });
    observer.observe(canvas);
    let printFitTimer = 0;
    const refitForPrint = () => {
      window.clearTimeout(printFitTimer);
      printFitTimer = window.setTimeout(fitGraph, 80);
    };
    window.addEventListener("beforeprint", refitForPrint);
    window.addEventListener("afterprint", refitForPrint);

    return () => {
      observer.disconnect();
      window.clearTimeout(printFitTimer);
      window.removeEventListener("beforeprint", refitForPrint);
      window.removeEventListener("afterprint", refitForPrint);
      if (fitFrame.current !== null) cancelAnimationFrame(fitFrame.current);
    };
  }, [fitGraph, layout.nodes]);

  return <section className="graph-panel" role="region" aria-label={`${visual.kind === "control_flow" ? "Interactive control flow diagram" : "Interactive concept map"}: ${visual.title}`}>
    {visual.nodes.length > 0 ? <div className={`flow-canvas ${theme === "dark" ? "theme-dark" : ""} ${compactLayout ? "compact-layout" : ""}`} ref={flowCanvas} style={compactCanvasHeight ? { height: `${compactCanvasHeight}px` } : undefined}>
      <ReactFlow
        nodes={layout.nodes}
        edges={layout.edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        colorMode={theme === "dark" ? "dark" : "light"}
        minZoom={0.12}
        maxZoom={1.5}
        nodesDraggable
        nodesConnectable={false}
        nodesFocusable
        elementsSelectable
        onInit={(instance) => { flowInstance.current = instance; fitGraph(); }}
        onMoveEnd={(_, viewport) => setShowFitControl((flowCanvas.current?.clientWidth ?? window.innerWidth) < 640 || viewport.zoom < 0.42)}
        onSelectionChange={({ nodes }) => setSelectedNodeId(nodes[0]?.id ?? null)}
        aria-label={visual.kind === "control_flow" ? "Control flow diagram" : "Concept map"}
      >
        <Background color={themeAccents[theme].canvas} gap={22} size={1} />
      </ReactFlow>
      {showFitControl && <button className="diagram-fit-control" type="button" onClick={fitGraph}>Fit diagram</button>}
    </div> : <p className="empty-graph">No visual relationships were generated.</p>}
    {visual.nodes.length > 0 && <details className="diagram-text-alternative">
      <summary>Read {visual.title} diagram as text</summary>
      <ol>{visual.nodes.map((node) => {
        const incomplete = incompleteItems.some((item) => item.source_ranges.some((range) => node.source_ranges.some((nodeRange) => nodeRange.start_line <= range.end_line && nodeRange.end_line >= range.start_line)));
        return <li key={node.id}>{incomplete && <strong>Incomplete implementation. </strong>}<strong>{node.label}.</strong> {node.description}</li>;
      })}</ol>
      {visual.edges.length > 0 && <ul aria-label="Diagram connections">{visual.edges.map((edge, index) => <li key={`${edge.from}-${edge.to}-${index}`}>{visual.nodes.find((node) => node.id === edge.from)?.label ?? edge.from} {edge.label ? `(${edge.label})` : "connects to"} {visual.nodes.find((node) => node.id === edge.to)?.label ?? edge.to}.</li>)}</ul>}
    </details>}
    {selectedNode ? <div className="selected-node-detail" role="status" aria-live="polite" aria-atomic="true" aria-label="Selected diagram item details">
      <div><span className="node-caption">SELECTED {visual.kind === "control_flow" ? "STEP" : "CONCEPT"}</span><h3>{selectedNode.label}</h3><p>{selectedNode.description}</p></div>
      {selectedNode.source_ranges.some((range) => isValidRange(range, lineCount)) ? <div className="selected-source"><span className="node-caption">RELATED SOURCE</span>{selectedNode.source_ranges.filter((range) => isValidRange(range, lineCount)).map((range, index) => <div className="selected-source-range" key={`${range.start_line}-${range.end_line}-${index}`}><button className="source-range" type="button" onClick={() => onSelectRange(range)} aria-label={`Select source lines ${range.start_line} through ${range.end_line}`}>Lines {range.start_line}{range.end_line !== range.start_line ? `–${range.end_line}` : ""} · Jump to source</button><pre>{sourceLines.slice(range.start_line - 1, range.end_line).map((text, lineIndex) => `${range.start_line + lineIndex}  ${text}`).join("\n")}</pre></div>)}</div> : <p className="empty-graph">No valid source lines are linked to this item.</p>}
    </div> : <p className="graph-hint">Select a node to inspect its explanation and related source lines. Drag or swipe to pan; scroll or pinch to zoom. Use Fit diagram to show all nodes.</p>}
    {invalidEdgeCount > 0 && <p className="empty-graph">{invalidEdgeCount} connection{invalidEdgeCount === 1 ? " was" : "s were"} omitted because they were duplicated, cyclic, or referenced missing nodes.</p>}
  </section>;
}

function ExplanationFlowNode({ data, selected }: NodeProps<FlowNode>) {
  const isVertical = data.orientation === "vertical";
  return <div className={`flow-card-node ${selected ? "selected" : ""} ${data.incomplete ? "incomplete" : ""}`}>
    <Handle type="target" position={isVertical ? Position.Top : Position.Left} />
    <span className="node-caption">{data.caption}</span>
    {data.incomplete && <span className="incomplete-node-badge">INCOMPLETE</span>}
    <strong>{data.label}</strong>
    <p>{data.description}</p>
    <Handle type="source" position={isVertical ? Position.Bottom : Position.Right} />
  </div>;
}

function ReadableFlowEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, style, data }: EdgeProps<FlowEdge>) {
  const [path, labelX, labelY] = getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  const label = data?.label;
  return <>
    <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
    {label && <EdgeLabelRenderer><div className="flow-edge-label" aria-hidden="true" title={label} style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}>{label}</div></EdgeLabelRenderer>}
  </>;
}
