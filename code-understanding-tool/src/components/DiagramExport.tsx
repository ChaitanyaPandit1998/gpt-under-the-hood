import { useEffect, useRef, useState } from "react";
import type { Explanation } from "../domain/explanation-types";
import type { ExportContext } from "./DocumentExport";
import { diagramExportLimits, type DiagramExportLayout } from "../domain/diagram-export-layout";

export type DiagramExportContext = ExportContext & { explanation: Explanation; visualIndex: number };

type DiagramExportProps = { source: string; context: DiagramExportContext; onIllustrate?: () => void };

export function DiagramExport(props: DiagramExportProps) {
  return <DiagramExportForm key={JSON.stringify([props.source, props.context])} {...props} />;
}

function DiagramExportForm({ source, context, onIllustrate }: DiagramExportProps) {
  const [open, setOpen] = useState(false);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [preview, setPreview] = useState("");
  const [layout, setLayout] = useState<DiagramExportLayout>("detailed");
  const job = useRef<AbortController | null>(null);
  const objectUrl = useRef("");
  const mounted = useRef(true);
  const visual = context.explanation.visuals[context.visualIndex];
  const limit = diagramExportLimits[layout];
  const supported = visual && visual.nodes.length > 0 && visual.nodes.length <= limit.nodes && visual.edges.length <= limit.edges;
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; job.current?.abort(); if (objectUrl.current) URL.revokeObjectURL(objectUrl.current); };
  }, []);
  useEffect(() => {
    if (!open) return;
    const check = new AbortController();
    void fetch("/api/diagram-export", { signal: check.signal }).then(async (response) => {
      const value = await response.json() as { available?: boolean };
      if (!check.signal.aborted) setAvailable(response.ok && value.available === true);
    }).catch(() => { if (!check.signal.aborted) setAvailable(false); });
    return () => check.abort();
  }, [open]);

  async function generate() {
    if (job.current) return;
    const controller = new AbortController(); job.current = controller;
    setBusy(true); setMessage("");
    try {
      const response = await fetch("/api/diagram-export", { method: "POST", signal: controller.signal, headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version: 1, source, ...context, layout }),
      });
      if (!response.ok) { const value = await response.json() as { error?: string }; throw new Error(value.error || "Diagram export failed."); }
      const blob = await response.blob();
      if (!mounted.current || controller.signal.aborted) return;
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = URL.createObjectURL(blob);
      setPreview(objectUrl.current);
      setMessage("PNG ready. The interactive diagram and explanation are unchanged.");
    } catch (error) {
      if (mounted.current) setMessage(controller.signal.aborted ? "Diagram export cancelled." : error instanceof Error ? error.message : "Diagram export failed.");
    } finally { if (job.current === controller) job.current = null; if (mounted.current) setBusy(false); }
  }

  return <details className="document-export diagram-export" onToggle={(event) => {
    const next = event.currentTarget.open;
    setOpen(next); if (!next) job.current?.abort();
  }}>
    <summary>Export diagram image <small>Existing diagram</small></summary>
    <p>Render this existing diagram as a PNG without model calls. All supplied relationships, source coordinates, and review/coverage notes are retained. Layout is not execution proof.</p>
    <p>No raw source appendix is included, but labels may quote code. Read the full explanation for descriptions and assumptions.</p>
    {layout === "automatic" && <p>Uses branching lanes when possible, otherwise a connection index. Decision labels sit on the arrows; other connection labels appear beneath the lanes.</p>}
    {!supported && <p>This layout supports 1-{limit.nodes} nodes and up to {limit.edges} connections. Try the other layout or read the diagram as text.</p>}
    {available === false && <p>VectorCraft export is unavailable. Configure the approved VectorCraft 0.24.1 executable and enable experimental diagram export on the local server.</p>}
    <div className="chapter-toolbar">
      <label>Diagram layout <select value={layout} disabled={busy} onChange={(event) => {
        if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
        objectUrl.current = "";
        setPreview(""); setMessage("");
        setLayout(event.target.value === "automatic" ? "automatic" : "detailed");
      }}><option value="detailed">Detailed</option><option value="automatic">Automatic branches</option></select></label>
      <button type="button" disabled={!available || !supported || busy} onClick={() => void generate()}>{busy ? "Rendering diagram..." : preview ? "Regenerate diagram PNG" : "Generate diagram PNG"}</button>
      {busy && <button type="button" onClick={() => job.current?.abort()}>Cancel diagram export</button>}
      {preview && <a href={preview} download="chapter-diagram.png">Download diagram PNG</a>}
      {onIllustrate && <button type="button" onClick={onIllustrate}>Create an illustration instead</button>}
    </div>
    <p role="status">{message || (available === null && open ? "Checking VectorCraft..." : "Rendering happens only when requested.")}</p>
    {preview && <figure><div className="diagram-image-scroll" role="region" aria-label="Diagram image preview" tabIndex={0}><img src={preview} style={layout === "automatic" ? { minWidth: 1100 } : undefined} alt={`${visual.title}: exported diagram with chapter citations and coverage notes`} /></div><figcaption>Scroll the preview to read it at a larger size, or download the PNG. Source-linked explanatory diagram, not a runtime trace. PNG is a raster image, not an editable vector.</figcaption></figure>}
  </details>;
}
