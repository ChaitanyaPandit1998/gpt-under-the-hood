import { useEffect, useRef, useState } from "react";
import type { Explanation } from "../domain/explanation-types";

export type ExportContext = { scope: string; provenance: string };

export function DocumentExport(props: { explanation: Explanation; source: string; context: ExportContext }) {
  return <ExportForm key={JSON.stringify([props.explanation, props.source, props.context])} {...props} />;
}

function ExportForm({ explanation, source, context }: { explanation: Explanation; source: string; context: ExportContext }) {
  const [available, setAvailable] = useState<boolean | null>(null);
  const [format, setFormat] = useState("docx");
  const [diagramImages, setDiagramImages] = useState(false);
  const [diagramIndex, setDiagramIndex] = useState("");
  const [includeSource, setIncludeSource] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const controller = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  const hasDiagram = explanation.visuals.some((visual) => visual.nodes.length > 0);
  useEffect(() => {
    mounted.current = true;
    const check = new AbortController();
    void fetch("/api/document-export", { signal: check.signal }).then(async (response) => {
      const data = await response.json() as { available?: boolean; diagramImages?: boolean };
      if (!check.signal.aborted) {
        setAvailable(response.ok && data.available === true);
        setDiagramImages(response.ok && data.diagramImages === true);
      }
    }).catch(() => { if (!check.signal.aborted) setAvailable(false); });
    return () => { mounted.current = false; check.abort(); controller.current?.abort(); };
  }, []);

  async function download() {
    if (controller.current) return;
    const current = new AbortController();
    controller.current = current;
    setBusy(true); setMessage("");
    try {
      const response = await fetch("/api/document-export", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: current.signal,
        body: JSON.stringify({ version: 1, explanation, source, ...context, includeSource, format,
          ...(format === "pdf" && diagramImages && diagramIndex !== "" ? { diagramIndex: Number(diagramIndex) } : {}) }),
      });
      if (!response.ok) {
        const error = await response.json() as { error?: string };
        throw new Error(error.error || "Export failed.");
      }
      const blob = await response.blob();
      if (current.signal.aborted || !mounted.current) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url; link.download = `study-guide.${format}`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 30_000);
      setMessage("Download ready. Your explanation is unchanged.");
    } catch (error) {
      if (mounted.current) setMessage(current.signal.aborted ? "Export cancelled." : error instanceof Error ? error.message : "Export failed.");
    } finally {
      if (controller.current === current) controller.current = null;
      if (mounted.current) setBusy(false);
    }
  }

  return <details className="document-export" data-tool-active={busy}>
    <summary>Export document <small>Experimental</small> <small className="tool-active-indicator">Preparing</small></summary>
    <p><strong>Scope:</strong> {context.scope}. Other chapters, variants, follow-ups, and conceptual artwork are excluded. Diagram text and citations are always included.</p>
    <p>Uses this existing explanation locally without model calls. Review and coverage warnings are retained; export does not verify code behaviour.</p>
    {!hasDiagram && <p>This explanation has no diagram to embed. A diagram note is text only.</p>}
    {hasDiagram && <p>For an illustrated PDF, use this panel rather than your browser's Print command.</p>}
    {available === false && <p>Export is unavailable. Set ENABLE_EXPERIMENTAL_DOCUMENT_EXPORT=1 and WORDCRAFT_CLI_PATH to the approved WordCraft 0.18.1 executable in the server environment, then restart.</p>}
    <div className="chapter-toolbar">
      <label>Document format <select value={format} disabled={busy} onChange={(event) => setFormat(event.target.value)}><option value="docx">Word (.docx)</option><option value="pdf">PDF</option></select></label>
      {format === "pdf" && explanation.visuals.length > 0 && <label>Diagram image <select value={diagramIndex} disabled={busy || !diagramImages} onChange={(event) => setDiagramIndex(event.target.value)}>
        <option value="">No image (text only)</option>
        {explanation.visuals.map((visual, index) => {
          const supported = visual.nodes.length > 0 && visual.nodes.length <= 8 && visual.edges.length <= 12;
          return <option key={index} value={index} disabled={!supported}>{visual.title}{supported ? "" : " (unsupported size)"}</option>;
        })}
      </select></label>}
      <label><input type="checkbox" checked={includeSource} disabled={busy} onChange={(event) => setIncludeSource(event.target.checked)} /> Include source text in this download</label>
      <button type="button" disabled={!available || busy} onClick={() => void download()}>{busy ? "Preparing document..." : "Download document"}</button>
      {busy && <button type="button" onClick={() => controller.current?.abort()}>Cancel export</button>}
    </div>
    {format === "pdf" && explanation.visuals.length > 0 && available && !diagramImages && <p>VectorCraft is unavailable. You can still download a text-only PDF.</p>}
    <p role="status">{message || (available === null ? "Checking local exporter..." : "The source appendix is omitted by default. Explanation text may still quote code.")}</p>
  </details>;
}
