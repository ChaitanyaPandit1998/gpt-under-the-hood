import { useEffect, useRef, useState } from "react";
import type { VisualBrief } from "../domain/visual-brief";
import { visualResolutions, type VisualResolution } from "../domain/visual-resolution";
import { ChoiceCards } from "./ChoiceCards";
import { IllustrationComparison, InlineIllustration } from "./InlineIllustration";
import type { RenderedIllustration } from "../domain/saved-study";
import { validateVisualScene, type VisualScene } from "../domain/visual-scene";
import { checkVisualLabels } from "../domain/visual-label-layout";
import { VisualReviewPanel, type VisualReviewContext } from "./VisualReviewPanel";
import { appendVisualReview, validateGenerationStage, type GenerationStage } from "../domain/visual-provenance";

const sizes = (Object.keys(visualResolutions) as VisualResolution[]).map((value) => {
  const { title, width, height } = visualResolutions[value];
  return { value, title, description: `${width} × ${height} px` };
});

export type IllustrationPlacement = {
  id: string;
  current?: RenderedIllustration;
  onRender: (image: RenderedIllustration) => void;
  onReview: (image: RenderedIllustration) => void;
  location?: "scene";
};

export function VisualBriefImage({ brief, placement, disabled = false, reviewContext, briefGeneration }: { brief: VisualBrief; placement?: IllustrationPlacement; disabled?: boolean; reviewContext?: VisualReviewContext; briefGeneration?: GenerationStage }) {
  const [preview, setPreview] = useState<RenderedIllustration>();
  const [previewScene, setPreviewScene] = useState<VisualScene>();
  const [renderRevision, setRenderRevision] = useState(0);
  const [resolution, setResolution] = useState<VisualResolution>(() => placement?.current?.resolution ?? "standard");
  const [phase, setPhase] = useState<"planning" | "rendering" | null>(null);
  const [error, setError] = useState("");
  const pending = useRef<AbortController | null>(null);
  const planned = useRef<VisualScene | null>(null);
  const plannedGeneration = useRef<GenerationStage | undefined>(undefined);
  const renderButton = useRef<HTMLButtonElement>(null);
  const busy = phase !== null;
  const current = placement?.current;
  const candidate = current && preview?.png !== current.png ? preview : undefined;
  useEffect(() => () => {
    pending.current?.abort();
  }, []);

  async function render() {
    if (pending.current || disabled) return;
    const controller = new AbortController();
    pending.current = controller;
    setError("");
    try {
      let scene = planned.current;
      if (!scene) {
        setPhase("planning");
        const response = await fetch("/api/visual-plan", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(brief), signal: controller.signal });
        const result = await response.json() as { scene?: unknown; error?: string; generation?: unknown };
        if (!response.ok) throw new Error(result.error || "Could not plan the drawing.");
        if (pending.current !== controller || controller.signal.aborted) return;
        scene = validateVisualScene(result.scene);
        const lineage = result.generation === undefined ? undefined : await validateGenerationStage(result.generation, JSON.stringify(scene));
        const canvas = document.createElement("canvas").getContext("2d");
        if (!canvas) throw new Error("Text layout checking is unavailable in this browser.");
        await document.fonts.ready;
        const issues = checkVisualLabels(scene, (item) => {
          canvas.font = `${item.size}px "Source Sans 3", sans-serif`;
          canvas.textAlign = item.align;
          const m = canvas.measureText(item.text);
          return { left: m.actualBoundingBoxLeft, right: m.actualBoundingBoxRight,
            ascent: m.actualBoundingBoxAscent, descent: m.actualBoundingBoxDescent };
        });
        if (issues.length) throw new Error(`Label layout check failed: ${issues.slice(0, 3).join(" ")} Retry for a new plan or simplify the brief.`);
        if (pending.current !== controller || controller.signal.aborted) return;
        planned.current = scene;
        plannedGeneration.current = lineage;
      }
      setPhase("rendering");
      const response = await fetch("/api/visual-render", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene, resolution }), signal: controller.signal });
      if (!response.ok) {
        const result = await response.json() as { error?: string };
        throw new Error(result.error || "Could not render the image.");
      }
      const png = await response.blob();
      if (png.type !== "image/png" || !png.size) throw new Error("The renderer did not return a PNG.");
      if (pending.current === controller && !controller.signal.aborted) {
        const image: RenderedIllustration = { png, alt: brief.altText, caption: brief.caption, resolution,
          ...(reviewContext ? { provenance: { version: 1, brief, scene, reviews: [],
            ...(briefGeneration || plannedGeneration.current ? { generation: {
              ...(briefGeneration ? { brief: briefGeneration } : {}), ...(plannedGeneration.current ? { plan: plannedGeneration.current } : {}),
            } } : {}), context: {
            explanation: reviewContext.explanation, scope: reviewContext.scope, audience: reviewContext.audience,
            ...(reviewContext.purpose ? { purpose: reviewContext.purpose } : {}),
          } } } : {}),
        };
        // Only the first illustration and resizes of the accepted drawing publish immediately.
        if (!current || current.png === preview?.png) placement?.onRender(image);
        setPreview(image);
        setPreviewScene(scene);
        setRenderRevision((revision) => revision + 1);
      }
    } catch (caught) {
      if (pending.current === controller && !controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Could not render the image.");
    } finally {
      if (pending.current === controller) { pending.current = null; setPhase(null); }
    }
  }

  return <div className="visual-image-renderer" data-tool-active={busy} data-tool-decision={!!candidate}>
    <details className="visual-builder-options">
      <summary>Rendering details</summary>
      <p>The first render plans the drawing; retries and other sizes reuse that plan without another model call.</p>
      <p>Preflight checks label clipping and overlap, not factual correctness. Font substitution and shapes can affect readability; inspect the PNG. Captions appear below it.</p>
    </details>
    <ChoiceCards label="Image resolution" choices={sizes} value={resolution} onChange={setResolution} disabled={busy || disabled} />
    <div className="chapter-toolbar">
      <button ref={renderButton} className="visual-primary-action" type="button" disabled={busy || disabled} onClick={() => void render()}>{error ? (planned.current ? "Retry PNG render" : "Retry drawing plan") : preview ? "Render image again" : "Render image"}</button>
      {busy && <button type="button" onClick={() => { pending.current?.abort(); pending.current = null; setPhase(null); }}>Cancel image render</button>}
      {current && placement && <a href={`#${placement.id}`}>View inline illustration</a>}
    </div>
    {phase && <p role="status">{phase === "planning" ? "Planning the drawing..." : "Rendering the PNG..."}</p>}
    {error && <p role="alert">{error}</p>}
    {candidate && current && placement && <>
      <p role="status">New version ready to compare. {placement.location === "scene" ? "The current image stays in the scene until you choose." : "The current illustration stays in the explanation and PDF until you choose."}</p>
      <IllustrationComparison current={current} candidate={candidate} />
      <div className="chapter-toolbar">
        <button type="button" disabled={busy || disabled} onClick={() => { setPreview(undefined); setError(""); renderButton.current?.focus(); }}>Keep current illustration</button>
        <button className="visual-primary-action" type="button" disabled={busy || disabled} onClick={() => { placement.onRender(candidate); setError(""); renderButton.current?.focus(); }}>Use new illustration</button>
      </div>
    </>}
    {preview && placement && !candidate && <p role="status">{placement.location === "scene" ? "Image ready in the scene." : "Illustration ready in the explanation."}</p>}
    {preview && !placement && <InlineIllustration image={preview} />}
    {preview && previewScene && reviewContext && <VisualReviewPanel key={renderRevision} brief={brief} scene={previewScene} image={preview} context={reviewContext} disabled={busy || disabled}
      onComplete={(review) => {
        if (!preview.provenance) return;
        const updated = { ...preview, provenance: appendVisualReview(preview.provenance, review) };
        setPreview(updated);
        placement?.onReview(updated);
      }} />}
  </div>;
}
