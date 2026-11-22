import { useEffect, useRef, useState } from "react";
import type { VisualBrief } from "../domain/visual-brief";
import { visualResolutions, type VisualResolution } from "../domain/visual-resolution";
import { ChoiceCards } from "./ChoiceCards";
import { IllustrationComparison, InlineIllustration } from "./InlineIllustration";
import type { RenderedIllustration } from "../domain/saved-study";
import { validateVisualScene, type VisualScene } from "../domain/visual-scene";

const sizes = (Object.keys(visualResolutions) as VisualResolution[]).map((value) => {
  const { title, width, height } = visualResolutions[value];
  return { value, title, description: `${width} × ${height} px` };
});

export type IllustrationPlacement = { id: string; current?: RenderedIllustration; onRender: (image: RenderedIllustration) => void; location?: "scene" };

export function VisualBriefImage({ brief, placement, disabled = false }: { brief: VisualBrief; placement?: IllustrationPlacement; disabled?: boolean }) {
  const [preview, setPreview] = useState<RenderedIllustration>();
  const [resolution, setResolution] = useState<VisualResolution>(() => placement?.current?.resolution ?? "standard");
  const [phase, setPhase] = useState<"planning" | "rendering" | null>(null);
  const [error, setError] = useState("");
  const pending = useRef<AbortController | null>(null);
  const planned = useRef<VisualScene | null>(null);
  const renderButton = useRef<HTMLButtonElement>(null);
  const busy = phase !== null;
  const current = placement?.current;
  const candidate = current && preview !== current ? preview : undefined;
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
        const result = await response.json() as { scene?: unknown; error?: string };
        if (!response.ok) throw new Error(result.error || "Could not plan the drawing.");
        if (pending.current !== controller || controller.signal.aborted) return;
        scene = validateVisualScene(result.scene);
        planned.current = scene;
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
        const image = { png, alt: brief.altText, caption: brief.caption, resolution };
        // Only the first illustration and resizes of the accepted drawing publish immediately.
        if (!current || current === preview) placement?.onRender(image);
        setPreview(image);
      }
    } catch (caught) {
      if (pending.current === controller && !controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Could not render the image.");
    } finally {
      if (pending.current === controller) { pending.current = null; setPhase(null); }
    }
  }

  return <>
    <p>The first render plans the drawing from the brief. Once planned, retries and other sizes reuse that drawing without another model call.</p>
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
  </>;
}
