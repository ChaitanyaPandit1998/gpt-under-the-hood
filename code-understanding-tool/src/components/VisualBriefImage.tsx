import { useEffect, useRef, useState } from "react";
import type { VisualBrief } from "../domain/visual-brief";
import { visualResolutions, type VisualResolution } from "../domain/visual-resolution";
import { ChoiceCards } from "./ChoiceCards";
import { InlineIllustration } from "./InlineIllustration";
import type { RenderedIllustration } from "../domain/saved-study";
import { validateVisualScene, type VisualScene } from "../domain/visual-scene";

const sizes = (Object.keys(visualResolutions) as VisualResolution[]).map((value) => {
  const { title, width, height } = visualResolutions[value];
  return { value, title, description: `${width} × ${height} px` };
});

export type IllustrationPlacement = { id: string; onRender: (image: RenderedIllustration) => void };

export function VisualBriefImage({ brief, placement, disabled = false }: { brief: VisualBrief; placement?: IllustrationPlacement; disabled?: boolean }) {
  const [preview, setPreview] = useState<RenderedIllustration>();
  const [resolution, setResolution] = useState<VisualResolution>("standard");
  const [phase, setPhase] = useState<"planning" | "rendering" | null>(null);
  const [error, setError] = useState("");
  const pending = useRef<AbortController | null>(null);
  const planned = useRef<VisualScene | null>(null);
  const busy = phase !== null;
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
        placement?.onRender(image);
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
      <button className="visual-primary-action" type="button" disabled={busy || disabled} onClick={() => void render()}>{error ? (planned.current ? "Retry PNG render" : "Retry drawing plan") : preview ? "Render image again" : "Render image"}</button>
      {busy && <button type="button" onClick={() => { pending.current?.abort(); pending.current = null; setPhase(null); }}>Cancel image render</button>}
      {preview && placement && <a href={`#${placement.id}`}>View inline illustration</a>}
    </div>
    {phase && <p role="status">{phase === "planning" ? "Planning the drawing..." : "Rendering the PNG..."}</p>}
    {error && <p role="alert">{error}</p>}
    {preview && placement && <p role="status">Illustration ready in the explanation.</p>}
    {preview && !placement && <InlineIllustration image={preview} />}
  </>;
}
