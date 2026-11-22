import { useEffect, useRef, useState } from "react";
import { visualResolutions } from "../domain/visual-resolution";
import type { RenderedIllustration } from "../domain/saved-study";

function usePngUrl(png: Blob) {
  const [preview, setPreview] = useState<{ png: Blob; url: string }>();
  useEffect(() => {
    const url = URL.createObjectURL(png);
    setPreview({ png, url });
    return () => URL.revokeObjectURL(url);
  }, [png]);
  return preview?.png === png ? preview.url : undefined;
}

function ComparisonFigure({ image, label }: { image: RenderedIllustration; label: string }) {
  const url = usePngUrl(image.png);
  const { width, height } = visualResolutions[image.resolution];
  return <figure aria-label={label}>
    <div className="section-caption">{label}</div>
    <div className="illustration-comparison-frame">{url && <img src={url} alt={image.alt} width={width} height={height} />}</div>
    <figcaption>{image.caption} <small>{width} × {height} px</small></figcaption>
  </figure>;
}

export function IllustrationComparison({ current, candidate }: { current: RenderedIllustration; candidate: RenderedIllustration }) {
  return <div className="illustration-comparison" role="group" aria-label="Compare illustrations">
    <ComparisonFigure image={current} label="Current illustration" />
    <ComparisonFigure image={candidate} label="New illustration" />
  </div>;
}

export function InlineIllustration({ image, id }: { image: RenderedIllustration; id?: string }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const url = usePngUrl(image.png);
  if (!url) return null;
  const { width, height } = visualResolutions[image.resolution];
  return <figure className="visual-brief-preview inline-illustration" id={id} tabIndex={-1}>
    <div className="section-caption">GENERATED ILLUSTRATION</div>
    <button className="illustration-zoom" type="button" aria-label="Enlarge illustration" onClick={() => dialog.current?.showModal()}>
      <img src={url} alt={image.alt} width={width} height={height} />
    </button>
    <figcaption>{image.caption} <small>{width} × {height} px. Review for accuracy before sharing.</small></figcaption>
    <a href={url} download="explanation-visual.png">Download image</a>
    <dialog className="illustration-dialog" ref={dialog} aria-label="Enlarged illustration">
      <form method="dialog"><button type="submit">Close preview</button></form>
      <img src={url} alt={image.alt} width={width} height={height} />
      <p>{image.caption}</p>
    </dialog>
  </figure>;
}
