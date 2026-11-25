import { useEffect, useRef, useState } from "react";
import { visualResolutions } from "../domain/visual-resolution";
import type { RenderedIllustration } from "../domain/saved-study";
import { SavedVisualReviews } from "./SavedVisualReviews";

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

function IllustrationLabels({ labels }: { labels: string[] }) {
  if (!labels.length) return null;
  const list = <ul>{labels.map(text => <li key={text}>{text}</li>)}</ul>;
  return labels.length > 6 ? <details className="illustration-labels">
    <summary>Image labels ({labels.length})</summary>{list}
  </details> : <div className="illustration-labels"><strong>Image labels</strong>{list}</div>;
}

export function InlineIllustration({ image, id, label = "GENERATED ILLUSTRATION", onChange }: { image: RenderedIllustration; id?: string; label?: string; onChange?: (image: RenderedIllustration) => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [actualSize, setActualSize] = useState(false);
  const url = usePngUrl(image.png);
  if (!url) return null;
  const { width, height } = visualResolutions[image.resolution];
  const labels = Array.from(new Set(image.provenance?.scene.items.flatMap(item => item.kind === "text" ? [item.text.trim()] : [])));
  return <figure className="visual-brief-preview inline-illustration" id={id} tabIndex={-1} aria-label={label}>
    <div className="section-caption">{label}</div>
    <button className="illustration-zoom" type="button" aria-label="Enlarge illustration" onClick={() => dialog.current?.showModal()}>
      <img src={url} alt={image.alt} width={width} height={height} />
      <span className="illustration-mobile-hint">Enlarge illustration</span>
    </button>
    <figcaption>
      {image.caption} <small>{width} × {height} px. Review for accuracy before sharing.</small>
      {labels.length > 0 && <div className="illustration-inline-labels"><IllustrationLabels labels={labels} /></div>}
    </figcaption>
    <a href={url} download="explanation-visual.png">Download image</a>
    <SavedVisualReviews image={image} onChange={onChange} />
    <dialog className="illustration-dialog" ref={dialog} aria-label="Enlarged illustration" onClose={() => setActualSize(false)}>
      <form method="dialog">
        <button type="submit">Close preview</button>
        <button type="button" onClick={() => setActualSize(value => !value)}>{actualSize ? "Fit to preview" : "Actual size"}</button>
      </form>
      <div className="illustration-dialog-image" data-actual-size={actualSize}
        role={actualSize ? "region" : undefined} aria-label={actualSize ? "Scrollable full-size illustration" : undefined} tabIndex={actualSize ? 0 : undefined}>
        <img src={url} alt={image.alt} width={width} height={height} />
      </div>
      <p>{image.caption}</p>
      <IllustrationLabels labels={labels} />
    </dialog>
  </figure>;
}
