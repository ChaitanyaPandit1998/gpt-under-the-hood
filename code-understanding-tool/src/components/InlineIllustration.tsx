import { useEffect, useRef, useState } from "react";
import { visualResolutions } from "../domain/visual-resolution";
import type { RenderedIllustration } from "../domain/saved-study";

export function InlineIllustration({ image, id }: { image: RenderedIllustration; id?: string }) {
  const [preview, setPreview] = useState<{ png: Blob; url: string }>();
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const url = URL.createObjectURL(image.png);
    setPreview({ png: image.png, url });
    return () => URL.revokeObjectURL(url);
  }, [image.png]);
  const url = preview?.png === image.png ? preview.url : undefined;
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
