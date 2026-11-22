import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { explanationChapters, type ExplanationChapter } from "../domain/explanation-chapters";
import type { Explanation } from "../domain/explanation-types";
import type { RenderedIllustration } from "../domain/saved-study";
import { isValidRange, type SourceRange } from "../domain/source-ranges";
import type { ExportContext } from "./DocumentExport";

export type StudyPdfImages = { overview?: RenderedIllustration; chapters?: Record<string, RenderedIllustration> };
type Props = { explanation: Explanation; source: string; context: ExportContext; collectImages: () => StudyPdfImages; onlyChapterId?: string };

export function StudyPdfExport({ collectImages, ...props }: Props) {
  const [images, setImages] = useState<StudyPdfImages>();
  return <>
    <button type="button" onClick={() => setImages(collectImages())}>{props.onlyChapterId ? "Chapter PDF preview" : "PDF preview"}</button>
    {images && createPortal(<PdfPreview {...props} images={images} onClose={() => setImages(undefined)} />, document.body)}
  </>;
}

export function PdfPreviewShell({ onClose, count, total, notes, controls, children }: {
  onClose: () => void; count: number; total: number; notes: string;
  controls: ReactNode; children: ReactNode;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const preview = useRef<HTMLElement>(null);
  const [printing, setPrinting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => { dialog.current?.showModal(); }, []);

  async function print() {
    if (!count || printing || !preview.current) return;
    const documentPreview = preview.current;
    setPrinting(true);
    setError("");
    try {
      await Promise.all(Array.from(documentPreview.querySelectorAll("img"), (image) => image.decode()));
      await document.fonts.ready;
      if (dialog.current?.open) window.print();
    } catch {
      setError("An illustration could not be prepared. Uncheck it or reopen the preview and try again.");
    } finally { setPrinting(false); }
  }

  return <dialog ref={dialog} className="study-pdf-dialog" aria-label="PDF preview" onClose={onClose}>
    <div className="study-pdf-controls">
      <header>
        <div><div className="section-caption">EXPORT</div><h2>PDF preview</h2></div>
        <form method="dialog"><button type="submit">Close PDF preview</button></form>
      </header>
      <p>Choose what to include, then select <strong>Save as PDF</strong> in your browser’s print dialog. No model calls or uploads are made.</p>
      <p>{notes}</p>
      <fieldset disabled={printing}>
        <legend>Chapters and inline illustrations</legend>
        {controls}
      </fieldset>
      <div className="study-pdf-actions">
        <span role="status">{count ? `${count} of ${total} chapters selected` : "Select at least one chapter to export."}</span>
        <button className="generate-button" type="button" disabled={printing || !count} onClick={() => void print()}>{printing ? "Preparing PDF preview…" : "Print / Save as PDF"}</button>
      </div>
      {error && <p role="alert">{error}</p>}
    </div>
    <article ref={preview} className="study-pdf-paper" aria-label="Selected PDF content">{children}</article>
  </dialog>;
}

export function togglePdfSelection(values: Set<string>, id: string) {
  const next = new Set(values);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
}

export function pdfChapters(explanation: Explanation) {
  return explanationChapters(explanation).map((chapter) =>
    chapter.id === "recap" ? { ...chapter, title: "Assumptions and limits" } : chapter);
}

export function PdfResultBody({ explanation, source, images, includeSource, selected, excludedImages, embedded = false }: {
  explanation: Explanation; source: string; images: StudyPdfImages;
  includeSource: boolean; selected?: Set<string>; excludedImages?: Set<string>; embedded?: boolean;
}) {
  const sourceLines = source.split(/\r\n|\r|\n/);
  const chapters = pdfChapters(explanation);
  const MainHeading = embedded ? "h3" : "h2";
  const SubHeading = embedded ? "h4" : "h3";
  const overviewImage = images.overview && !excludedImages?.has("whole") ? images.overview : undefined;
  function citations(ranges: SourceRange[]) {
    const valid = ranges.filter((range) => isValidRange(range, sourceLines.length));
    return valid.length > 0 && <p className="study-pdf-citations">{embedded ? "Chapter source lines: " : "Source lines: "}{valid.map(({ start_line, end_line }) =>
      start_line === end_line ? String(start_line) : `${start_line}–${end_line}`).join(", ")}.</p>;
  }

  function chapterContent(chapter: ExplanationChapter) {
    if (chapter.id === "overview") return <p>{explanation.summary}</p>;
    if (chapter.id === "concepts") return explanation.concepts.map((item, index) => <div key={index}>
      <SubHeading>{item.term}</SubHeading><p>{item.explanation}</p>{citations(item.source_ranges)}
    </div>);
    if (chapter.id.startsWith("step-")) {
      const step = explanation.walkthrough[Number(chapter.id.slice(5))];
      return <><p>{step.explanation}</p>{citations(step.source_ranges)}</>;
    }
    if (chapter.id === "visuals") return explanation.visuals.length ? explanation.visuals.map((visual, index) => <div key={index}>
      <SubHeading>{visual.title}</SubHeading>
      <ul>{visual.nodes.map((node) => <li key={node.id}><strong>{node.label}:</strong> {node.description}{citations(node.source_ranges)}</li>)}</ul>
      {visual.edges.length > 0 && <ul>{visual.edges.map((edge, edgeIndex) => <li key={edgeIndex}>
        {visual.nodes.find((node) => node.id === edge.from)?.label ?? edge.from} → {visual.nodes.find((node) => node.id === edge.to)?.label ?? edge.to}{edge.label && ` (${edge.label})`}
      </li>)}</ul>}
    </div>) : <p>No technical diagram was provided. See the coverage note above.</p>;
    return explanation.assumptions.length ? <ul>{explanation.assumptions.map((item, index) => <li key={index}>
      <strong>{item.statement}</strong> {item.reason}
    </li>)}</ul> : <p>No additional assumptions were listed.</p>;
  }

  return <>
    <aside className="study-pdf-review">
      <p><strong>Review: {explanation.quality_review_status ?? "not provided"}.</strong> Model review is not runtime verification.</p>
      {explanation.quality_review_note && <p>{explanation.quality_review_note}</p>}
      {explanation.visual_note && <p><strong>Diagram coverage:</strong> {explanation.visual_note}</p>}
      {explanation.incomplete_items?.map((item, index) => <div key={index}>
        <p><strong>Incomplete code: {item.title}.</strong> {item.explanation}</p>{citations(item.source_ranges)}
      </div>)}
      {!includeSource && !embedded && <p>Original source appendix omitted. Explanation text may still quote the source.</p>}
    </aside>
    {!embedded && overviewImage && <PdfIllustration image={overviewImage} />}
    {chapters.filter(({ id }) => !selected || selected.has(id)).map((chapter) => <section key={chapter.id}>
      <MainHeading>{chapter.title}</MainHeading>
      {chapterContent(chapter)}
      {embedded && chapter.id === "overview" && overviewImage && <PdfIllustration image={overviewImage} />}
      {images.chapters?.[chapter.id] && !excludedImages?.has(chapter.id) && <PdfIllustration image={images.chapters[chapter.id]} />}
    </section>)}
    {includeSource && source.trim() && <section><MainHeading>{embedded ? "Chapter source appendix" : "Original source appendix"}</MainHeading>
      <pre>{sourceLines.map((line, index) => `${index + 1} | ${line}`).join("\n")}</pre>
    </section>}
  </>;
}

function PdfPreview({ explanation, source, context, images, onClose, onlyChapterId }: Omit<Props, "collectImages"> & {
  images: StudyPdfImages; onClose: () => void;
}) {
  const chapters = pdfChapters(explanation).filter(({ id }) => !onlyChapterId || id === onlyChapterId);
  const [selected, setSelected] = useState(() => new Set(chapters.map((chapter) => chapter.id)));
  const [excludedImages, setExcludedImages] = useState(new Set<string>());
  const [includeSource, setIncludeSource] = useState(false);
  return <PdfPreviewShell onClose={onClose} count={selected.size} total={chapters.length}
    notes={onlyChapterId ? "Only this chapter is included, with review and coverage notes. Follow-ups, quizzes, and decorative covers are omitted." : "Only the current explanation is included. Follow-ups, quizzes, and decorative covers are omitted."}
    controls={<>
        {chapters.map(({ id, title }) => {
          const chapterImage = images.chapters?.[id];
          return <div className="study-pdf-option" key={id}>
            <label><input type="checkbox" checked={selected.has(id)} onChange={() => setSelected(togglePdfSelection(selected, id))} />{title}</label>
            {chapterImage && <label className="study-pdf-image-option"><input type="checkbox" disabled={!selected.has(id)}
              checked={selected.has(id) && !excludedImages.has(id)} onChange={() => setExcludedImages(togglePdfSelection(excludedImages, id))} />{title} illustration</label>}
            {!chapterImage && <small className="study-pdf-no-image">No inline illustration</small>}
          </div>;
        })}
        {images.overview && <label><input type="checkbox" checked={!excludedImages.has("whole")}
          onChange={() => setExcludedImages(togglePdfSelection(excludedImages, "whole"))} />Whole-explanation illustration</label>}
        {!onlyChapterId && <label><input type="checkbox" checked={includeSource} disabled={!source.trim()}
          onChange={(event) => setIncludeSource(event.target.checked)} />Include original source appendix</label>}
    </>}>
      <div className="study-pdf-kicker">UNDERSTAND / STUDY GUIDE</div>
      <h1>{explanation.title}</h1>
      <p><strong>Scope:</strong> {context.scope}</p>
      {context.provenance && <p className="study-pdf-provenance">{context.provenance}</p>}
      <PdfResultBody explanation={explanation} source={source} images={images} includeSource={includeSource}
        selected={selected} excludedImages={excludedImages} />
  </PdfPreviewShell>;
}

function PdfIllustration({ image }: { image: RenderedIllustration }) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    const next = URL.createObjectURL(image.png);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [image.png]);
  return <figure>
    {url && <img src={url} alt={image.alt} />}
    <figcaption>{image.caption} <span>Generated illustration. Review for accuracy.</span></figcaption>
  </figure>;
}
