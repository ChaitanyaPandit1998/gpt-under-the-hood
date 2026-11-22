import { useState } from "react";
import { createPortal } from "react-dom";
import type { ChapterSettings } from "../domain/chapter-settings";
import type { PullRequestChapter, PullRequestPlan } from "../domain/pull-request-plan";
import type { CachedPrResult } from "../domain/saved-study";
import { PdfPreviewShell, PdfResultBody, togglePdfSelection, type StudyPdfImages } from "./StudyPdfExport";

export type PrPdfChapter = { chapter: PullRequestChapter; settings: ChapterSettings; result?: CachedPrResult };

export function PullRequestPdfExport({ plan, available, collect }: {
  plan: PullRequestPlan; available: boolean; collect: () => PrPdfChapter[];
}) {
  const [chapters, setChapters] = useState<PrPdfChapter[]>();
  return <>
    <button type="button" disabled={!available} onClick={() => setChapters(collect())}>PR study PDF preview</button>
    {chapters && createPortal(<PrPdfPreview plan={plan} chapters={chapters} onClose={() => setChapters(undefined)} />, document.body)}
  </>;
}

function resultImages(result?: CachedPrResult): StudyPdfImages {
  return { overview: result?.reading?.illustration, chapters: result?.reading?.chapter?.images };
}

function PrPdfPreview({ plan, chapters, onClose }: { plan: PullRequestPlan; chapters: PrPdfChapter[]; onClose: () => void }) {
  const [selected, setSelected] = useState(() => new Set(chapters.filter(({ result }) => result).map(({ chapter }) => chapter.id)));
  const [excludedImages, setExcludedImages] = useState(new Set<string>());
  const [includeSource, setIncludeSource] = useState(false);
  const unavailable = chapters.filter(({ result }) => !result);
  const omitted = chapters.filter(({ chapter, result }) => result && !selected.has(chapter.id));

  return <PdfPreviewShell onClose={onClose} count={selected.size} total={chapters.length}
    notes="Uses the completed result for each chapter’s selected settings. Other variants, follow-ups, quizzes, and decorative covers are omitted."
    controls={<>
      {chapters.map(({ chapter, result }) => {
        const { id, title } = chapter;
        const images = resultImages(result);
        const hasImages = Boolean(images.overview || Object.keys(images.chapters ?? {}).length);
        return <div className="study-pdf-option" key={id}>
          <label><input type="checkbox" checked={selected.has(id)} disabled={!result}
            onChange={() => setSelected(togglePdfSelection(selected, id))} />{title}</label>
          {!result ? <small className="study-pdf-no-image">No completed result for the selected settings</small> :
            hasImages ? <label className="study-pdf-image-option"><input type="checkbox" disabled={!selected.has(id)}
              checked={selected.has(id) && !excludedImages.has(id)} onChange={() => setExcludedImages(togglePdfSelection(excludedImages, id))} />{title} illustrations</label> :
              <small className="study-pdf-no-image">No inline illustrations</small>}
        </div>;
      })}
      <label><input type="checkbox" checked={includeSource} onChange={(event) => setIncludeSource(event.target.checked)} />Include source appendices for selected chapters</label>
    </>}>
    <div className="study-pdf-kicker">UNDERSTAND / PR STUDY GUIDE</div>
    <h1>{plan.title}</h1>
    <p><strong>Pull request:</strong> <a href={plan.url}>{plan.url}</a></p>
    <p className="study-pdf-provenance"><strong>Pinned head:</strong> {plan.headSha}<br />
      <strong>Pinned base:</strong> {plan.baseSha ?? "Not recorded"}</p>
    <aside className="study-pdf-review">
      <p><strong>Scope:</strong> {selected.size} of {chapters.length} planned chapters selected. This is a pinned snapshot, not a review of the latest PR. Model review is not runtime verification.</p>
      <p><strong>Initial coverage:</strong> {plan.files.length} {plan.totalFiles == null ? "changed files listed; total unknown." : `of ${plan.totalFiles} changed files listed.`}
        {" "}{plan.filesComplete ? "File inventory complete." : "File inventory incomplete."}</p>
      {plan.warnings.length > 0 && <ul>{plan.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul>}
      <p>Citation line numbers are local to each chapter’s imported source, not GitHub file line numbers.</p>
      {!includeSource && <p>Source appendices omitted. Explanation text may still quote the source.</p>}
    </aside>
    {unavailable.length > 0 && <p><strong>No completed result for selected settings:</strong> {unavailable.map(({ chapter }) => chapter.title).join("; ")}.</p>}
    {omitted.length > 0 && <p><strong>Not selected for export:</strong> {omitted.map(({ chapter }) => chapter.title).join("; ")}.</p>}
    {chapters.map(({ chapter, settings, result }, index) => result && selected.has(chapter.id) && <section className="study-pdf-pr-chapter" key={chapter.id}>
      <header>
        <h2>{index + 1}. {chapter.title}</h2>
        <p><strong>{result.explanation.title}</strong></p>
        <p className="study-pdf-provenance">Depth: {settings.depth} · Audience: {settings.learnerLevel} · Tone: {settings.tone} · Focus: {settings.focus}</p>
        {chapter.files.length ? <p className="study-pdf-provenance">Chapter evidence: {chapter.files.join(", ")}</p> :
          <p className="study-pdf-provenance">Inventory only. No implementation patches in this chapter.</p>}
      </header>
      <PdfResultBody explanation={result.explanation} source={chapter.sourceText} embedded includeSource={includeSource}
        images={excludedImages.has(chapter.id) ? {} : resultImages(result)} />
    </section>)}
  </PdfPreviewShell>;
}
