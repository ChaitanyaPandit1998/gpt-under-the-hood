import { useEffect, useRef, useState } from "react";
import type { Explanation } from "../domain/explanation-types";
import { chapterVariantKey, defaultChapterSettings, type ChapterSettings } from "../domain/chapter-settings";
import { ChapterControls } from "./ChapterControls";
import type { PlannedFile, PullRequestChapter, PullRequestEvidence, PullRequestPlan } from "../domain/pull-request-plan";
import type { ThemeMode } from "../domain/theme";
import { sourceOffsetsForRange } from "../domain/source-ranges";
import { generationPhaseCopy } from "../domain/generation-progress";
import { useExplanationSession } from "../hooks/useExplanationSession";
import { ExplanationResult } from "./ExplanationResult";
import { ChapterIllustration } from "./ChapterIllustration";
import type { ChapterArtwork } from "../domain/chapter-illustrations";

export function PullRequestStudy({ plan, theme, onClose }: { plan: PullRequestPlan; theme: ThemeMode; onClose: () => void }) {
  const [index, setIndex] = useState(0);
  const [chapterSettings, setChapterSettings] = useState<Record<number, ChapterSettings>>({});
  const [extraChapters, setExtraChapters] = useState<PullRequestChapter[]>([]);
  const [evidenceProgress, setEvidenceProgress] = useState<Record<string, { nextOffset: number | null; sections: number; totalCharacters: number; loadedCharacters: number }>>({});
  const [loadingFile, setLoadingFile] = useState<string | null>(null);
  const [evidenceError, setEvidenceError] = useState("");
  const [covers, setCovers] = useState<Record<string, ChapterArtwork>>({});
  const evidenceController = useRef<AbortController | null>(null);
  const evidenceRequestId = useRef(0);
  const cache = useRef(new Map<string, Explanation>());
  const editor = useRef<HTMLTextAreaElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const chapters = [...plan.chapters, ...extraChapters];
  const chapter = chapters[index];
  const source = chapter.sourceText;
  const settingsFor = (chapterIndex: number) => chapterSettings[chapterIndex] ?? { ...defaultChapterSettings, depth: "detailed" as const };
  const settings = settingsFor(index);
  const session = useExplanationSession({ source, mode: "pull_request", ...settings });
  const cacheKey = (chapterIndex: number, preferences: ChapterSettings) => `${chapterIndex}:${chapterVariantKey(preferences)}`;

  useEffect(() => () => {
    evidenceRequestId.current += 1;
    evidenceController.current?.abort();
  }, []);

  function cancelEvidence() {
    evidenceRequestId.current += 1;
    evidenceController.current?.abort();
    evidenceController.current = null;
    setLoadingFile(null);
    setEvidenceError("");
  }

  function selectChapter(next: number, nextSettings = settingsFor(next)) {
    cancelEvidence();
    if (session.explanation) cache.current.set(cacheKey(index, settings), session.explanation);
    session.resetExplanation();
    session.setExplanation(cache.current.get(cacheKey(next, nextSettings)) ?? null);
    setChapterSettings((current) => ({ ...current, [next]: nextSettings }));
    setIndex(next);
    if (next !== index) heading.current?.focus();
  }

  async function loadEvidence(file: PlannedFile) {
    if (loadingFile || evidenceProgress[file.path]?.nextOffset === null) return;
    if (extraChapters.length >= 100) { setEvidenceError("This session has reached its 100-section limit. Start a new study to load more sections."); return; }
    session.invalidatePendingRequests();
    const requestId = ++evidenceRequestId.current;
    const controller = new AbortController();
    evidenceController.current = controller;
    setLoadingFile(file.path); setEvidenceError("");
    const offset = evidenceProgress[file.path]?.nextOffset ?? 0;
    try {
      const response = await fetch("/api/pull-request-evidence", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ url: plan.url, headSha: plan.headSha, baseSha: plan.baseSha, filename: file.path, page: file.page, offset }),
      });
      const payload = await response.json() as PullRequestEvidence & { error?: string };
      if (requestId !== evidenceRequestId.current) return;
      if (!response.ok || !payload.chapter || payload.filename !== file.path || payload.offset !== offset) throw new Error(payload.error || "The patch section response was invalid.");
      setExtraChapters((current) => [...current, payload.chapter]);
      setEvidenceProgress((current) => ({ ...current, [file.path]: { nextOffset: payload.nextOffset, totalCharacters: payload.totalCharacters, loadedCharacters: payload.endOffset, sections: (current[file.path]?.sections ?? 0) + 1 } }));
      selectChapter(chapters.length);
    } catch (caught) {
      if (requestId === evidenceRequestId.current && !controller.signal.aborted) setEvidenceError(caught instanceof Error ? caught.message : "Could not load this patch section.");
    } finally {
      if (requestId === evidenceRequestId.current) { evidenceController.current = null; setLoadingFile(null); }
    }
  }

  async function copy() {
    if (!session.explanation) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify({ chapter: chapter.title, head: plan.headSha, preferences: settings, source, explanation: session.explanation }, null, 2));
      session.setCopyStatus("Copied chapter with source");
    } catch { session.setCopyStatus("Clipboard unavailable"); }
  }

  return <section className="pr-study" aria-label="Pull request study plan">
    <header className="pr-study-header"><div><div className="section-caption">COVERAGE-AWARE READING PLAN</div><h2>{plan.title}</h2><a href={plan.url} target="_blank" rel="noreferrer">View PR on GitHub</a></div><button type="button" onClick={onClose}>Back to input</button></header>
    <p>{plan.files.length} / {plan.totalFiles ?? "unknown total"} files listed · {plan.files.filter((file) => file.coverage === "included").length} patches included without local trimming in the initial import · {chapters.length} chapters · {plan.filesComplete ? "File inventory complete" : "File inventory incomplete"}</p>
    <p className="pr-revision">Snapshot: {plan.headSha}. Chapters are generated only when requested. Results stay in this page session; no repository-wide analysis is implied.</p>
    <p aria-live="polite">{chapters.filter((_, itemIndex) => itemIndex === index ? !!session.explanation : cache.current.has(cacheKey(itemIndex, settingsFor(itemIndex)))).length} / {chapters.length} chapters generated for their selected preferences.</p>
    <aside className="incomplete-code-note" aria-label="PR coverage limits"><strong>Initial import coverage limits</strong><ul>{plan.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul><p>Load additional sections below for excluded or excerpted files. Retrieved sections remain separate evidence chapters; no chapter automatically sees all other sections.</p></aside>
    <details className="pr-manifest"><summary>File manifest and evidence coverage</summary><ul>{plan.files.map((file) => <li key={file.path}><code>{file.path}</code> · {file.status} · +{file.additions}/-{file.deletions} · <strong>Initial: {file.coverage.replaceAll("_", " ")}</strong>
      {evidenceProgress[file.path] && <p role="status">{evidenceProgress[file.path].sections} section(s) retrieved · {evidenceProgress[file.path].loadedCharacters} / {evidenceProgress[file.path].totalCharacters} patch characters · {evidenceProgress[file.path].nextOffset === null ? "All GitHub-returned patch sections retrieved" : "More sections remain"}</p>}
      {file.coverage !== "included" && file.page && <button type="button" disabled={!!loadingFile || evidenceProgress[file.path]?.nextOffset === null || extraChapters.length >= 100} onClick={() => void loadEvidence(file)} aria-label={`Load patch section for ${file.path}`}>{loadingFile === file.path ? "Loading patch…" : evidenceProgress[file.path]?.nextOffset === null ? "Patch sections loaded" : evidenceProgress[file.path] ? "Load next patch section" : "Load patch section"}</button>}
    </li>)}</ul></details>
    {loadingFile && <div className="chapter-toolbar"><span role="status">Loading patch section for {loadingFile}</span><button type="button" onClick={cancelEvidence}>Cancel patch loading</button></div>}
    {evidenceError && <p role="alert">{evidenceError}</p>}
    <div className="chapter-layout">
      <nav className="chapter-sidebar" aria-label="PR chapter outline"><ol>{chapters.map((item, itemIndex) => <li key={item.id}><button type="button" aria-current={index === itemIndex ? "step" : undefined} onClick={() => selectChapter(itemIndex)}><span>{itemIndex + 1}</span>{item.title}</button></li>)}</ol></nav>
      <div className="chapter-body">
        <h3 ref={heading} tabIndex={-1}>{chapter.title}</h3>
        <ChapterIllustration key={chapter.id} cached={covers[chapter.id]} onChange={(artwork) => setCovers((current) => {
          const next = { ...current };
          if (artwork) next[chapter.id] = artwork;
          else delete next[chapter.id];
          return next;
        })} />
        <p>{chapter.files.length ? `${chapter.files.length} file patch(es). Citations refer to the chapter source below.` : "Inventory-only overview: implementation behavior must be explained in the evidence chapters."}</p>
        <details><summary>Review chapter source</summary><textarea ref={editor} readOnly aria-label="Chapter source" value={source} rows={10} /></details>
        <ChapterControls settings={settings} onChange={(next) => selectChapter(index, next)} disabled={!!loadingFile} />
        <div className="chapter-toolbar">
          <button type="button" disabled={session.isGenerating || !!loadingFile} onClick={() => { cache.current.delete(cacheKey(index, settings)); void session.explainSource(); }}>{session.isGenerating ? "Generating chapter…" : session.explanation ? "Regenerate chapter" : "Generate this chapter"}</button>
          {session.isGenerating && <button type="button" onClick={session.resetExplanation}>Cancel</button>}
        </div>
        {session.generationPhase && session.isGenerating && <p role="status">{generationPhaseCopy[session.generationPhase].detail}</p>}
        {session.error && <p role="alert">{session.error}</p>}
        {session.explanation && <ExplanationResult allowChapterArtwork={false} explanation={session.explanation} source={source} theme={theme} lineCount={source.split("\n").length}
          isAskingFollowUp={session.isAskingFollowUp} copyStatus={session.copyStatus} followUpQuestion={session.followUpQuestion} selectedTextForQuestion="" followUpMessages={session.followUpMessages} followUpError={session.followUpError} textSelection={null}
          onCopy={() => void copy()} onSelectRange={(range) => {
            const element = editor.current;
            const offsets = sourceOffsetsForRange(source, range);
            if (!element || !offsets) return;
            const details = element.closest("details");
            if (details) details.open = true;
            element.focus(); element.setSelectionRange(offsets.start, offsets.end); element.scrollIntoView({ block: "center" });
          }} onFollowUpQuestionChange={session.setFollowUpQuestion} onAskFollowUp={session.askFollowUp} onClearSelectedTextContext={() => session.setSelectedTextForQuestion("")} onAskAboutSelectedText={() => {}} />}
        <nav className="chapter-pagination" aria-label="PR chapter navigation"><button type="button" disabled={index === 0} onClick={() => selectChapter(index - 1)}>Previous chapter</button><span>{index + 1} / {chapters.length}</span><button type="button" disabled={index === chapters.length - 1} onClick={() => selectChapter(index + 1)}>Next chapter</button></nav>
      </div>
    </div>
  </section>;
}
