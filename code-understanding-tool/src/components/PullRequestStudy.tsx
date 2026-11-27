import { useEffect, useId, useRef, useState } from "react";
import { chapterVariantKey, defaultChapterSettings, type ChapterSettings } from "../domain/chapter-settings";
import { ChapterControls } from "./ChapterControls";
import { ChapterOutline } from "./ChapterOutline";
import type { PlannedFile, PullRequestChapter, PullRequestEvidence, PullRequestPlan } from "../domain/pull-request-plan";
import type { ThemeMode } from "../domain/theme";
import { generationPhaseCopy } from "../domain/generation-progress";
import { useExplanationSession } from "../hooks/useExplanationSession";
import { ExplanationResult, type ExplanationResultHandle } from "./ExplanationResult";
import { ChapterIllustration } from "./ChapterIllustration";
import type { ChapterArtwork } from "../domain/chapter-illustrations";
import type { CachedPrResult, PrReadingState } from "../domain/saved-study";
import { PullRequestPdfExport, type PrPdfChapter } from "./PullRequestPdfExport";

const initialChapterSettings: ChapterSettings = { ...defaultChapterSettings, depth: "detailed" };
const automaticReviewKey = "understand-pr-automatic-review";

export function PullRequestStudy({ plan, theme, onClose, initialReading, onSave, saving }: {
  plan: PullRequestPlan; theme: ThemeMode; onClose: () => void;
  initialReading?: PrReadingState; onSave: (reading: PrReadingState) => void; saving: boolean;
}) {
  const [index, setIndex] = useState(() => Math.max(0, [...plan.chapters, ...(initialReading?.extraChapters ?? [])].findIndex((chapter) => chapter.id === initialReading?.chapterId)));
  const [chapterSettings, setChapterSettings] = useState<Record<number, ChapterSettings>>(() => initialReading?.settings ?? {});
  const [extraChapters, setExtraChapters] = useState<PullRequestChapter[]>(() => initialReading?.extraChapters ?? []);
  const [evidenceProgress, setEvidenceProgress] = useState<PrReadingState["evidenceProgress"]>(() => initialReading?.evidenceProgress ?? {});
  const [loadingFile, setLoadingFile] = useState<string | null>(null);
  const [evidenceError, setEvidenceError] = useState("");
  const [automaticReview, setAutomaticReview] = useState(() => {
    try { return localStorage.getItem(automaticReviewKey) === "true"; } catch { return false; }
  });
  const [covers, setCovers] = useState<Record<string, ChapterArtwork>>(() => initialReading?.covers ?? {});
  const evidenceController = useRef<AbortController | null>(null);
  const evidenceRequestId = useRef(0);
  const [cache] = useState(() => new Map<string, CachedPrResult>(Object.entries(initialReading?.results ?? {})));
  const resultReader = useRef<ExplanationResultHandle>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const refinementTarget = useId();
  const diagramNote = useId();
  const chapters = [...plan.chapters, ...extraChapters];
  const chapter = chapters[index];
  const source = chapter.sourceText;
  const cacheKey = (chapterIndex: number, preferences: ChapterSettings) => `${chapterIndex}:${chapterVariantKey(preferences)}`;
  const settingsFor = (chapterIndex: number): ChapterSettings => {
    const saved = chapterSettings[chapterIndex];
    if (saved) return saved;
    // Older studies could save a generated chapter without saving its default settings.
    if (cache.has(cacheKey(chapterIndex, initialChapterSettings))) return initialChapterSettings;
    return { ...initialChapterSettings, visualPreference: "no" };
  };
  const settings = settingsFor(index);
  const activeKey = cacheKey(index, settings);
  const savedResult = cache.get(activeKey);
  const session = useExplanationSession({ source, mode: "pull_request", ...settings }, savedResult?.explanation);
  const completedCount = chapters.filter((_, itemIndex) =>
    itemIndex === index ? !!session.explanation : cache.has(cacheKey(itemIndex, settingsFor(itemIndex)))).length;

  function updateAutomaticReview(enabled: boolean) {
    setAutomaticReview(enabled);
    try { localStorage.setItem(automaticReviewKey, String(enabled)); } catch { /* Keep the preference for this session. */ }
  }

  function collectPdfChapters(): PrPdfChapter[] {
    return chapters.map((item, itemIndex) => {
      const preferences = settingsFor(itemIndex);
      const result = itemIndex === index
        ? session.explanation ? { explanation: session.explanation, reading: resultReader.current?.snapshot() } : undefined
        : cache.get(cacheKey(itemIndex, preferences));
      return { chapter: item, settings: preferences, result };
    });
  }

  function cacheActiveResult() {
    if (session.explanation) cache.set(activeKey, { explanation: session.explanation, reading: resultReader.current?.snapshot() });
  }

  function save() {
    cacheActiveResult();
    onSave({ chapterId: chapter.id, settings: chapterSettings, extraChapters, evidenceProgress, results: Object.fromEntries(cache), covers });
  }

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
    cacheActiveResult();
    session.resetExplanation();
    session.setExplanation(cache.get(cacheKey(next, nextSettings))?.explanation ?? null);
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
    <p className="pr-revision">Snapshot: {plan.headSha}. Chapters are generated only when requested. A saved study restores this snapshot, not the latest PR revision.</p>
    <div className="chapter-toolbar"><button type="button" disabled={saving || !!loadingFile || session.isGenerating} aria-describedby="pr-study-save-note" onClick={save}>Save PR study locally</button>
      <PullRequestPdfExport plan={plan} available={completedCount > 0} collect={collectPdfChapters} />
      <span id="pr-study-save-note">Stores the imported patches, completed explanations and stories, and accepted illustrations in this browser only when you save.</span></div>
    <p aria-live="polite">{completedCount} / {chapters.length} chapters generated for their selected preferences.</p>
    <aside className="incomplete-code-note" aria-label="PR coverage limits"><strong>Initial import coverage limits</strong><ul>{plan.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul><p>Load additional sections below for excluded or excerpted files. Retrieved sections remain separate evidence chapters; no chapter automatically sees all other sections.</p></aside>
    <details className="pr-manifest"><summary>File manifest and evidence coverage</summary><ul>{plan.files.map((file) => <li key={file.path}><code>{file.path}</code> · {file.status} · +{file.additions}/-{file.deletions} · <strong>Initial: {file.coverage.replaceAll("_", " ")}</strong>
      {evidenceProgress[file.path] && <p role="status">{evidenceProgress[file.path].sections} section(s) retrieved · {evidenceProgress[file.path].loadedCharacters} / {evidenceProgress[file.path].totalCharacters} patch characters · {evidenceProgress[file.path].nextOffset === null ? "All GitHub-returned patch sections retrieved" : "More sections remain"}</p>}
      {file.coverage !== "included" && file.page && <button type="button" disabled={!!loadingFile || evidenceProgress[file.path]?.nextOffset === null || extraChapters.length >= 100} onClick={() => void loadEvidence(file)} aria-label={`Load patch section for ${file.path}`}>{loadingFile === file.path ? "Loading patch…" : evidenceProgress[file.path]?.nextOffset === null ? "Patch sections loaded" : evidenceProgress[file.path] ? "Load next patch section" : "Load patch section"}</button>}
    </li>)}</ul></details>
    {loadingFile && <div className="chapter-toolbar"><span role="status">Loading patch section for {loadingFile}</span><button type="button" onClick={cancelEvidence}>Cancel patch loading</button></div>}
    {evidenceError && <p role="alert">{evidenceError}</p>}
    <div className="chapter-layout">
      <ChapterOutline label="PR chapter outline" chapters={chapters} index={index} onSelect={selectChapter} />
      <div className="chapter-body">
        <h3 ref={heading} tabIndex={-1}>{chapter.title}</h3>
        <ChapterIllustration key={chapter.id} cached={covers[chapter.id]} onChange={(artwork) => setCovers((current) => {
          const next = { ...current };
          if (artwork) next[chapter.id] = artwork;
          else delete next[chapter.id];
          return next;
        })} />
        <p>{chapter.files.length ? `${chapter.files.length} file patch(es). Citations refer to the chapter source below.` : "Inventory-only overview: implementation behavior must be explained in the evidence chapters."}</p>
        <details><summary>Review chapter source</summary><textarea readOnly aria-label="Chapter source" value={source} rows={10} /></details>
        <div id={refinementTarget} className="refinement-target" tabIndex={-1} role="group" aria-label="PR chapter preferences">
          <ChapterControls settings={settings} onChange={(next) => selectChapter(index, next)} disabled={!!loadingFile} />
        </div>
        <div className="chapter-toolbar">
          <button type="button" disabled={session.isGenerating || !!loadingFile} onClick={() => void session.explainSource(automaticReview ? "draft-and-review" : "draft")}>{session.isReviewing ? "Reviewing chapter…" : session.isGenerating ? "Generating chapter…" : session.explanation ? "Regenerate chapter" : session.error ? "Retry chapter generation" : "Generate this chapter"}</button>
          {session.explanation && (session.explanation.quality_review_status === "unreviewed" || session.explanation.quality_review_status === "incomplete") &&
            <button type="button" disabled={session.isGenerating || !!loadingFile} onClick={() => void session.explainSource("review")}>Review this chapter</button>}
          <label className="pr-diagram-option"><input type="checkbox" checked={settings.visualPreference !== "no"} disabled={session.isGenerating || !!loadingFile} aria-describedby={diagramNote}
            onChange={(event) => selectChapter(index, { ...settings, visualPreference: event.target.checked ? "auto" : "no" })} />Include diagrams</label>
          <small id={diagramNote}>Optional. Diagrams are off by default.</small>
          <label className="pr-diagram-option"><input type="checkbox" checked={automaticReview} disabled={session.isGenerating || !!loadingFile}
            onChange={(event) => updateAutomaticReview(event.target.checked)} />Review automatically</label>
          {session.isGenerating && <button type="button" onClick={session.invalidatePendingRequests}>Cancel</button>}
        </div>
        {session.generationPhase && session.isGenerating && <p role="status">{generationPhaseCopy[session.generationPhase].detail}</p>}
        {session.error && <p role="alert">{session.error}</p>}
        {session.explanation && <ExplanationResult key={activeKey} ref={resultReader} initialReading={savedResult?.explanation === session.explanation ? savedResult.reading : undefined}
          allowChapterArtwork={false} materialMode="pull_request" refinementTarget={refinementTarget} explanation={session.explanation} source={source} theme={theme} lineCount={source.split("\n").length}
          storyPreferences={{ depth: settings.depth, learnerLevel: settings.learnerLevel }}
          exportContext={{ scope: `PR chapter: ${chapter.title}`, provenance: `PR: ${plan.url}\nHead: ${plan.headSha}\nBase: ${plan.baseSha}\nPreferences: ${JSON.stringify(settings)}\nPR coverage limits: ${plan.warnings.join("; ")}` }}
          isRegenerating={session.isGenerating} isReviewing={session.isReviewing} isAskingFollowUp={session.isAskingFollowUp} copyStatus={session.copyStatus} followUpQuestion={session.followUpQuestion} selectedTextForQuestion="" followUpMessages={session.followUpMessages} followUpError={session.followUpError} textSelection={null}
          onCopy={() => void copy()} onFollowUpQuestionChange={session.setFollowUpQuestion} onAskFollowUp={session.askFollowUp} onClearSelectedTextContext={() => session.setSelectedTextForQuestion("")} onAskAboutSelectedText={() => {}} />}
        <nav className="chapter-pagination" aria-label="PR chapter navigation"><button type="button" disabled={index === 0} onClick={() => selectChapter(index - 1)}>Previous chapter</button><span>{index + 1} / {chapters.length}</span><button type="button" disabled={index === chapters.length - 1} onClick={() => selectChapter(index + 1)}>Next chapter</button></nav>
      </div>
    </div>
  </section>;
}
