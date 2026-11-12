import { useRef, useState } from "react";
import type { Explanation, ExplanationDepth, LearnerLevel } from "../domain/explanation-types";
import type { PullRequestPlan } from "../domain/pull-request-plan";
import type { ThemeMode } from "../domain/theme";
import { sourceOffsetsForRange } from "../domain/source-ranges";
import { generationPhaseCopy } from "../domain/generation-progress";
import { useExplanationSession } from "../hooks/useExplanationSession";
import { ExplanationResult } from "./ExplanationResult";

export function PullRequestStudy({ plan, theme, onClose }: { plan: PullRequestPlan; theme: ThemeMode; onClose: () => void }) {
  const [index, setIndex] = useState(0);
  const [depth, setDepth] = useState<ExplanationDepth>("detailed");
  const [learnerLevel, setLearnerLevel] = useState<LearnerLevel>("beginner");
  const cache = useRef(new Map<string, Explanation>());
  const editor = useRef<HTMLTextAreaElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const chapter = plan.chapters[index];
  const source = chapter.sourceText;
  const session = useExplanationSession({ source, mode: "pull_request", depth, learnerLevel });
  const cacheKey = (chapterIndex: number, chapterDepth: ExplanationDepth, level: LearnerLevel) => `${chapterIndex}:${chapterDepth}:${level}`;

  function selectChapter(next: number, nextDepth = depth, nextLevel = learnerLevel) {
    if (session.explanation) cache.current.set(cacheKey(index, depth, learnerLevel), session.explanation);
    session.resetExplanation();
    session.setExplanation(cache.current.get(cacheKey(next, nextDepth, nextLevel)) ?? null);
    setIndex(next); setDepth(nextDepth); setLearnerLevel(nextLevel);
    heading.current?.focus();
  }

  async function copy() {
    if (!session.explanation) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify({ chapter: chapter.title, head: plan.headSha, source, explanation: session.explanation }, null, 2));
      session.setCopyStatus("Copied chapter with source");
    } catch { session.setCopyStatus("Clipboard unavailable"); }
  }

  return <section className="pr-study" aria-label="Pull request study plan">
    <header className="pr-study-header"><div><div className="section-caption">COVERAGE-AWARE READING PLAN</div><h2>{plan.title}</h2><a href={plan.url} target="_blank" rel="noreferrer">View PR on GitHub</a></div><button type="button" onClick={onClose}>Back to input</button></header>
    <p>{plan.files.length} / {plan.totalFiles ?? "unknown total"} files listed · {plan.files.filter((file) => file.coverage === "included").length} patches included without local trimming · {plan.chapters.length} chapters · {plan.filesComplete ? "File inventory complete" : "File inventory incomplete"}</p>
    <p className="pr-revision">Snapshot: {plan.headSha}. Chapters are generated only when requested. Results stay in this page session; no repository-wide analysis is implied.</p>
    <p aria-live="polite">{plan.chapters.filter((_, itemIndex) => itemIndex === index ? !!session.explanation : cache.current.has(cacheKey(itemIndex, depth, learnerLevel))).length} / {plan.chapters.length} chapters generated for the selected depth and learner level.</p>
    <aside className="incomplete-code-note" aria-label="PR coverage limits"><strong>Coverage limits</strong><ul>{plan.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></aside>
    <details className="pr-manifest"><summary>File manifest and evidence coverage</summary><ul>{plan.files.map((file) => <li key={file.path}><code>{file.path}</code> · {file.status} · +{file.additions}/-{file.deletions} · <strong>{file.coverage.replaceAll("_", " ")}</strong></li>)}</ul></details>
    <div className="chapter-layout">
      <nav className="chapter-sidebar" aria-label="PR chapter outline"><ol>{plan.chapters.map((item, itemIndex) => <li key={item.id}><button type="button" aria-current={index === itemIndex ? "step" : undefined} onClick={() => selectChapter(itemIndex)}><span>{itemIndex + 1}</span>{item.title}</button></li>)}</ol></nav>
      <div className="chapter-body">
        <h3 ref={heading} tabIndex={-1}>{chapter.title}</h3>
        <p>{chapter.files.length ? `${chapter.files.length} file patch(es). Citations refer to the chapter source below.` : "Inventory-only overview: implementation behavior must be explained in the evidence chapters."}</p>
        <details><summary>Review chapter source</summary><textarea ref={editor} readOnly aria-label="Chapter source" value={source} rows={10} /></details>
        <div className="chapter-toolbar">
          <label>Depth <select aria-label="Chapter depth" value={depth} onChange={(event) => selectChapter(index, event.target.value as ExplanationDepth)}><option value="overview">Overview</option><option value="balanced">Balanced</option><option value="detailed">Detailed</option></select></label>
          <label>Explain for <select aria-label="Chapter learner level" value={learnerLevel} onChange={(event) => selectChapter(index, depth, event.target.value as LearnerLevel)}><option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option></select></label>
          <button type="button" disabled={session.isGenerating} onClick={() => { cache.current.delete(cacheKey(index, depth, learnerLevel)); void session.explainSource(); }}>{session.isGenerating ? "Generating chapter…" : session.explanation ? "Regenerate chapter" : "Generate this chapter"}</button>
          {session.isGenerating && <button type="button" onClick={session.resetExplanation}>Cancel</button>}
        </div>
        {session.generationPhase && session.isGenerating && <p role="status">{generationPhaseCopy[session.generationPhase].detail}</p>}
        {session.error && <p role="alert">{session.error}</p>}
        {session.explanation && <ExplanationResult explanation={session.explanation} source={source} theme={theme} lineCount={source.split("\n").length}
          isAskingFollowUp={session.isAskingFollowUp} copyStatus={session.copyStatus} followUpQuestion={session.followUpQuestion} selectedTextForQuestion="" followUpMessages={session.followUpMessages} followUpError={session.followUpError} textSelection={null}
          onCopy={() => void copy()} onSelectRange={(range) => {
            const element = editor.current;
            const offsets = sourceOffsetsForRange(source, range);
            if (!element || !offsets) return;
            const details = element.closest("details");
            if (details) details.open = true;
            element.focus(); element.setSelectionRange(offsets.start, offsets.end); element.scrollIntoView({ block: "center" });
          }} onFollowUpQuestionChange={session.setFollowUpQuestion} onAskFollowUp={session.askFollowUp} onClearSelectedTextContext={() => session.setSelectedTextForQuestion("")} onAskAboutSelectedText={() => {}} />}
        <nav className="chapter-pagination" aria-label="PR chapter navigation"><button type="button" disabled={index === 0} onClick={() => selectChapter(index - 1)}>Previous chapter</button><span>{index + 1} / {plan.chapters.length}</span><button type="button" disabled={index === plan.chapters.length - 1} onClick={() => selectChapter(index + 1)}>Next chapter</button></nav>
      </div>
    </div>
  </section>;
}
