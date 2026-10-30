import { useState, type FormEvent, type KeyboardEvent } from "react";
import type { Explanation, FollowUpMessage } from "../domain/explanation-types";
import type { SourceRange } from "../domain/source-ranges";
import type { ThemeMode } from "../domain/theme";
import { formatMathText } from "../domain/format-math-text";
import { GraphView } from "./GraphView";
import { SpeechReader } from "./SpeechReader";
import { SourceRanges } from "./SourceRanges";
import { UnderstandingCheck } from "./UnderstandingCheck";

export type TextSelection = { text: string; top: number; left: number; wasTruncated: boolean };

type Props = {
  explanation: Explanation;
  source: string;
  theme: ThemeMode;
  lineCount: number;
  isAskingFollowUp: boolean;
  copyStatus: string;
  followUpQuestion: string;
  selectedTextForQuestion: string;
  followUpMessages: FollowUpMessage[];
  followUpError: string;
  textSelection: TextSelection | null;
  onCopy: () => void;
  onSelectRange: (range: SourceRange) => void;
  onFollowUpQuestionChange: (value: string) => void;
  onAskFollowUp: (event: FormEvent<HTMLFormElement>) => void;
  onClearSelectedTextContext: () => void;
  onAskAboutSelectedText: () => void;
};

export function ExplanationResult({
  explanation, source, theme, lineCount, isAskingFollowUp, copyStatus, followUpQuestion,
  selectedTextForQuestion, followUpMessages, followUpError, textSelection, onCopy, onSelectRange,
  onFollowUpQuestionChange, onAskFollowUp, onClearSelectedTextContext, onAskAboutSelectedText,
}: Props) {
  const [activeVisualIndex, setActiveVisualIndex] = useState(0);
  const activeVisual = explanation.visuals[activeVisualIndex] ?? explanation.visuals[0];
  const selectVisualFromKeyboard = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const lastIndex = explanation.visuals.length - 1;
    let nextIndex = index;
    if (event.key === "ArrowRight") nextIndex = index === lastIndex ? 0 : index + 1;
    else if (event.key === "ArrowLeft") nextIndex = index === 0 ? lastIndex : index - 1;
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = lastIndex;
    else return;
    event.preventDefault();
    setActiveVisualIndex(nextIndex);
    document.getElementById(`diagram-tab-${nextIndex}`)?.focus();
  };
  return <>
    <section className="result-card" aria-labelledby="explanation-title" aria-busy={isAskingFollowUp}>
      <div className="result-topline"><span className="result-label"><span className="result-spark">✳</span> YOUR EXPLANATION</span><div className="result-actions"><button className="copy-explanation" type="button" onClick={onCopy}>{copyStatus || "Copy explanation"}</button><span className="preview-badge">EXPLANATION</span></div></div>
      <div className="explanation-content">
        <h2 id="explanation-title" tabIndex={-1}>{explanation.title}</h2>
        <p className="result-summary">{explanation.summary}</p>
        {explanation.incomplete_items && explanation.incomplete_items.length > 0 && <aside className="incomplete-code-note" role="status" aria-labelledby="incomplete-code-heading">
          <div className="section-caption" id="incomplete-code-heading">INCOMPLETE CODE DETECTED</div>
          <p>The source contains unfinished work. Its intended behavior is not specified, so the explanation does not infer it.</p>
          <ul>{explanation.incomplete_items.map((item, index) => <li key={`${item.kind}-${item.source_ranges[0]?.start_line}-${index}`}>
            <strong>{item.title}.</strong> {item.explanation}
            <SourceRanges ranges={item.source_ranges} lineCount={lineCount} onSelectRange={onSelectRange} />
          </li>)}</ul>
        </aside>}
        {explanation.quality_review_note && <aside className={`quality-review-note ${explanation.quality_review_status ?? "incomplete"}`} role="status" aria-live="polite">
          <strong>Second-pass review · {explanation.quality_review_status === "passed" ? "passed" : explanation.quality_review_status === "revised" ? "revised · final check passed" : "not fully verified"}</strong>
          <span>{explanation.quality_review_note}</span>
        </aside>}
        <SpeechReader explanation={explanation} />
        {explanation.concepts.length > 0 && <div className="result-section"><div className="section-caption">KEY IDEAS</div><div className="concept-grid">{explanation.concepts.map((concept, index) => <article className="concept-card" key={`${concept.term}-${index}`}><h3>{concept.term}</h3><p>{concept.explanation}</p><SourceRanges ranges={concept.source_ranges} lineCount={lineCount} onSelectRange={onSelectRange} /></article>)}</div></div>}
        {explanation.walkthrough.length > 0 && <div className="result-section walkthrough-section"><div className="section-caption">WALKTHROUGH</div><div className="walkthrough-list">{explanation.walkthrough.map((step, index) => <article className="walkthrough-item" key={`${step.heading}-${index}`}><div className="walkthrough-heading"><span className="number-badge">{index + 1}</span><div><h3>{step.heading}</h3></div></div><p className="result-body">{step.explanation}</p><SourceRanges ranges={step.source_ranges} lineCount={lineCount} onSelectRange={onSelectRange} /></article>)}</div></div>}
        {activeVisual && <div className="result-section">
          <div className="section-caption diagram-section-caption">{activeVisual.kind === "control_flow" ? "CONTROL FLOW MAP" : "CONCEPT MAP"}</div>
          {explanation.visuals.length > 1 && <div className="diagram-view-tabs" role="tablist" aria-label="Diagram views">{explanation.visuals.map((visual, index) => <button key={`${visual.title}-${index}`} type="button" role="tab" id={`diagram-tab-${index}`} aria-selected={index === activeVisualIndex} aria-controls="active-diagram-panel" tabIndex={index === activeVisualIndex ? 0 : -1} onKeyDown={(event) => selectVisualFromKeyboard(event, index)} onClick={() => setActiveVisualIndex(index)}>{visual.title}</button>)}</div>}
          <div id="active-diagram-panel" role={explanation.visuals.length > 1 ? "tabpanel" : undefined} aria-labelledby={explanation.visuals.length > 1 ? `diagram-tab-${activeVisualIndex}` : undefined} tabIndex={explanation.visuals.length > 1 ? 0 : undefined}><GraphView visual={activeVisual} theme={theme} lineCount={lineCount} source={source} onSelectRange={onSelectRange} incompleteItems={explanation.incomplete_items} /></div>
        </div>}
        {explanation.visuals.length > 0 && <div className="diagram-gallery-print" aria-hidden="true">{explanation.visual_note && <p className="diagram-print-note"><strong>Coverage note:</strong> {explanation.visual_note}</p>}{explanation.visuals.map((visual, index) => <section className="diagram-print-alternative" key={`${visual.title}-${index}`}>
          <div className="diagram-print-heading">{index === 0 && <div className="section-caption">{visual.kind === "control_flow" ? "CONTROL FLOW MAP" : "CONCEPT MAP"}</div>}<h3>{visual.title}</h3></div>
          <ol>{visual.nodes.map((node) => {
            const incomplete = explanation.incomplete_items?.some((item) => item.source_ranges.some((range) => node.source_ranges.some((nodeRange) => nodeRange.start_line <= range.end_line && nodeRange.end_line >= range.start_line)));
            return <li key={node.id}>{incomplete && <strong>Incomplete implementation. </strong>}<strong>{node.label}.</strong> {node.description}</li>;
          })}</ol>
          {visual.edges.length > 0 && <ul>{visual.edges.map((edge, edgeIndex) => <li key={`${edge.from}-${edge.to}-${edgeIndex}`}>{visual.nodes.find((node) => node.id === edge.from)?.label ?? edge.from} {edge.label ? `(${edge.label})` : "connects to"} {visual.nodes.find((node) => node.id === edge.to)?.label ?? edge.to}.</li>)}</ul>}
        </section>)}</div>}
        {explanation.visual_note && <p className={`visual-omitted-note ${explanation.visuals.length > 0 ? "coverage-screen-note" : ""}`}><strong>{explanation.visuals.length === 0 ? "Diagram note." : "Coverage note."}</strong> {explanation.visual_note}</p>}
        {explanation.assumptions.length > 0 && <div className="assumptions"><div className="section-caption">WHAT THE INPUT DOESN’T TELL US</div>{explanation.assumptions.map((assumption, index) => <p key={`${assumption.statement}-${index}`}><strong>{assumption.statement}</strong> {assumption.reason}</p>)}</div>}
        <UnderstandingCheck source={source} explanation={explanation} />
      </div>
      <section className="followup-section" aria-label="Ask a follow-up question">
        <div className="section-caption">KEEP EXPLORING</div>
        <h3>Ask a follow-up</h3>
        <p className="followup-intro">Ask about the source or this explanation. The conversation stays in this page session.</p>
        {followUpMessages.length > 0 && <div className="followup-thread" aria-live="polite" aria-relevant="additions">{followUpMessages.map((message, index) => <article className={`followup-message ${message.role}`} key={`${message.role}-${index}`}>
          <span className="followup-speaker">{message.role === "learner" ? "YOU" : "ANSWER"}</span>
          <p>{message.role === "assistant" ? formatMathText(message.text) : message.text}</p>
          {message.role === "learner" && message.selected_text && <blockquote className="followup-selection-quote">{message.selected_text}</blockquote>}
          {message.role === "assistant" && <><SourceRanges ranges={message.source_ranges} lineCount={lineCount} onSelectRange={onSelectRange} />{message.assumptions.length > 0 && <ul>{message.assumptions.map((assumption, assumptionIndex) => <li key={`${assumption.statement}-${assumptionIndex}`}><strong>{assumption.statement}</strong> {assumption.reason}</li>)}</ul>}</>}
        </article>)}</div>}
        {selectedTextForQuestion && <div className="selected-question-context"><span><strong>Asking about:</strong> “{selectedTextForQuestion}”</span><button type="button" onClick={onClearSelectedTextContext} aria-label="Remove selected text context">×</button></div>}
        <form className="followup-form" onSubmit={onAskFollowUp}>
          <label className="visually-hidden" htmlFor="followup-question">Your follow-up question</label>
          <textarea id="followup-question" value={followUpQuestion} onChange={(event) => onFollowUpQuestionChange(event.target.value)} placeholder={selectedTextForQuestion ? "Ask about the selected passage…" : "What would you like to explore further?"} maxLength={4000} rows={2} />
          <button className="followup-submit followup-submit-icon" type="submit" disabled={!followUpQuestion.trim() || isAskingFollowUp} aria-label={isAskingFollowUp ? "Sending question" : "Ask follow-up question"} title={isAskingFollowUp ? "Sending question" : "Ask"}>
            {isAskingFollowUp
              ? <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="8" strokeDasharray="18 32" /></svg>
              : <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 12h15M13 6l6 6-6 6" /></svg>}
          </button>
        </form>
        {followUpError && <div className="followup-error" role="alert">{followUpError}</div>}
      </section>
      <div className="result-bottom"><span>✳ Keep exploring</span><span>Ask another question above <b>↑</b></span></div>
    </section>

    {textSelection && <div className="selection-action" role="toolbar" aria-label="Ask about selected text" style={{ top: textSelection.top, left: textSelection.left }}>
      <span>{textSelection.wasTruncated ? "Selection shortened to 2,000 characters" : `“${textSelection.text.slice(0, 72)}${textSelection.text.length > 72 ? "…" : ""}”`}</span>
      <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={onAskAboutSelectedText}>Ask about this</button>
    </div>}
  </>;
}
