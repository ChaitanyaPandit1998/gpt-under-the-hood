import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { isValidRange, sourceOffsetsForRange, type SourceRange } from "./domain/source-ranges";
import { UnderstandingCheck } from "./components/UnderstandingCheck";
import { GraphView } from "./components/GraphView";
import { SourceRanges } from "./components/SourceRanges";

type Mode = "code" | "docs";
type Explanation = {
  title: string;
  summary: string;
  concepts: { term: string; explanation: string; source_ranges: SourceRange[] }[];
  walkthrough: { heading: string; explanation: string; source_ranges: SourceRange[] }[];
  assumptions: { statement: string; reason: string }[];
  visual: null | {
    kind: "control_flow" | "concept_map";
    nodes: { id: string; label: string; description: string; source_ranges: SourceRange[] }[];
    edges: { from: string; to: string; label: string }[];
  };
};
type Transcription = { transcription: string; uncertainties: { text: string; reason: string }[] };
type FollowUpMessage =
  | { role: "learner"; text: string; selected_text?: string }
  | { role: "assistant"; text: string; source_ranges: SourceRange[]; assumptions: { statement: string; reason: string }[] };
type FollowUpAnswer = { answer: string; source_ranges: SourceRange[]; assumptions: { statement: string; reason: string }[] };
type TextSelection = { text: string; top: number; left: number; wasTruncated: boolean };

const exampleCode = `function retry<T>(operation: () => Promise<T>, attempts = 3): Promise<T> {
  return operation().catch((error) => {
    if (attempts <= 1) throw error;
    return retry(operation, attempts - 1);
  });
}`;

function App() {
  const [mode, setMode] = useState<Mode>("code");
  const [source, setSource] = useState("");
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isAskingFollowUp, setIsAskingFollowUp] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");
  const [followUpQuestion, setFollowUpQuestion] = useState("");
  const [selectedTextForQuestion, setSelectedTextForQuestion] = useState("");
  const [textSelection, setTextSelection] = useState<TextSelection | null>(null);
  const [followUpMessages, setFollowUpMessages] = useState<FollowUpMessage[]>([]);
  const [followUpError, setFollowUpError] = useState("");
  const [screenshotPreview, setScreenshotPreview] = useState<string | null>(null);
  const [screenshotFile, setScreenshotFile] = useState<File | null>(null);
  const [transcription, setTranscription] = useState<Transcription | null>(null);
  const [needsApproval, setNeedsApproval] = useState(false);
  const [approved, setApproved] = useState(false);
  const [error, setError] = useState("");
  const screenshotInput = useRef<HTMLInputElement>(null);
  const sourceEditor = useRef<HTMLTextAreaElement>(null);
  const explanationRequestId = useRef(0);
  const followUpRequestId = useRef(0);
  const transcriptionRequestId = useRef(0);
  const lineCount = useMemo(() => Math.max(source.split("\n").length, 1), [source]);

  useEffect(() => () => {
    if (screenshotPreview) URL.revokeObjectURL(screenshotPreview);
  }, [screenshotPreview]);

  useEffect(() => {
    function updateTextSelection() {
      const selection = window.getSelection();
      const selected = selection?.toString().trim() ?? "";
      const content = document.querySelector(".explanation-content");
      const toolbar = document.querySelector(".selection-action");
      if ((!selection || !selected || !content || !selection.anchorNode || !selection.focusNode ||
          !content.contains(selection.anchorNode) || !content.contains(selection.focusNode))) {
        if (toolbar?.contains(document.activeElement)) return;
        setTextSelection(null);
        return;
      }
      if (selection.rangeCount === 0) return;
      const range = selection.getRangeAt(0);
      if (!content.contains(range.commonAncestorContainer)) {
        setTextSelection(null);
        return;
      }
      const rect = range.getBoundingClientRect();
      const maxTextLength = 2_000;
      setTextSelection({
        text: selected.slice(0, maxTextLength),
        wasTruncated: selected.length > maxTextLength,
        top: Math.max(8, Math.min(rect.bottom + 8, window.innerHeight - 56)),
        left: Math.max(8, Math.min(rect.left, window.innerWidth - 296)),
      });
    }
    document.addEventListener("selectionchange", updateTextSelection);
    return () => document.removeEventListener("selectionchange", updateTextSelection);
  }, []);

  function invalidatePendingRequests() {
    explanationRequestId.current += 1;
    followUpRequestId.current += 1;
    setIsGenerating(false);
    setIsAskingFollowUp(false);
  }

  function resetExplanation() {
    invalidatePendingRequests();
    setExplanation(null);
    setFollowUpMessages([]);
    setFollowUpQuestion("");
    setSelectedTextForQuestion("");
    setFollowUpError("");
    setError("");
  }

  function updateMode(nextMode: Mode) {
    if (nextMode === mode) return;
    resetExplanation();
    setMode(nextMode);
  }

  function askAboutSelectedText() {
    if (!textSelection) return;
    setSelectedTextForQuestion(textSelection.text);
    setFollowUpQuestion("Could you explain this?");
    setTextSelection(null);
    requestAnimationFrame(() => {
      const questionInput = document.getElementById("followup-question");
      questionInput?.scrollIntoView({ behavior: "smooth", block: "center" });
      questionInput?.focus();
    });
  }

  function clearSelectedTextContext() {
    setSelectedTextForQuestion("");
    setFollowUpQuestion("");
  }

  function updateSource(value: string) {
    transcriptionRequestId.current += 1;
    setIsTranscribing(false);
    resetExplanation();
    setSource(value);
  }

  function jumpToSource(range: SourceRange) {
    const editor = sourceEditor.current;
    if (!editor) return;
    const offsets = sourceOffsetsForRange(source, range);
    if (!offsets) return;
    editor.focus();
    editor.setSelectionRange(offsets.start, offsets.end);
    editor.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  async function copyExplanation() {
    if (!explanation) return;
    const formatRanges = (ranges: SourceRange[]) => ranges.filter((range) => isValidRange(range, lineCount)).map((range) => `Lines ${range.start_line}${range.end_line !== range.start_line ? `–${range.end_line}` : ""}`).join(", ");
    const sections = [
      `# ${explanation.title}`,
      explanation.summary,
      explanation.concepts.length ? `## Key ideas\n${explanation.concepts.map((concept) => `### ${concept.term}\n${concept.explanation}${formatRanges(concept.source_ranges) ? `\n_Source: ${formatRanges(concept.source_ranges)}_` : ""}`).join("\n\n")}` : "",
      explanation.walkthrough.length ? `## Walkthrough\n${explanation.walkthrough.map((step, index) => `### ${index + 1}. ${step.heading}\n${step.explanation}${formatRanges(step.source_ranges) ? `\n_Source: ${formatRanges(step.source_ranges)}_` : ""}`).join("\n\n")}` : "",
      explanation.visual ? `## ${explanation.visual.kind === "control_flow" ? "Control flow" : "Concept map"}\n${explanation.visual.nodes.map((node) => `- **${node.label}:** ${node.description}${formatRanges(node.source_ranges) ? ` (${formatRanges(node.source_ranges)})` : ""}`).join("\n")}${explanation.visual.edges.length ? `\n\nConnections:\n${explanation.visual.edges.map((edge) => `- ${edge.from} → ${edge.to}${edge.label ? ` (${edge.label})` : ""}`).join("\n")}` : ""}` : "",
      explanation.assumptions.length ? `## What the input doesn’t tell us\n${explanation.assumptions.map((item) => `- **${item.statement}** ${item.reason}`).join("\n")}` : "",
      followUpMessages.length ? `## Follow-up\n${followUpMessages.map((message) => message.role === "learner" ? `**Question:** ${message.text}${message.selected_text ? `\n> ${message.selected_text.replaceAll("\n", "\n> ")}` : ""}` : `**Answer:** ${message.text}${formatRanges(message.source_ranges) ? `\n_Source: ${formatRanges(message.source_ranges)}_` : ""}`).join("\n\n")}` : "",
    ].filter(Boolean).join("\n\n");
    try {
      await navigator.clipboard.writeText(sections);
      setCopyStatus("Copied");
      window.setTimeout(() => setCopyStatus(""), 1800);
    } catch {
      setCopyStatus("Clipboard unavailable");
      window.setTimeout(() => setCopyStatus(""), 2500);
    }
  }

  function loadExample() {
    clearInput();
    setMode("code");
    updateSource(exampleCode);
  }

  function clearInput() {
    updateSource("");
    setScreenshotPreview(null);
    setScreenshotFile(null);
    setTranscription(null);
    setNeedsApproval(false);
    setApproved(false);
    if (screenshotInput.current) screenshotInput.current.value = "";
  }

  async function transcribeScreenshot(file: File) {
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      setError("Choose a PNG, JPEG, or WebP screenshot.");
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      setError("Screenshots must be smaller than 8 MB.");
      return;
    }

    resetExplanation();
    const requestId = ++transcriptionRequestId.current;
    setSource("");
    setScreenshotFile(file);
    setScreenshotPreview(URL.createObjectURL(file));
    setTranscription(null);
    setNeedsApproval(true);
    setApproved(false);
    setIsTranscribing(true);
    try {
      const response = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: await fileToBase64(file), mime_type: file.type }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not read text from this screenshot.");
      if (requestId !== transcriptionRequestId.current) return;
      const result = payload as Transcription;
      setSource(result.transcription || "");
      setTranscription(result);
    } catch (caught) {
      if (requestId !== transcriptionRequestId.current) return;
      setError(caught instanceof Error ? caught.message : "Unable to read text from this screenshot.");
    } finally {
      if (requestId === transcriptionRequestId.current) setIsTranscribing(false);
    }
  }

  async function explainSource() {
    const requestId = ++explanationRequestId.current;
    const requestSource = source;
    const requestMode = mode;
    setIsGenerating(true);
    setError("");
    setExplanation(null);
    setFollowUpMessages([]);
    setFollowUpQuestion("");
    setSelectedTextForQuestion("");
    setFollowUpError("");
    try {
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_text: requestSource,
          material_kind: requestMode,
          teaching_mode: "auto",
          visual_preference: "auto",
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not generate an explanation.");
      if (requestId !== explanationRequestId.current) return;
      setExplanation(payload as Explanation);
    } catch (caught) {
      if (requestId !== explanationRequestId.current) return;
      setError(caught instanceof Error ? caught.message : "Unable to reach the local explanation service.");
    } finally {
      if (requestId === explanationRequestId.current) setIsGenerating(false);
    }
  }

  async function askFollowUp(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = followUpQuestion.trim();
    if (!question || !explanation || isAskingFollowUp) return;
    const requestId = ++followUpRequestId.current;
    const requestSource = source;
    const requestExplanation = explanation;
    const requestHistory = followUpMessages;
    setIsAskingFollowUp(true);
    setFollowUpError("");
    try {
      const response = await fetch("/api/followup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_text: requestSource,
          explanation: requestExplanation,
          history: requestHistory,
          selected_text: selectedTextForQuestion || undefined,
          question,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not answer this follow-up.");
      if (requestId !== followUpRequestId.current) return;
      const answer = payload as FollowUpAnswer;
      setFollowUpMessages((current) => [
        ...current,
        { role: "learner", text: question, ...(selectedTextForQuestion ? { selected_text: selectedTextForQuestion } : {}) },
        { role: "assistant", text: answer.answer, source_ranges: answer.source_ranges, assumptions: answer.assumptions },
      ]);
      setFollowUpQuestion("");
      setSelectedTextForQuestion("");
    } catch (caught) {
      if (requestId !== followUpRequestId.current) return;
      setFollowUpError(caught instanceof Error ? caught.message : "Unable to reach the local explanation service.");
    } finally {
      if (requestId === followUpRequestId.current) setIsAskingFollowUp(false);
    }
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Understand home">
          <span className="brand-mark"><span /><span /><span /></span>
          <span>understand<span className="brand-period">.</span></span>
        </a>
        <div className="topbar-center"><span className="local-dot" /> Local workspace</div>
        <button className="avatar" aria-label="Your workspace">C</button>
      </header>

      <main id="main-content" className="page-content">
        <section className="intro">
          <div className="eyebrow"><span className="eyebrow-spark">✳</span> YOUR CODE, MADE CLEARER</div>
          <h1>Understand what<br className="desktop-break" /> you’re working with.</h1>
          <p>Bring a piece of code or documentation. Get an explanation that helps you build a mental model, not just skim the surface.</p>
        </section>

        <section className="workspace" aria-label="Learning workspace">
          <div className="workspace-heading">
            <div>
              <div className="step-label"><span>01</span> START WITH SOMETHING</div>
              <h2>What would you like to understand?</h2>
            </div>
            <button className="example-link" onClick={loadExample}>Try an example <span>↗</span></button>
          </div>

          <div className="input-card">
            <div className="input-toolbar">
              <div className="segmented" role="tablist" aria-label="Input type">
                <button role="tab" id="tab-code" aria-controls="source-input" aria-selected={mode === "code"} tabIndex={mode === "code" ? 0 : -1} className={mode === "code" ? "active" : ""} onKeyDown={(event) => { if (event.key === "ArrowRight" || event.key === "ArrowLeft") { event.preventDefault(); updateMode(mode === "code" ? "docs" : "code"); document.getElementById(mode === "code" ? "tab-docs" : "tab-code")?.focus(); } }} onClick={() => updateMode("code")}>⌘ <span>Code</span></button>
                <button role="tab" id="tab-docs" aria-controls="source-input" aria-selected={mode === "docs"} tabIndex={mode === "docs" ? 0 : -1} className={mode === "docs" ? "active" : ""} onKeyDown={(event) => { if (event.key === "ArrowRight" || event.key === "ArrowLeft") { event.preventDefault(); updateMode(mode === "code" ? "docs" : "code"); document.getElementById(mode === "code" ? "tab-docs" : "tab-code")?.focus(); } }} onClick={() => updateMode("docs")}>▤ <span>Documentation</span></button>
              </div>
              <div className="input-tools"><span className="language-pill">{mode === "code" ? "Language detected automatically" : "Markdown or plain text"}</span><span className="toolbar-divider" /><button className="tool-button" onClick={clearInput} disabled={isTranscribing || isGenerating || (!source && !screenshotPreview)} aria-label="Clear input" title="Clear input">⌫</button></div>
            </div>

            <div className={`editor-wrap ${source || (screenshotPreview && !isTranscribing) ? "has-source" : ""}`}>
              {source || (screenshotPreview && !isTranscribing) ? (
                <>
                  {mode === "code" && <div className="line-numbers" aria-hidden="true">{Array.from({ length: lineCount }, (_, i) => <span key={i}>{i + 1}</span>)}</div>}
                  <textarea id="source-input" ref={sourceEditor} aria-label={mode === "code" ? "Code input" : "Documentation input"} spellCheck={false} value={source} onChange={(event) => updateSource(event.target.value)} placeholder={mode === "code" ? "Paste a code snippet here…" : "Paste documentation here…"} className={mode === "code" ? "source-editor code-editor" : "source-editor docs-editor"} />
                </>
              ) : (
                <div className="editor-empty">
                  {isTranscribing ? <><div className="empty-icon">◌</div><strong>Reading the screenshot…</strong><span>The model is extracting visible code or documentation.</span></> : <><div className="empty-icon">{mode === "code" ? "{ }" : "Aa"}</div><strong>{mode === "code" ? "Paste a code snippet to get started" : "Paste a passage or document to explore"}</strong><span>{mode === "code" ? "Any language is welcome. A few lines are enough." : "Technical docs, API references, or concepts you want to unpack."}</span><button className="paste-hint" onClick={loadExample}>Or try our example <span>→</span></button></>}
                </div>
              )}
            </div>

            <div className="input-footer">
              <span>{source ? `${source.length.toLocaleString()} characters · ${lineCount} lines` : "Your input stays in this local session"}</span>
              <input ref={screenshotInput} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => { const file = event.target.files?.[0]; if (file) void transcribeScreenshot(file); event.currentTarget.value = ""; }} />
              <button className="upload-button" onClick={() => screenshotInput.current?.click()} disabled={isTranscribing || isGenerating}><span>▧</span> Add screenshot</button>
            </div>
          </div>

          {screenshotPreview && <div className="screenshot-review" aria-live="polite">
            <img src={screenshotPreview} alt="Screenshot provided for text extraction" />
            <div className="screenshot-review-copy">
              <strong>{isTranscribing ? "Reading screenshot…" : transcription ? "Review the extracted text" : "Screenshot ready"}</strong>
              <span>{isTranscribing ? "The model is reading visible code or documentation." : "Check and edit the text above. It will not be explained until you approve it."}</span>
              {transcription && transcription.uncertainties.length > 0 && <ul className="transcription-uncertainties">{transcription.uncertainties.map((item, index) => <li key={`${item.text}-${index}`}><strong>{item.text}</strong> {item.reason}</li>)}</ul>}
            </div>
            {needsApproval && !isTranscribing && <button className="approve-transcription" onClick={() => setApproved(true)} disabled={approved || !source.trim()}>{approved ? "Text reviewed ✓" : transcription ? "I reviewed this text" : "Use the text I entered"}</button>}
          </div>}

          <div className="action-row">
            <span className="privacy-note"><span>◈</span> Nothing is saved automatically</span>
            <button className="generate-button" disabled={!source.trim() || isGenerating || isTranscribing || (needsApproval && !approved)} onClick={explainSource}>{isTranscribing ? "Reading screenshot…" : isGenerating ? "Working with Codex…" : "Help me understand"} <span>{isGenerating || isTranscribing ? "◌" : "→"}</span></button>
          </div>
          {error && <div className="error-message" role="alert"><span>!</span><p>{error}</p><button onClick={() => screenshotFile && needsApproval && !transcription ? void transcribeScreenshot(screenshotFile) : void explainSource()} disabled={isGenerating || isTranscribing}>Try again</button></div>}
        </section>

        {explanation && <section className="result-card" aria-labelledby="explanation-title" aria-busy={isAskingFollowUp}>
          <div className="result-topline"><span className="result-label"><span className="result-spark">✳</span> YOUR EXPLANATION</span><div className="result-actions"><button className="copy-explanation" type="button" onClick={() => void copyExplanation()}>{copyStatus || "Copy explanation"}</button><span className="preview-badge">EXPLANATION</span></div></div>
          <div className="explanation-content">
          <h2 id="explanation-title" tabIndex={-1}>{explanation.title}</h2>
          <p className="result-summary">{explanation.summary}</p>
          {explanation.concepts.length > 0 && <div className="result-section"><div className="section-caption">KEY IDEAS</div><div className="concept-grid">{explanation.concepts.map((concept, index) => <article className="concept-card" key={`${concept.term}-${index}`}><h3>{concept.term}</h3><p>{concept.explanation}</p><SourceRanges ranges={concept.source_ranges} lineCount={lineCount} onSelectRange={jumpToSource} /></article>)}</div></div>}
          {explanation.walkthrough.length > 0 && <div className="result-section"><div className="section-caption">WALKTHROUGH</div><div className="walkthrough-list">{explanation.walkthrough.map((step, index) => <article className="walkthrough-item" key={`${step.heading}-${index}`}><div className="walkthrough-heading"><span className="number-badge">{index + 1}</span><div><h3>{step.heading}</h3></div></div><p className="result-body">{step.explanation}</p><SourceRanges ranges={step.source_ranges} lineCount={lineCount} onSelectRange={jumpToSource} /></article>)}</div></div>}
          {explanation.visual && <div className="result-section"><div className="section-caption">{explanation.visual.kind === "control_flow" ? "CONTROL FLOW MAP" : "CONCEPT MAP"}</div><GraphView visual={explanation.visual} lineCount={lineCount} source={source} onSelectRange={jumpToSource} /></div>}
          {explanation.assumptions.length > 0 && <div className="assumptions"><div className="section-caption">WHAT THE INPUT DOESN’T TELL US</div>{explanation.assumptions.map((assumption, index) => <p key={`${assumption.statement}-${index}`}><strong>{assumption.statement}</strong> {assumption.reason}</p>)}</div>}
          <UnderstandingCheck source={source} explanation={explanation} />
          </div>
          <section className="followup-section" aria-label="Ask a follow-up question">
            <div className="section-caption">KEEP EXPLORING</div>
            <h3>Ask a follow-up</h3>
            <p className="followup-intro">Ask about the source or this explanation. The conversation stays in this page session.</p>
            {followUpMessages.length > 0 && <div className="followup-thread" aria-live="polite" aria-relevant="additions">{followUpMessages.map((message, index) => <article className={`followup-message ${message.role}`} key={`${message.role}-${index}`}>
              <span className="followup-speaker">{message.role === "learner" ? "YOU" : "ANSWER"}</span>
              <p>{message.text}</p>
              {message.role === "learner" && message.selected_text && <blockquote className="followup-selection-quote">{message.selected_text}</blockquote>}
              {message.role === "assistant" && <><SourceRanges ranges={message.source_ranges} lineCount={lineCount} onSelectRange={jumpToSource} />{message.assumptions.length > 0 && <ul>{message.assumptions.map((assumption, assumptionIndex) => <li key={`${assumption.statement}-${assumptionIndex}`}><strong>{assumption.statement}</strong> {assumption.reason}</li>)}</ul>}</>}
            </article>)}</div>}
            {selectedTextForQuestion && <div className="selected-question-context"><span><strong>Asking about:</strong> “{selectedTextForQuestion}”</span><button type="button" onClick={clearSelectedTextContext} aria-label="Remove selected text context">×</button></div>}
            <form className="followup-form" onSubmit={askFollowUp}>
              <label className="visually-hidden" htmlFor="followup-question">Your follow-up question</label>
              <textarea id="followup-question" value={followUpQuestion} onChange={(event) => setFollowUpQuestion(event.target.value)} placeholder={selectedTextForQuestion ? "Ask about the selected passage…" : "What would you like to explore further?"} maxLength={4000} rows={2} />
              <button className="followup-submit" type="submit" disabled={!followUpQuestion.trim() || isAskingFollowUp}>{isAskingFollowUp ? "Thinking…" : "Ask"}<span>{isAskingFollowUp ? "◌" : "→"}</span></button>
            </form>
            {followUpError && <div className="followup-error" role="alert">{followUpError}</div>}
          </section>
          <div className="result-bottom"><span>✳ Keep exploring</span><span>Ask another question above <b>↑</b></span></div>
        </section>}

        {textSelection && explanation && <div className="selection-action" role="toolbar" aria-label="Ask about selected text" style={{ top: textSelection.top, left: textSelection.left }}>
          <span>{textSelection.wasTruncated ? "Selection shortened to 2,000 characters" : `“${textSelection.text.slice(0, 72)}${textSelection.text.length > 72 ? "…" : ""}”`}</span>
          <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={askAboutSelectedText}>Ask about this</button>
        </div>}

        <footer className="page-footer"><span>Made for curious minds.</span><span>Understand <i>·</i> local prototype</span></footer>
      </main>
    </div>
  );
}

async function fileToBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}


export default App;
