import type { ChangeEvent, RefObject } from "react";
import type { ExplanationDepth, LearnerLevel, Mode, Transcription } from "../domain/explanation-types";

type Props = {
  mode: Mode;
  source: string;
  lineCount: number;
  sourceEditor: RefObject<HTMLTextAreaElement | null>;
  screenshotInput: RefObject<HTMLInputElement | null>;
  isTranscribing: boolean;
  isGenerating: boolean;
  screenshotPreview: string | null;
  transcription: Transcription | null;
  needsApproval: boolean;
  approved: boolean;
  explanationDepth: ExplanationDepth;
  learnerLevel: LearnerLevel;
  error: string;
  onModeChange: (mode: Mode) => void;
  onSourceChange: (source: string) => void;
  onExample: () => void;
  onClear: () => void;
  onScreenshotSelected: (file: File) => void;
  onApproveTranscription: () => void;
  onDepthChange: (depth: ExplanationDepth) => void;
  onLearnerLevelChange: (level: LearnerLevel) => void;
  onExplain: () => void;
  onRetry: () => void;
};

export function LearningWorkspace({
  mode, source, lineCount, sourceEditor, screenshotInput, isTranscribing, isGenerating,
  screenshotPreview, transcription, needsApproval, approved, explanationDepth, learnerLevel, error,
  onModeChange, onSourceChange, onExample, onClear, onScreenshotSelected, onApproveTranscription,
  onDepthChange, onLearnerLevelChange, onExplain, onRetry,
}: Props) {
  return <section className="workspace" aria-label="Learning workspace">
      <div className="workspace-heading">
        <div>
          <div className="step-label"><span>01</span> START WITH SOMETHING</div>
          <h2>What would you like to understand?</h2>
        </div>
        <button className="example-link" onClick={onExample}>Try an example <span>↗</span></button>
      </div>

      <div className="input-card">
        <div className="input-toolbar">
          <div className="segmented" role="tablist" aria-label="Input type">
            <button role="tab" id="tab-code" aria-controls="source-input" aria-selected={mode === "code"} tabIndex={mode === "code" ? 0 : -1} className={mode === "code" ? "active" : ""} onKeyDown={(event) => { if (event.key === "ArrowRight" || event.key === "ArrowLeft") { event.preventDefault(); onModeChange(mode === "code" ? "docs" : "code"); document.getElementById(mode === "code" ? "tab-docs" : "tab-code")?.focus(); } }} onClick={() => onModeChange("code")}>⌘ <span>Code</span></button>
            <button role="tab" id="tab-docs" aria-controls="source-input" aria-selected={mode === "docs"} tabIndex={mode === "docs" ? 0 : -1} className={mode === "docs" ? "active" : ""} onKeyDown={(event) => { if (event.key === "ArrowRight" || event.key === "ArrowLeft") { event.preventDefault(); onModeChange(mode === "code" ? "docs" : "code"); document.getElementById(mode === "code" ? "tab-docs" : "tab-code")?.focus(); } }} onClick={() => onModeChange("docs")}>▤ <span>Documentation</span></button>
          </div>
          <div className="input-tools"><span className="language-pill">{mode === "code" ? "Language detected automatically" : "Markdown or plain text"}</span><span className="toolbar-divider" /><button className="tool-button" onClick={onClear} disabled={isTranscribing || isGenerating || (!source && !screenshotPreview)} aria-label="Clear input" title="Clear input">⌫</button></div>
        </div>

        <div className={`editor-wrap ${source || (screenshotPreview && !isTranscribing) ? "has-source" : ""}`}>
          {mode === "code" && (source || (screenshotPreview && !isTranscribing)) && <div className="line-numbers" aria-hidden="true">{Array.from({ length: lineCount }, (_, i) => <span key={i}>{i + 1}</span>)}</div>}
          <textarea id="source-input" ref={sourceEditor} aria-label={mode === "code" ? "Code input" : "Documentation input"} spellCheck={false} value={source} onChange={(event) => onSourceChange(event.target.value)} placeholder={mode === "code" ? "Paste a code snippet here…" : "Paste documentation here…"} className={mode === "code" ? "source-editor code-editor" : "source-editor docs-editor"} disabled={isTranscribing} />
          {!source && (!screenshotPreview || isTranscribing) && <div className="editor-empty" aria-hidden="true">
            {isTranscribing ? <><div className="empty-icon">◌</div><strong>Reading the screenshot…</strong><span>The model is extracting visible code or documentation.</span></> : <><div className="empty-icon">{mode === "code" ? "{ }" : "Aa"}</div><strong>{mode === "code" ? "Paste a code snippet to get started" : "Paste a passage or document to explore"}</strong><span>{mode === "code" ? "Any language is welcome. A few lines are enough." : "Technical docs, API references, or concepts you want to unpack."}</span></>}
          </div>}
        </div>

        <div className="input-footer">
          <span>{source ? `${source.length.toLocaleString()} characters · ${lineCount} lines` : "Your input stays in this local session"}</span>
          <input ref={screenshotInput} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={(event: ChangeEvent<HTMLInputElement>) => { const file = event.target.files?.[0]; if (file) onScreenshotSelected(file); event.currentTarget.value = ""; }} />
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
        {needsApproval && !isTranscribing && <button className="approve-transcription" onClick={onApproveTranscription} disabled={approved || !source.trim()}>{approved ? "Text reviewed ✓" : transcription ? "I reviewed this text" : "Use the text I entered"}</button>}
      </div>}

      <div className="explanation-preferences" aria-label="Explanation preferences">
        <label>Depth
          <select value={explanationDepth} onChange={(event) => onDepthChange(event.target.value as ExplanationDepth)}>
            <option value="overview">Quick overview</option>
            <option value="balanced">Balanced</option>
            <option value="detailed">Detailed walkthrough</option>
          </select>
        </label>
        <label>Explain for
          <select value={learnerLevel} onChange={(event) => onLearnerLevelChange(event.target.value as LearnerLevel)}>
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </label>
      </div>

      <div className="action-row">
        <span className="privacy-note"><span>◈</span> Nothing is saved automatically</span>
        <button className="generate-button" disabled={!source.trim() || isGenerating || isTranscribing || (needsApproval && !approved)} onClick={onExplain}>{isTranscribing ? "Reading screenshot…" : isGenerating ? "Working with Codex…" : "Help me understand"} <span>{isGenerating || isTranscribing ? "◌" : "→"}</span></button>
      </div>
      {error && <div className="error-message" role="alert"><span>!</span><p>{error}</p><button onClick={onRetry} disabled={isGenerating || isTranscribing}>Try again</button></div>}
  </section>;
}
