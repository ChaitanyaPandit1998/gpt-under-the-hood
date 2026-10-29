import { useEffect, useMemo, useRef, useState } from "react";
import { isValidRange, sourceOffsetsForRange, type SourceRange } from "./domain/source-ranges";
import type { ExplanationDepth, LearnerLevel, Mode, Transcription } from "./domain/explanation-types";
import { isThemeMode, type ThemeMode } from "./domain/theme";
import { useExplanationSession } from "./hooks/useExplanationSession";
import { LearningWorkspace } from "./components/LearningWorkspace";
import { ExplanationResult, type TextSelection } from "./components/ExplanationResult";
import { PageDisplayControls } from "./components/PageDisplayControls";

const exampleCode = `function retry<T>(operation: () => Promise<T>, attempts = 3): Promise<T> {
  return operation().catch((error) => {
    if (attempts <= 1) throw error;
    return retry(operation, attempts - 1);
  });
}`;

function App() {
  const [pageZoom, setPageZoom] = useState(100);
  const [pageTheme, setPageTheme] = useState<ThemeMode>(() => {
    try {
      const savedTheme = localStorage.getItem("understand-theme");
      return isThemeMode(savedTheme) ? savedTheme : "light";
    } catch {
      return "light";
    }
  });
  const [mode, setMode] = useState<Mode>("code");
  const [explanationDepth, setExplanationDepth] = useState<ExplanationDepth>("balanced");
  const [learnerLevel, setLearnerLevel] = useState<LearnerLevel>("beginner");
  const [source, setSource] = useState("");
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [textSelection, setTextSelection] = useState<TextSelection | null>(null);
  const [screenshotPreview, setScreenshotPreview] = useState<string | null>(null);
  const [screenshotFile, setScreenshotFile] = useState<File | null>(null);
  const [transcription, setTranscription] = useState<Transcription | null>(null);
  const [needsApproval, setNeedsApproval] = useState(false);
  const [approved, setApproved] = useState(false);
  const screenshotInput = useRef<HTMLInputElement>(null);
  const sourceEditor = useRef<HTMLTextAreaElement>(null);
  const transcriptionRequestId = useRef(0);
  const transcriptionRequestController = useRef<AbortController | null>(null);
  const session = useExplanationSession({ source, mode, depth: explanationDepth, learnerLevel });
  const { explanation, isGenerating, generationPhase, isAskingFollowUp, copyStatus, setCopyStatus, followUpQuestion, setFollowUpQuestion, selectedTextForQuestion, setSelectedTextForQuestion, followUpMessages, followUpError, error, setError, resetExplanation, explainSource, askFollowUp } = session;
  const lineCount = useMemo(() => Math.max(source.split("\n").length, 1), [source]);

  useEffect(() => {
    document.documentElement.dataset.theme = pageTheme;
    try {
      localStorage.setItem("understand-theme", pageTheme);
    } catch {
      // The theme still applies for this session when storage is unavailable.
    }
  }, [pageTheme]);

  useEffect(() => () => {
    if (screenshotPreview) URL.revokeObjectURL(screenshotPreview);
  }, [screenshotPreview]);

  useEffect(() => () => {
    transcriptionRequestId.current += 1;
    transcriptionRequestController.current?.abort();
  }, []);

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
    transcriptionRequestController.current?.abort();
    transcriptionRequestController.current = null;
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
      ...explanation.visuals.map((visual) => `## ${visual.title}\n${visual.nodes.map((node) => `- **${node.label}:** ${node.description}${formatRanges(node.source_ranges) ? ` (${formatRanges(node.source_ranges)})` : ""}`).join("\n")}${visual.edges.length ? `\n\nConnections:\n${visual.edges.map((edge) => `- ${edge.from} → ${edge.to}${edge.label ? ` (${edge.label})` : ""}`).join("\n")}` : ""}`),
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
    transcriptionRequestController.current?.abort();
    transcriptionRequestController.current = null;
    const requestId = ++transcriptionRequestId.current;
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      setIsTranscribing(false);
      setError("Choose a PNG, JPEG, or WebP screenshot.");
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      setIsTranscribing(false);
      setError("Screenshots must be smaller than 8 MB.");
      return;
    }

    resetExplanation();
    const requestController = new AbortController();
    transcriptionRequestController.current = requestController;
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
        signal: requestController.signal,
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
      if (transcriptionRequestController.current === requestController) transcriptionRequestController.current = null;
      if (requestId === transcriptionRequestId.current) setIsTranscribing(false);
    }
  }

  return (
    <div className="app-shell" data-theme={pageTheme}>
      <div className="page-scaled-content" style={{ zoom: pageZoom / 100 }}>
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

          <LearningWorkspace
            mode={mode}
            source={source}
            lineCount={lineCount}
            sourceEditor={sourceEditor}
            screenshotInput={screenshotInput}
            isTranscribing={isTranscribing}
            isGenerating={isGenerating}
            generationPhase={generationPhase}
            screenshotPreview={screenshotPreview}
            transcription={transcription}
            needsApproval={needsApproval}
            approved={approved}
            explanationDepth={explanationDepth}
            learnerLevel={learnerLevel}
            error={error}
            onModeChange={updateMode}
            onSourceChange={updateSource}
            onExample={loadExample}
            onClear={clearInput}
            onScreenshotSelected={(file) => void transcribeScreenshot(file)}
            onApproveTranscription={() => setApproved(true)}
            onDepthChange={setExplanationDepth}
            onLearnerLevelChange={setLearnerLevel}
            onExplain={() => void explainSource()}
            onRetry={() => screenshotFile && needsApproval && !transcription ? void transcribeScreenshot(screenshotFile) : void explainSource()}
          />

          {explanation && <ExplanationResult
            explanation={explanation}
            source={source}
            theme={pageTheme}
            lineCount={lineCount}
            isAskingFollowUp={isAskingFollowUp}
            copyStatus={copyStatus}
            followUpQuestion={followUpQuestion}
            selectedTextForQuestion={selectedTextForQuestion}
            followUpMessages={followUpMessages}
            followUpError={followUpError}
            textSelection={textSelection}
            onCopy={() => void copyExplanation()}
            onSelectRange={jumpToSource}
            onFollowUpQuestionChange={setFollowUpQuestion}
            onAskFollowUp={askFollowUp}
            onClearSelectedTextContext={clearSelectedTextContext}
            onAskAboutSelectedText={askAboutSelectedText}
          />}

          <footer className="page-footer"><span>Made for curious minds.</span><span>Understand <i>·</i> local prototype</span></footer>
        </main>
      </div>
      <PageDisplayControls zoom={pageZoom} theme={pageTheme} onZoomChange={setPageZoom} onThemeChange={setPageTheme} />
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
