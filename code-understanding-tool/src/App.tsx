import { useEffect, useMemo, useRef, useState } from "react";
import { isValidRange, type SourceRange } from "./domain/source-ranges";
import type { ExplanationDepth, LearnerLevel, Mode, Transcription } from "./domain/explanation-types";
import { isThemeMode, type ThemeMode } from "./domain/theme";
import { useExplanationSession } from "./hooks/useExplanationSession";
import { LearningWorkspace } from "./components/LearningWorkspace";
import { ExplanationResult, type TextSelection } from "./components/ExplanationResult";
import { PageDisplayControls } from "./components/PageDisplayControls";
import { PullRequestStudy } from "./components/PullRequestStudy";
import type { PullRequestPlan } from "./domain/pull-request-plan";
import { accessSavedStudy, type ResultReadingState, type SavedStudy } from "./domain/saved-study";
import type { StarterExample } from "./domain/starter-examples";
import { PortableStudyTools } from "./components/PortableStudyTools";

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
  const [pullRequestUrl, setPullRequestUrl] = useState("");
  const [isLoadingPullRequest, setIsLoadingPullRequest] = useState(false);
  const [prPlan, setPrPlan] = useState<PullRequestPlan | null>(null);
  const [savedStudy, setSavedStudy] = useState<SavedStudy>();
  const [restoredStudy, setRestoredStudy] = useState<SavedStudy>();
  const [restoreRevision, setRestoreRevision] = useState(0);
  const [storageBusy, setStorageBusy] = useState(true);
  const [storageMessage, setStorageMessage] = useState("");
  const pullRequestLoadId = useRef(0);
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
    let active = true;
    void accessSavedStudy({ type: "read" }).then((study) => {
      if (active) setSavedStudy(study);
    }).catch(() => {
      if (active) setStorageMessage("Local study storage is unavailable.");
    }).finally(() => {
      if (active) setStorageBusy(false);
    });
    return () => { active = false; };
  }, []);

  async function saveStudy(study: SavedStudy) {
    if (storageBusy) return;
    if (savedStudy && !window.confirm("Replace the saved study on this browser?")) return;
    setStorageBusy(true);
    try {
      await accessSavedStudy({ type: "save", study });
      setSavedStudy(study); setStorageMessage("Study saved in this browser.");
    } catch { setStorageMessage("Could not save this study. Browser storage may be unavailable or full."); }
    finally { setStorageBusy(false); }
  }

  function saveSourceStudy(reading: ResultReadingState) {
    if (explanation && mode !== "pull_request") void saveStudy({ version: 1, source, mode, depth: explanationDepth, level: learnerLevel, explanation, reading });
  }

  function resumeStudy(study: SavedStudy) {
    if ((source.trim() || prPlan || isLoadingPullRequest || isTranscribing) && !window.confirm("Replace the current workspace with the saved study?")) return;
    clearInput();
    setMode(study.mode);
    if (study.mode === "pull_request") {
      setPullRequestUrl(study.plan.url); setPrPlan(study.plan);
    } else {
      setSource(study.source); setExplanationDepth(study.depth); setLearnerLevel(study.level);
      session.setExplanation(study.explanation);
    }
    setRestoredStudy(study); setRestoreRevision((revision) => revision + 1); setStorageMessage("Saved study restored.");
  }

  async function deleteStudy() {
    if (storageBusy) return;
    setStorageBusy(true);
    try {
      await accessSavedStudy({ type: "delete" });
      setSavedStudy(undefined); setStorageMessage("Saved study deleted from this browser.");
    } catch { setStorageMessage("Could not delete the saved study."); }
    finally { setStorageBusy(false); }
  }

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
    pullRequestLoadId.current += 1;
    setIsLoadingPullRequest(false);
    setPrPlan(null);
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
    pullRequestLoadId.current += 1;
    setIsLoadingPullRequest(false);
    setPrPlan(null);
    transcriptionRequestId.current += 1;
    transcriptionRequestController.current?.abort();
    transcriptionRequestController.current = null;
    setIsTranscribing(false);
    resetExplanation();
    setSource(value);
  }

  async function copyExplanation() {
    if (!explanation) return;
    const formatRanges = (ranges: SourceRange[]) => ranges.filter((range) => isValidRange(range, lineCount)).map((range) => `Lines ${range.start_line}${range.end_line !== range.start_line ? `–${range.end_line}` : ""}`).join(", ");
    const sections = [
      `# ${explanation.title}`,
      explanation.summary,
      explanation.concepts.length ? `## Key ideas\n${explanation.concepts.map((concept) => `### ${concept.term}\n${concept.explanation}${formatRanges(concept.source_ranges) ? `\n_Source: ${formatRanges(concept.source_ranges)}_` : ""}`).join("\n\n")}` : "",
      explanation.walkthrough.length ? `## Walkthrough\n${explanation.walkthrough.map((step, index) => `### ${index + 1}. ${step.heading}\n${step.explanation}${formatRanges(step.source_ranges) ? `\n_Source: ${formatRanges(step.source_ranges)}_` : ""}`).join("\n\n")}` : "",
      explanation.incomplete_items?.length ? `## Incomplete code detected\n${explanation.incomplete_items.map((item) => `- **${item.title}:** ${item.explanation}${formatRanges(item.source_ranges) ? ` (${formatRanges(item.source_ranges)})` : ""}`).join("\n")}` : "",
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

  function loadExample(example: StarterExample) {
    clearInput();
    setMode(example.mode);
    if (example.mode === "pull_request") setPullRequestUrl(example.input);
    else setSource(example.input);
  }

  function clearInput() {
    updateSource("");
    setScreenshotPreview(null);
    setScreenshotFile(null);
    setTranscription(null);
    setNeedsApproval(false);
    setApproved(false);
    setPullRequestUrl("");
    if (screenshotInput.current) screenshotInput.current.value = "";
  }

  async function loadPullRequest() {
    const requestId = ++pullRequestLoadId.current;
    setError("");
    setIsLoadingPullRequest(true);
    try {
      const response = await fetch("/api/pull-request-plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: pullRequestUrl }),
      });
      const payload = await response.json() as { plan?: PullRequestPlan; error?: string };
      if (requestId !== pullRequestLoadId.current) return;
      if (!response.ok || !payload.plan?.chapters.length) throw new Error(payload.error || "Could not load this pull request.");
      resetExplanation();
      setMode("pull_request");
      setSource("");
      setPrPlan(payload.plan);
    } catch (caught) {
      if (requestId === pullRequestLoadId.current) setError(caught instanceof Error ? caught.message : "Unable to load this pull request.");
    } finally {
      if (requestId === pullRequestLoadId.current) setIsLoadingPullRequest(false);
    }
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

          <PortableStudyTools study={savedStudy} disabled={storageBusy} onImport={saveStudy} />
          {(savedStudy || storageMessage) && <section className="saved-study-tools" aria-label="Local saved study">
            {savedStudy && <div className="chapter-toolbar">
              <span><strong>Saved on this browser:</strong> {savedStudy.mode === "pull_request" ? savedStudy.plan.title : savedStudy.explanation.title}</span>
              <button type="button" disabled={storageBusy} onClick={() => resumeStudy(savedStudy)}>Resume saved study</button>
              <button type="button" disabled={storageBusy} onClick={() => void deleteStudy()}>Delete saved study</button>
            </div>}
            {storageMessage && <p role="status">{storageMessage}</p>}
          </section>}

          {prPlan ? <PullRequestStudy key={restoreRevision} plan={prPlan} theme={pageTheme} onClose={() => setPrPlan(null)}
            initialReading={restoredStudy?.mode === "pull_request" && restoredStudy.plan === prPlan ? restoredStudy.reading : undefined}
            saving={storageBusy} onSave={(reading) => void saveStudy({ version: 1, mode: "pull_request", plan: prPlan, reading })} /> : <LearningWorkspace
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
            pullRequestUrl={pullRequestUrl}
            isLoadingPullRequest={isLoadingPullRequest}
            explanationDepth={explanationDepth}
            learnerLevel={learnerLevel}
            error={error}
            onModeChange={updateMode}
            onPullRequestUrlChange={setPullRequestUrl}
            onLoadPullRequest={loadPullRequest}
            onSourceChange={updateSource}
            onExample={loadExample}
            onClear={clearInput}
            onScreenshotSelected={(file) => void transcribeScreenshot(file)}
            onApproveTranscription={() => setApproved(true)}
            onDepthChange={setExplanationDepth}
            onLearnerLevelChange={setLearnerLevel}
            onExplain={() => void explainSource()}
            onCancelExplanation={session.invalidatePendingRequests}
            onRetry={() => mode === "pull_request" ? void loadPullRequest() : screenshotFile && needsApproval && !transcription ? void transcribeScreenshot(screenshotFile) : void explainSource()}
          />}

          {!prPlan && explanation && <ExplanationResult
            key={restoreRevision}
            initialReading={restoredStudy && restoredStudy.mode !== "pull_request" && restoredStudy.explanation === explanation && restoredStudy.source === source ? restoredStudy.reading : undefined}
            onSaveStudy={mode === "pull_request" ? undefined : saveSourceStudy}
            savingStudy={storageBusy || isGenerating}
            materialMode={mode}
            storyPreferences={{ depth: explanationDepth, learnerLevel }}
            explanation={explanation}
            source={source}
            theme={pageTheme}
            lineCount={lineCount}
            isAskingFollowUp={isAskingFollowUp}
            isRegenerating={isGenerating}
            copyStatus={copyStatus}
            followUpQuestion={followUpQuestion}
            selectedTextForQuestion={selectedTextForQuestion}
            followUpMessages={followUpMessages}
            followUpError={followUpError}
            textSelection={textSelection}
            onCopy={() => void copyExplanation()}
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
