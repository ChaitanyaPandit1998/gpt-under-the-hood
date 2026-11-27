import { useEffect, useId, useImperativeHandle, useRef, useState, type FormEvent, type Ref } from "react";
import { buildVisualPrompt, defaultVisualPreferences, sceneVisualGuidance, validateVisualBrief, visualForms, visualInputLimits, type SceneVisualPurpose, type VisualBrief, type VisualPreferences } from "../domain/visual-brief";
import type { Explanation } from "../domain/explanation-types";
import { VisualPreferenceFields } from "./VisualPreferenceFields";
import { VisualBriefImage, type IllustrationPlacement } from "./VisualBriefImage";
import { combineVisualPresets, type VisualPresetSelection } from "../domain/visual-presets";
import { VisualDesignerStep } from "./VisualDesignerStep";
import type { VisualReviewContext } from "./VisualReviewPanel";
import { validateGenerationStage, type GenerationStage } from "../domain/visual-provenance";

export type VisualBriefPanelHandle = { openForDiagram: (index: number) => void };
type Props = { explanation: Explanation; source: string; scope: string; label?: string; placement?: IllustrationPlacement; ref?: Ref<VisualBriefPanelHandle>;
  purpose?: SceneVisualPurpose; initialPreferences?: VisualPreferences; embedded?: boolean };

export function VisualBriefPanel(props: Props) {
  return <BriefForm key={JSON.stringify([props.explanation, props.source, props.scope, props.purpose, props.initialPreferences])} {...props} />;
}

function BriefForm({ explanation, source, scope, label = "Design a visual", placement, ref, purpose, initialPreferences = defaultVisualPreferences, embedded }: Props) {
  const [preferences, setPreferences] = useState(initialPreferences);
  const [presets, setPresets] = useState<VisualPresetSelection>({ direction: [], additionalContext: [] });
  const [brief, setBrief] = useState<VisualBrief | null>(null);
  const [briefRevision, setBriefRevision] = useState(0);
  const [briefGeneration, setBriefGeneration] = useState<GenerationStage>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [showPrompt, setShowPrompt] = useState(false);
  const [promptOverride, setPromptOverride] = useState<string | null>(null);
  const [copyStatus, setCopyStatus] = useState("");
  const [diagramIndex, setDiagramIndex] = useState<number | null>(null);
  const panel = useRef<HTMLDetailsElement>(null);
  const summary = useRef<HTMLElement>(null);
  const pending = useRef<AbortController | null>(null);
  const promptId = useId();
  const promptInputId = useId();
  const formId = useId();
  const sharingId = useId();
  const renderId = useId();
  const sourceAvailable = Boolean(source.trim()) && source.length <= visualInputLimits.source;
  const includeSource = sourceAvailable && preferences.includeSource;
  const direction = combineVisualPresets("direction", presets.direction, preferences.direction);
  const additionalContext = combineVisualPresets("additionalContext", presets.additionalContext, preferences.additionalContext);
  const generatedPrompt = buildVisualPrompt({ ...preferences, direction, additionalContext });
  const promptDraft = promptOverride ?? generatedPrompt;
  const diagram = diagramIndex === null ? undefined : explanation.visuals[diagramIndex];
  const serializedContext = JSON.stringify({
    scope, title: explanation.title, summary: explanation.summary, notes: explanation.visual_note,
    selectedDiagram: diagram, reviewStatus: explanation.quality_review_status, reviewNote: explanation.quality_review_note,
    assumptions: explanation.assumptions, incomplete: explanation.incomplete_items, concepts: explanation.concepts, walkthrough: explanation.walkthrough,
  });
  const context = serializedContext.length > visualInputLimits.context ?
    "Excerpt of explanation (truncated):\n" + serializedContext.slice(0, visualInputLimits.context - 80) : serializedContext;
  useImperativeHandle(ref, () => ({
    openForDiagram(index) {
      if (!panel.current || !explanation.visuals[index]) return;
      if (diagramIndex !== index || preferences.style !== "expressive") {
        pending.current?.abort(); pending.current = null;
        setBusy(false); setError(""); setBrief(null); setDiagramIndex(index);
        setPreferences((previous) => ({ ...previous, style: "expressive" }));
      }
      panel.current.open = true;
      summary.current?.focus();
    },
  }), [diagramIndex, explanation.visuals, preferences.style]);
  useEffect(() => () => {
    pending.current?.abort();
    pending.current = null;
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (pending.current) return;
    if (promptOverride !== null && !promptOverride.trim()) {
      setShowPrompt(true);
      setError("Enter design instructions or reset the prompt to your selections.");
      return;
    }
    if (promptOverride === null && (direction.length > visualInputLimits.direction || additionalContext.length > visualInputLimits.additionalContext)) {
      setError("Shorten your custom notes or deselect a preset. The combined art direction and teaching notes must fit their respective limits.");
      return;
    }
    const controller = new AbortController();
    pending.current = controller;
    setBusy(true); setError("");
    const submittedSource = includeSource ? source : undefined;
    try {
      const response = await fetch("/api/visual-brief", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({
          context,
          preferences: { ...preferences, direction: promptOverride === null ? direction : "",
            additionalContext: promptOverride === null ? additionalContext : "", includeSource: submittedSource !== undefined },
          designPrompt: promptOverride ?? undefined,
          source: submittedSource, purpose,
        }),
      });
      const data = await response.json() as { brief?: unknown; error?: string; generation?: unknown };
      if (!response.ok) throw new Error(data.error || "Could not prepare a visual brief.");
      if (pending.current === controller && !controller.signal.aborted) {
        const validated = validateVisualBrief(data.brief, submittedSource?.split(/\r\n|\r|\n/).length);
        const lineage = data.generation === undefined ? undefined : await validateGenerationStage(data.generation, JSON.stringify(validated));
        if (pending.current !== controller || controller.signal.aborted) return;
        setBrief(validated);
        setBriefGeneration(lineage);
        setBriefRevision((revision) => revision + 1);
      }
    } catch (caught) {
      if (pending.current === controller && !controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Could not prepare a visual brief.");
    } finally {
      if (pending.current === controller) { pending.current = null; setBusy(false); }
    }
  }

  function cancel() {
    pending.current?.abort(); pending.current = null; setBusy(false);
  }

  function editPrompt(next: string | null) {
    setPromptOverride(next); setCopyStatus(""); setBrief(null); setError("");
  }

  async function copyPrompt() {
    try {
      await navigator.clipboard.writeText(promptDraft);
      setCopyStatus("Prompt copied.");
    } catch {
      setCopyStatus("Clipboard unavailable. You can select and copy the prompt manually.");
    }
  }

  const builder = <>
    <p>{purpose ? sceneVisualGuidance[purpose] : "Use the defaults or customize your illustration. The existing diagram stays unchanged."}</p>
    <div className="visual-designer-flow">
    <form id={formId} className="visual-configuration-form" onSubmit={(event) => void submit(event)}>
      {diagram && <div className="chapter-toolbar">
        <span><strong>Illustrate this diagram:</strong> {diagram.title}</span>
        <button type="button" disabled={busy} onClick={() => { setDiagramIndex(null); setBrief(null); setError(""); }}>Use whole explanation</button>
      </div>}
      <VisualPreferenceFields value={preferences} onChange={(next) => { setPreferences(next); setBrief(null); setError(""); setCopyStatus(""); }}
        presets={presets} onPresetsChange={(next) => { setPresets(next); setBrief(null); setError(""); setCopyStatus(""); }}
        disabled={busy} sourceAvailable={sourceAvailable} />
    </form>
      <VisualDesignerStep number="03" title="Generate">
        <div className="chapter-toolbar">
          <button type="button" aria-expanded={showPrompt} aria-controls={promptId} onClick={() => { setShowPrompt(!showPrompt); setCopyStatus(""); }}>
            {showPrompt ? "Hide prompt" : "Preview prompt"}
          </button>
        </div>
        <section id={promptId} hidden={!showPrompt} className="visual-prompt-editor" aria-label="Visual prompt preview">
          <div className="visual-brief-field">
            <label htmlFor={promptInputId}>Visual design prompt</label>
            <textarea id={promptInputId} form={formId} rows={5} maxLength={visualInputLimits.prompt} value={promptDraft} disabled={busy}
              onChange={(event) => editPrompt(event.target.value)} />
          </div>
          <p>Preview only; no model call. Explanation context and opted-in source are sent separately when generating.</p>
          {promptOverride !== null && <p>Using your edited draft. Add exact values here, or reset to apply changes from the cards and fields.</p>}
          <div className="chapter-toolbar">
            <button type="button" disabled={!promptDraft.trim()} onClick={() => void copyPrompt()}>Copy prompt</button>
            {promptOverride !== null && <button type="button" disabled={busy} onClick={() => editPrompt(null)}>Reset to selections</button>}
            <span role="status">{copyStatus}</span>
          </div>
        </section>
        <div className="visual-context-summary" id={sharingId}>
          <strong>What the model receives</strong>
          <ul>
            <li>{purpose === "factual-diagram" ? "This scene's factual mapping, not its narrative. The mapping may quote the original input." : purpose ? "This scene's narrative and factual mapping, which may quote the original input." : "The explanation for this scope, which may quote the original input."}</li>
            <li>{promptOverride === null ? "Your design selections and any extra details you entered." : "Your edited prompt, instead of the current design selections and notes."}</li>
            <li>{includeSource ? "Original source: included with numbered lines." : "Original source: not attached."}</li>
          </ul>
        </div>
        <div className="chapter-toolbar">
          <button className="visual-primary-action" type="submit" form={formId} aria-describedby={sharingId} disabled={busy}>{busy ? "Designing visual..." : brief ? "Regenerate visual brief" : "Generate visual brief"}</button>
          {busy && <button type="button" onClick={cancel}>Cancel visual brief</button>}
          {brief && !busy && <a href={`#${renderId}`}>{brief.disposition === "render" ? "Review brief and render" : "Review text-only brief"}</a>}
        </div>
        {busy && <p role="status">{brief ? "Preparing a replacement brief. The last completed brief is still available below." : "Preparing the design brief."}</p>}
        {error && <p role="alert">{error}</p>}
        {brief && <section className="visual-generation-result" aria-label="Render the image" id={renderId} tabIndex={-1}>
          <h5>{brief.disposition === "render" ? "Render the image" : "Text-only recommendation"}</h5>
          <BriefResult key={briefRevision} brief={brief} placement={placement} disabled={busy} generation={briefGeneration}
            reviewContext={{ explanation: context, scope: diagram ? `${scope}: ${diagram.title}` : scope, audience: preferences.audience, purpose }} />
        </section>}
      </VisualDesignerStep>
    </div>
  </>;
  if (embedded) return <section className="visual-brief-panel scene-visual-builder" aria-label={label}>{builder}</section>;
  return <details ref={panel} className="visual-brief-panel" data-tool-active={busy}>
    <summary ref={summary}>{label} <small className="tool-active-indicator">Working</small></summary>
    {builder}
  </details>;
}

function BriefResult({ brief, placement, disabled, reviewContext, generation }: { brief: VisualBrief; placement?: IllustrationPlacement; disabled: boolean; reviewContext: VisualReviewContext; generation?: GenerationStage }) {
  const form = visualForms.find(([id]) => id === brief.visualType)?.[1] ?? "Text only";
  return <section aria-label="Visual design brief">
    <h5>{form}</h5>
    <p><strong>Takeaway:</strong> {brief.takeaway}</p>
    <details className="visual-builder-options">
      <summary>Read the full brief</summary>
      <p><strong>Composition:</strong> {brief.composition}</p>
      {brief.content.length > 0 && <ul>{brief.content.map((item, index) => <li key={index}>
        <strong>{item.label}</strong> ({item.role}): {item.statement}{item.chapterLines.length > 0 && ` (lines ${item.chapterLines.join(", ")})`}
      </li>)}</ul>}
      {brief.guardrails.length > 0 && <><p><strong>Guardrails:</strong></p><ul>{brief.guardrails.map((rule, index) => <li key={index}>{rule}</li>)}</ul></>}
      <p><strong>Caption:</strong> {brief.caption}</p>
      <p><strong>Text alternative:</strong> {brief.altText}</p>
    </details>
    {brief.disposition === "render" ? <VisualBriefImage brief={brief} placement={placement} disabled={disabled} reviewContext={reviewContext} briefGeneration={generation} /> : <p>This brief recommends text instead of an image. To try another approach, adjust your selections and generate a new brief.</p>}
  </section>;
}
