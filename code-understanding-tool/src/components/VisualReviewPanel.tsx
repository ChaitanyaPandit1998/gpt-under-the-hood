import { useEffect, useId, useRef, useState } from "react";
import type { VisualBrief, VisualPreferences, SceneVisualPurpose } from "../domain/visual-brief";
import type { VisualScene } from "../domain/visual-scene";
import type { RenderedIllustration } from "../domain/saved-study";
import { createVisualReviewDraft, reviewAudienceChoices, reviewLocations, type VisualReviewDraft } from "../domain/visual-review-draft";
import { visualEvaluationVerdict } from "../domain/visual-evaluation";
import { ChoiceCards } from "./ChoiceCards";
import { requestVisualReview } from "./visual-review-client";
import type { SavedVisualReview } from "../domain/visual-provenance";

export type VisualReviewContext = { explanation: string; scope: string; audience: VisualPreferences["audience"]; purpose?: SceneVisualPurpose };
const stageLabels = { brief: "Draft quality", plan: "Drawing fidelity", raster: "Image readability", alignment: "Factual alignment", audience: "Audience suitability" };
const statusLabels = { pass: "Checks passed", revise: "Needs revision", fail: "Blocking issue", uncertain: "Uncertain", "not-run": "Not reviewed" };

export function VisualReviewPanel({ brief, scene, image, context, disabled, onComplete }: {
  brief: VisualBrief; scene: VisualScene; image: RenderedIllustration; context: VisualReviewContext; disabled?: boolean;
  onComplete?: (review: SavedVisualReview) => void;
}) {
  const [draft, setDraft] = useState(() => createVisualReviewDraft(brief, context.audience, context.purpose));
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [completed, setCompleted] = useState<(Awaited<ReturnType<typeof requestVisualReview>> & { snapshot: string; png: Blob })>();
  const pending = useRef<AbortController | null>(null);
  const id = useId();
  const snapshot = JSON.stringify({ draft, explanation: context.explanation, brief, scene, caption: image.caption, resolution: image.resolution });
  const stale = completed && (completed.snapshot !== snapshot || completed.png !== image.png);
  const locked = busy || disabled;
  useEffect(() => () => { pending.current?.abort(); pending.current = null; }, []);

  function edit(next: VisualReviewDraft) { setDraft(next); setConfirmed(false); setError(""); setNotice(""); }
  function requirement(index: number, change: Partial<VisualReviewDraft["requirements"][number]>) {
    edit({ ...draft, requirements: draft.requirements.map((item, i) => i === index ? { ...item, ...change } : item) });
  }
  function cancel() {
    if (!pending.current) return;
    pending.current.abort(); pending.current = null; setBusy(false); setNotice("Review cancelled. Your image is unchanged.");
  }
  async function review() {
    if (pending.current || !confirmed || disabled) return;
    const controller = new AbortController(); pending.current = controller;
    setBusy(true); setError(""); setNotice("");
    try {
      const result = await requestVisualReview({ draft, explanation: context.explanation, brief, scene, image }, controller.signal);
      if (pending.current === controller && !controller.signal.aborted) {
        setCompleted({ ...result, snapshot, png: image.png });
        onComplete?.({ ...result, reviewedAt: new Date().toISOString() });
      }
    } catch (caught) {
      if (pending.current === controller && !controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Could not review the image.");
    } finally {
      if (pending.current === controller) { pending.current = null; setBusy(false); }
    }
  }

  return <details className="visual-review-panel visual-builder-options" data-tool-active={busy} onToggle={(event) => { if (!event.currentTarget.open) cancel(); }}>
    <summary>Review this visual <small>Optional model review</small></summary>
    <p><strong>Scope:</strong> {context.scope}</p>
    <p>Check this rendered version, not a different accepted image. Review does not replace, approve or repair your illustration.</p>
    <p><strong>Caption being reviewed:</strong> {image.caption}</p>
    <p className="visual-context-summary">A checklist drafted from the brief is not independent evidence. Edit it to match this scene and audience. On request, the configured model receives this PNG, caption, explanation context, brief, drawing plan and checklist. No separate original-source attachment is sent, but the context may quote it.</p>
    <form onSubmit={(event) => { event.preventDefault(); void review(); }}>
      <ChoiceCards label="Review audience" value={draft.audience} choices={reviewAudienceChoices} disabled={locked}
        onChange={(audience) => edit({ ...draft, audience: audience as VisualReviewDraft["audience"] })} />
      <ChoiceCards label="Role of this visual" value={draft.purpose} disabled={locked} choices={[
        { value: "story-illustration", title: "Story illustration", description: "An explicitly fictional analogy supported by its caption." },
        { value: "factual-diagram", title: "Factual diagram", description: "Technical relationships grounded in the explanation." },
      ]} onChange={(purpose) => edit({ ...draft, purpose })} />
      <div className="visual-brief-field"><label htmlFor={`${id}-goal`}>What should this visual teach?</label>
        <textarea id={`${id}-goal`} rows={3} maxLength={2000} required value={draft.teachingGoal} disabled={locked} onChange={(event) => edit({ ...draft, teachingGoal: event.target.value })} /></div>
      <div className="visual-brief-field"><label htmlFor={`${id}-objective`}>What should this audience understand?</label>
        <textarea id={`${id}-objective`} rows={3} maxLength={2000} required value={draft.audienceObjective} disabled={locked} onChange={(event) => edit({ ...draft, audienceObjective: event.target.value })} /></div>
      <fieldset className="visual-review-checklist" disabled={locked}>
        <legend>Required content checklist</legend>
        <p>Include only what this visual should convey. Choose where each requirement belongs; a story image need not carry every technical detail itself.</p>
        {draft.requirements.map((item, index) => <div className="visual-review-requirement" key={item.id}>
          <label className="visual-brief-consent"><input type="checkbox" checked={item.enabled} onChange={(event) => requirement(index, { enabled: event.target.checked })} />Include requirement {index + 1}</label>
          <div className="visual-brief-field"><label htmlFor={`${id}-requirement-${index}`}>Requirement {index + 1}</label>
            <textarea id={`${id}-requirement-${index}`} rows={3} maxLength={1900} required={item.enabled} value={item.statement} onChange={(event) => requirement(index, { statement: event.target.value })} /></div>
          <div className="visual-brief-field"><label htmlFor={`${id}-location-${index}`}>Where requirement {index + 1} belongs</label>
            <select id={`${id}-location-${index}`} value={item.location} onChange={(event) => requirement(index, { location: event.target.value as typeof item.location })}>
              {reviewLocations.map((location) => <option key={location}>{location}</option>)}
            </select></div>
          <small>{item.evidenceReference}</small>
        </div>)}
        <div className="chapter-toolbar"><button type="button" disabled={draft.requirements.length >= 40} onClick={() => edit({ ...draft, requirements: [...draft.requirements,
          { id: crypto.randomUUID(), kind: "claim", statement: "", evidence: "background", evidenceReference: "User-provided requirement; verify against the supplied explanation.", enabled: true, location: "Image and caption" }] })}>Add requirement</button>
        </div>
      </fieldset>
      <label className="visual-brief-consent"><input type="checkbox" checked={confirmed} disabled={locked} onChange={(event) => setConfirmed(event.target.checked)} />I confirm this checklist and agree to send these materials for model review.</label>
      <div className="chapter-toolbar">
        <button className="visual-primary-action" type="submit" disabled={locked || !confirmed || !draft.audience || !draft.requirements.some((item) => item.enabled)}>{busy ? "Reviewing visual..." : "Review visual"}</button>
        {busy && <button type="button" onClick={cancel}>Cancel visual review</button>}
      </div>
      {busy && <p role="status">Checking the image, caption and explanation. Your current illustration stays unchanged.</p>}
      {notice && <p role="status">{notice}</p>}
      {error && <p role="alert">{error}</p>}
    </form>
    {completed && <section aria-label="Visual review results" className="visual-review-results">
      <h5>{stale ? "Previous review (outdated)" : statusLabels[visualEvaluationVerdict(completed.report, completed.contract)]}</h5>
      {stale && <p role="status">The checklist or materials changed. These findings do not apply to the current draft. Confirm and review again.</p>}
      <p>Model assessment, not proof of correctness or human approval. Results apply only to the reviewed materials and confirmed checklist. Up to five recent reports, within the storage budget, are kept when you explicitly save the study; candidate reports are saved only if you accept that image.</p>
      {completed.report.checks.map((check) => <details className="visual-review-stage" key={check.stage}>
        <summary>{stageLabels[check.stage]} <span>{statusLabels[check.status]}</span></summary>
        <p>{check.summary}</p>
        {check.findings.map((finding, index) => <div className="visual-review-finding" key={index}>
          <strong>{finding.severity}: {finding.message}</strong><p>{finding.evidence}</p>
        </div>)}
        <small>Reviewer: {check.reviewer.name}. Version: {check.reviewer.version}.</small>
      </details>)}
    </section>}
  </details>;
}
