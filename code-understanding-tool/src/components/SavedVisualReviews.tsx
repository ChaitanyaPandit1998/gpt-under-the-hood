import { useEffect, useState } from "react";
import type { RenderedIllustration } from "../domain/saved-study";
import { appendVisualReview, validateVisualProvenance, type VisualProvenance } from "../domain/visual-provenance";
import { visualEvaluationVerdict } from "../domain/visual-evaluation";
import { VisualReviewPanel } from "./VisualReviewPanel";

export function SavedVisualReviews({ image, onChange }: { image: RenderedIllustration; onChange?: (image: RenderedIllustration) => void }) {
  const [checked, setChecked] = useState<{ image: RenderedIllustration; provenance?: VisualProvenance }>();
  const [reviewOpen, setReviewOpen] = useState(false);
  useEffect(() => {
    let active = true;
    if (image.provenance) void validateVisualProvenance(image.provenance, image).then(
      provenance => { if (active) setChecked({ image, provenance }); },
      () => { if (active) setChecked({ image }); },
    );
    return () => { active = false; };
  }, [image]);
  if (!image.provenance) return null;
  if (checked?.image !== image) return <p role="status">Checking saved review bindings...</p>;
  if (!checked.provenance) return <p role="alert">Saved review metadata does not match this image. No saved verdict is shown.</p>;
  return <details className="saved-visual-reviews visual-builder-options">
    <summary>Review history ({checked.provenance.reviews.length}) <small className="tool-active-indicator">Working</small></summary>
    <p>Historical assessments, not approval. Hashes check artifact consistency, not reviewer identity or factual correctness. Imported reports may be edited. Opening history makes no model call; use Save study locally to retain it after closing.</p>
    <details className="visual-builder-options">
      <summary>Generation skills and profiles</summary>
      <p>Recorded instruction hashes bind to the brief/plan, not proof of authorship. Missing metadata means unknown, not the current installed skill version.</p>
      {(["brief", "plan"] as const).map(stage => <section key={stage}><h5>{stage === "brief" ? "Visual brief" : "Drawing plan"}</h5>
        {checked.provenance?.generation?.[stage] ? <ul>{checked.provenance.generation[stage].instructions.map(stamp =>
          <li key={stamp.id}>{stamp.id} · version {stamp.version} · SHA256: {stamp.sha256}</li>)}</ul> : <p>Not recorded for this artifact.</p>}
      </section>)}
    </details>
    {onChange && <>
      <div className="chapter-toolbar"><button type="button" aria-expanded={reviewOpen} onClick={() => setReviewOpen(open => !open)}>
        {reviewOpen ? "Close re-review controls" : "Review this image again"}
      </button></div>
      {reviewOpen && <p>Re-review uses the recorded explanation context, brief and drawing plan, not later chapter edits. Confirm the checklist before sending these materials.</p>}
      {reviewOpen && <VisualReviewPanel image={image} brief={checked.provenance.brief} scene={checked.provenance.scene}
        context={checked.provenance.context} onComplete={review => {
          setReviewOpen(false);
          onChange({ ...image, provenance: appendVisualReview(checked.provenance!, review) });
        }} />}
    </>}
    {[...checked.provenance.reviews].reverse().map((entry, index) => <details key={`${entry.contract.id}-${index}`} className="visual-review-stage">
      <summary>{entry.reviewedAt} · Historical result: {visualEvaluationVerdict(entry.report, entry.contract)}</summary>
      <p>{entry.contract.audience} · {entry.contract.purpose} · {entry.contract.teachingGoal}</p>
      {entry.report.checks.map(check => <section key={check.stage}>
        <h5>{check.stage}: {check.status}</h5><p>{check.summary}</p>
        {check.findings.map((finding, i) => <p key={i}><strong>{finding.severity}: {finding.message}</strong> {finding.evidence}</p>)}
        <small>Recorded reviewer: {check.reviewer.name}. Version: {check.reviewer.version}.</small>
      </section>)}
    </details>)}
  </details>;
}
