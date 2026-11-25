# Prioritized Reader and Skills Roadmap

Status: Priorities 1-6 are implemented. Priority 7 is in progress.

Integrated validation: lint, build, 345 unit/API tests and 88 mocked desktop/
mobile browser tests passed. Coverage includes legacy restore, scoped tools,
the three-stage builder, review consent, lineage, saved-image re-review and
out-of-order review completion. Re-review uses recorded context; old images
without provenance cannot reconstruct missing data.

The external Intermediate and Advanced Balanced pilot passed live story
generation, VectorCraft illustration rendering and save/restore. Mobile
now shows readable labels and a full-screen preview. Drawing planning
still dominates latency. Compact encoding gave mixed results and remains
opt-in. A separate bounded-complexity pilot was faster on repeated runs
for one technical diagram and one story scene without apparent loss of
meaning. The app default remains unchanged pending broader evaluation.
The wider audience/depth matrix, broader factual-diagram evaluation, live
review calibration and full release acceptance remain open. A ResizeObserver
warning was observed on diagram paths despite passing browser tests. The
compact mobile chapter picker is a separate follow-up. Bundled skill-format
validation remains blocked by missing PyYAML.

Scope: consolidate the agreed reader simplification and story/visual skills
plans. Implement priorities 1-6 first, then perform rigorous integrated testing
in priority 7. Completion gates below describe intended acceptance criteria,
not claims that the changes have already been fully tested.

| Order | Work | Completion gate |
| --- | --- | --- |
| 1 | Chapter-only reading | Audit chapter coverage and full-view assets; remove the Chapters / Full explanation switch. Older saved/imported studies retain content, stories, refinements, images and reading position. Keep whole-study export and read-aloud. |
| 2 | Reduce UI clutter | Prioritize chapter text and inline images. Use a compact scoped toolbar, shorter helper text and collapsed advanced controls. Tool drafts, active status, consent, keyboard access and mobile layout remain intact. |
| 3 | Simplify the visual builder | Present What to teach, Look and feel, then Generate with defaults and an optional prompt preview. Preserve existing brief/plan/render contracts, cancellation and retry; opening controls makes no model call. |
| 4 | Extract reusable story skills | Introduce app-owned story-planning, story-writing and story-review instructions with shared audience/depth profiles and an allowlisted backend loader. Planning may share the writing call; do not add calls merely to match file boundaries. |
| 5 | Reuse and extend visual skills | Reuse the existing design-explanation-visual skill for visual direction. Add VectorCraft composition guidance aligned with supported scene primitives. Rendering stays in the validated adapter; generated commands or paths are never executed directly. |
| 6 | Consolidate reviews and provenance | Extract visual-review guidance without weakening deterministic checks or opt-in consent. Record skill/profile versions and artifact bindings; persist review history only through explicit study saving with backward-compatible validation. No automatic acceptance. |
| 7 | End-to-end evaluation and tuning | Compare audience/depth combinations, factual mappings, story continuity, actual PNG alignment, readability, latency and model-call cost. Use generic committed fixtures and an external Riemann Hypothesis pilot; separate mocked, live-provider and manual evidence. |

## Delivery Rules

- Preserve the distinction between audience (Beginner, Intermediate, Advanced)
  and depth (Overview, Balanced, Detailed).
- Keep skills topic-independent and user preferences configurable. Do not
  duplicate a complete skill for every audience/depth combination.
- During implementation, use lightweight build/type checks and narrow checks
  for high-risk changes such as saved-study compatibility or consent handling.
  Do not run the full end-to-end suite or live-provider pilots after each slice.
- Write or update regression coverage alongside changed behavior, but defer
  broad execution until priorities 1-6 are integrated. Track unrun checks and
  known gaps explicitly; implemented does not mean fully validated.
- Priority 7 runs rigorous unit/API and desktop/mobile browser regressions,
  saved/imported-study compatibility, accessibility and cancellation checks,
  plus controlled live visual/story evaluation, latency and cost comparisons.
  Resolve blocking findings before marking the work release-ready.
- Keep independent review separate from generation instructions. Existing
  schema, privacy, cancellation and rendering safeguards stay in application code.
- Update DEVLOG.md with actual outcomes, limitations and validation results.

## Supporting Plans

- [Reader UI Simplification](reader-ui-simplification-plan.md)
- [Story and Visual Skills](story-visual-skills-plan.md)
- [Visual Validation](visual-validation-plan.md)
- Existing visual-direction skill: `skills/design-explanation-visual/SKILL.md`.

Next step: test the bounded planner on unseen multi-panel illustrations
and exact-scale technical figures before offering a quick planning mode.
