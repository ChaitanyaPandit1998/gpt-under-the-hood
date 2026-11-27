# Story and Visual Skills Plan

Status: Story skills, shared learning profiles, visual direction and VectorCraft
composition and independent visual-review wiring implemented. Optional saved
image provenance/history, visual-generation skill/profile lineage and restored
image re-review are implemented. Broader end-to-end validation remains pending.

## Implemented Story Wiring

- `skills/plan-chapter-story/SKILL.md`, `skills/write-chapter-story/SKILL.md`
  and `skills/review-chapter-story/SKILL.md` contain the reusable story guidance.
- `skills/learning-profiles/audience/` and `skills/learning-profiles/depth/`
  hold three profiles each. Audience and depth remain independent.
- `src/server/application-instructions.ts` loads only allowlisted app-owned
  files from the server's configured root. Requests cannot supply file paths.
- Planning and writing share the draft/repair call. Review loads its own skill
  plus the same selected profiles, not the generation skills. Existing JSON
  responses, source validation and the one-repair/final-review limit remain.
- Instruction IDs, version 1 and exact content hashes accompany loaded prompt
  sections. These are internal metadata, not yet persisted with saved stories.
- Build, targeted lint and eight focused loader/service tests passed. Full
  regressions and live story quality checks remain deferred. The bundled skill
  format validator could not run because Python's PyYAML dependency is absent.

## Capability Brief

Related UI work: [Reader UI Simplification Plan](reader-ui-simplification-plan.md).

Delivery order: [Prioritized Roadmap](prioritized-roadmap.md). Reuse the existing
`skills/design-explanation-visual/SKILL.md` for visual direction rather than
creating a duplicate skill.

1. Create six reusable skills rather than separate skills for every complexity level.
2. **Story planning:** Choose the teaching goal, analogy, outline, and analogy limits.
3. **Story writing:** Generate connected scenes with consistent characters and a clear takeaway.
4. **Story review:** Check factual accuracy, continuity, citations, and audience suitability.
5. **Visual direction:** Choose what to illustrate and create the visual brief.
6. **Visual composition:** Produce the layout, shapes, labels, and captions for VectorCraft.
7. **Visual review:** Inspect the rendered image for correctness, readability, and explanation alignment.
8. Use shared audience profiles: Beginner, Intermediate, Advanced; depth profiles: Overview, Balanced, Detailed.
9. Let users select story style and visual preferences, keeping every skill topic-independent.
10. **Next step:** Run the deferred integrated validation, including live quality checks; do not equate implementation with release readiness.

## Implemented Visual Wiring

- The existing `skills/design-explanation-visual/SKILL.md` is reused through the
  instruction allowlist, not copied into a second direction skill.
- Explicit visual audiences load the shared audience profile. Match explanation
  leaves audience inference to the supplied context, not a hidden beginner default.
- `skills/compose-vectorcraft-visual/SKILL.md` guides composition with supported
  text, rectangle and path primitives. Its compact-scene reference loads only
  when compact planning is enabled. It does not add new renderer capabilities.
- Canvas bounds, schema selection/validation, cancellation, the one structural
  repair limit and renderer invocation remain in application code. Numerical
  canvas dimensions come from the canonical scene definition.
- APIs return optional version/hash stamps from the resources actually loaded
  into each brief/plan request, bound to the exact returned artifact. New images
  retain these records, including explicitly loaded audience and compact profiles.
  Older responses/images remain supported with unknown lineage, never relabelled
  with the current installed skill version. No instruction text is duplicated
  into the metadata.
- Restored images with provenance can be reviewed again using their recorded
  context, explicit checklist confirmation and no automatic acceptance. Metadata
  updates merge into the same image only; late results cannot replace a newer PNG.
- Build, targeted lint and 40 focused checks passed for the completion slice;
  actual generated image quality and end-to-end behavior remain unverified.
