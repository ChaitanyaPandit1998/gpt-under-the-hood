# Development Log

Track meaningful changes, validation, limitations, and the next action here. Add a
dated entry for future work; never record secrets or private source material.

## 2026-11-21 - Chapter-local explanation controls

### Changes

- Added this development log and a project workflow rule to keep it current.
- Added Simplify / Go deeper and a visual depth meter, plus a compact Customize
  panel for depth, audience, tone, and focus (concepts, behaviour, worked example).
- PR chapters retain independent preferences and cache variants by all four
  preferences. Changing settings does not automatically start a paid request.
- Code/documentation readers offer section-specific variants in chapter mode.
  The full original source remains available for valid line citations; the model
  is asked to explain only the selected section. Original sections, warnings,
  diagrams, and understanding checks remain available below the variant.
- Added tone/focus validation and passed preferences through drafting, review,
  repair, and final review. Existing clients retain default behaviour.
- Used SVG control icons, a depth graphic, and existing source-linked diagrams
  instead of decorative raster images for factual code behaviour. Labels,
  uncertainty, and evidence are not replaced by ambiguous pictures.

### Validation

- `npm run check`: lint, 130 unit/API tests, TypeScript, and production build passed.
- `PLAYWRIGHT_CHANNEL=chrome npm run test:e2e`: all 26 desktop/mobile browser
  tests passed, including code/docs section variants, chapter isolation, cached
  preferences, citations, cancellation, coverage warnings, covers, and layout.
- Inspected the mobile controls screenshot for readability and layout.
- Existing Vite native-config warnings about extensionless imports remain.
- Browser checks use mocked API responses, not live model/image requests.

### Limitations / Next Steps

- Variants and preferences last only for the current page session.
- Section variants supplement the original; copying and speech still use the
  original explanation. No claim of proven learning improvement is made.
- Next: test a real long PR, evaluate explanation quality, and consider explicit
  variant export and persisted reading progress after validation.

## 2026-11-21 - Post-Push Large-PR Validation

- Reviewed chapter preference isolation, cached variants, cancellation, artwork
  consent, server-side credentials, evidence bounds, and revision checks. No
  blocking issue identified in this focused review; this is not an independent
  security audit or a human assessment of generated explanations.
- Live import of `https://github.com/fastapi/fastapi/pull/9816` passed: 274 unique
  files, complete inventory, 29 chapters; 139 included patches, 2 partial,
  2 unavailable, and 131 excluded by the initial evidence budget.
- Loaded 2,243 patch characters for an excluded test file on demand and verified
  its revision context. No model or image requests were made.
- Explanation quality on a real large PR still needs live generation and human
  assessment. Next: open a review PR and inspect CI before considering merge.

## 2026-11-21 - Live Explanation Quality Check (Merge Held)

- Generated one real chapter from public FastAPI PR #9816 through the local API:
  `Implementation: fastapi - part 4`, containing the `fastapi/params.py` patch
  excerpt (26,045 source characters). Balanced depth, beginner audience, plain
  tone, behaviour focus. No images generated. Request duration: about 189 seconds.
- The pipeline revised the draft but returned `quality_review_status: incomplete`.
  Its final review identified missing diagram branches for the example/examples/
  regex checks, extras selection, v2 pattern fallback, and per-argument unset
  filter. The response exposes this limitation rather than claiming a pass.
- Agent spot-check of cited patch lines supported the main text claims about
  warning conditions, version switching, v1 extras precedence, unset filtering,
  and truncation. This is not an independent human quality assessment, execution
  test of FastAPI, or proof that every generated statement is correct.
- PR #5 remains unmerged: live generation works, but the selected example did not
  pass its final completeness review. No automatic generation retry was made.
- Next: improve focused diagram coverage and add a regression case for these
  multiple independent decisions, then rerun the live check before merging.

## 2026-11-21 - Multi-Decision Diagram Planning

- Added a bounded lexical planning aid for Python-like conditions, assertions,
  short-circuit selections, conditional expressions, and comprehension filters.
  Diff removals, comments, strings, and docstrings are excluded; source line
  coordinates are preserved. Repeated fragments retain occurrence coordinates.
- The aid is explicitly untrusted and non-exhaustive, not a new correctness
  verifier. Drafting, review, repair, and final audit receive the same candidate
  list and instructions to allocate focused views and recheck the whole list.
- Expanded the explanation skill's guidance for sequential warning checks,
  truthiness versus None checks, extras precedence, and per-item unset filtering.
- Added a synthetic field-options evaluation case and eight automated regression
  tests, including preservation of incomplete status after failed final review.
- `npm run check` passed: lint, 138 unit/API tests, TypeScript, production build.
- All 26 desktop/mobile mocked browser regression tests passed.
- Reran the exact saved source and settings from the previously failing public
  FastAPI chapter. The request completed in about 196 seconds with status
  `revised`: it was repaired and passed the final model review. Five views now
  expose warning/skip, extras truthiness, version-specific pattern fallbacks,
  the path assertion, and keep/omit filtering, with separate non-path entry.
- A source-link diagnostic found branching nodes for all 17 candidate locations
  across nine groups and no graph cycles. Agent spot-check supported sequencing,
  extras overwrite behaviour, per-item filtering, and excerpt limitations.
  These checks are not semantic proof or an independent human assessment; one
  successful rerun does not establish general model reliability.
- No images were generated. Next: push the fix and recheck PR #5 for merge;
  continue monitoring additional languages and complex cases after integration.

## Existing Work - Baseline Recorded 2026-11-21

This is a retrospective inventory, not a claim that all work occurred today.

- Coverage-aware PR import, chapter plans, on-demand generation, revision checks,
  additional patch evidence, source citations, request cancellation, and caching.
- Multi-page explanation reader with full-explanation view and print support.
- Optional opt-in isometric chapter covers labelled "Conceptual illustration".
  Only reviewed generic themes are sent to Cloudflare, not chapter/code content.
- Sanitized Cloudflare image benchmarks: decorative covers are promising; the
  small technical-image pilot was unsuitable as evidence of code behaviour.
  See `evals/IMAGE_BEHAVIOUR_ASSESSMENT.md` for limitations.
- Enlarged desktop input card/editor while retaining a compact mobile layout.
- Added API/unit tests and isolated desktop/mobile Playwright regression tests.

Future entries should distinguish code changes, automated validation, live
provider experiments, and human quality assessments.
