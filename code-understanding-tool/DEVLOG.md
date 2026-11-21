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
