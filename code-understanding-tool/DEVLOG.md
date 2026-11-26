# Development Log

Track meaningful changes, validation, limitations, and the next action here. Add a
dated entry for future work; never record secrets or private source material.

## 2026-11-26 - Reading-first layout and simpler tools

- Grouped local save, PDF preview, and document export under a closed-by-default
  Save & export control. Moved save help into the group. Chapter and PR
  tool scopes are unchanged.
- Hide idle status badges; active export and visual work remain visible, and
  opening another tool does not hide an unresolved image comparison.
- Put chapter text and accepted chapter illustrations below Direct/Story
  and above chapter tools. Move whole-explanation actions below the reader
  while retaining review and incomplete-code notices above it.
- Validation: lint, build, 345 unit/API tests, and 98 mocked desktop and mobile
  browser tests passed, including keyboard access, save/restore, PDF preview,
  export cancellation, and the new reading order. No live provider calls
  were needed.
- A separate Chromium walkthrough on commit a9d6159 used a synthetic
  12-chapter explanation with a 600-word overview. Text was visible without
  scrolling from the explanation title. Chapter navigation, chapter and
  whole-study PDF previews, and local save worked at 1440 x 900 and 390 x 844
  without horizontal overflow or another explanation request.
- The whole-study toolbar started 1345px and 2545px below the title on
  desktop and mobile. This is a layout measurement, not a usability study.
- Added a Study tools jump link beside the Code and Documentation result
  title. It scrolls and focuses the existing toolbar without opening a panel.
  Lint, build, 345 unit/API tests, and 28 targeted mocked browser tests passed,
  including keyboard order and desktop/mobile viewport checks.
- Added visible paragraph spacing to direct explanations, stories, and
  refined sections. Authored breaks are preserved; long unbroken prose
  is split at sentence boundaries where supported, without changing the
  saved, copied, or read-aloud text.
- Validation: lint, build, 348 unit/API tests, and 102 mocked browser tests
  passed. Captured the same long mock overview at desktop and mobile widths.
  The mobile outline checks now measure clipping in a single browser frame;
  mobile chapter navigation itself is unchanged.
- New Direct and Story responses request selective bold for key terms and
  italics for contrasts and caveats. The reader, refined sections, and
  browser PDF preview render supported emphasis as text formatting, not HTML.
  Read-aloud and WordCraft prose omit the markers; existing unmarked studies
  are unchanged.
- Validation: lint, build, 350 unit/API tests, and 104 mocked desktop and
  mobile browser tests passed before the live pilot.
- Live Riemann Hypothesis pilot: the first Direct overview omitted emphasis
  and the first Story request returned a 502. Clarified the prose prompts and
  enabled italic synthesis only for reader emphasis. The final live Direct
  and Story responses both bolded two key terms and italicized one word;
  verified in screenshots. The focused desktop and mobile regression passed.
- Next: review and push the reading updates. The mobile chapter picker
  remains deferred.

## 2026-11-26 - Integrated checks and two live evaluation cases

- On revision `45f4f23`, lint, the production build, 345 unit/API tests
  and 98 mocked Chromium tests across desktop and mobile passed. The
  existing ResizeObserver warning still appeared in diagram workflows.
- Ran two new synthetic fixtures through the actual local API with Codex
  and VectorCraft: Beginner/Overview queue priority, and Advanced/Detailed
  slope versus intercept. Both explanation reviews passed; both stories
  returned three scenes. Both briefs and plans returned skill lineage
  and rendered as 1920 x 1080 PNGs.
- Manual PNG inspection confirmed the urgent/regular/empty branches and
  the three supplied coordinates without extrapolation. Plan inspection
  confirmed equal adjacent intervals on each axis and aligned endpoints.
  Planning took 30.11s and 68.28s; rendering took 0.66s and 0.59s.
- This is a two-case API pilot, not a full live browser matrix, independent
  reviewer calibration, bounded-planner comparison, or cost benchmark.
  Raw outputs and the pilot harness stayed outside the repository.
- Next: compare default and bounded planning on unseen multi-panel and
  exact-scale cases; continue the audience/depth and reviewer evaluations.

## 2026-11-24 - Architecture diagram refinement

- Reworked the VectorCraft layout into a numbered generation path, with
  browser storage, local output adapters, and outbound requests on separate
  branches. Updated the architecture guide's reading order and alt text.
- Set the headline in Georgia and the labels in Avenir Next. Shortened
  several labels to fit the wider lettering without moving the request paths.
- Labeled the browser-facing return and external payloads, and added a
  companion VectorCraft diagram for explanation review and PNG generation.
- Added three pull quotes highlighting limits of model review, the
  distinction between recovery and saves, and external model requests.
- Validation: inspected the rendered PNG for label fit and arrow routing,
  and checked both SVGs and their documentation links. No runtime code changed.
- Next: review the updated SVG in the rendered GitHub documentation.

## 2026-11-23 - Application architecture documentation

- Rendered a VectorCraft system diagram and added the SVG to the
  architecture guide. It distinguishes browser storage, the loopback API,
  local rendering tools, and requests to external services.
- Updated the README and implementation plan to point readers to current
  flows. The guide distinguishes refresh recovery from explicit saves.
  No runtime code changed.
- Validation: visually inspected the VectorCraft PNG and checked SVG XML,
  Markdown link targets, and the SVG's lack of active references.
- Next: review the diagram with the architecture docs before publishing.

## 2026-11-23 - Temporary explanation recovery follow-up

- Validate incomplete-code findings, source ranges, and review metadata
  before restoring a completed explanation. Invalid snapshots are discarded
  so the editor remains usable instead of crashing on refresh.
- Validation: lint, build, 345 unit/API tests, and all 10 targeted mocked
  browser tests passed. The browser tests cover malformed findings, invalid
  citations, and non-text review notes on both desktop and mobile.
- Next: push the follow-up fix for review before merging.

## 2026-11-22 - Temporary explanation recovery

- Completed Code and Documentation explanations now recover in the same
  tab after refresh, with their source, preferences, and active chapter.
  Recovery uses session storage and does not make a model request.
- Clearing the input or turning recovery off removes the temporary copy.
  Draft inputs, PR studies, follow-ups, variants, stories, and images
  are outside recovery. The explicit local study save remains separate.
- Validation: lint, build, 345 unit/API tests and 92 mocked browser tests
  passed. The new checks cover refresh without regeneration, independent
  tabs, clearing the input, and resuming a manual save with recovery off.
- Next: try the refresh flow with an existing study before deciding
  whether to include completed PR chapters.

## 2026-11-22 - Bounded drawing-plan benchmark

- Ran an external experiment on a synthetic cache decision flow and the
  existing Intermediate story illustration brief. Each brief ran twice with
  the ordinary planner and twice with an experimental budget of 26 items,
  18 points per path and 144 path points total. Paired modes used the same
  brief, composition skill, CLI and approved renderer, ran concurrently,
  and alternated launch order. Planning times include structural repairs.
- Mean planning times (ordinary / bounded): technical diagram 22.83s /
  17.88s; story illustration 102.56s / 46.42s. Both ordinary story runs
  needed one structural repair; none of the bounded runs did. Mean
  reported output tokens, including retries, fell from 909 to 684 for
  the flow and from 4,745 to 2,095 for the story. Rendering took
  0.27-0.46s per Full HD PNG.
- Every PNG cleared browser label preflight. Manual inspection confirmed
  that all technical images preserve the hit bypass, ordered miss path
  and shared return. The story images preserve the mirror geometry,
  character and visible fiction and uncertainty qualifiers; the bounded
  drawings use fewer decorative layers without losing the teaching point.
- This is two repeated briefs, not a general quality or latency guarantee.
  Provider caching varied, and there was no independent blind review.
  Kept benchmark logic, fixtures and artifacts external, left the app
  defaults unchanged, and did not rerun app regressions for this
  documentation-only change.
- Next: test on unseen multi-panel scenes and exact-scale technical
  figures before deciding whether to offer an opt-in quick planning mode.

## 2026-11-22 - Mobile image labels and compact-planning comparison

- On mobile, inline illustrations show the exact text from their drawing
  plan alongside the caption. Dense label lists collapse rather than
  truncate. Legacy images without a plan keep their original caption and
  alt text. The preview opens full-screen on mobile, exposes labels as
  readable text, and offers a scrollable actual-size view on both layouts.
- Validation: lint, build, 345 unit/API tests and 88 mocked browser tests
  passed. A separate 390px browser check reused the actual Advanced PNG:
  caption and labels were readable, and actual size exposed all 1920px
  within the scrollable preview. No model call was needed for that check.
- Compared ordinary and compact planning on the same two previously
  generated public-pilot briefs, one run per mode per brief, alternating
  mode order. Intermediate took 68.92s ordinary / 77.97s compact.
  Advanced took 94.87s ordinary / 61.45s compact. PNG delivery took
  0.57-0.71s. All four Full HD images passed label preflight and manual
  inspection confirmed the requested relationships and qualifiers. The
  API returns expanded plans, so it does not expose how much compact
  encoding the model used. Neither repeated timing nor token counts were
  measured. Compact planning remains opt-in.
- Next: benchmark bounded scene complexity with repeated runs, including
  technical diagrams and expressive scenes, before changing the default.

## 2026-11-22 - Live story and VectorCraft pilot

- Ran the external Riemann Hypothesis fixture through the live browser app
  at Intermediate and Advanced (both Balanced). Both produced three scenes
  and one 1920 x 1080 fictional illustration, then restored the story and
  image after explicit save and reload. No browser page errors were recorded.
  The returned brief and scene hashes matched their artifacts, with the
  selected audience profile on each brief and the composition skill on each plan.
- Browser timings in seconds (Intermediate / Advanced): explanation 69.9 /
  45.1, story 25.7 / 37.9, visual brief 22.7 / 22.8, drawing plan 59.9 /
  65.0. Planning through PNG delivery took 60.9 / 66.0 seconds; the remaining
  client checks and render together added about one second per run.
- Manual inspection of both stories and PNGs: the intermediate illustration
  shows a mirrored pair and a fixed point; the advanced triptych adds an
  aligned coordinate grid, with the exact change of variables in the text
  and caption. Both depict fictional pins rather than computed zeros, and
  the stories distinguish the known symmetry from the conjecture. The
  intermediate mapping is still dense in places. On mobile, captions are
  readable but embedded labels are small in both inline previews; the
  advanced PNG cannot reflow its panels as the brief suggested.
- This was a two-sample live and manual check, not a reviewer calibration,
  factual-diagram run, or full audience/depth benchmark. Pilot artifacts
  and topic-specific test logic stayed outside the repository.
- Next: test caption-backed labels and a larger mobile preview, then
  benchmark a faster drawing-plan path against the same quality criteria.

## 2026-11-22 - Integrated regression testing

- Updated browser and API assertions for chapter-only navigation, scoped
  controls, overwrite confirmation and compact planning. The artifact hash
  assertion now uses the exact returned scene, not a reconstructed fixture
  with potentially different JSON key order.
- Builder and inline visual reviews now merge into the latest history for
  the same accepted image. A late result cannot replace a different PNG or
  erase an assessment that finished in another panel.
- Validation: lint, TypeScript, production build and 345 unit/API tests
  passed. All 86 mocked browser tests passed across desktop and mobile,
  including legacy restore, fresh re-review consent, out-of-order reviews,
  PR citations, visual builder cancellation and export.
- Diagram navigation/export still emits a ResizeObserver warning in the dev
  browser, although the workflows passed. No new live generation or manual
  image-quality assessment was performed in this pass.
- Next: rerun the Riemann Hypothesis as an external live story/VectorCraft
  pilot and compare factual accuracy, audience fit, visual readability and
  stage latency. Investigate the diagram resize warning separately.

## 2026-11-22 - Visual-generation lineage and restored-image re-review

- Brief/plan services expose the exact instruction resources loaded for that
  request, including the selected audience or enabled compact reference. APIs
  add optional generation metadata with version/content hashes bound to the
  returned artifact, including repaired plans and scene-policy-adjusted briefs.
  Metadata is not reconstructed by rereading potentially changed skill files.
- Clients validate artifact bindings before retaining lineage with accepted
  images and reuse plan lineage for resizes. Missing metadata from older
  responses/studies remains unknown; no current-version attribution is invented.
  Portable/save validation accepts the optional extension and rejects malformed,
  duplicate or artifact-mismatched stamps without executing imported IDs.
- Inline image history now shows recorded generation stamps and opt-in
  re-review for images with valid provenance, including restored studies.
  Re-review uses recorded context, not later chapter edits; opening controls
  makes no call and a fresh checklist confirmation is mandatory.
- Connected metadata updates for whole results, chapters, adapted variants and
  story scene images. Reviews merge by identity in chronological bounded
  history and cannot replace a newer PNG or unrelated brief/context. Candidate
  acceptance remains explicit and review does not approve the artwork.
- Validation: production build with TypeScript, targeted lint and 40 focused
  loader/generation/provenance/portable tests passed. New checks cover exact
  resources used on repaired plans, artifact-bound lineage, duplicate stamps,
  metadata merging/idempotence and late results against replacement images.
  Updated API fixtures for additive metadata and prepared mocked browser
  save/reload/lineage/re-review/fresh-consent coverage, but did not execute full
  API/browser regressions, live provider calls or manual visual checks.
- Limits: hashes check consistency, not authorship or reviewer truth. Images
  without original provenance cannot be re-reviewed using reconstructed inputs.
  Story-generation instruction metadata is still internal; the new persisted
  lineage describes the visual brief/plan generation stages. Existing bundle/
  import warnings and missing PyYAML skill-format dependency remain.
- Next: run the deferred rigorous integrated validation across the implemented
  workstreams, then fix findings before marking release readiness. The compact
  mobile chapter picker remains a documented UI follow-up.

## 2026-11-22 - Visual review skill and explicit saved review history

- Extracted independent image-review guidance into an allowlisted SKILL.md.
  The reviewer records its exact skill hash in version metadata. Deterministic
  preflight, attachment handling, strict response validation, conservative
  failure policy and opt-in consent remain in application code. Standalone
  review defaults to the trusted module-relative app root; API uses server root.
- New rendered images retain their brief, validated scene and exact scoped
  explanation context. Successful, artifact-bound reports append to a recent
  history capped at five entries and a bounded metadata budget. Metadata-only
  changes compare PNG identity so they cannot turn an accepted image into a
  candidate or automatically publish a different candidate image.
- Explicit save and portable transfer include accepted images' context/history.
  Save, IndexedDB restore, portable import/export and history display validate
  shapes and exact explanation/brief/plan/PNG/caption bindings. Older images
  without metadata remain supported. Mismatched portable/saved metadata rejects
  rather than showing a saved pass; traversal is bounded and cycle-checked.
- Added collapsed read-only history to inline images, labelled as historical
  assessments, not approval. Hash consistency does not authenticate reviewer
  identity or establish truth; imports may contain edited assertions. Opening
  history calls no model, and persistence is never automatic. Save/transfer
  disclosures now mention review context, which may quote original source.
- Unified bounded explanation context across brief generation, review and
  provenance so a long context cannot silently bind different submitted text.
- Validation: production build with TypeScript, targeted lint and 34 focused
  provenance/portable/reviewer tests passed. They cover exact binding, modified
  artifacts/context, malformed or excessive history, cycle detection, bounded
  appending, old images, transfer round trips, conservative review failure and
  cancellation. Fixed two initial unknown-value narrowing compile errors.
- Prepared a mocked browser case for explicit-save/reload history without a
  second review call; did not execute browser/full regression/live provider
  or manual visual checks. Existing Vite/bundle warnings and missing PyYAML
  skill-format validation dependency remain.
- Limits: restored history is read-only; re-review controls and exact generation
  skill/profile lineage still need implementation. Review hashes bind recorded
  artifacts, not the current editable checklist or independent human approval.
- Next: complete generation-skill lineage and restored-image review controls,
  then perform the deferred integrated testing.

## 2026-11-22 - Reusable visual direction and VectorCraft composition

- Reused the existing visual-direction SKILL.md through the allowlisted
  instruction loader. Explicit audiences reuse shared learning profiles;
  Match explanation does not silently select the beginner profile.
- Added a generic VectorCraft composition skill for supported text, rectangle
  and path scene data, semantic composition, readable labels, restrained text
  and visible qualifiers. An optional compact-encoding reference loads only
  with the compact response schema; unsupported renderer features are excluded.
- Moved reusable composition guidance out of the service, retained host/schema
  constraints in code and sourced canvas dimensions from the canonical scene
  definition. Both planning routes pass the configured application root.
  Structural validation, one repair, cancellation, safe adapter invocation and
  API response formats remain unchanged. Repair guidance now matches the
  normal prompt's 60-character/40-word teaching-label guidance.
- Shared instruction formatting includes version/content hashes internally;
  these are not persisted with accepted images yet. No new model stage or
  arbitrary command execution was introduced.
- Validation: production build with TypeScript, targeted lint and 16 focused
  loader/visual-service tests passed. Coverage includes default/compact schema
  selection, direction reuse, explicit/auto audience handling, missing resources,
  structural repair limits, provider failure and cancellation. No browser suite,
  live image generation or manual visual-quality judgment was performed.
- Limits: skill-format validation still needs the previously missing PyYAML
  dependency; broad/live evaluation remains deferred. Richer composition is
  guidance, not evidence that generated images have improved, and the renderer
  still supports only its existing scene vocabulary.
- Next: extract independent visual-review guidance and save artifact-bound
  provenance/history with explicit study saving, preserving old studies.

## 2026-11-22 - App-owned story skills and learning profiles

- Added generic planning, writing and independent review SKILL.md files plus
  shared Beginner/Intermediate/Advanced and Overview/Balanced/Detailed profiles.
  No topic-specific production logic or committed pilot material was added.
- Added an explicit instruction-ID allowlist rooted in server configuration.
  Unknown IDs (including traversal/prototype names), blank/oversized resources
  and cancellation fail without falling back to arbitrary paths. Loaded text
  carries an ID, version and exact SHA256 for later provenance persistence.
- Story generation now composes planning/writing with only the selected
  profiles. Review receives its own skill and the profiles, not the generation
  skills. Existing source/range/schema validation, untrusted-data boundaries,
  cancellation, temporary-file cleanup and one-repair limit remain in code.
  Successful generation still uses two model calls, or at most four with repair.
- Passed the configured root through story API wiring and preserved narrowed
  audience/depth types in the validated parser. Response/save formats and UI
  preferences are unchanged; internal skill metadata is not persisted yet.
- Validation: production build with TypeScript and targeted lint passed; eight
  focused loader/service tests passed using generic fixtures and mocked runners.
  Fixed an initial parser type-widening build error. The three bundled skill
  format-validation attempts could not run because Python lacks PyYAML; no new
  dependency was installed. Existing Vite/bundle warnings remain.
- Limits: no full API/browser regression, live provider run, independent skill
  behavioral evaluation or manual story-quality assessment was performed in
  this implementation slice. Skills are explicitly loaded by the app, not
  installed as global Codex skills. Missing skill resources fail the request.
- Next: reuse the visual-direction skill and add VectorCraft composition
  guidance through the loader; retain strict scene validation and rendering.

## 2026-11-22 - Three-stage visual builder

- Consolidated the visual builder into three numbered stages: What to teach,
  Look and feel, and Generate. Optional prompt preview/editing and the brief/
  render result now live within Generate rather than separate numbered stages.
- Kept the default generation path, explicit brief-then-render actions, edited
  prompt behavior, source inclusion, privacy disclosure, resolution choices,
  cancellation and image review/comparison contracts unchanged. Result/review
  forms are outside the configuration form to avoid nested HTML forms; the
  generate action and prompt textarea explicitly reference that form.
- Prepared a generic mocked browser case for three-stage structure, optional
  prompt editing without calls, exact edited-prompt submission, source opt-out,
  and no automatic drawing-plan call after brief generation. Corrected a
  text-only fixture's visual type in the pending toolbar regression.
- Validation: production build with TypeScript passed after the structural
  change; targeted component/browser-test lint passed. No browser suite,
  live-provider calls or manual visual inspection was run, per the agreed
  implementation-first cadence. Existing build warnings remain.
- Limits: three-stage presentation is implemented, not fully validated; it
  intentionally retains separate brief and render actions rather than adding
  automatic model calls or bypassing review.
- Next: extract generic story planning/writing/review skills and shared
  audience/depth profiles with explicit backend loading.

## 2026-11-22 - Progressive reading tools and concise helper copy

- Reading toolbars collapse idle sibling disclosures when another tool opens.
  Nested options do not trigger toolbar switching; active brief generation,
  image rendering/review, export and playback are protected from automatic
  collapse. Open panels move below the compact action row via CSS.
- Kept tool contents mounted so drafts and speech settings survive collapse.
  Manually closing a visual builder no longer cancels brief generation; its
  summary shows a Working badge while brief/render/review activity continues.
  Export and speech summaries show activity too. Explicit cancel controls and
  existing unmount cleanup remain available.
- Collapsed successful explanation-review commentary, while unresolved review
  notes remain expanded. Shortened save/visual helper text, moved rendering
  implementation notes into a disclosure, removed redundant prompt guidance
  and hid the empty render stage until a brief exists. Generation data/source
  disclosures and explicit visual-review consent remain intact.
- Prepared a generic browser case for draft retention, idle-panel switching,
  active-work visibility and explicit cancellation; updated empty-render-stage
  expectations. Browser tests were not executed, per the agreed cadence.
- Validation: production build (including TypeScript) and targeted component/
  browser-test lint passed. Fixed an initial type error by using native capture
  for the non-bubbling details toggle event. Existing import-extension and
  bundle-size warnings remain. No full regressions, live calls or manual
  desktop/mobile visual checks were run.
- Limits: progressive visual steps still need consolidation; browser behavior
  and screenshot quality remain unverified until integrated testing. These
  changes are implemented, not release-validated.
- Next: consolidate the visual builder into What to teach, Look and feel,
  and Generate with an optional prompt preview, preserving current contracts.

## 2026-11-22 - Chapter-only reader implementation

- Removed the Chapters / Full explanation switch and length-based default.
  Every result now uses the existing chapter outline and pagination, including
  short results. Chapter-scoped Direct/Story and refinement remain available.
- Legacy saved/imported `chapterMode: false` is ignored when restoring the
  view; snapshots retain the version-1 field as `true`. Chapter IDs, variants,
  stories, covers and images keep their existing storage structure. Whole-study
  illustrations remain outside the reader and whole-study export/read-aloud
  remain available rather than being assigned to an unrelated chapter.
- Prepared a generic browser regression for legacy full-view restoration,
  chapter/whole-study image access and normalized resaving. Updated explicit
  reading-mode selectors in existing browser tests and the citation regression
  to navigate to its containing chapter. Other full-view assumptions still
  require review during integrated testing.
- Validation: production build (including TypeScript) passed; targeted lint
  passed; four chapter-outline unit tests passed. Existing Vite import-extension
  and bundle-size warnings remain. Browser, full regression, live-provider and
  manual visual checks were not run, per the implementation-first cadence.
- Limits: this slice removes the reading toggle only; it does not yet simplify
  tool panels or add the compact mobile chapter picker. Browser compatibility
  coverage is prepared but unexecuted; not fully validated or release-ready.
- Next: simplify scoped tool presentation and helper copy while preserving
  tool drafts, active generation status and essential consent disclosures.

## 2026-11-22 - Story and visual skills plan

- Saved the ten-point proposal in `docs/story-visual-skills-plan.md`, covering
  six reusable skills, shared audience/depth profiles and user preferences.
- Added `docs/reader-ui-simplification-plan.md` for less verbose progressive
  tools and chapter-only reading, including saved-study compatibility,
  full-view asset access, accessibility and regression acceptance checks.
- Consolidated delivery in `docs/prioritized-roadmap.md`: chapter-only reading,
  clutter reduction, guided visuals, story skills, reuse of the existing visual
  skill plus VectorCraft composition, review provenance, then broader evaluation.
  Updated the validation cadence at the user's request: implement first with
  lightweight build/type and narrow high-risk checks, prepare regression
  coverage alongside changes, and defer full end-to-end runs and live pilots
  to the integrated testing phase. Track unrun checks explicitly; implementation
  completion is not release readiness.
- Validation: reviewed the Markdown and ran `git diff --check`; no runtime
  changes or application tests required for this documentation-only update.
- Limits: proposed architecture only; no skill loader or generation changes.
- Next: implement chapter-only navigation and compatibility handling, then
  simplify the visual builder; extract story instructions into app-owned
  skills before extending the generation pipeline.

## 2026-11-22 - Opt-in visual review panel

- Added a reusable themed review panel after new image renders. Users edit a
  brief-derived checklist, audience, visual role, teaching goal and learning
  objective; requirements can be included, edited, added and assigned to the
  image/caption/surrounding explanation. Unknown audiences require an explicit
  choice. This draft is clearly not independently verified evidence.
- Explicit confirmation discloses the PNG, caption, explanation context,
  brief, plan and checklist sent to the model. Opening the panel does not call
  the model. Original source is not separately attached; context can quote it.
- Connected a bounded, validated `/api/visual-review` endpoint through existing
  origin, concurrency and cancellation guards. The browser verifies exact
  returned artifact/caption/explanation bindings. New server-side imports use
  explicit TypeScript extensions to avoid adding native-loader warnings.
- Five expandable result sections show status, findings, evidence and reviewer
  metadata. Editing the checklist resets confirmation and marks the report
  outdated. Cancellation, failure or review findings never replace/approve an
  image; new renders clear temporary review state. Automatic acceptance and
  repair remain disabled.
- Validation: lint, build and 317 unit/API tests passed, including draft,
  transport, consent/origin/method and corrupt-PNG regressions. Six new mocked
  desktop/mobile browser tests passed after correcting an initial test-only
  submit-button selector. They cover opt-in behavior, scoped requirements,
  findings, stale reports, cancellation/late responses, failure, artifact
  mismatches and keeping the accepted image unchanged. Full regression passed
  all 76 mocked desktop/mobile browser tests; all 91 focused visual tests
  passed. Desktop/mobile screenshot inspection found no horizontal overflow;
  aligned the Add requirement button with the existing toolbar controls.
- Limits: temporary reports only; saved/imported images without brief/plan
  provenance cannot use this panel yet. No new live provider run or independent
  human calibration was performed for this integration. A selected scene
  review does not audit unrelated scenes. Existing bundle/config warnings
  remain; no topic-specific logic or committed pilot material was added.
- Next: save review provenance/history with images, then controlled repair and
  comparison; keep automatic acceptance disabled pending broader calibration.

## 2026-11-22 - Live visual-review pilot

- Ran seven real vision-review calls on the external public-math pilot with
  actual PNG attachments. The CLI identified the reviewer as gpt-6.2-sol.
  Reports, raw outputs, inputs and deliberately modified VectorCraft images
  remain outside the repository; no topic-specific production logic or
  committed evaluation fixtures were introduced.
- The reviewer rejected the existing Advanced narrative transformation error.
  With that narrative corrected only in the external pilot, the original
  Advanced image/caption passed the configured story-illustration checks.
  The original app study was not repaired or silently replaced.
- Two image-only negative cases retained honest briefs, plans, explanations
  and captions. Both the incorrect coordinate label and false proof-status
  label were detected as critical raster/alignment contradictions.
- Initial Intermediate review requested revision because the pilot contract
  incorrectly required an Advanced coordinate equivalence. Preserved that
  result, scoped the external contract to its intended audience, and reran
  the original/negative Intermediate cases: pass with a minor gaze note for
  the original, fail for the misleading image. This was a contract correction,
  not evidence of an image defect or an automatic image repair.
- Live review latency across seven calls was 22.6-31.7 seconds, median 24.8
  seconds, excluding variant rendering. These are exploratory timings, not
  a general latency guarantee. All seven provider calls completed.
- Manual comparison was performed by the coding assistant, not an independent
  human expert. This small, single-topic pilot does not establish accuracy,
  repeatability, mobile readability or cross-topic audience calibration.
  Passing images were assessed as captioned story illustrations, not as
  standalone technical diagrams. Reviewer/generator model independence is
  not established. No production code changed in this increment.
- Validation: live provider calls, actual attachment delivery/inspection,
  two VectorCraft variant renders, manual artifact inspection and saved
  reports. No fresh unit/browser suite was run for this documentation-only
  increment; the previous 299-test result remains the latest code check.
- Next: an opt-in in-app review panel with scoped contracts and stage-specific
  findings; leave automatic acceptance disabled pending broader independent
  human calibration.

## 2026-11-22 - PNG decode and independent visual-review service

- Replaced header-only evaluation with bounded decoding of static,
  non-interlaced 8-bit PNGs via pinned pngjs. Checks reject malformed chunks,
  duplicate headers, oversized inputs/pixels, bad checksums, truncation,
  unexpected dimensions and excess scanlines. Uniform/transparent output
  requires revision; successful decoding alone remains uncertain.
- Added an opt-in image/caption review service using the existing runner's
  image attachment support. It evaluates all five stages against the contract,
  binds the caption digest, retains deterministic blockers and requires full
  alignment coverage. Invalid/unseen-image responses and provider failures do
  not approve the image. Cancellation propagates; temporary files are removed.
- New generic real-PNG fixtures and mocked reviewer tests cover decoding,
  attachment delivery, metadata binding, corruption, limits, unknown coverage,
  failure/cancellation cleanup and preservation of deterministic findings.
  These mocked reviews are not evidence of live semantic accuracy.
- Replayed decoding on the previously generated Intermediate and Advanced
  public-math pilot PNGs: both decoded at 1920 x 1080 with nonuniform visible
  pixels. This is artifact compatibility only, not content approval, and does
  not resolve the previously identified Advanced narrative error.
- Validation: `npm run check` passed lint, all 299 unit/API tests and build;
  `npm run test:visual` passed all 76 focused tests. `git diff --check` passed.
  Existing Vite configuration/bundle warnings remain. No browser or live model
  review was run for this increment. npm audit reports nine moderate findings
  in the existing ESLint/TypeScript-ESLint/
  minimatch toolchain, none attributed to pngjs or its type declarations.
  No broad dependency upgrades were made.
- Limits: no separate OCR engine, live semantic benchmark, human calibration,
  persistence, auto-repair or app/UI integration. Decoder format/resource
  restrictions are conservative. Model inspection assertions are not proof
  of visual understanding; synchronous decoding needs deployment isolation
  for high concurrency.
- Next: run a consented live PNG/caption review against human-labeled positive
  and deliberately misleading cases, then surface calibrated results in-app.

## 2026-11-22 - Deterministic visual evaluation runner

- Added a reusable opt-in server runner that hashes exact supplied artifacts,
  rejects stale explanation bindings and validates drawing plans. Shared label
  bounds/overlap checks accept explicitly versioned font metrics; additional
  checks flag small exported text, low solid-background contrast and possible
  shape occlusion. No topic-specific production rules or fixtures were added.
- PNG signature/IHDR/dimension checks detect malformed metadata. Matching
  headers deliberately remain uncertain rather than claiming decoded images.
  Brief/alignment/audience reviews remain unrun with no claimed requirement
  coverage, so deterministic-only reports never produce an overall pass.
- Added 22 generic offline tests with defective scene/header variants and
  `npm run test:visual` for the focused suite. `npm run check` passed lint,
  all 276 unit/API tests and build. Existing Vite/bundle warnings remain.
  No live provider or browser tests were run for this increment.
- Limits: no PNG decoding, CRC, OCR, semantic review or mobile legibility
  validation. Contrast skips mixed/unknown backgrounds; path occlusion uses
  conservative bounds and can yield false positives. The runner is not wired
  into image generation/acceptance, persistence or UI yet.
- Next: real decoded raster fixtures and independent PNG/caption semantic
  and audience review, then integrate results into the app.

## 2026-11-22 - Visual evaluation foundation

- Recorded the phased pipeline validation plan in
  `docs/visual-validation-plan.md`, separating brief, plan, actual raster,
  explanation alignment and audience suitability reviews.
- Added generic versioned contract/report types and strict runtime validators.
  Contracts describe teaching goals, audience objectives, required claims,
  elements, qualifiers and prohibitions with evidence references. Reports bind
  explanation/artifact digests, reviewer versions, coverage, findings, timings
  and a maximum of one recorded repair attempt.
- Conservative derived verdicts block critical findings and failed checks;
  missing/unrun reviews and uncovered passing reports remain uncertain.
  Aesthetic quality cannot average away a factual failure.
- Validation: `npm run check` passed lint, 254 unit/API tests (25 new generic
  schema/verdict cases) and production build. Existing Vite configuration and
  bundle-size warnings remain. No new live provider or browser run was made.
- Limits: foundation only, not wired into generation or acceptance. Digests,
  reviewer identity, evidence references and declared coverage are not proof
  of truth. No OCR, actual-PNG review, persistence, repair execution or UI added.
  The public-math pilot is not included in production logic or test fixtures.
- Next: build the deterministic evaluation runner and deliberately defective
  visual fixtures, then add independent PNG/alignment/audience review.

## 2026-11-22 - Caption separation and label preflight

- Drawing prompts now explicitly keep captions, alt text, and narrative
  paragraphs outside PNGs. Short labels and essential qualifiers remain inside.
  Existing app captions now wrap long content with consistent line spacing.
- Added reusable label geometry checks and a browser canvas measurement adapter
  before rendering or accepting a newly planned illustration. The UI rejects
  clipped bounds, text-to-text overlap, labels over 60 characters, and drawings
  exceeding 40 words, with a retry that requests a fresh plan. Accepted images
  and their captions remain unchanged when a new plan fails.
- Passed lint, build, 229 unit/API tests and all 70 mocked desktop/mobile browser
  tests. Browser coverage rejects a clipped label without a render request,
  then successfully retries and preserves the existing save/restore workflow.
- This is a browser layout preflight, not OCR or final-raster validation.
  Browser/renderer font substitution can affect measurements. Geometry covering
  text, contrast, factual accuracy, and mobile raster legibility still require
  visual review. Direct API clients do not run the browser preflight.
- Follow-up live pilots generated fresh Intermediate and Advanced explanations,
  three-scene overview stories, and one Full HD scene illustration per level.
  Both passed the browser label preflight, displayed captions outside the PNG,
  and restored their illustrations after save/reload. Screenshots and inputs
  remain outside the repository. These are selected-chapter workflows, not
  generation of every chapter or every scene image.
- Manual review found an Advanced narrative error despite its model review:
  the stated half-turn centre did not match the accompanying transformation.
  The workflow passed technically, but that story is not content-approved.
  This demonstrates that label checks and a model review do not establish
  mathematical fidelity.
- Next: run the public-math pilot with the new planner policy and inspect the
  image on desktop/mobile; evaluate renderer-side font measurements before
  extending automatic validation to the raster output.

## 2026-11-22 - Story Mode audience validation

- Added generic API regression coverage that checks Beginner, Intermediate,
  and Advanced reach both the story draft and review prompts. Lint, build,
  222 unit/API tests, and 10 mocked desktop/mobile Story Mode tests passed.
- Separate live browser pilots used the same public-math source for Intermediate
  and Advanced. Verified the selected audience in explanation, story, and
  visual-brief requests. Each run returned a reviewed three-scene story and one
  inline 1920 x 1080 illustration that decoded again after save and reload.
- Story requests took 44 seconds (Intermediate) and 23 seconds (Advanced);
  illustration brief and plan took about 79 and 92 seconds, respectively,
  excluding PNG rendering. These are single-run observations, not benchmarks.
- Manual inspection found both stories preserved the conjecture's qualifiers,
  but the technical-depth difference was modest on the short source. The
  Intermediate image had a partially obscured label. Pilot inputs and outputs
  remain outside the repository.
- A second, richer public-math pilot held audience depth fixed and added
  continuation, the completed function, and symmetry arguments. Intermediate
  emphasized a mirror analogy; Advanced explicitly composed conjugation and
  the functional equation. This is one qualitative comparison, not evidence
  of reliable audience adaptation across topics.
- Intermediate completed live with a Full HD image and storage restoration.
  Advanced explanation and story succeeded, but its first visual brief returned
  502. Retrying only the visual stage with cached explanation/story succeeded;
  the generated Full HD image also restored after reload. Do not count that
  retry as a fresh live explanation/story generation.
- Short-label art direction produced unobscured main labels in both second-pilot
  images. Long captions were still embedded in the PNGs and become small on
  mobile. No automatic visual-legibility gate was implemented. The pilot and
  its topic-specific material remain outside the repository.
- Next: test audience adaptation on a richer source and check label legibility
  in rendered illustrations rather than relying only on plan validation.

## 2026-11-22 - Portable code and document studies

- Added local JSON export of explicit saved studies and validated import before
  storage replacement. Transfers preserve source, explanation, chapter state,
  adapted variants, completed stories and accepted PNGs. Export warns about
  source disclosure; importing does not automatically replace the workspace.
- Files are versioned and limited to 40 MB. Import checks reading structure,
  explanations, story citations, PNG signatures and dimensions, and rejects
  dangerous object keys. PR studies and decorative covers remain explicitly
  unsupported rather than being silently dropped.
- Passed lint, build, 219 unit/API tests, and all 70 desktop/mobile browser
  tests. Transfer regressions use mocked providers and real browser storage.
- A separate pilot replayed previously generated public-math assets to export
  a three-scene study. Its JSON imported into a fresh browser context with all
  three Full HD images decoded and no model POST requests. This verification
  did not regenerate content; topic-specific files stay outside the repository.
- Retested that exact pilot file through the Import button's file chooser in
  fresh desktop and mobile browser contexts. Verified three scene images after
  reload, exported JSON equality with the original file, retained save after a
  malformed import and declined replacement, no model POSTs, and no page errors.
- Next: open the portable pilot in the user's browser; review PR-study and cover
  validation before extending transfer support.

## 2026-11-22 - Bounded drawing repair and story-image validation

- Drawing planning now makes at most one simpler replacement after structural
  validation fails. The original brief and qualifiers remain in the prompt.
  Provider errors and cancellation do not automatically retry; invalid repairs
  still fail without rendering. This does not establish factual accuracy or
  guarantee recovery from the earlier live planning failure.
- Passed lint, build, 210 unit/API tests and 32 desktop/mobile story and PR
  browser tests. Tests cover ordinary and compact repair, bounded failure,
  provider errors, and cancellation.
- A separate browser pilot replayed saved responses and the actual generated
  Full HD public-math story PNG. Verified inline decoding at 1920 x 1080,
  IndexedDB save/reload, and restoration without additional generation calls.
  This storage check made no live provider requests. Pilot artifacts and
  topic-specific inputs remain outside the repository.
- Next: validate repair effectiveness with live providers and review the changes.

## 2026-11-22 - Resume saved chapter stories

- Explicit local saves now keep completed stories and accepted scene images
  by chapter and reading preferences. Resume restores the saved Story view
  without regenerating it, including in cached PR chapter variants.
- Existing snapshots without stories remain supported. Drafts, in-flight
  work, and unaccepted comparison images are not saved. Document and
  PDF export are unchanged.
- Passed lint, build, 205 unit/API tests, and 68 desktop/mobile browser
  tests. The storage round trips used mocked providers and real IndexedDB.
- Next: open a PR for Story Mode persistence.

## 2026-11-22 - Chapter Story Mode

- Added an opt-in Direct / Story view. Each scene separates narrative from
  factual meaning and links to the original source where it supports a claim.
  Story requests use the selected audience and depth, share the local model
  limit, and keep the direct view available during generation or failure.
- Validated story structure and citation bounds, with a fidelity review and
  at most one repair. Only reviewed stories are shown. Completed variants are
  cached in the current reader; they are not saved, narrated, or exported.
- Passed lint, build, 204 unit/API tests, and 60 desktop/mobile browser tests.
  A separate live public-math pilot initially failed review. The retry passed
  in 24.1 s and kept the distinction between known facts and the conjecture.
- Rendered and inspected two Full HD VectorCraft pilot images from curated
  briefs: editorial illustration (57.3 s planning) and coordinate schematic
  (46.6 s planning). Rasterization took about 0.6 s each. The coordinate
  version labeled the midpoint and did not plot invented data.
  Pilot inputs and outputs are outside the repository.
- Added an on-demand scene image action with separate fictional illustration
  and factual diagram modes. Both reuse the visual builder and VectorCraft.
  Approved images stay separate by scene and mode in the current reader;
  downloads work, but the images are not persisted or included in exports.
- The live scene pilot produced both modes at Full HD. The first factual
  render marked a hypothetical point, so the mode now asks for hypothetical
  outcomes to stay in text unless supplied as schematic examples. The revised
  render had no point markers. Generation remains model-guided; review
  factual images before sharing.
- Passed lint, build, 205 unit/API tests, and 64 desktop/mobile browser tests
  in CI mode. Two earlier browser runs had intermittent failures in existing
  desktop reading tests; the new scene tests passed in each run.
- During PR review, fixed direct chapter audio continuing after the story
  replaces the direct view. Passed lint, build, 205 unit/API tests, and 66
  browser tests, including the new desktop/mobile playback regression.
- Next: merge the Story Mode PR after the required check passes on the fix.

## 2026-11-22 - Visual API route extraction

- Moved visual brief, scene planning, PNG rendering, and chapter cover routes
  into their own module. Kept the same endpoint contracts and shared model
  request limiter, cancellation, phase diagnostics, and timeout handling.
- Passed lint, build, 202 unit/API tests, and 56 desktop/mobile browser tests.
  The new API regression verifies that an active visual render uses the same
  concurrency slot as an explanation and releases it on completion.
  No live image providers were called.
- Next: extract shared model request lifecycle handling from the local API.

## 2026-11-22 - Automated app checks

- Added a GitHub Actions job for PRs targeting main and pushes to main.
  It installs locked dependencies, runs lint, unit/API tests and the build,
  then runs both Playwright projects. Failed browser runs retain diagnostics
  for seven days. The job uses read-only permissions and no provider secrets.
- Passed actionlint. A clean macOS checkout passed npm ci, lint, build,
  201 unit/API tests, and 56 browser tests with Playwright's bundled Chromium
  and CI mode. This does not substitute for a hosted Ubuntu run.
- Next: verify the first hosted run and consider making App checks required.

## 2026-11-22 - Reading workflow review fixes

- Isolated visual drafts by adapted chapter settings and result identity.
  Switching cached variants or regenerating identical text cannot retain
  the previous result's draft. Saved illustrations stay variant-specific.
- Citation excerpts scroll locally to the first highlighted line, so long
  wrapped context cannot push it out of view.
- Passed lint, build, 201 unit/API tests, and 56 desktop/mobile browser tests.
  New browser cases use mocked model responses to cover both regressions.
- Next: review the fresh PR and its checks before merging.

## 2026-11-22 - Release walkthrough

- Imported the existing public FastAPI starter: 274 files and 29 chapters.
  Chapter navigation worked; a repeat import timed out.
- A temporary public math pilot generated a live explanation in 27.7 s.
  Verified citation highlighting and focus restoration. The first drawing
  took 85.0 s to plan and about 0.6 s to rasterize; the second planning
  request was cancelled before a usable result returned.
- Completed the browser walkthrough using captured public-pilot responses
  and an alternate palette with real VectorCraft rendering. Verified side-by-side
  selection, local save and restore, and a two-page Chromium PDF with one
  embedded 1920 x 1080 illustration. Inspected both printed pages. No pilot
  inputs or outputs were added to the repository.
- Lint, build, and 201 unit/API tests passed again; the latest browser suite
  passed all 52 tests. Next: check branch CI and review before merging.

## 2026-11-22 - Scoped reading toolbars

- Reused one toolbar for whole explanations, individual chapters, and
  generated PR chapters. Refinement links lead to the right preferences.
  Individual chapters now have their own read-aloud and PDF preview; the
  chapter preview keeps citations, review notes, and the chapter illustration
  without including other chapters or the full-source appendix.
- Starting another reader or leaving the speaking chapter stops the prior
  playback. Document export and combined PR study PDF remain separate.
- Passed lint, build, 201 unit/API tests, and 52 desktop/mobile browser tests.
  Speech used a browser mock; tests verified scope, playback handoff,
  navigation cleanup, and chapter PDF inclusion and exclusion.
- Next: do an end-to-end walkthrough on real material before publishing the
  remaining local changes.

## 2026-11-22 - Sticky chapter outlines

- Shared the contents outline between explanations and PR studies. It stays
  visible while scrolling through a chapter on wide screens. Long lists scroll
  independently and bring the active chapter into view. On narrow screens the
  outline stays in the page flow above the chapter.
- Passed lint, build, 201 unit/API tests, and 50 desktop/mobile browser tests.
  The new browser cases use mocked outlines and explanations to verify actual
  scroll placement, list overflow, active selection, and heading focus.
- Next: unify chapter and whole-explanation toolbars and clarify their scope.

## 2026-11-22 - Workspace starter cards

- Added selectable code, long PR, and technical concept examples. Each card
  fills and focuses the relevant input without fetching or generating. Replacing
  existing input requires confirmation; cards are disabled during active work.
- Passed lint, build, 201 unit/API tests, and 46 desktop/mobile browser tests.
  The new test uses mocked PR import to verify keyboard selection, cancellation,
  mobile layout, and that only the explicit Load PR action requests an import.
  The public FastAPI example uses the PR from the earlier live import; no live
  import was repeated for this change.
- Next: keep chapter outlines visible while scrolling on wide screens.

## 2026-11-22 - Inspect citations beside the reader

- Citations open a local, read-only excerpt beside the explanation on wide
  screens and in a modal bottom sheet on narrow screens. Highlighted lines
  include nearby context, and closing returns keyboard focus to the citation.
- PR citations use the imported chapter text and its line numbers, not original
  repository file coordinates. Opening a citation does not fetch new evidence.
- Passed lint, build, 201 unit/API tests, and 44 desktop/mobile browser tests.
  Browser tests verified highlighted code, documentation and mocked PR evidence,
  responsive placement and focus restoration.
- Next: add starter cards for code, a long PR, and a technical concept.

## 2026-11-22 - Compare illustration versions

- New drawings can be compared against the current illustration at equal
  display sizes, stacked on narrow screens. Keep the current image or accept
  the new one for inline reading, local saves, and PDF previews. Unselected
  drafts are not saved. Resizing the accepted drawing still reuses its plan
  and updates the image without requiring another choice.
- Passed lint, build, 201 unit/API tests, and 44 desktop/mobile browser tests.
  The new browser test uses mocked generation to verify layout, both choices,
  drawing reuse, and PDF selection before and after acceptance.
- Next: open citation evidence alongside the reader on desktop and in a
  bottom sheet on mobile.

## 2026-11-21 - Selectable illustrated PDF preview

- Added a browser-local PDF preview for the current explanation, generated
  PR chapter, or adapted section. Readers select chapters and existing inline
  illustrations, with an optional source appendix. The preview prints the
  same content without a server export or a model call. Review, coverage,
  and citation context is retained; WordCraft and diagram export are unchanged.
- Passed lint, build, 201 unit/API tests, and 12 desktop/mobile browser tests.
  Produced and visually inspected a real two-page Chromium PDF using a
  temporary public illustration. Verified the Full HD image was embedded and
  that the workspace UI and unselected source appendix were excluded.
- Next: verify the browser's print dialog on additional engines and consider
  a cross-chapter PDF for a whole saved PR workspace.

### Combined PR study PDF

- Added one browser-local PDF for selected completed PR chapters in active
  or resumed studies. It uses each chapter's selected settings, cached
  illustrations, and chapter-local citations. The pinned head/base and scope
  warnings are retained; missing and deselected results are distinguished.
- Passed lint, build, 201 unit/API tests, and 42 desktop/mobile browser tests.
  Printed and visually inspected a four-page Chromium sample from a synthetic
  saved PR workspace, with two embedded Full HD illustrations. No GitHub
  fetch or model call was made to create the sample.
- Next: add side-by-side illustration comparison without replacing the
  current inline image until the reader chooses a version.

## 2026-11-21 - Compact visual scene planning

- Added opt-in compact scene planning with reusable repeated geometry and
  evenly spaced labels. Server expansion is bounded and uses the existing
  drawing validator; browser and renderer still receive ordinary shapes.
- Passed lint, build, 201 unit/API tests, and eight focused desktop/mobile
  browser tests. Coverage includes unsafe and out-of-bounds requests,
  expanded item limits, and the opt-in planning boundary.
- Two temporary live pilots rendered valid Full HD PNGs and preserved key
  facts on visual inspection. The comparison planned in 39.46 s vs. 70.05 s
  previously, and the coordinate figure in 40.19 s vs. 61.51 s previously.
  In the instrumented sample, output fell from 3,268 to 1,580 tokens. These
  are single observations against earlier runs, not a controlled benchmark.
- Next: repeat across more prompts and visual forms before enabling by default.

## 2026-11-21 - Visual context disclosure and scene profiling

- The brief step now shows which explanation, design instructions, and source
  attachment are sent. The optional notes explain how to provide exact values
  without attaching the original input. No request or consent behavior changed.
- Passed lint, build, 196 unit/API tests, and eight focused mocked desktop and
  mobile browser tests.
- Profiled one temporary public brief with the existing scene prompt and schema,
  once at CLI defaults and once with lower reasoning. Planning took 70.05 s
  and 75.02 s; the CLI provider turn accounted for 68.12 s and 74.12 s. Full
  HD rasterization took 0.47 s and 0.45 s. Both rendered correctly on visual
  inspection. This is one comparison; no production defaults changed.
- Next: prototype compact reusable scene primitives to reduce generated
  output, then compare multiple runs and formats for latency and accuracy.

## 2026-11-21 - Guided visual designer sequence

- Made the prompt, brief, and image stages explicit after the teaching and
  style cards. Prompt review is optional; rendering is only available after
  a renderable brief. Added a direct jump to the brief and render controls.
- Kept full brief details available on demand and explained the next move
  when a brief recommends text only.
- Added browser assertions for the locked rendering state and focus on the
  render stage. The initial test pass was deferred until feature work was done.

### Live pilot and recovery checks

- Used an external, temporary public mathematics example. No topic-specific
  source, prompt, fixture, or image was added to the project.
- Live explanation, visual brief, and scene planning took 46.17 s, 22.53 s,
  and 61.51 s respectively in this single run. Verified actual HD and Full
  HD PNG dimensions from the same scene; local rasterization took 0.34 s and
  0.38 s. Inspected the preview for the requested mathematical qualifiers.
- Passed lint and build. Chrome desktop replay used the captured live model
  responses with real VectorCraft rasterization and verified prompt editing,
  render focus, resolution reuse, zoom, local restore/delete, and failure
  and cancellation recovery. Mobile replay verified adapted variant and
  image restore, retained brief on failure, and pinned PR snapshot restore
  with a mocked PR and no new retrieval or generation on resume.
- Next: run the same workflow on other user-provided material before
  generalizing about explanation or illustration quality.

### Second live topic

- Ran the same desktop workflow on temporary algorithm-learning material.
  The planner chose aligned comparisons instead of reusing the first pilot's
  coordinate figure. Both resolutions shared one scene; the chapter and PNG
  restored without a new model call.
- With original source excluded, the initial brief used placeholders for data
  omitted by the explanation. Supplying exact values as optional context
  produced a complete and visually verified illustration without sharing the
  original source. No topic-specific artifacts were added to the project.
- Scene planning took 87.83 s and 88.23 s across the two renders; Full HD
  rasterization took 0.46 s and 0.57 s. Small sample only. Next: clarify
  which data the designer sees and profile scene-planning latency.

## 2026-11-21 - Explanation retry and cancellation

- Kept the last completed explanation or variant visible throughout
  regeneration, including on failure or cancellation. A successful replacement
  starts a fresh follow-up conversation.
- Added cancellation for the standalone generator and clearer retry actions
  for chapters and adapted sections. PR import retry now returns to PR import.
- Deferred test execution until feature work is complete. Verify cancellation,
  retries, and source or chapter changes during generation in the test pass.

## 2026-11-21 - Saved studies and a faster visual workflow

- Added an explicit browser-local save, resume, and delete for one code or
  docs study, with chapter position and rendered PNGs. PR workspaces and
  adapted variants are not included in this first version.
- Grouped visual options into teaching and style steps, with custom controls
  and prompt editing available but optional.
- Split drawing planning from PNG generation. The current brief keeps its
  validated drawing when retrying PNG export or selecting another resolution.
- Verified lint, build, 196 unit/API tests, and 18 desktop/mobile browser
  tests including save/restore/delete and PNG failure recovery. Browser
  generation used mocked responses.

### PR and adapted-study follow-up

- Extended the existing explicit local save to pinned PR evidence, chapter
  settings, cached reader results, and code/docs adapted variants. Restore
  does not refetch or generate. Only one study is stored; no follow-ups or
  pending work are included.
- Live test with one synthetic batching brief: one model call planned 34 scene
  items in 32.91 s. Real VectorCraft rendered the same scene at 1280 x 720
  in 0.69 s and 1920 x 1080 in 0.43 s. Checked PNG dimensions and viewed
  the Full HD composition at preview size. This is one sample, not a latency
  benchmark.
- Verified lint, build, 196 unit/API tests and 28 desktop/mobile browser
  tests with mocked generation, including pinned PR restore, patch offset
  continuation, and restored adapted-section illustrations. Next: try a
  saved study with a substantial public PR and measure local storage use.

## 2026-11-21 - Inline illustrations and reading tools

- Rendered visuals appear in the reader, with a caption, size, full-screen
  preview, and PNG download. Chapter visuals stay attached to the current
  reading session when switching chapters or reading modes.
- Grouped whole-explanation visual, speech, and export controls and gave
  individual chapters a scoped illustration action. Document export
  behavior is unchanged.
- Verified lint, build, 196 unit/API tests, and 18 focused desktop/mobile
  browser tests. Browser generation was mocked.

## 2026-11-21 - Editable visual prompt preview

- Added a local prompt preview from visual selections and notes, with copy,
  edit, and reset. An edited draft replaces preset design instructions for
  brief generation; source sharing still requires separate opt-in.
- Verified lint, build, 196 unit/API tests, and eight desktop/mobile browser
  tests. Browser generation and clipboard were mocked.

## 2026-11-21 - Visual designer presets and styling

- Added reusable multi-select cards for art direction and teaching presets.
  Selections combine with optional custom notes when generating a brief,
  without changing source opt-in or sending requests on selection.
- Aligned the visual designer and export panel with workspace theme colors,
  controls, and borders. No renderer or export format changes.
- Verified lint, build, 196 unit/API tests, and 14 focused desktop/mobile
  browser tests, including keyboard selection and preset removal.

## 2026-11-21 - Reusable explanation visual design skill

- Added a repo-local design skill that chooses the visual form from the
  teaching question, separates factual content from styling, and specifies
  source, readability, and accessibility checks. It produces a brief, not
  executable renderer input.
- Parsed and checked the YAML frontmatter using Ruby; the bundled validator
  could not run without PyYAML. No visual-generation behavior was tested, and
  the app does not invoke this skill yet. Next: evaluate briefs across multiple
  domains, then wire an opt-in Illustration mode to validated scenes.

### Three-visual rendering pilot

- Hand-applied the skill to a critical-strip illustration, a branch view
  grounded in the saved FastAPI #9816 chapter, and a synthetic attention
  heatmap. This tests the design approach and rendering, not automated brief
  selection or generation quality.
- Real VectorCraft rendered PNG and landscape PDF for each. Verified the PR
  graph and saved patch lines, attention row sums, required PDF labels and
  caveats, artboard bounds, and PNG dimensions. Inspected all three PNGs and
  all three PDF previews; adjusted the decision-label placement after review.
- Latest renderer-only runs took 98-234 ms per pair locally. Lint and diff
  checks passed. App behavior is unchanged. Next: evaluate model-generated
  briefs against typed scene constraints and the three visual references.

### Model-written brief smoke test

- Added a schema-constrained, read-only CLI evaluation of the three briefs.
  It does not execute or render model output. Used public patch lines and
  educational examples only. Lint and diff checks passed.
- All three live responses selected the intended visual form. Reviewed their
  factual content: correct unproved critical-line claim with no plotted zeros,
  correct empty-mapping fallback and next branch with chapter citations, and
  correct illustrative attention values, row maximum and total.
- First run passed all mechanical checks in two of three cases. Attention
  mislabeled unnumbered illustrative inputs as source-backed, so clarified
  evidence roles in the skill and reran only that case; all checks then passed.
  Individual model calls took about 18-24 seconds. The three first-run briefs
  chose conservative compositions. Two topics appear in the skill examples,
  so this is not a held-out or creativity benchmark.
- No app integration or automated render yet. Next: test unseen subjects and
  explicit style briefs against a bounded scene schema before enabling export.

### Reusable app brief controls

- Added a shared **Design a visual** panel, preference fields, and result view
  for regular explanations, PR chapters, and adapted sections. The learner can
  set the question, audience, visual form, style, output target, and art
  direction, or keep defaults. Editing controls does not call the model.
- App and evaluator now share a bounded request/response contract. The
  opt-in request includes the current explanation; raw source is a separate
  checkbox. The endpoint uses existing local origin, concurrency, cancellation,
  and timeout handling. It returns text only and never executes a scene.
- Lint, build, 190 unit/API tests, and diff checks passed. The real local
  endpoint returned a correct synthetic stack/queue comparison using the
  requested expressive warm style and print target in about 22 seconds.
- The 20 focused PR and brief browser tests passed on desktop/mobile after
  scoping the cover caption assertion to its own panel. The other 16 browser
  cases passed in the full run before that test-only correction.
- Existing PNG/PDF behavior is unchanged. Next: translate reviewed briefs into
  validated reusable scenes and add an explicit render action.

### Topic-independent prompts and learner data

- Removed named benchmark scenarios and identifiers from the production
  explanation and visual skills and replaced case-derived flow stages in
  the planning aid with source-derived stages. Concrete cases remain in
  tests and evaluations. Language-specific lexical hints remain bounded aids,
  not required scenarios or authoritative program analysis.
- Added optional **Additional context or data** to the shared visual
  preferences. The bounded input is sent only on explicit generation and
  handled as evidence separate from art direction and opted-in original
  source. Insufficient or conflicting inputs are not silently filled in.
- Lint, build, 191 unit/API tests, eight focused desktop/mobile browser tests,
  and diff checks passed. Live no-preset synthetic experiment brief selected
  a coordinate figure, preserved all three user-supplied measurements, and
  explicitly avoided interpolation (about 17 seconds). This verified the brief
  only, not rendering or generalization across all domains.

### Render an approved brief

- Added an explicit **Render image** action below a renderable brief. It sends
  the brief in a separate model request, then previews and downloads a PNG.
  Text-only briefs stay text-only, and changing the explanation does not
  generate an image. Diagram exports and PDF embedding are unchanged.
- The model emits bounded text, rectangles, and paths, not renderer commands.
  The server validates the scene and builds fixed VectorCraft commands, reusing
  executable checks, dimensions, cancellation, and temporary-file cleanup.
- Lint, TypeScript/build, 194 unit/API tests, and ten focused desktop/mobile
  tests passed. Live saved public math brief produced a 1000×700 PNG in
  about 28 seconds; visual review caught a label overlapping a line and
  prompted tighter label-spacing guidance. No guarantee of factual or layout
  correctness across arbitrary model scenes.
- The follow-up live render took about 24 seconds; inspected it and confirmed
  that the label cleared the line and the conjecture remained marked unproved.

### Make illustration and diagram export distinct

- Labeled the graph export as an existing diagram and the designer as a new
  illustration. Added a keyboard-focusable shortcut from the graph export to
  the same designer, carrying the selected diagram and an editable expressive
  style preset. Learners can switch back to the whole explanation. The
  shortcut itself sends no model request and leaves export behavior unchanged.
- Lint, build, 194 unit/API tests, and eight focused desktop/mobile browser
  tests passed, including selected-view context, source opt-in, focus, and the
  whole-explanation fallback. Next: allow a reviewed illustration in chapter PDFs.

### Audience and depth cards plus HD illustrations

- Replaced the main input's audience and depth dropdowns with two optional,
  independently navigable card steps. Shared native radio cards preserve the
  defaults, support keyboard navigation, and do not trigger generation.
- Reused the card component for Standard, HD, and Full HD illustration PNGs.
  VectorCraft uses exact output artboards and scales the validated vector
  scene uniformly, with centered background margins as needed. Existing
  technical diagram and PDF exports are unchanged.
- Lint, TypeScript/build, 196 unit/API tests, and six focused desktop/mobile
  browser tests passed. Real VectorCraft CLI rendered and verified dimensions
  for all three presets. Next: use the shared cards for Focus and Visuals
  and connect their selections to an editable prompt preview.

## 2026-11-21 - Clarify when PDFs have no diagram

- The export panel now distinguishes explanations without a renderable diagram
  from illustrated PDF exports. Browser print no longer includes interactive
  export controls or their download status.
- Verified the reported PDF was browser-produced with no embedded image and
  only a diagram note. Created a separate one-off VectorCraft companion page
  and verified the original pages' extracted text was preserved in the copy.
  No automatic diagram generation was added.
- Lint, build, 186 unit/API tests and four focused desktop/mobile browser
  tests passed. Next: offer an explicit diagram request when no visual exists.

## 2026-11-21 - Automatic VectorCraft branch layout prototype

- Added an opt-in evaluation script that lays out nested binary splits and
  joins in colored lanes, with decision labels and source-line references.
  It emits one route per supplied connection and rejects unsupported graphs
  rather than discarding edges. App and PDF export behavior are unchanged.
- Two focused Node tests and lint passed. Rendered the saved FastAPI PR #9816
  version-selection view with the real local CLI: 10 nodes, 12 connections,
  about 0.27 seconds for rendering and output checks in one run. Visually
  inspected the lanes, arrow labels, joins, and citation placement.
- Only structured, acyclic, reconverging binary branches are supported; cycles,
  bypass edges directly to joins, and independent terminating paths are not.
  Only decision edge labels are printed in the image. Next: test more PRs,
  tighten spacing, and design an explicit fallback before wiring it into PDFs.

### Compact layout and fallback follow-up

- Reduced row spacing from 155 to 132 and moved the footer closer. On the
  original nested FastAPI view the artboard went from 990 to 839 pixels tall.
- Added a connection-index fallback for valid graphs the lane layout cannot
  represent. It lists every node reference, source citation, directed edge,
  and edge label, including loops and parallel edges. Invalid input still fails.
- Rendered and visually inspected four views across three public FastAPI PRs:
  #9816 version selection (10 nodes, 12 edges, lanes), #9816 warnings (12/14,
  index), #15813 JSONL headers (5/5, lanes), and #16424 response length (5/6,
  index). The last two are hand-transcribed diff fixtures, not model-generated.
  Local renderer runs took 88-280 ms; no model or image-provider calls.
- Five focused Node tests, lint, and diff checks passed. Next: test both
  layouts at PDF page scale before changing the app default.

### Portrait PDF fit check

- Added an offline evaluation using the production WordCraft command builder
  and the same portrait sizing rule, without relaxing live PNG limits.
- Embedded the #9816 version lanes and warnings connection index in separate
  eight-page study PDFs. Poppler confirmed one uncropped image on page 8 in each;
  selectable explanation text, evidence references, and review status on
  pages 1-7 were identical. Inspected both image pages at 96 dpi.
- At 480 points wide, the lane chart's 16 px decision labels and 11 px
  citations become 5.0 pt and 3.4 pt. The index's 15 px detail text becomes
  about 6.5 pt. Both fit but are too small to make default print views.
- Lint passed. Keep the app default unchanged. Next: check a landscape or split
  lane figure and a selectable, paginated connection index.

### Larger synthetic job-flow check

- Added a 16-node synthetic fixture with nested cache/provider decisions and
  shared completion steps. The real VectorCraft CLI rendered all 18 connections
  in lanes (310 ms); adding one retry edge switched to the connection index
  without losing any of the 19 connections (150 ms).
- Inspected both PNGs at full size. The lane chart is readable as a standalone
  image, but its long shared tail would benefit from a separate section in print.
  Six focused Node tests, lint, and diff checks passed. No app default changed.

### Local app integration and live rendering

- Added an opt-in **Automatic branches** layout in the diagram PNG panel.
  It shares the evaluator's lane and connection-index planner; the existing
  detailed PNG and illustrated PDF defaults and limits are unchanged.
- The automatic PNG accepts up to 16 nodes and 24 connections and retains
  source references, supplied edge labels, review and coverage notes. Shared
  renderer checks enforce the planned dimensions and reject partial outputs.
- Both desktop and mobile Chrome previewed and downloaded real PNGs from
  the running app for the 16-node synthetic workflow (18 edges, lanes) and
  its retry variant (19 edges, index). Only explanation generation was
  stubbed; the local API and pinned VectorCraft CLI were live. The saved
  public FastAPI chapter also rendered via the live API (420 ms).
- Lint, TypeScript/build, 186 unit/API tests, six standalone layout tests,
  and 34 mocked desktop/mobile browser tests passed. Next: test PDF-specific
  split-page diagrams and a selectable connection index.

## 2026-11-21 - Diagram images in chapter PDFs

- PDF export now optionally embeds one selected VectorCraft diagram on its own
  page. The text-only default, selectable diagram descriptions, citations,
  review warnings, and source appendix preference are preserved. DOCX is
  unchanged in scope. Rendering reuses the existing explanation with no model
  calls; only server-rendered PNGs are accepted.
- Lint, TypeScript, build, 184 unit/API tests, and four focused mocked
  desktop/mobile browser tests passed. Exported a real synthetic two-page PDF
  through the local API, confirmed the embedded image with Poppler, and inspected
  both rendered pages.
- Next: evaluate multi-diagram and multi-chapter exports. Existing VectorCraft
  size limits apply; very tall diagrams are scaled to fit the image page.

## 2026-11-21 - Local activation and VectorCraft PNG integration

### Changes

- Enabled the two optional renderers for this workspace in ignored
  `.env.development.local`. Copied the previously verified official executables
  and accompanying notices into ignored `.local-tools/`, rather than depending
  on `/tmp` binaries. Checked both capability endpoints on the running app at
  port 5173. Provider credentials were neither read nor changed.
- Added opt-in PNG preview/download below original and adapted diagrams, keeping
  React Flow and evidence selection intact. Rendering is explicit, uses the
  chosen existing diagram, and makes no model/image-provider calls. Previews
  are scrollable and keyboard focusable, including on narrow mobile screens.
- Added a checksum/version-pinned VectorCraft 0.24.1 adapter using safe mode,
  no-prefs, fixed typed commands, isolated temporary HOME, restricted environment,
  bounded output and artifact sizes, cancellation, and cleanup. Shared download
  lifecycle keeps local host/origin guards and one concurrent export per type.
- Initial PNG layout retains every supplied edge, including loops/duplicates,
  with exact connection labels, node descriptions, chapter coordinates, source
  hash, and full review/coverage/incomplete-code notes. Node ordering is explicitly
  not an inferred execution sequence. Supports 1-8 nodes and at most 12 edges;
  excessive height, renderer warnings, overflow, out-of-artboard geometry and
  invalid/cropped PNG output are rejected rather than silently truncated.
- Updated `EXPORTS.md`, setup examples and current implementation plan. Binary
  and environment files are ignored; nothing is downloaded automatically at run
  time, and the repository still defaults to disabled for new clones.

### Validation

- Final `npm run check`: lint, all 180 unit/API tests, TypeScript and build
  passed, including exact-PNG-height hardening. `git diff --check` passed.
- All 34 mocked desktop/mobile browser tests passed, including chosen diagram,
  PNG preview/download, failures, cancellation, missing renderer and no extra
  generation. After the scrollable-preview adjustment, its four focused browser
  tests passed again.
- Real local HTTP downloads passed for a synthetic branch/retry/self-loop
  fixture: one measured run was DOCX 294 ms, PDF 268 ms, PNG 471 ms. Final
  post-hardening downloads also passed (254/277/485 ms). The first DOCX
  run was 2271 ms, so these few samples are not a p95 or a latency guarantee.
- Separate desktop/mobile Chrome acceptance used stubbed explanation generation
  but **real local renderer APIs**: DOCX downloads and 1000-pixel-wide PNG
  previews/downloads passed; explanation remained available, no horizontal page
  overflow, and no extra generation. Inspected the real PNG and both screenshots.
- Verified `.env.development.local` and both executable paths are ignored.
  Existing extensionless-import warnings from Vite remain.

### Limitations and next steps

- Experimental trusted-local use only, not an OS sandbox. SVG/native editable
  downloads, document image embedding, arbitrary large/multilingual layouts and
  production performance remain unverified or deferred. A PNG can explain a
  supplied graph but does not verify source behaviour or runtime outcomes.
- Next: collect user feedback on the active exports, then embed selected PNGs
  into Word/PDF study guides while retaining text alternatives and evidence.

## 2026-11-21 - Optional WordCraft explanation export

### Changes

- Added experimental DOCX/PDF export for existing original explanations, adapted
  sections, and selected PR chapters. Export scope is explicit; adapted results
  are not silently replaced by originals. No model calls or regeneration.
- Added versioned snapshot validation, evidence/graph checks, source hash,
  opt-in source appendix, exact review notes, incomplete-code findings, and
  visible coverage/assumption limits. Diagrams currently export as text.
- Added a disabled-by-default, checksum/version-pinned WordCraft adapter with
  minimal environment, isolated temporary HOME, fixed typed commands, bounded
  diagnostics/artifacts, strict PDF conversion, and cleanup before download.
- Used a cancellable direct-response API rather than retained export jobs for
  this one-result slice; one concurrent request, 2 MB input and 120-second cap.
  Existing local origin/host guards apply. Setup is documented in `EXPORTS.md`.
- Left unrelated existing README/docs and sibling-project changes untouched.

### Validation

- Final `npm run check`: lint, 167 unit/API tests, TypeScript and production
  build passed. Existing Vite native-config extensionless-import warnings remain.
- Desktop/mobile browser suite passed all 30 tests with mocked APIs, including
  selected-adapted-result export, explicit consent, no regeneration, missing CLI,
  renderer-error recovery and cancellation. Inspected both export UI screenshots.
- Real pinned CLI generated both formats with and without a source appendix
  from a synthetic fixture through the new adapter. Text checks retained review
  and coverage warnings and verified appendix inclusion/omission. Measured
  101-329 ms including binary checks, rendering and text extraction; four small
  local runs are not a general latency benchmark. No provider calls were made.
- Independently rendered both DOCX variants with bundled LibreOffice and
  inspected all four pages, plus all four WordCraft PDF pages using Poppler.
  Headings, warnings, evidence and source were readable without clipping.
  Short fixtures leave a sparse final page; arbitrary layout quality is unproven.
- Started a temporary local Vite server with the approved CLI enabled and sent
  a synthetic fixture to the real endpoint. DOCX and PDF HTTP downloads passed
  (206 ms and 253 ms respectively), no-store headers were retained, and an
  untrusted Origin was rejected with 403. Stopped the test server afterwards.
  Normal local configuration remains unchanged and disabled unless opted in.

### Limitations and next steps

- Export remains opt-in, macOS WordCraft 0.18.1 only; not an OS sandbox or a
  hosted-service security boundary. No automatic installation or redistribution.
- No rendered diagram assets, conceptual covers, multi-result bundles, or native
  Microsoft Word validation. Source omission is not redaction of explanation
  text. Arbitrary large/multilingual layout quality is not established.
- Next: enable the approved CLI locally for user acceptance, then implement
  sanitized VectorCraft diagram assets without replacing evidence or warnings.

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

## 2026-11-21 - Actual DeckCraft Export Pilot

- Downloaded the official standalone macOS CLI 0.18.1 into a temporary folder,
  without a system-wide installation. Its archive SHA-256 matched the GitHub
  release asset digest: `10992f73c88a4c0361dd0fec3a4d7490731ffb3ea9658ee2274b396e3341af00`.
- Added a repeatable, deliberately case-specific export benchmark for the saved
  public FastAPI params chapter. No generic converter or app UI was added.
- Built five slides with native editable text/branch diagrams and exported
  `.deckcraft`, `.pptx`, and PDF. Full source and original explanation accompany
  the files. Speaker notes retain the exact review note and revision; each slide
  visibly includes chapter citations and partial-patch/verification caveats.
- Native and PPTX inspections passed five-slide, notes, source, warning, and bounds
  checks. All five DeckCraft-rendered preview images were identical after PPTX
  round-trip. Poppler extracted five PDF pages with citations and warnings.
- Agent inspected all native slide previews and all independently Poppler-rendered
  PDF pages. This is not a Microsoft PowerPoint compatibility test or an
  independent human quality assessment. Generic source selection/layout is not
  tested by this curated example. No model/image-provider calls were made.
- `npm run check` passed: lint, 138 unit/API tests, TypeScript, production build.
  Existing Vite native-config warnings remain. No new browser tests were run;
  this experiment does not change the app UI. Next: evaluate a bounded,
  optional export adapter and independent PowerPoint compatibility before adding
  a user-facing export action. Keep DeckCraft outside the core explanation path.

## 2026-11-21 - WordCraft and VectorCraft Pilots and Integration Plan

- WordCraft CLI 0.18.1 archive matched published SHA-256
  `34397cf6fcf73ee8ae8d53cab08fbdfc213262dbb6753c2965ebb7cde37b8616`.
  Built a four-page, five-chapter DOCX/PDF guide from the same saved public PR
  result. Heading styles, citation text, exact original review property, and DOCX
  text round-trip checks passed. Agent inspected all native pages and all four
  pages independently rendered with bundled LibreOffice. The optional figure
  reuses a DeckCraft raster image; it is not an editable WordCraft vector.
- VectorCraft CLI 0.24.1 archive matched published SHA-256
  `7f0c8261e5c831d9681e48272f9ba73bf8702dbf6ceb84f98177c7c007faddf2`.
  Native commands created a four-node/four-edge view with live editable text,
  citations and source metadata. Native text editing, SVG round-trip, bounds,
  basic SVG content checks, and PDF citations passed. Agent inspected native,
  Chrome SVG, and Poppler PDF renderings, moving labels clear of connectors.
- The initial VectorCraft font fallback was resolved by explicitly selecting its
  Source Sans 3 font. All binaries remain temporary, not installed system-wide
  or committed. No model/image-provider calls were made by these pilots.
- Added repeatable case-specific benchmark scripts and the phased integration
  plan in PLAN.md: typed snapshots, optional adapters, bounded local jobs, UI and
  consent, source/quality preservation, SVG safety, and release gates. This does
  not yet add a generic converter, public export endpoint, or app Export button.
- Scope: one public case on macOS arm64. Native Microsoft Office, other operating
  systems, generic/dense/multilingual layouts, and production isolation remain
  unverified. These agent inspections are not independent human assessments.
- Final checks passed: lint, 144 unit/API tests (including six fail-fast pilot
  guard tests), TypeScript, production build, and 26 desktop/mobile browser tests.
  All three native CLI pilots were rerun successfully. Existing Vite native-config
  warnings remain. Next: review and merge the pilots/plan, then implement the
  bounded WordCraft adapter behind a feature flag as described in PLAN.md.

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
