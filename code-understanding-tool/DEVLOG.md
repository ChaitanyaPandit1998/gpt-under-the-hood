# Development Log

Track meaningful changes, validation, limitations, and the next action here. Add a
dated entry for future work; never record secrets or private source material.

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
