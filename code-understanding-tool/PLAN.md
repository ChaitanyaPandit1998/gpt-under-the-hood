# Implementation Plan

## Product goal

Build a small interactive workspace that helps a learner understand a piece of code or documentation. It should combine an explanation, a useful visual when appropriate, and a way to check understanding. It should make clear what the input supports and where the explanation is inferring missing context.

## MVP decisions

- Support pasted code and documentation as the first input path.
- Support a screenshot input path with editable text read by the model before generation. Never treat uncertain extraction as confirmed source text.
- Start with one **Help me understand** action; let the model choose the most useful explanation style. Add explicit workflow choices later if learners need them.
- Let learners set explanation depth (overview, balanced, detailed) and assumed level (beginner, intermediate, advanced) without changing the single explanation action or response schema.
- Generate structured explanation content and optional diagram data; render diagrams as interactive React Flow elements.
- Keep generated output tied to source text using line references where possible.
- Do not require sign-in or save submissions by default in the first version.
- Run locally during initial development and use the user's configured Codex CLI for LLM requests.

## Proposed technical shape

- **Frontend:** React + TypeScript web application.
- **Interactive diagrams:** React Flow renders navigable nodes and labeled edges. Selecting a node reveals its explanation and linked source lines; nodes can be rearranged and the view can be zoomed or fitted.
- **Generation endpoint:** `src/server/local-api.ts` contains the local API routes and Codex CLI adapter; `vite.config.ts` only registers the plugin. Keep the CLI integration behind an injectable adapter so it can be tested and replaced or upgraded later. Use one-shot `codex exec` with structured output; consider Codex app-server only if the product later needs persistent agent conversations or streaming controls.
- **Task skills:** provider-neutral Markdown files for explanation, transcription, follow-up answers, and knowledge checks. Each API route loads only its task's skill; output schemas remain enforced separately by the host. Skills define the teaching approach, not executable code.
- **Output contract:** structured sections for summary, concepts, walkthrough, assumptions, and optional diagram nodes/edges. Validate the response and handle malformed or incomplete output gracefully.
- **Screenshot understanding:** send the screenshot to Codex as visual input in transcription-only mode. Do not add a separate OCR service for the MVP. Show extracted text and uncertainty cues for user correction and explicit approval before explanation; allow paste/edit as a fallback. Store the image only in a temporary directory and remove it when the request ends.
- **Execution boundary:** treat submitted code as input to explain, never as code to run. The local Codex CLI invocation ignores personal config and disables shell/code execution, plugins, apps, hooks, and multi-agent features; it also uses the read-only sandbox and an isolated temporary working directory.

## Milestones

**Status:** Milestones 1–4 are complete. Milestone 5 is in progress.

### 1. Input and workspace shell

- [x] Create the app shell and a focused learning workspace.
- [x] Add paste/edit input with a mode for code or documentation.
- [x] Add screenshot upload and a review/edit step for text read from the image.
- [x] Let the model infer language or document type; ask the user only when ambiguity materially affects the explanation.

**Done:** a user can submit pasted text or a screenshot, review and edit the source text (including text read from an image), then start an explanation with one action.

### 2. Skill-based explanations

- [x] Implement the provider-neutral `explain-material` skill, which chooses a suitable teaching approach while honoring the learner's depth and level settings.
- [x] Define and validate a shared structured output schema; show the explanation in a readable result view.
- [x] Include source line references and a visible assumptions/unknowns section.
- [x] Add loading, retry, and useful error states.

**Done:** the explanation workflow supports code and documentation, and references point back to the submitted text.

### 3. Interactive visuals

- [x] Generate optional control-flow or concept maps where a visual helps explain the material.
- [x] Render diagram data in React Flow; selecting a node reveals its explanation and linked source lines.
- [x] Improve diagram readability with labeled edges, separated labels, and a layered layout that reduces crossings and avoids overlapping cards.
- [x] Omit the visual when it would add little value.
- [x] Add layout tests for branching, convergence, uneven paths, crossings, and malformed/cyclic connections.

**Done:** generated diagrams are navigable, their nodes map to source text, and a diagram is not forced for unsuitable input.

### 4. Understanding checks and follow-up

- [x] Let the learner ask follow-up questions about the current source and explanation.
- [x] Let the learner select part of an explanation and ask a follow-up focused on that passage.
- [x] Preserve follow-up conversation context for the current page session only.
- [x] Add an optional three-question understanding check using `skills/knowledge-check/SKILL.md`, with immediate feedback and explanations.

**Done when:** learners can ask a focused, source-grounded follow-up without re-entering the source and optionally check understanding without saving scores.

### 5. Refinement

- [x] Refactor the page into focused input, explanation, and display-control components so `App.tsx` coordinates state rather than owning all page markup.
- [x] Add automated diagram-layout coverage for branching/converging graphs and awkward edge cases.
- [x] Add a branch-heavy pricing-flow regression case that retains the address, threshold, expedited-shipping, and membership outcomes through layout.
- [x] Add explicit skill guidance and a concrete branch-heavy example requiring distinct outcomes and visible convergence for every condition, including ternary expressions.
- [x] Add conservative source-range and outcome checks for supported control-flow syntax; keep diagrams visible and label incomplete or unverifiable coverage so learners can compare them with the walkthrough.
- [x] Move generated-response schemas into a dedicated module and enforce bounded response sizes, valid source ranges, and valid diagram edge endpoints before returning model output to the UI.
- [x] Strengthen documentation source-attribution guidance so each source-specific concept and walkthrough step points to its supporting text range.
- [x] Add a bounded second-pass explanation review that checks coverage, math, grounding, and diagram consistency; allow one repair attempt and disclose when review is incomplete.
- [x] Exercise the full workflows on representative short snippets, longer code, and documentation; record any explanation or source-mapping failures.
- [ ] Revisit screenshot correction, diagram rendering in the browser, and explanation quality using learner feedback.
- [ ] Decide whether session export, persistence, additional languages, or repository context is valuable for the next release.

#### Manual workflow check

- Short code (`Array.reduce`): the explanation and walkthrough correctly describe the accumulation, both link to source line 1, and the app correctly skips an unnecessary diagram.
- Branching code (12-line order flow): the output includes empty-order, inventory, payment-failure, confirmation, and sequential shipping outcomes, with overview and focused diagram views. The app warns that automated checks could not verify every source decision; address that coverage limitation in the next diagram/source-mapping pass.
- Documentation (rate limits): the summary, key ideas, walkthrough, quiz, feedback, and a source-grounded follow-up all worked. Two of three key ideas and both walkthrough steps lacked source-line links, while one key idea linked only to line 3. Improve range attribution for prose/documentation.
- The documentation example did not generate a diagram; a linear explanation did not need one.

### Diagram fit and print behavior

The React Flow canvas uses automatic fitting, resize observation, a lower zoom floor, a narrow-layout fit control, and print-specific styles. A remaining risk is visual behavior across real browser sizes and print engines:

- [x] Refit the graph when it mounts and whenever its container size changes, including transitions into print layout.
- [x] Revisit the zoom floor: allow enough zoom-out to fit, but avoid making labels unreadable. If the fit would fall below a readable scale, use the narrow-layout fallback instead.
- [x] Add a narrow-layout fallback: keep the graph pannable and provide a clearly visible **Fit diagram** control on narrow canvases or when the fitted zoom is small.
- [x] Add print-specific styles that replace the interactive canvas with a complete, readable text version of the diagram, avoiding clipped graphs and awkward page breaks.
- [x] Verify representative linear and branching graphs in a browser at wide and narrow sizes; confirm nodes and labels remain within the canvas or readable through the narrow-layout fit and scroll path.
- [x] Inspect actual Chrome print preview for a representative branching graph; confirm the complete text alternative and its connections appear on the printed page without clipping.

**Done:** visual verification confirmed representative linear and branching diagrams fit at wide and narrow widths, offer a usable fit/pan path on narrow screens, and print a complete text alternative without clipping. The branching example produced a two-page Chrome print preview; no PDF was saved.

## First implementation slice

Milestones 1–4 are implemented. The local app supports pasted code/documentation and screenshots, one explanation action, automatic teaching-mode selection, adjustable explanation depth and learner level, and local Codex CLI endpoints for transcription, explanation, follow-up questions, and an optional three-question understanding check. Each route loads its own provider-neutral skill. Quiz responses are session-only and are not persisted. Screenshot text is editable and must be explicitly approved; uploaded images are removed with their temporary request directory. Interactive React Flow diagrams show labeled edges and use a layered layout; selecting a node reveals its explanation and valid source excerpts. Branch, convergence, uneven-path, crossing, and malformed-graph layout cases have automated coverage. Selecting text in the explanation opens an action to ask a question about that passage; follow-up history is held in page state and discarded when the source changes or the page session ends.

## Product and quality requirements

- Keep source code and uploaded screenshots transient by default; do not add persistence without a clear user-facing choice.
- Do not expose Codex credentials or invoke the CLI from browser code; keep the integration in the local server process.
- Never execute user-submitted code as part of explaining it.
- Clearly distinguish what the submitted material states from assumptions based on missing context.
- Make the source text easy to correct and revisit while reading the explanation.
- Provide a useful text explanation even if diagram generation fails.
- Keep keyboard navigation and narrow-screen layouts usable.
- Provide a skip link, keyboard-operable input tabs, visible focus indicators, reduced-motion support, and semantic quiz controls with live feedback.
- Keep UI components in focused modules so the main app can coordinate state without containing every reusable view.

## Not in the first release

- Whole-repository indexing or automatic codebase ingestion
- Accounts, collaboration, or saved learning history
- Arbitrary diagram editing or a general-purpose visual programming environment
- Automatic code execution

## Current assumptions

- Development starts as a local application using the user's existing Codex CLI setup.
- Let the model identify languages and document types from the input; do not gate the workflow on a fixed supported-language list.
- Keep deployment, alternative model providers, and persistent agent sessions out of the first implementation slice.

## Craft export integration plan

### Current implementation

Opt-in WordCraft DOCX/PDF and VectorCraft PNG adapters with explicit
original/adapted/PR-result export controls are implemented. See `EXPORTS.md`
for setup, bounded PNG layout limits, local activation, and exact scope.
This first slice uses a cancellable direct download request instead of retained
jobs: one selected result, no artifact store, and cleanup before responding.
The job API below remains a proposal for larger asynchronous exports. Editable
SVG/native VectorCraft downloads, DOCX image embedding, DeckCraft
integration, multi-chapter bundles, and hosted deployment isolation remain
deferred. Pilot findings below are historical feasibility data,
not a claim that these deferred features shipped.

### Decision and evidence

Use Craft apps as optional local export backends, not explanation engines or
mandatory runtime dependencies. Keep React Flow as the in-app diagram renderer.
These pilots do not implement an Export button, production adapter, or MCP server.

| Backend | Pilot result on macOS arm64 | Intended role |
| --- | --- | --- |
| DeckCraft 0.18.1 | Five editable slides; native/PPTX round-trip and PDF checks passed | Presentations and slide PDFs |
| WordCraft 0.18.1 | Five chapters over four pages; DOCX text/metadata checks and independent LibreOffice render passed | Study guides and document PDFs |
| VectorCraft 0.24.1 | Four-node view; native text editing, SVG round-trip, and browser/PDF checks passed | Editable SVG and diagram image assets |

All pilots used one saved public FastAPI PR chapter and curated layouts. They do
not establish arbitrary-input layout quality, general correctness, performance
guarantees, Windows/Linux support, or native Microsoft Office compatibility.
WordCraft's optional illustration reuses the earlier DeckCraft image; its body
text remains editable, but that image is not an editable vector diagram.

### Phase 1 - Shared export snapshot and safety boundary

Create a versioned `StudyExportSnapshot` containing only the user-selected,
already-generated chapter variants. Include chapter IDs/titles, source text or
explicit omission markers, source hashes, PR URL/head/base, depth/audience/tone/
focus, concepts, walkthrough, graph nodes/edges, source ranges, coverage warnings,
incomplete-code findings, and the exact quality-review status/note. Record which
chapters were not selected or not generated. Never silently substitute the
original explanation for an adapted variant or trigger generation during export.

Validate the snapshot before dispatch: source ranges, graph endpoints, text and
node/edge limits, supported format, and safe artifact references. Start with one
selected chapter and a 2 MB snapshot cap; use a separate explicit consent option
for bundling source text. Without source, retain PR/revision links and say that
chapter-coordinate evidence is omitted. A supplied review label is provenance,
not independently verified proof or authorization to execute anything.

### Phase 2 - Optional adapters and local export jobs

Introduce an injectable `ExportAdapter` with capability/version detection,
`export(snapshot, destination, signal)`, and structured artifact/warning results.
Implement WordCraft DOCX/PDF first, then DeckCraft PPTX/PDF; keep VectorCraft as a
separate optional asset adapter. Missing executables disable only their formats,
with setup guidance; the learner can still read, copy, and study normally.
Configure approved executable paths server-side. Do not download binaries on
startup, bundle them in the browser, or inherit model/provider credentials into
renderer processes. Pin tested versions and review licences, notices, fonts,
binary checksums, and distribution requirements before shipping integrations.

Generate only allowlisted command JSON from typed templates. Never interpolate
source into shell commands, executable scripts, paths, or operation names. Use
`spawn`/`execFile` with argument arrays, `shell: false`, a minimal environment,
isolated temporary HOME/work directories, and no user plug-ins or GUI connection.
Use safe mode/no-prefs where supported. A temporary directory or CLI read root is
not an OS sandbox: assess renderer parsers and enforce filesystem/network
confinement before treating hostile files as safe. Initially accept only the
app's own structured content and server-owned assets, not uploaded Office files,
arbitrary SVG, remote images, templates, plug-ins, or native application commands.

Proposed local-only API: `POST /api/exports`, `GET /api/exports/:id`,
`GET /api/exports/:id/download`, and `DELETE /api/exports/:id`. Return an opaque job
ID, not paths or arbitrary file access. Reuse host/origin guards; they are not
authentication for a future shared deployment. Begin with one concurrent export,
120-second job deadline, bounded stdout/stderr and artifacts (25 MB), explicit
cancel, termination escalation, and cleanup after cancellation/failure or a
10-minute retention period. No model/image-provider requests or automatic retries.

### Phase 3 - UI and evidence-preserving layout

Add an optional Export action to the active chapter with detected formats,
chapter/variant selection, include-source consent, progress, cancel, errors, and
download. Keep it keyboard accessible and usable on mobile. Exporting must not
mutate the study, history, selected preferences, citations, or cached variants.

Use semantic document headings and editable slide text/shapes. Split dense views
and long chapters rather than shrinking text or deleting branches. Preserve
warnings visibly in every output, not only speaker notes; notes, captions, and
the manifest retain precise revision and source coordinates. Label partial and
unverified material consistently. A condensed export must state that its coverage
is selective. Package the selected output with source/evidence companions when
approved. Raster figures must be identified as such, not advertised as vectors.

For VectorCraft, preserve source-linked node IDs/edges and editable text. Validate
and sanitize exported SVG with an allowlist before preview/download: no scripts,
event handlers, foreignObject, external assets, or network font loads. Prefer a
safe PNG preview while retaining the sanitized SVG download. Decide on embedded
licensed fonts versus explicit font-substitution warnings; cross-render checks
are required. Do not enable arbitrary SVG editing or tool execution in the app.

### Phase 4 - Release gates

- Unit tests: snapshot validation, exact variant selection, source omission,
  missing/partial/incomplete warnings, hostile labels, path traversal, SVG safety,
  executable/version detection, cancellation, limits, and cleanup.
- Adapter tests with fake CLIs: argv/JSON correctness, no shell interpolation or
  leaked environment secrets, failures/timeouts, bounded diagnostics, malformed
  output, and no partial artifacts becoming downloadable.
- Opt-in real-CLI tests: repeatable synthetic fixtures, short/long/multilingual
  labels, dense branches, code blocks, images, incomplete reviews, source links,
  editable objects, page/slide overflow, and round-trip metadata preservation.
- Browser tests: feature unavailable, export scope/consent, unchanged study state,
  progress/cancel/download, retries only by user action, and mobile accessibility.
- Visual checks: every generated page/slide, an independent PDF/SVG renderer,
  native Microsoft Word/PowerPoint opening and editing, and target OS coverage.
  Keep untested combinations experimental; do not treat one pilot as a pass.

Completed slice: a validated selected-result snapshot, injected WordCraft and
VectorCraft PNG adapters, cancellable direct downloads, and explicit controls.
PDF now optionally embeds one selected server-generated diagram PNG. Next:
assess multi-diagram and multi-chapter exports and DeckCraft. Editable SVG needs a real
sanitization boundary before release. MCP orchestration, cloud rendering, video,
and hosted-service isolation remain deferred.
