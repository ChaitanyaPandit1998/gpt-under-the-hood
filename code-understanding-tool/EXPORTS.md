# Experimental local exports

## Design a visual

An existing explanation, PR chapter, or adapted section can draft a visual
brief. Open **Design a visual** and follow **What to teach**, **How it should
look**, optional **Preview prompt**, **Generate visual brief**, then **Render
image**. You can keep the defaults and skip straight to generation.
**Customize visual** offers the teaching question, form, audience, style,
output target, art direction, and additional facts or examples. Presets are
general visual forms, not scenarios. No request runs until you generate.

**Export diagram image** preserves the existing graph; **Design a visual**
creates a new illustration. Use **Create an illustration instead** inside the
diagram export panel to open the designer with the selected view as context and
the expressive style selected. You can change the style or choose **Use
whole explanation** before generating.

The model receives the current explanation and preferences; explanation text
may quote material. Raw source is sent only if **Include original source for
line references** is checked. **What the model receives** shows that choice
before generation. If the explanation omitted exact values or labels, paste
just those under **Additional context or data**. If you edited the prompt,
add them to the draft or reset it to use the latest notes.
The result is a reviewable brief with a caption,
text alternative, and guardrails. Select **Render image** to plan the drawing
with a second model request and render the validated scene through the local
VectorCraft CLI. The app retains that scene while the brief is open, so
changing size or retrying a failed PNG render does not call the model again.
The preview can be downloaded as a PNG. Review the drawing
before sharing; the scene checks enforce geometry and command safety, not
factual correctness. Existing diagram PNG and PDF export paths are unchanged.

Before rendering, choose **Standard** (1000 × 700), **HD** (1280 × 720), or
**Full HD** (1920 × 1080). The HD sizes are rendered directly from the scene.
The composition is scaled uniformly and centered rather than stretching or
cropping it to fit widescreen. These sizes apply to new illustrations only.

To try experimental compact scene planning, set
`ENABLE_EXPERIMENTAL_COMPACT_VISUAL_PLANNING=1` in the server's local
environment and restart Vite. The planner can describe repeated geometry and
labels once; the server expands and validates each item before rendering.
The default planner, browser API, image sizes, and export formats are unchanged.

The explanation screen and adapted section results offer **Export document**.
PR study results export only the current generated chapter and its selected
preferences. Adapted sections export the adapted result, not a fallback to the
original explanation. The original explanation's export is explicitly labelled
as original and excludes adapted variants. Export never generates content or
makes model/image-provider calls.

## Local PDF preview

Use **PDF preview** in the explanation tools to select sections and include
already rendered inline illustrations. The preview and printed pages use the
same content. Use the browser's **Save as PDF** destination to download it;
this option uses neither a server export nor another model call.

The preview is scoped to the current result (including a generated PR chapter
or an adapted section when opened there). It preserves review and coverage
notes and valid citations. The original source appendix is opt-in. Follow-up
threads, quizzes, decorative covers, and interactive diagram images are not
included. The separate **Export document** and diagram options are unchanged.

## Local setup

This integration is off by default. Set these server-only variables in
`.env.local` and restart the Vite development server:

```dotenv
ENABLE_EXPERIMENTAL_DOCUMENT_EXPORT=1
WORDCRAFT_CLI_PATH=/absolute/path/to/wordcraft-cli
```

Use the macOS universal executable from the official WordCraft 0.18.1 release
at <https://github.com/storytold/wordcraft/releases/tag/v0.18.1>. The adapter
checks the **executable** SHA256 and runtime version. It does not download or
install software or use a PATH fallback. Other builds and platforms are not
approved by this prototype. The executable is not included in the repository.

Tested executable SHA256:
`9d110decf7f99871cdc981798f11fa06d2daaa6f73621db71177bc5f5aee565c`.
This differs from the release archive checksum recorded in the pilot.

## Export behaviour

- Select Word or PDF and explicitly download; cancellation aborts the local
  request and terminates the renderer. Changing the exported result also cancels
  its pending request. There are no automatic retries.
- Source appendix inclusion is off by default and resets for each result.
  Omission is not redaction: the explanation itself may quote source code.
  The local server receives the current source to validate evidence and compute
  a source hash, whether or not the appendix is included in the download.
- Review status, exact review note, incomplete-code findings, assumptions,
  coverage notes, and chapter-local coordinates are included. PR exports retain
  URL, head/base when available, preferences, and import coverage warnings.
  Normal explanation exports say when preferences were not recorded.
- Diagrams are always preserved as node/edge text with citations. For PDF,
  choose one supported diagram under **Diagram image** to embed a VectorCraft
  PNG on its own page. This is optional and does not require a prior PNG export.
  DOCX remains text-only; the standalone PNG is still available. Follow-ups,
  other variants, chapter covers, and full-PR coverage are excluded.
- `GET /api/document-export` reports capability. `POST` accepts a version-1
  selected-result snapshot and returns the chosen attachment directly. Existing
  local host/origin guards apply. No job IDs, server artifact store, or extra
  download endpoint is needed for this first synchronous slice.
- One export per type at a time; 2 MB request limit; 120-second request deadline; bounded
  CLI output; 25 MB artifact limit. Temporary directories are removed after
  success, failure, and cancellation, before returning the artifact bytes.
  A forced process/server crash may leave an OS temporary directory behind.

## VectorCraft diagram PNG

Set `ENABLE_EXPERIMENTAL_DIAGRAM_EXPORT=1` and `VECTORCRAFT_CLI_PATH` to the
macOS universal executable from VectorCraft 0.24.1. The tested executable SHA256
is `d40e68ac59768c3957abe23c674484ad46a3abbe606e92a747001c8e4082edc4`.
It runs in safe mode with `VECTORCRAFT_NO_PREFS=1`, the same restricted environment
and cancellation/cleanup boundary as document export. No SVG or user-uploaded
vector files are opened, embedded or downloaded by this integration.

Expand **Export diagram image** below the chosen interactive diagram, then
generate the PNG. The app previews it in a scrollable region and offers download.
Switching diagram/result discards the old preview and cancels pending rendering.
The original interactive diagram, text alternative, and source selection remain.
The PNG is a raster export, not an editable vector or execution trace.

**Detailed** (the default) supports 1-8 nodes, up to 12 connections, and an
artboard at most 1000 by 8000 pixels. It retains all supplied edges, including cycles, self-loops,
and parallel connections, rather than silently dropping them. Node numbers are
layout references, not an inferred execution order. Connection labels have an
explicit legend. Review notes, coverage notes, incomplete-code findings and
chapter-local source coordinates are retained. Raw source is not appended, but
explanation labels may quote code. Full explanation assumptions remain in the
explanation/document, not this selected diagram image.

**Automatic branches** is opt-in for standalone PNG preview and download. It
supports up to 16 nodes and 24 connections, uses lanes for structured binary
branches, and otherwise uses a connection index with every supplied edge.
Decision labels sit beside the arrows; other supplied labels and the review
and coverage notes are retained below the lane chart. The detailed interactive
explanation remains unchanged. Illustrated PDF still uses the detailed layout
and its original limits.

Renderer warnings, text overflow, out-of-artboard bounds, unsupported dimensions,
or invalid PNG output fail the export instead of serving a partial image.
Larger views stay usable interactively and through textual document export.
`GET/POST /api/diagram-export` uses the same local guards and request lifecycle
as document export. Rendering consumes existing structured explanation data;
there are no LLM or image-provider calls.

## This workspace

Both integrations have been explicitly enabled in the ignored
`.env.development.local` for local acceptance testing. Verified executables and
their accompanying notices/licences are copied under ignored `.local-tools/`,
so this workspace does not depend on temporary-directory executables. Existing
provider credentials are unchanged. Neither local configuration nor binaries
are tracked or distributed. Delete the local override or set its flags to `0`
to disable the integrations. New clones remain disabled by default.

## Safety and limitations

Only typed, app-generated commands and plain text are accepted. No uploaded
documents, templates, arbitrary commands, images, or remote assets are opened.
The renderer uses an isolated temporary HOME, a minimal environment without
provider credentials, argument arrays with `shell: false`, and a pinned binary.
These are **not an OS sandbox**. This remains an opt-in trusted-local-workspace
experiment, not a secure shared-hosted export service. Do not use the endpoint
as authentication or enable arbitrary file uploads. Binaries/fonts are not
redistributed; licence and deployment review is required before packaging.

The API is part of the existing Vite development-server backend. The static
production build alone does not host this or the other local APIs. Native Word
compatibility, arbitrary long/multilingual layouts, platform support, and
production p95 performance remain unverified.
