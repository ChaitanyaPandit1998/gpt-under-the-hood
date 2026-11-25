# Visual validation plan

Status: contract/report foundation, deterministic runner, bounded PNG decoding
and an opt-in app image-review panel. Not an automatic image-approval gate.

## Goal

Evaluate the complete explanation -> brief -> drawing plan -> PNG pipeline.
Keep factual fidelity separate from aesthetics, successful rendering, and
technical density. A polished picture is not evidence of correctness.
All production logic and committed fixtures must remain topic-independent.

## Evaluation layers

| Stage | Question | Evidence |
| --- | --- | --- |
| Brief | Does the draft preserve the teaching goal, claims and qualifiers? | Explanation, contract and brief |
| Plan | Does the drawing encode the brief without inventing relationships? | Brief, contract and scene items |
| Raster | Is the actual output readable and faithful to the plan? | PNG, dimensions, measured bounds, OCR and visual inspection |
| Alignment | Does the image convey the explanation accurately? | Explanation, narrative/factual mapping, contract and actual PNG |
| Audience | Does it teach at the requested level? | Explicit audience objectives and actual PNG/caption |

Beginner objectives emphasize understandable relationships without removing
qualifiers. Intermediate objectives may require mechanisms and assumptions.
Advanced objectives may require precise notation, domains and limitations.
These are guidance, not fixed equation counts. A story illustration may teach
through analogy, but must not present fictional geometry as factual evidence.
Review narrative transformations against their factual mappings as well.

## Contract and report

`src/domain/visual-evaluation.ts` defines versioned runtime-validated contracts
and reports. A contract binds the explanation digest to a teaching goal,
audience objectives, and uniquely identified claims, elements, qualifiers and
prohibitions. Each requirement records its evidence role and reference.
References are descriptive metadata, not automatically verified citations.

Scope contracts to the selected scene, its audience and the intended role of
the visual. State whether required material belongs in the image, caption or
surrounding explanation. Do not accidentally require every advanced detail
from the full source in an intermediate story scene. A contract can create
false rejections even when the reviewer correctly follows it; changes to a
contract require a separately recorded evaluation, not overwriting the first.

A report binds brief, plan and PNG digests to the contract. Image-review reports
also bind the separately displayed caption digest (older preflight reports may
omit it). Each stage records
reviewer kind/name/version, elapsed time, coverage, findings and observed
evidence. Compute digests from exact evaluated artifacts, not model output.
Never render report text as HTML. These structures are untrusted inputs;
runtime validation checks shape and linkage, not truth or reviewer identity.

Outcomes are derived, not supplied by the model:

- Fail: any failed check or critical finding, including factual contradictions.
- Uncertain: missing/unrun/uncertain stage, or an otherwise passing report with uncovered requirements.
- Revise: requested changes or a major finding after required reviews ran.
- Pass: all five stages pass, every requirement is covered and no blocking findings remain.

Minor findings may accompany a pass. Coverage declarations need independent
review: listing a requirement ID does not establish that it was met. A single
bounded repair can be recorded; each repaired artifact needs a fresh report.
No scalar aesthetic average can override these gates.

## Delivery sequence

1. Foundation: contract/report validators, conservative verdicts and generic unit tests.
2. Deterministic suite: share existing label checks with the renderer; add bounds,
   contrast, text-size and shape-occlusion checks where measurable. Test invalid
   output dimensions, clipped/overlapping labels and excessive text. Record
   unsupported checks as uncertain rather than treating geometry as OCR.
3. PNG review: inspect the actual raster plus its separately displayed caption.
   Add OCR and an independent multimodal reviewer for claims, qualifiers,
   invented data, misleading arrows/scales and narrative/mapping consistency.
   Model failure or unavailable vision must never silently approve an image.
4. Recovery/UI: show findings and evaluation stages; offer one bounded repair,
   preserve accepted images, and retain original and replacement reports.
5. Calibration: human-review a topic-diverse benchmark across audience levels,
   purposes and resolutions; measure false passes/false rejects before enabling
   an approval gate. Keep the public-math pilot external to committed fixtures.

## Test and release policy

Fast offline PR tests use generic fixtures and intentionally defective variants:
missing qualifier, reversed relation, invented measurement, omitted mechanism,
unreadable label, and analogy presented as observation. Validators must reject
unknown fields, stale bindings and invalid requirement references. Missing
stages and critical findings must never yield a pass.

Live provider evaluations run separately with explicit consent for submitted
material. Store explanation, contract, brief, plan, PNG, caption, checks,
reviewer versions, timings and repair history in a local opt-in run directory.
Do not commit private source, credentials or raw private prompts. Human labels
must remain distinguishable from model findings and mocked test results.

Acceptance requires actual-PNG and caption inspection, no unresolved critical
findings, all required stages run and all requirements reviewed. Automated
pass means the configured checks passed, not proof of technical correctness.

## Current limits and next step

Run the offline suite with `npm run test:visual`. The reusable server runner is
`src/server/visual-evaluation-runner.ts`. It hashes the exact explanation, brief,
plan and PNG bytes, rejects stale explanation bindings, validates scene shape,
and uses shared label checks when supplied with font-versioned measurements.
It checks nominal exported text size, contrast against known solid backgrounds,
and possible occlusion by later shapes. Path intersection is a conservative
bounding-box warning, not precise polygon or glyph coverage. Unknown mixed
backgrounds are not assigned a contrast ratio. Contrast uses a uniform 4.5:1
preflight threshold, not a claim of accessibility compliance.

`src/server/visual-png.ts` now decodes static non-interlaced 8-bit PNGs using
[pngjs](https://github.com/pngjs/pngjs). Evaluation limits are 10 MiB input,
16 million pixels and 2048 chunks. It rejects duplicate headers, animation,
truncation/trailing bytes, unexpected dimensions and excess decompressed
scanlines before decoding with CRC checks enabled. Uniform or fully transparent
rasters require revision. Unsupported formats conservatively fail this gate;
that does not imply a factual content error. Decoding is synchronous and bounded,
but a deployed high-concurrency service should isolate it from request handling.

Real encoded generic PNG fixtures cover pixel decoding, uniform/transparent
output, checksum corruption, truncation, duplicate headers and limits. A
header-only input now fails instead of remaining uncertain. Scene fixtures
exercise clipping, overlap, excessive text, contrast, occlusion and small text.
These are not semantic benchmark images or human quality judgments.

`src/server/visual-review-service.ts` takes the same artifacts plus the exact
caption. It runs preflight, then uses the existing runner's image attachment
argument for an independent model call. The caller supplies reviewer name and
version; the model supplies five structured stage checks and observed evidence.
The rubric checks narrative/factual consistency, invented evidence, qualifiers,
legibility and explicit audience objectives, not equation count or decoration.
Alignment cannot pass with missing requirement coverage. Deterministic blockers
are retained even if the reviewer claims pass. Missing vision, invalid output or
provider failures remain unreviewed; there is no automatic retry. Cancellation
propagates, and private temporary attachments are removed after the call.
The single model-call elapsed time is associated with each jointly reviewed
stage; do not sum those per-stage times as independent calls.

The model's assertion that it inspected the image is not independently provable.
Mocked service tests validate attachment plumbing, response validation and
failure policy, not reviewer accuracy. There is no separate OCR engine yet.

The app now exposes `Review this visual` after a successful new render, inside
the visual builder. The editable checklist is drafted deterministically from
the brief, not independently verified. Users select an audience if it was
unknown, choose the visual role, edit the teaching goal and audience objective,
include/edit/add requirements and choose their location (image, caption, joint
image/caption or surrounding explanation). Explicit checklist confirmation and
submission are required; merely opening the panel makes no model request.

`POST /api/visual-review` uses the same loopback/origin protections, concurrency
guard and cancellation infrastructure as other model requests. It validates a
bounded JSON/base64 request, consent flag and explanation digest before calling
the reviewer. The original source is not separately attached, but explanation
context can quote it. The model/version defaults are labeled unverified rather
than pretending the app knows the underlying model version.

The browser verifies returned explanation/artifact/caption bindings. Five
expandable result sections show findings and evidence using app theme tokens.
Editing the checklist marks its last completed report outdated and resets
confirmation. A new render clears temporary review state. Cancellation/failure
does not replace the image, and late responses cannot publish results after
cancellation/unmount. A successful review never auto-accepts a candidate or
changes which image is exported. New images now retain their brief, drawing
plan, scoped explanation context and up to five recent reports within a bounded
metadata budget. Reports for a candidate stay with that candidate until accepted.
Persistence requires the existing explicit Save study locally action; code/docs
portable transfers also include these materials. Images and review reports are
not included in same-tab explanation recovery.

Only newly rendered images have a locally available drawing plan and brief for
this panel. Restored images without that provenance cannot be reviewed through
it yet. Newer restored images expose read-only historical reports after checking
exact explanation/brief/plan/PNG/caption hashes. Save, restore and portable
import/export also validate these bindings; older images without metadata remain
supported. Hash integrity does not establish reviewer identity or truth, and
imports can contain user-edited assertions. No saved report auto-approves an
image. Restored images with provenance now offer explicit re-review using their
recorded context; fresh checklist confirmation is mandatory and old images
without provenance cannot reconstruct missing input. Optional visual-generation
lineage records exact loaded skill/profile stamps bound to brief/plan artifacts.
Automatic repair and independent contract generation remain pending. A selected scene
review does not audit unrelated story scenes.
Deterministic-only reports claim no requirement coverage, leave semantic stages
unrun and never return overall pass. A model-reviewed pass means the configured
reviewer accepted the artifacts, not proof of correctness or human approval.
Browser label preflight remains the automatic layout gate; review is optional.
An initial external single-topic pilot ran seven live calls, including actual
PNGs with deliberately misleading labels while their plans stayed honest.
The reviewer detected both image-label contradictions and an existing
narrative/mapping contradiction. Audience-scoping errors in an initial contract
also demonstrated the need to review the contract itself. After external
contract/narrative corrections, the original captioned story images passed.
These results were compared with coding-assistant inspection, not independent
human annotations. They do not establish calibrated quality or standalone
technical-diagram usefulness.

Review guidance now loads from the allowlisted review-explanation-visual skill;
the exact skill hash is recorded in reviewer version metadata. Deterministic
checks and response validation remain in code. Focused persistence/reviewer
checks passed, but restored-history browser coverage is prepared and unrun.

Next run rigorous integrated evaluation of generation lineage, restored review,
consent/cancellation and actual visual quality. Keep automatic acceptance disabled
until broader cross-topic and independent human evaluation.
