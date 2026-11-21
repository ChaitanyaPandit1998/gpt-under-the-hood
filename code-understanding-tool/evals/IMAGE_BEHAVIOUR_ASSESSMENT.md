# Can generated images explain code behaviour?

## Finding

Use FLUX.1 Schnell artwork as optional conceptual chapter decoration, not as evidence of execution behaviour. The existing pilot did not establish that images improve code comprehension. No user study, runtime trace evaluation, or broad model comparison has been performed.

## Observed pilot evidence

Two request-validation images asked for validation before processing, an accepted route into processing, and a rejected route that stops before processing. Direct visual inspection by the Codex assistant found ambiguous routes in the concept card and ambiguous arrows plus garbled text in the storyboard. Both received subjective semantic-accuracy scores of 1/4 and were rejected for technical teaching. Generation took approximately 3.0 and 2.5 seconds.

Two subsequent decorative illustrations asked only for a blank document and magnifying glass. Both produced recognisable, text-free artwork in approximately 1.6 seconds. The isometric style was preferred for composition and object recognisability. This supports its suitability as artwork in those two samples, not correctness of technical explanations or consistency across repeated generations.

The raw outputs, generation metadata, and separate visual-review notes remain local under `evals/image-results/2026-11-13T11-03-24-414Z/` and `evals/image-results/2026-11-13T11-06-41-365Z/`. They are ignored generated artifacts, not committed evidence. Costs were not measured; no free-allocation availability is assumed. The generations were unseeded because the live REST schema rejected seed.

## Product boundary

- Artwork can suggest a topic, such as examining a request or adapting an interface.
- Artwork must not claim branch coverage, ordering, retries, state changes, or security guarantees.
- For behaviour, use the source-backed walkthrough, explicit control-flow maps, and exact source citations already in the explanation. Those remain model-assisted and should still be checked against the source.
- Label every cover **Conceptual illustration** and keep it removable.
- Never send real chapter material to the image provider through this feature. The optional UI selects a fixed generic theme instead.

## Verification added

Unit and API tests enforce the fixed-prompt privacy boundary, explicit approval, request limits, provider-error redaction, and cancellation. Desktop/mobile browser regressions confirm that image generation leaves the original explanation and source citation navigation intact, restores only the correct chapter's cached cover, and ignores cancelled or late images. Mocked browser tests do not establish real image quality or learning outcomes.

## Future behavioural evaluation

To claim a learning benefit, compare explanation-only, explanation-plus-artwork, and explanation-plus-structured-diagram conditions using the same independently checked code examples. Measure correct predictions of outputs and branches, not visual appeal. Include invalid inputs, early exits, bounded retries, and asynchronous error paths. Do not infer a benefit from image recognisability alone.
