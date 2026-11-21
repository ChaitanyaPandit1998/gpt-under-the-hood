# Cloudflare image pilot

This is an offline-first, six-image pilot, not an integration into the explanation UI and not a statistically representative model benchmark.

## What gets sent

Only the three manually authored synthetic cases in `image-cases.json` are used: version compatibility, request validation, and bounded retries. Each case has a concept-card prompt and a storyboard prompt. All use FLUX.1 Schnell and four steps. Generation is unseeded: the live REST schema supports only prompt and steps and rejects seed, despite some documentation examples showing it. Reports record seed as null; results are not a controlled-seed comparison.

No real PRs, source code, filenames, repository names, URLs, or personal data are loaded. The runner accepts a case ID but no arbitrary input-file path. Pattern checks reject some obvious identifying content; these checks are not a general-purpose privacy sanitizer. New cases require manual review before live use.

## Commands

`npm run eval:images` prepares exact prompts, a JSON report, and a local gallery without any API calls.

For live use, configure `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in the local environment or an ignored `.env`/`.env.local` file. Do not put them in client-side `VITE_*` variables, commit them, or paste them into chat. Use a token scoped to Workers AI access for the intended account. The `.env.example` contains empty placeholders only.

`npm run eval:images -- --live` makes at most six sequential requests. `npm run eval:images -- --live --case request-validation` makes at most two. There are no automatic retries; the batch stops at its first error. Missing credentials block live execution before any API request.

`npm run eval:images -- --live --suite decorative` runs a separate two-image artwork experiment with flat editorial and isometric styles. Its curated brief is in `image-illustration-cases.json`. Judge these only as decorative chapter covers: object recognizability, relevance, clean composition, and absence of misleading technical claims. A successful decorative cover is not a successful teaching diagram. The original technical cases and earlier run reports remain unchanged.

Outputs are written to the ignored `evals/image-results/<timestamp>/` directory: exact prompts and metadata in `report.json`, a contact-sheet-style `gallery.html`, and image files for successful requests. Credentials and raw provider error bodies are not persisted.

## Review rubric

Score each image from 0 (unusable) to 4 (strong):

| Field | Question |
| --- | --- |
| Relevance | Does the image communicate this chapter's actual concept? |
| Semantic accuracy | Are ordering, alternatives, stopping conditions, and outcomes represented correctly? |
| Clarity | Can the relationships be understood at a small chapter-card size, without relying on color alone? |
| Teaching usefulness | Does it add understanding beyond decorative appeal? |

Inspect the case's required elements and failure modes. List misleading elements explicitly. A contradiction of the summary is a rejection regardless of visual appeal. A provisional pass requires semantic accuracy and clarity of at least 3, with no critical misconception. Scores and decisions remain `null`/`not_reviewed` until a human or an explicitly identified reviewer inspects the actual images; failed requests are not scored as images. Even passing images remain conceptual illustrations, not source-backed diagrams.

Record latency, image bytes, request ID, and provider-reported billing if available. `billedUsd` and `billedNeurons` are unknown by default, not zero. Check the Cloudflare dashboard for actual usage. Shared free allocation may already be consumed by other workloads; the runner does not assume generation is free or change billing settings.

## Sources

- [Cloudflare FLUX.1 Schnell input schema](https://developers.cloudflare.com/workers-ai/models/flux-1-schnell/schema-input.json) documents the prompt limit and steps and disallows additional fields. The [model guide](https://developers.cloudflare.com/workers-ai/models/flux-1-schnell/) documents the base64 image response.
- [Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/) documents the shared daily Neuron allocation and current model rates. Check it again before interpreting cost results.
