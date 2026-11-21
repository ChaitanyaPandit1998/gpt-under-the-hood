# Coverage-aware PR study

Loading a public GitHub PR now builds a plan before any model request:

1. Fetch PR metadata and up to five pages of changed files (500 files maximum).
2. Check the head revision and reported file count again after pagination. Reject an import if the PR changed.
3. Retain a file manifest, including files without usable patches.
4. Group available patches deterministically by file role and directory. Show implementation chapters before dependencies, automation, documentation, and tests.
5. Generate only the selected chapter using the existing draft/review/repair pipeline.

## Scope and limits

- Each GitHub response is bounded to 1 MB; import has a 30-second timeout.
- The patch evidence budget is 600,000 characters across the plan.
- Individual patches are excerpted at 24,000 characters when necessary.
- Each evidence chunk contains at most eight files and approximately 36,000 patch characters, plus shared context.
- A chapter includes the PR description, revision, outline, coverage limits, and only its own patch evidence.
- The overview has an inventory, not implementation patches. Its prompt explicitly prohibits inferring behavior from filenames.
- File coverage distinguishes `included`, `partial`, `unavailable`, and `omitted_budget`. An included patch is not a complete repository file.
- `filesComplete` describes the fetched inventory, not whether all patches were available or analyzed.
- Groups are organizational heuristics, not semantic dependency analysis. No repository cloning, arbitrary URL fetching, or code execution is performed by the importer.

## Reading sessions

Completed explanations are cached in memory by chapter, depth, and learner level. Switching chapters cancels pending generation and resets follow-up conversation context. Returning to a completed chapter restores its explanation without a model call. Regeneration replaces that cached result.

Citations select lines in the displayed chapter source, not GitHub file line numbers. Copy exports the chapter explanation together with its source and PR head revision as JSON.

Reloading or leaving the study clears the session. Reimport the PR to inspect newer commits. Code and documentation inputs retain the existing explanation workflow; the legacy `/api/pull-request` endpoint remains available.

## Additional patch evidence

Open the file manifest and select **Load patch section** for an excluded, excerpted, or unavailable file. The guarded `/api/pull-request-evidence` endpoint retrieves only its original manifest page, verifies the file path, and checks the head revision before and after fetching. The base revision is also pinned when present in the plan, since a changed target branch can alter the diff even without new PR commits.

Oversized patches are split into consecutive sections of at most 24,000 characters, preferring newline boundaries. **Load next patch section** advances without gaps. Each section becomes a separate evidence chapter; fetching never automatically invokes the model. Retrieval status is tracked separately from initial import coverage and from explanation generation.

Requests retain the 1 MB GitHub-response limit and 30-second timeout. GitHub-unavailable patches cannot be reconstructed. A page session allows at most 100 additional sections. Retrieval remains limited to the first five manifest pages; this does not bypass the inventory cap. Navigating away, cancelling, or closing the study ignores late responses.

## Verification

Run `npm run check` for lint, tests, and the production build. Planner tests cover pagination, safe URLs, rate limits, snapshot changes, inventory mismatches, evidence limits, and missing patches. API integration tests verify that plan creation does not invoke the model and remains behind the local-origin guard.

Run `npm run test:e2e` for Playwright browser regressions, or `npm run check:all` for both suites. First install the browser with `npx playwright install chromium`; alternatively, use an already installed Chrome with `PLAYWRIGHT_CHANNEL=chrome npm run test:e2e`.

Browser tests run at desktop and mobile sizes on an isolated server at port 5175. They mock GitHub/model endpoints, so they do not spend model credits or require GitHub credentials. Tests cover import, generation, cache restoration and preference isolation, citations, oversized-file section navigation, cancellation, stale responses, revision errors, and narrow-screen overflow. These UI tests complement API tests; they do not benchmark real model quality. Failure screenshots and traces are saved under `test-results/`.

## Optional chapter covers

PR study chapters and the chapter-reading mode for standalone code/documentation offer an optional cover. Open **Optional chapter cover**, select a reviewed generic theme, inspect the exact prompt, and explicitly approve sending it to Cloudflare and any applicable charges. Nothing is generated on page load or chapter selection.

The server accepts only a theme identifier and boolean approval, not chapter titles, PR URLs, source code, summaries, or custom prompts. It builds an isometric prompt from fixed public presets and calls FLUX.1 Schnell with four steps and no seed. Configure `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` locally and restart Vite; credentials are used by the server plugin, not client code. Missing configuration produces a clear error without a provider call.

Covers are labelled **Conceptual illustration** and explicitly described as unverified decorative artwork, not code-behaviour diagrams. They are cached by chapter for the page session and can be hidden or removed. A new explanation, study closure, or page reload clears the relevant session. A server instance allows at most 12 cover requests, including failed requests, and shares the existing model-concurrency limit. There are no automatic retries. Cancellation aborts the local request but cannot guarantee that provider-side work is not billed.

Covers never modify explanation text, control-flow diagrams, citations, or review status. See `evals/IMAGE_BEHAVIOUR_ASSESSMENT.md` for the pilot findings and their limitations. Browser tests use mocked image responses to verify consent, privacy boundaries, chapter isolation, cancellation, and intact source navigation without spending credits.

## Chapter-Local Explanation Preferences

Each PR chapter has independent depth, audience, tone, and focus settings.
Use **Simplify** / **Go deeper**, or expand **Customize explanation**. Settings
alone do not invoke a model: choose **Generate this chapter** to create a variant.
Returning to an already generated preference combination restores its cached
result without a new request. Copies include the selected preferences and source.

For code and documentation, switch to **Chapters** to generate an adapted section
alongside its original. Citations continue to refer to the original source, and
warnings remain available. These section variants are supplemental: copy, speech,
and understanding checks still use the original explanation. Navigating away
cancels in-flight section generation. All preferences and variants are page-session
only; refreshing clears them.

Icons and a depth meter improve scanning, while factual behaviour stays in
source-linked text and diagrams. Optional generated cover images are conceptual,
not evidence of execution or correctness. See [DEVLOG.md](DEVLOG.md) for changes
and validation history.
