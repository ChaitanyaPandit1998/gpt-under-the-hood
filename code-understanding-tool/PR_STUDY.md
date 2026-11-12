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

## Verification

Run `npm run check` for lint, tests, and the production build. Planner tests cover pagination, safe URLs, rate limits, snapshot changes, inventory mismatches, evidence limits, and missing patches. API integration tests verify that plan creation does not invoke the model and remains behind the local-origin guard.
