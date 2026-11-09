# Explanation quality evaluation

This is a small, deterministic regression corpus for the explanation pipeline. It covers source-grounded control flow, unfinished code, mathematics, and prose documentation. The cases are synthetic and safe to keep in version control.

## Run the baseline

Run the current local Codex-backed pipeline for each case:

```sh
npm run eval:baseline
```

This writes a timestamped JSON file under `evals/results/` without overwriting earlier baselines. It records the source revision, elapsed time, complete API response, and automatic checks. The script calls the real local `/api/explain` middleware directly without opening a network port. It still invokes the configured Codex CLI and may use model quota.

The runner uses Node's built-in TypeScript stripping and requires Node 22.6 or newer.

To rerun one case or choose a result path explicitly:

```sh
node --experimental-strip-types evals/run.mjs --case documented-upload-retry-policy --output evals/results/retry.json
```

## Score each case manually

Compare the explanation with that case's `expectations` in `cases.json`. Score each expectation:

- **2** — stated accurately and clearly.
- **1** — partly covered or ambiguous, with no direct contradiction.
- **0** — missing or contradicted.

Then score these dimensions from 0 to 2:

- **Grounding:** factual statements are supported by the supplied input; linked source ranges are relevant; uncertainty is explicit where the input is incomplete.
- **Diagram fidelity:** for cases requiring a visual, every important condition/outcome is represented in the right order and with correct edge labels. Use N/A for cases without a visual requirement.
- **Incomplete-code handling:** for cases with unfinished code, explicit markers and actual fall-through are described without inventing intent. Use N/A otherwise.

Per-case factual score is `sum(expectation points) / (2 × number of expectations) × 100`. Also record each dimension separately; do not hide a diagram or grounding failure inside the average. An overall corpus score is the mean of case factual scores, with the separate dimension averages shown alongside it. Do not count the model's second-pass `passed` label as proof of correctness.

The runner leaves `manual_scores` as `null` by design. Fill them only after a human has read the captured response against the source and rubric. Retain the raw responses so later changes can be scored against the same cases. Add cases when a real failure is found, and preserve the old baseline for comparison.

The `visual_present` field is descriptive; it does not count as a passed/failed assertion. `visual_requirement_met` checks whether a diagram was produced when the case expects one. This avoids penalizing a clear prose-only explanation when no diagram is needed.

## First measured baseline

Measured on 2026-10-30 at revision `47343a7` with four synthetic cases and the configured Codex CLI:

- **Generation availability:** 3/4 cases succeeded (75%). The documentation-policy case failed twice with `The generated explanation was incomplete or invalid`; its content score is unavailable.
- **Factual criteria on successful outputs:** 26/28 points (92.9%). The end-to-end case score, counting the failed case as zero, is 68.8%.
- **Source grounding:** 6/6 across completed outputs.
- **Diagram fidelity:** 3/4 (75%) across the two cases requiring diagrams. The retry diagram does not draw its recursive loop back to the operation.
- **Incomplete-code handling:** 2/2 on the relevant case.
- **Review completion:** 3/3 successful outputs were marked passed or revised; one needed a repair and final review.
- **Mechanical checks:** 15/18 boolean checks passed.

The main quality gap in this run was the quadratic explanation: it named the discriminant but did not explain the positive, zero, and negative cases. The documentation failure was diagnosed as an out-of-range source citation (`explanation_concept_source_range_out_of_bounds`). The prompt now shows explicit one-based source line numbers for inputs up to 1,000 lines; a targeted rerun of that case succeeded, passed final review after a diagram repair, and scored 10/10 on factual criteria. See `results/diagnostic-retry-2026-10-30.json`. The original raw outputs and scores are in `results/baseline-2026-10-30.json`; the later full-corpus rerun is summarized below.

## Full rerun after source-line prompt fix

Measured on 2026-11-09 with four cases through the current working tree based on revision `47343a7`; the raw responses and human scores are in `results/baseline-2026-11-09T09-03-33-404Z.json`. The generated report marks `working_tree_dirty: true`, because the prompt fix and evaluation work had not yet been committed when this run was captured.

- **Generation availability:** 4/4 (100%).
- **Factual score:** 96.9% mean across the four cases; the quadratic formula case earned 7/8 because it listed the discriminant outcome categories without mapping positive, zero, and negative values to them.
- **Source grounding:** 100% across all four cases.
- **Diagram fidelity:** 3/4 (75%) across the two cases requiring diagrams. The recursive retry map still omits the return loop to the operation.
- **Incomplete-code handling:** 2/2 (100%) on the relevant case.
- **Review completion:** 3/4. The upload-policy output included an unnecessary cyclic diagram; the second-pass review marked it incomplete because the automatic repair could not satisfy the forward-only diagram constraint.
- **Mechanical assertions:** 16/16. Diagram presence is reported separately and only required where the case expects a visual.

The remaining issues are explicit: map the three discriminant signs to their outcomes, represent recursion without losing the retry relationship, and avoid generating a diagram for a prose policy when its cycle cannot be represented under the current diagram rules. These results are a small regression sample, not a production reliability claim.

These are small-sample baseline measurements, not a claim of production reliability. Keep the set fixed for comparisons, and add new cases when a real failure reveals a missing category.
