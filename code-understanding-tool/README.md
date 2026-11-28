# Code Understanding Tool

An interactive workspace that helps people understand code and documentation through clear explanations, useful visuals, and follow-up questions.

See [PLAN.md](PLAN.md) for the implementation plan, milestones, and current decisions.

See the [architecture guide](docs/architecture.md) for the VectorCraft system diagram, runtime flows, and local/external boundaries.

## Run locally

```sh
npm install
npm run dev
```

The app loads a task-specific skill and sends generation requests to Codex CLI through the local Vite server. Paste code or documentation and select **Help me understand**, or load a public GitHub PR to see its chapter outline and generate explanations on demand. Private PRs are not supported. Codex CLI must be installed and signed in. The server looks for `codex` on its `PATH`, checks the packaged macOS Codex CLI location, and supports `CODEX_CLI_PATH` for other installations. One-shot requests ignore personal Codex configuration and disable shell/code execution and extension tools. Screenshot text is shown for editing and requires explicit approval before explanation; image files are temporary.

If a request reports that Codex CLI cannot reach its service, check network access for the terminal running `npm run dev`. Restricted or offline development environments may block the CLI connection even when Codex is signed in; start the local server from a terminal with network access and retry.

New PR chapters show an **Unreviewed draft** first. Choose **Review this
chapter** to check the draft against the diff, or enable **Review automatically**
to start that check as soon as the draft is ready. Select **Include diagrams**
to opt in to diagrams for that chapter. Saved studies keep results for each
diagram setting separate. Story and illustration tools remain on demand.

## Story Mode

After generating an explanation, select **Story** in the chapter reader.
Each story scene separates the fictional analogy from its factual meaning and
links to supporting source lines when available. Generating a story sends the
selected chapter context and original source to the configured Codex CLI.
Switch back to **Direct** at any time. Open **Illustrate this scene** to
choose a fictional **Story illustration** or a **Factual diagram** based on
the scene's factual mapping. Preview the design prompt, generate a brief, and
render with the configured VectorCraft CLI. No image is generated automatically.
Fictional illustrations can reuse built-in faces and hair when the scene calls
for people. Bodies, gestures, and props are planned per scene; the app does
not yet lock a character assignment across chapters.
Saving a local study keeps completed stories and accepted scene images. If
the saved chapter was showing a completed story, it reopens in Story Mode.
Stories and their images are not included in document/PDF exports. Individual
images can be downloaded.

## Portable Studies

Save a code or document study locally, then use **Export saved study** to
download a JSON file containing the saved source, explanation, reading state,
completed stories, and accepted PNGs. Unsaved changes are not included.
Use **Import study file**, then **Resume saved study** on another browser.
Transfer and resume do not generate new content or upload the file.
Import validates the file before saving and asks before replacing an existing
save. Treat exported files as sensitive: they contain the original source.
The transfer limit is 40 MB. PR studies and decorative chapter covers are not
supported in this first version; studies with covers are rejected rather than
silently losing them. Story scene images are supported.

## Automated checks

GitHub Actions runs **App checks** on pull requests targeting `main` and on
pushes to `main`. The job runs lint, unit/API tests, the production build,
and the desktop and mobile Playwright projects in Chromium. Provider calls
are mocked; no API keys or Craft installations are required. Download
`playwright-failures` from a failed run to inspect browser traces or screenshots.

Run the same app checks locally with Node.js 26:

```sh
npm ci
./node_modules/.bin/playwright install chromium
npm run check:all
```

## Security and privacy boundary

This is a local-only prototype. Both Vite development and preview servers bind to `127.0.0.1`; the API accepts loopback-host requests and rejects cross-origin browser requests. Do not expose it with `--host`, a reverse proxy, a tunnel, or a public deployment. The host/origin check is a browser-request guard, not user authentication or protection for a shared server.

The app passes submitted source text, follow-up questions, quiz context, and screenshot images to the Codex CLI signed in on the same machine. The CLI may send that content to its configured service. When loading a PR, the local server makes outbound requests to the GitHub API for public PR metadata and file patches; that content is then passed to Codex when you request an explanation.

Completed Code and Documentation explanations, their source, and current chapter are recoverable after refreshing the same tab via browser session storage. Uncheck **Keep source and completed explanation after refresh in this tab** or clear the input to remove the temporary copy. Invalid recovery data is discarded. Unsubmitted drafts, PR studies, stories, illustrations, and follow-ups are not part of that copy. **Save study locally** is separate and stores a study in this browser's IndexedDB for later use.

On the server, temporary request files are created while processing and removed afterward. Operational diagnostics include random request IDs, endpoint names, phase durations, outcomes, and broad failure categories; they must not include source text, prompts, screenshots, or raw provider errors.

Before hosting this for multiple users, replace the Vite middleware with a production API and establish a real security boundary: authenticate users, protect state-changing requests against CSRF, isolate provider credentials per server-side identity, enforce per-user and global rate/concurrency/request-size limits, keep uploaded files in isolated temporary storage, define retention and deletion policies, and add redacted structured observability and abuse monitoring. Also review provider data handling, network egress, secret storage, dependency/build hardening, and deployment access controls. The current prototype is not production-ready for shared or public use.

## Evaluate explanation quality

Run the synthetic regression corpus against the local Codex-backed explanation pipeline:

```sh
npm run eval:baseline
```

This requires Node 22.6 or newer, a signed-in Codex CLI, and access to its configured service. It sends test inputs through the real explanation and second-pass review flow, may use model quota, and saves raw responses with automatic checks under `evals/results/`. Human rubric scores are added only after comparing each response with its written expectations. The corpus, scoring rubric, run instructions, and measured results are in [evals/README.md](evals/README.md). This small synthetic set is useful for regression tracking; it does not establish production reliability.

The latest full run generated all four explanations and scored 96.9% on factual criteria. Its remaining gaps were the missing recursive return edge in a retry diagram, an incomplete review for an unnecessary cyclic diagram, and missing explicit positive/zero/negative discriminant outcomes. Use later runs against the same corpus to check whether changes address these gaps.

Select a passage in the explanation to ask a follow-up focused on that text. The selected passage is sent with the question and checked against the full source.

Use **Start a quick check** below an explanation to generate three multiple-choice questions from the submitted material. Answers and score stay in the current page session and are not saved.

Use **Save study locally** or **Save PR study locally** to keep one study in
this browser. The saved copy includes the source or pinned PR evidence,
chapter position, completed explanations and variants, completed stories,
and accepted illustrations.
Resume or delete it above the workspace. This is opt-in, not encrypted
storage or a backup. Follow-up conversations, quizzes, and in-progress work
are not saved, including unaccepted images in comparison. To keep recent
changes, save again explicitly.

Before generating an explanation, set **Depth** to Quick overview, Balanced, or Detailed walkthrough, and choose the learner level (Beginner, Intermediate, or Advanced). The model adapts detail and terminology while keeping the same structured response format.

The UI keeps diagram rendering (`src/components/GraphView.tsx`), source citations (`src/components/SourceRanges.tsx`), and the understanding check (`src/components/UnderstandingCheck.tsx`) separate from the workspace orchestration in `src/App.tsx`. The diagram also has a text alternative for readers who do not use the interactive canvas.

Run `npm run check` to run ESLint, unit and API integration tests, and the production build. Integration tests exercise each local API route using a stubbed Codex runner; they do not require network access or a signed-in Codex CLI.

## Starting scope

### Inputs

- Pasted code snippets
- Screenshots of code or documentation, with model-read text shown for correction and approval before explanation

### Explanation skills

Use a small set of reusable skills to guide the teaching approach and output, such as:

- Plain-language walkthrough
- Concept-first explanation
- Control-flow explanation
- Knowledge check quiz

### Visuals

Generate a visual only when it helps explain the material. Start with:

- Interactive control-flow diagrams
- Component or concept maps

Keep code and explanation linked, so selecting a visual element can reveal the relevant code. Clearly label inferences when the input lacks surrounding project context.

## MVP sequence

1. Build the input and editable text workspace.
2. Load the explanation skill and produce structured explanations through Codex CLI.
3. Add interactive visuals and link their elements back to source lines.
4. Add screenshot input and a model transcription review step. (Implemented)
5. Add grounded follow-up questions that use the current source and explanation. (Implemented)
6. Add optional quizzes and more documentation context based on feedback. (Quiz check implemented)

## Initial product principle

Treat this as a code learning workspace, not just a code-to-diagram generator. Explanations, visuals, and questions should work together to build understanding.
