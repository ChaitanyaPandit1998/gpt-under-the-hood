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
- **Generation endpoint:** a small local server-side adapter invokes Codex CLI for bounded explanation requests. Keep the CLI integration behind an adapter so it can be replaced or upgraded later. For the first version, use one-shot `codex exec` with structured output; consider Codex app-server only if the product later needs persistent agent conversations or streaming controls.
- **Task skills:** provider-neutral Markdown files for explanation, transcription, follow-up answers, and knowledge checks. Each API route loads only its task's skill; output schemas remain enforced separately by the host. Skills define the teaching approach, not executable code.
- **Output contract:** structured sections for summary, concepts, walkthrough, assumptions, and optional diagram nodes/edges. Validate the response and handle malformed or incomplete output gracefully.
- **Screenshot understanding:** send the screenshot to Codex as visual input in transcription-only mode. Do not add a separate OCR service for the MVP. Show extracted text and uncertainty cues for user correction and explicit approval before explanation; allow paste/edit as a fallback. Store the image only in a temporary directory and remove it when the request ends.
- **Execution boundary:** treat submitted code as input to explain, never as code to run. The local Codex CLI invocation ignores personal config and disables shell/code execution, plugins, apps, hooks, and multi-agent features; it also uses the read-only sandbox and an isolated temporary working directory.

## Milestones

### 1. Input and workspace shell

- Create the app shell and a focused learning workspace.
- Add paste/edit input with a mode for code or documentation.
- Add screenshot upload and a review/edit step for text read from the image.
- Let the model infer language or document type; ask the user only when ambiguity materially affects the explanation.

**Done when:** a user can submit pasted text or a screenshot, review and edit the source text (including text read from an image), then start an explanation with one action.

### 2. Skill-based explanations

- Implement the first two skills: **Plain-language walkthrough** and **Concept-first explanation**.
- Define a shared structured output schema and show sections in a readable result view.
- Include source line references and a visible assumptions/unknowns section.
- Add loading, retry, and useful error states.

**Done when:** both skills produce a coherent explanation for code and documentation, and references point back to the submitted text.

### 3. Interactive visuals

- Add a control-flow diagram skill for code where control flow is meaningful.
- Add a concept/component map for material better explained through relationships.
- Render diagram data in React Flow; selecting a node highlights its related source lines and explanation.
- Omit the visual when it would add little value, and explain that choice briefly.

**Done when:** a generated diagram is navigable, its nodes map to source text, and a diagram is not forced for unsuitable input.

### 4. Understanding checks and follow-up

- [x] Let the learner ask follow-up questions about the current source and explanation.
- [x] Let the learner select part of an explanation and ask a follow-up focused on that passage.
- [x] Preserve follow-up conversation context for the current page session only.
- [x] Add an optional three-question understanding check using `skills/knowledge-check/SKILL.md`, with immediate feedback and explanations.

**Done when:** learners can ask a focused, source-grounded follow-up without re-entering the source and optionally check understanding without saving scores.

### 5. Refinement

- Try the workflows on representative short snippets, longer code, and documentation.
- Improve screenshot text correction, source mapping, diagram readability, and explanation quality based on observed failures.
- Decide whether session export, persistence, additional languages, or repository context is valuable for the next release.

## First implementation slice

The current local slice supports pasted code/documentation and screenshots, one explanation action, automatic teaching-mode selection, adjustable explanation depth and learner level, and local Codex CLI endpoints for transcription, explanation, follow-up questions, and an optional three-question understanding check. Each route loads its own provider-neutral skill. Quiz responses are session-only and are not persisted. Screenshot text is editable and must be explicitly approved; uploaded images are removed with their temporary request directory. Interactive React Flow diagrams show labeled edges; selecting a node reveals its explanation and valid source excerpts. Selecting text in the explanation opens an action to ask a question about that passage; follow-up history is held in page state and discarded when the source changes or the page session ends.

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
