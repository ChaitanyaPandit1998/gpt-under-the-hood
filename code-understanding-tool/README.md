# Code Understanding Tool

An interactive workspace that helps people understand code and documentation through clear explanations, useful visuals, and follow-up questions.

See [PLAN.md](PLAN.md) for the implementation plan, milestones, and current decisions.

## Run locally

```sh
npm install
npm run dev
```

The app reads `skills/explain-material/SKILL.md` and `skills/knowledge-check/SKILL.md`, and sends requests to Codex CLI through the local Vite development server. The model chooses an explanation style that fits the material; the interface currently has one “Help me understand” action. Codex CLI must be installed and signed in. The server looks for `codex` on its `PATH`, checks the packaged macOS Codex CLI location, and supports `CODEX_CLI_PATH` when it is installed elsewhere. The one-shot request ignores personal Codex configuration and disables shell/code execution and extension tools. Screenshot text is read by the model, shown for editing, and requires explicit approval before explanation; image files are temporary.

The API only accepts loopback-host requests and rejects cross-origin browser requests. Keep this prototype on your local machine; it is not intended to be exposed as a shared or public server.

Select a passage in the explanation to ask a follow-up focused on that text. The selected passage is sent with the question and checked against the full source.

Use **Start a quick check** below an explanation to generate three multiple-choice questions from the submitted material. Answers and score stay in the current page session and are not saved.

The UI keeps diagram rendering (`src/components/GraphView.tsx`), source citations (`src/components/SourceRanges.tsx`), and the understanding check (`src/components/UnderstandingCheck.tsx`) separate from the workspace orchestration in `src/App.tsx`. The diagram also has a text alternative for readers who do not use the interactive canvas.

Run `npm run check` to run ESLint, the unit tests, and the production build.

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
