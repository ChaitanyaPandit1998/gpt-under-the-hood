# Application architecture

This diagram describes the current local prototype and its request/data flow. The Vite middleware is the API server; there is no separate production backend or database.

```mermaid
flowchart LR
  subgraph Browser[Browser · React application]
    UI[App and LearningWorkspace]
    Session[useExplanationSession<br/>request state, cancellation, progress polling]
    Result[ExplanationResult<br/>citations, walkthrough, follow-ups]
    Diagram[GraphView<br/>React Flow]
    Reader[SpeechReader<br/>browser SpeechSynthesis]
    Check[UnderstandingCheck]
    UI --> Session
    Session --> Result
    Result --> Diagram
    Result --> Reader
    Result --> Check
  end

  subgraph Local[Vite local server · loopback only]
    Guard[Local host/origin guard]
    API[Local API middleware<br/>request limits · concurrency · timeouts<br/>cancellation · safe diagnostics]
    ExplainRoute[POST /api/explain]
    ProgressRoute[GET /api/explain-progress]
    OtherRoutes[POST /api/transcribe<br/>POST /api/followup<br/>POST /api/quiz<br/>POST /api/pull-request]
    ExplanationService[createReviewedExplanation]
    ExplanationValidation[Explanation schema validation]
    TaskValidation[Transcription · follow-up · quiz validation]
    Domain[Domain annotations<br/>incomplete-code and control-flow coverage]
    Guard --> API
    API --> ExplainRoute
    API --> ProgressRoute
    API --> OtherRoutes
    ExplainRoute --> ExplanationService
    ExplanationService --> ExplanationValidation --> Domain
    OtherRoutes --> TaskValidation
  end

  GitHub[GitHub public REST API<br/>public PR metadata and file patches]
  OtherRoutes --> GitHub

  subgraph Model[Local model execution]
    ExplainSkill[Explanation and review skills]
    TaskSkills[Follow-up · quiz · transcription skills]
    CLI[Codex CLI<br/>signed-in local account]
    Draft[Explanation draft]
    ReviewPrompt[Second-pass review]
    ReviewDecision{Review result}
    Repair[One bounded repair]
    Repaired[Repaired explanation]
    FinalReview[Final review]
    TaskResponse[Transcription · follow-up<br/>or quiz response]
    ExplainSkill --> CLI --> Draft --> ReviewPrompt --> CLI --> ReviewDecision
    ReviewDecision -- pass or uncertain --> ExplanationValidation
    ReviewDecision -- revise --> Repair --> CLI --> Repaired --> FinalReview --> CLI --> ExplanationValidation
    TaskSkills --> CLI --> TaskResponse --> TaskValidation
  end

  Session <-->|HTTP JSON and progress| Guard
  ExplanationService --> ExplainSkill
  OtherRoutes --> TaskSkills
  ProgressRoute --> Session
  Domain -->|annotated explanation| Session
  TaskValidation -->|validated response| Session

  subgraph Quality[Separate evaluation workflow]
    Eval[evals/run.mjs<br/>fixed synthetic cases]
    Corpus[Raw JSON results<br/>human rubric scores]
    Eval --> API
    Eval --> Corpus
  end
```

## Main code areas

- `src/App.tsx` and `src/components/` compose the browser interface; `src/hooks/useExplanationSession.ts` coordinates explanation and follow-up requests.
- `src/server/local-api.ts` registers the local API routes, validates request boundaries, invokes the model runner, tracks progress, and emits privacy-safe diagnostics.
- `src/server/explanation-service.ts` builds explanation prompts and manages the bounded draft, review, repair, and final-review sequence.
- `src/server/github-pull-request.ts` validates public GitHub PR URLs and fetches bounded metadata and file patches for review in the input editor.
- `src/server/response-schemas.ts` and `src/server/response-validation.ts` define and validate structured model responses.
- `src/domain/` contains reusable types and transformations, including source-range handling, incomplete-code annotation, control-flow coverage, and generation phases.
- `skills/` holds task-specific model instructions; `evals/` holds the synthetic regression corpus and recorded evaluations.

## Trust and deployment boundary

The app is intended to run locally. Submitted text, follow-up context, and screenshot images are passed to the Codex CLI configured on the same machine; screenshots and intermediate request files use temporary storage and are removed after processing. The host/origin guard is not user authentication. Do not expose the Vite server to a network or treat this prototype as a multi-user service.
