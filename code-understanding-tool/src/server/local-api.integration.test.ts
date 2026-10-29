import { Readable } from "node:stream";
import { EventEmitter } from "node:events";
import type { IncomingMessage, ServerResponse } from "node:http";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ViteDevServer } from "vite";
import { createLocalExplanationApi, findCodexCliForRequest, type CodexRunner } from "./local-api";

type Middleware = (request: IncomingMessage, response: ServerResponse, next: () => void) => unknown;
type RegisteredMiddleware = { path: string; handler: Middleware };

const explanation = {
  title: "Retry helper",
  summary: "Retries a rejected operation.",
  concepts: [],
  walkthrough: [],
  assumptions: [],
  visual_note: "",
  visuals: [],
};
const passedReview = { status: "pass", findings: [] };
const reviseReview = { status: "revise", findings: ["The summary omits why the retry stops after the final attempt."] };
const reviewedExplanation = {
  ...explanation,
  quality_review_status: "passed",
  quality_review_note: "A second model pass checked this explanation against your source and request, including math and visuals where relevant.",
};

const responsesBySchemaKey: Record<string, unknown> = {
  title: explanation,
  transcription: { transcription: "const answer = 42;", uncertainties: [] },
  answer: { answer: "It retries after rejection.", source_ranges: [{ start_line: 1, end_line: 2 }], assumptions: [] },
  questions: { questions: [
    { prompt: "What happens first?", options: ["A", "B", "C", "D"], correct_index: 0, explanation: "The source shows this." },
    { prompt: "What happens next?", options: ["A", "B", "C", "D"], correct_index: 1, explanation: "The source shows this." },
    { prompt: "When does it stop?", options: ["A", "B", "C", "D"], correct_index: 2, explanation: "The source shows this." },
  ] },
};

function createApiHarness(runCodex: CodexRunner, limits?: NonNullable<Parameters<typeof createLocalExplanationApi>[0]>["limits"]) {
  const stack: RegisteredMiddleware[] = [];
  const plugin = createLocalExplanationApi({ runCodex, limits });
  const configureServer = plugin.configureServer as (server: ViteDevServer) => void;
  configureServer({
    middlewares: {
      use(pathOrHandler: string | Middleware, handler?: Middleware) {
        stack.push(typeof pathOrHandler === "string"
          ? { path: pathOrHandler, handler: handler! }
          : { path: "/", handler: pathOrHandler });
      },
    },
  } as unknown as ViteDevServer);

  return async function request(
    path: string,
    body: unknown,
    origin = "http://127.0.0.1:5173",
    method = "POST",
    onResponseReady?: (disconnect: () => void) => void,
  ) {
    const incoming = Readable.from([Buffer.from(JSON.stringify(body))]) as Readable & Partial<IncomingMessage>;
    incoming.method = method;
    incoming.url = path;
    incoming.headers = { host: "127.0.0.1:5173", origin, "content-type": "application/json" };
    let responseBody = "";
    const responseState = {
      statusCode: 200,
      headersSent: false,
      writableEnded: false,
      destroyed: false,
      headers: {} as Record<string, string>,
      setHeader(name: string, value: string) { responseState.headers[name.toLowerCase()] = value; },
      end(value?: string) { responseBody = value ?? ""; responseState.headersSent = true; responseState.writableEnded = true; },
    };
    const responseEvents = new EventEmitter();
    const response = Object.assign(responseState, {
      once: responseEvents.once.bind(responseEvents),
      removeListener: responseEvents.removeListener.bind(responseEvents),
    }) as unknown as ServerResponse;
    onResponseReady?.(() => {
      responseState.destroyed = true;
      responseEvents.emit("close");
    });

    let cursor = 0;
    while (cursor < stack.length && !response.headersSent) {
      const entry = stack[cursor++];
      if (!path.startsWith(entry.path)) continue;
      const originalUrl: string | undefined = path;
      if (entry.path !== "/") incoming.url = path.slice(entry.path.length) || "/";
      let shouldContinue = false;
      await entry.handler(incoming as IncomingMessage, response, () => { shouldContinue = true; });
      incoming.url = originalUrl;
      if (!shouldContinue) break;
    }
    return {
      status: responseState.statusCode,
      headers: responseState.headers,
      json: () => JSON.parse(responseBody) as unknown,
    };
  };
}

describe("local explanation API integration", () => {
  const runCodex = vi.fn<CodexRunner>(async (rawPrompt) => {
    const prompt = String(rawPrompt);
    if (prompt.includes("You are the second-pass quality reviewer")) return passedReview;
    if (prompt.includes("# Knowledge Check Skill")) return responsesBySchemaKey.questions;
    if (prompt.includes("# Answer a Follow-up Question")) return responsesBySchemaKey.answer;
    if (prompt.includes("# Transcribe a Screenshot")) return responsesBySchemaKey.transcription;
    return responsesBySchemaKey.title;
  });
  const request = createApiHarness(runCodex);

  beforeEach(() => runCodex.mockClear());

  it("does not start CLI work when cancellation happens during CLI lookup", async () => {
    const controller = new AbortController();
    let resolveCli: ((path: string) => void) | undefined;
    const cliLookup = findCodexCliForRequest(controller.signal, () => new Promise((resolve) => { resolveCli = resolve; }));

    controller.abort(new Error("The model request was cancelled by the client."));
    resolveCli?.("/fake/codex");

    await expect(cliLookup).rejects.toThrow("The model request was cancelled by the client.");
  });

  it("validates explain requests and returns a structured explanation", async () => {
    const invalid = await request("/api/explain", { source_text: "  " });
    expect(invalid.status).toBe(400);

    const response = await request("/api/explain", {
      source_text: "function retry() {}",
      material_kind: "code",
      explanation_depth: "overview",
      learner_level: "intermediate",
    });
    expect(response.status).toBe(200);
    expect(response.json()).toEqual(reviewedExplanation);
    expect(runCodex).toHaveBeenCalledTimes(2);
    expect(runCodex.mock.calls[0]?.[0]).toContain("explanation_depth: overview");
    expect(runCodex.mock.calls[0]?.[0]).toContain("learner_level: intermediate");
  });

  it("keeps an incomplete control-flow diagram and flags uncertain coverage", async () => {
    const incompleteExplanation = {
      ...explanation,
      visuals: [{
        title: "Access flow",
        kind: "control_flow",
        nodes: [
          { id: "start", label: "Check access", description: "Check the user.", source_ranges: [] },
          { id: "denied", label: "Denied", description: "Return early.", source_ranges: [] },
          { id: "allowed", label: "Continue", description: "Keep going.", source_ranges: [] },
        ],
        edges: [
          { from: "start", to: "denied", label: "missing" },
          { from: "start", to: "allowed", label: "present" },
        ],
      }],
    };
    const runner = vi.fn<CodexRunner>(async (prompt) => String(prompt).includes("You are the second-pass quality reviewer") ? passedReview : incompleteExplanation);
    const isolatedRequest = createApiHarness(runner);
    const response = await isolatedRequest("/api/explain", {
      source_text: "if (!user) return false;\nconst allowed = ready ? true : false;",
      teaching_mode: "control_flow",
    });

    expect(response.status).toBe(200);
    expect(response.json()).toEqual({
      ...reviewedExplanation,
      visuals: incompleteExplanation.visuals,
      visual_note: "Automated checks could not verify every source decision. The diagram is shown; compare it with the walkthrough and linked source lines.",
    });
  });

  it("shows the reviewer finding when an automatic repair fails", async () => {
    const runner = vi.fn<CodexRunner>(async (prompt) => {
      if (String(prompt).includes("You are the second-pass quality reviewer")) return reviseReview;
      if (String(prompt).includes("Revise the draft explanation")) throw new Error("simulated repair failure");
      return explanation;
    });
    const isolatedRequest = createApiHarness(runner);
    const response = await isolatedRequest("/api/explain", { source_text: "function retry() {}" });
    const result = response.json() as typeof explanation & { quality_review_status: string; quality_review_note: string };

    expect(response.status).toBe(200);
    expect(result.quality_review_status).toBe("incomplete");
    expect(result.quality_review_note).toContain("The summary omits why the retry stops after the final attempt.");
    expect(result.quality_review_note).toContain("this draft has not been revised");
  });

  it("says a repaired explanation is unverified if its final review fails", async () => {
    const runner = vi.fn<CodexRunner>(async (prompt) => {
      const content = String(prompt);
      if (content.includes("You are the second-pass quality reviewer")) return reviseReview;
      if (content.includes("Revise the draft explanation")) return { ...explanation, summary: "The repaired explanation." };
      if (content.includes("You are performing the final quality check after a repair")) throw new Error("simulated final review failure");
      return explanation;
    });
    const isolatedRequest = createApiHarness(runner);
    const response = await isolatedRequest("/api/explain", { source_text: "function retry() {}" });
    const result = response.json() as typeof explanation & { quality_review_status: string; quality_review_note: string };

    expect(response.status).toBe(200);
    expect(result.summary).toBe("The repaired explanation.");
    expect(result.quality_review_status).toBe("incomplete");
    expect(result.quality_review_note).toContain("The summary omits why the retry stops after the final attempt.");
    expect(result.quality_review_note).toContain("The answer was revised, but the final check could not be completed");
  });

  it("transcribes a valid screenshot payload", async () => {
    const pngSignature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
    const response = await request("/api/transcribe", {
      image_data: pngSignature.toString("base64"),
      mime_type: "image/png",
    });
    expect(response.status).toBe(200);
    expect(response.json()).toEqual(responsesBySchemaKey.transcription);
    expect(runCodex.mock.calls[0]?.[0]).toContain("Transcribe a Screenshot");
  });

  it("answers follow-ups with the selected passage included as context", async () => {
    const response = await request("/api/followup", {
      source_text: "function retry() {\n  return operation();\n}",
      explanation,
      question: "Why retry?",
      selected_text: "return operation();",
    });
    expect(response.status).toBe(200);
    expect(response.json()).toEqual(responsesBySchemaKey.answer);
    expect(runCodex.mock.calls[0]?.[0]).toContain("BEGIN_SELECTED_TEXT_");
  });

  it("creates a knowledge check from the supplied source", async () => {
    const response = await request("/api/quiz", { source_text: "retry after rejection", explanation });
    expect(response.status).toBe(200);
    expect(response.json()).toEqual(responsesBySchemaKey.questions);
    expect(runCodex.mock.calls[0]?.[0]).toContain("Knowledge Check Skill");
  });

  it("rejects cross-origin API requests before invoking the model", async () => {
    const response = await request("/api/explain", { source_text: "const x = 1;" }, "http://attacker.example");
    expect(response.status).toBe(403);
    expect(runCodex).not.toHaveBeenCalled();
  });

  it("terminates an explanation when its end-to-end deadline expires", async () => {
    const runner = vi.fn<CodexRunner>((_prompt, _schemaPath, _outputPath, _runDirectory, _imagePath, signal) => {
      if (signal?.aborted) return Promise.reject(signal.reason);
      return new Promise((_resolve, reject) => {
        signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
      });
    });
    const isolatedRequest = createApiHarness(runner, { explanationTimeoutMs: 15 });
    const response = await isolatedRequest("/api/explain", { source_text: "const answer = 42;" });

    expect(response.status).toBe(500);
    expect(response.json()).toEqual({ error: "Model request timed out after 1 second. Please try again." });
  });

  it("cancels active generation and frees its concurrency slot when the client disconnects", async () => {
    let holdFirstCall = true;
    let observedAbort = false;
    const runner = vi.fn<CodexRunner>((prompt, _schemaPath, _outputPath, _runDirectory, _imagePath, signal) => {
      if (holdFirstCall) {
        holdFirstCall = false;
        return new Promise((_resolve, reject) => {
          if (signal?.aborted) {
            observedAbort = true;
            reject(signal.reason);
            return;
          }
          signal?.addEventListener("abort", () => {
            observedAbort = true;
            reject(signal.reason);
          }, { once: true });
        });
      }
      return Promise.resolve(String(prompt).includes("You are the second-pass quality reviewer") ? passedReview : explanation);
    });
    const isolatedRequest = createApiHarness(runner, { maxConcurrentModelRequests: 1 });
    const progressId = "0a7f9cb3-ef92-4a20-bd9a-f379241f03f1";
    let disconnect = () => {};
    const cancelled = isolatedRequest(
      "/api/explain",
      { source_text: "const abandoned = true;", progress_id: progressId },
      undefined,
      "POST",
      (close) => { disconnect = close; },
    );

    await vi.waitFor(() => expect(runner).toHaveBeenCalledTimes(1));
    disconnect();
    await cancelled;
    expect(observedAbort).toBe(true);
    const progress = await isolatedRequest(`/api/explain-progress?id=${progressId}`, {}, undefined, "GET");
    expect(progress.status).toBe(404);

    const next = await isolatedRequest("/api/explain", { source_text: "const next = true;" });
    expect(next.status).toBe(200);
  });

  it.each([
    ["/api/transcribe", { image_data: Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).toString("base64"), mime_type: "image/png" }],
    ["/api/followup", { source_text: "const x = 1;", explanation, question: "What is x?" }],
    ["/api/quiz", { source_text: "const x = 1;", explanation }],
  ])("cancels %s work when its client disconnects", async (path, body) => {
    let observedAbort = false;
    const runner = vi.fn<CodexRunner>((_prompt, _schemaPath, _outputPath, _runDirectory, _imagePath, signal) => new Promise((_resolve, reject) => {
      if (signal?.aborted) {
        observedAbort = true;
        reject(signal.reason);
        return;
      }
      signal?.addEventListener("abort", () => {
        observedAbort = true;
        reject(signal.reason);
      }, { once: true });
    }));
    const isolatedRequest = createApiHarness(runner);
    let disconnect = () => {};
    const pending = isolatedRequest(path, body, undefined, "POST", (close) => { disconnect = close; });

    await vi.waitFor(() => expect(runner).toHaveBeenCalledTimes(1));
    disconnect();
    await pending;
    expect(observedAbort).toBe(true);
  });

  it.each([
    ["/api/transcribe", { image_data: Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).toString("base64"), mime_type: "image/png" }],
    ["/api/followup", { source_text: "const x = 1;", explanation, question: "What is x?" }],
    ["/api/quiz", { source_text: "const x = 1;", explanation }],
  ])("times out %s model work and aborts its runner", async (path, body) => {
    let observedAbort = false;
    const runner = vi.fn<CodexRunner>((_prompt, _schemaPath, _outputPath, _runDirectory, _imagePath, signal) => new Promise((_resolve, reject) => {
      signal?.addEventListener("abort", () => {
        observedAbort = true;
        reject(signal.reason);
      }, { once: true });
    }));
    const isolatedRequest = createApiHarness(runner, { modelRequestTimeoutMs: 15 });
    const response = await isolatedRequest(path, body);

    expect(response.status).toBe(500);
    expect(response.json()).toEqual({ error: "Model request timed out after 1 second. Please try again." });
    expect(observedAbort).toBe(true);
  });

  it("applies one concurrency limit across different model endpoints", async () => {
    let finishFollowup: ((value: unknown) => void) | undefined;
    const runner: CodexRunner = async (prompt) => {
      if (String(prompt).includes("# Answer a Follow-up Question")) {
        return new Promise((resolve) => { finishFollowup = resolve; });
      }
      return responsesBySchemaKey.questions;
    };
    const isolatedRequest = createApiHarness(runner, { maxConcurrentModelRequests: 1 });
    const followup = isolatedRequest("/api/followup", { source_text: "const x = 1;\nreturn x;", explanation, question: "What is x?" });

    await vi.waitFor(() => expect(finishFollowup).toBeTypeOf("function"));
    const rejectedQuiz = await isolatedRequest("/api/quiz", { source_text: "const x = 1;", explanation });
    expect(rejectedQuiz.status).toBe(503);
    finishFollowup?.(responsesBySchemaKey.answer);
    expect((await followup).status).toBe(200);
  });

  it("returns a request ID and logs only bounded metadata", async () => {
    const log = vi.spyOn(console, "info").mockImplementation(() => {});
    try {
      const privateSource = "PRIVATE_SOURCE_DO_NOT_LOG_91a2";
      const privateQuestion = "PRIVATE_QUESTION_DO_NOT_LOG_b37c";
      const isolatedRequest = createApiHarness(async () => { throw new Error("simulated provider failure"); });
      const response = await isolatedRequest("/api/followup", {
        source_text: privateSource,
        explanation,
        question: privateQuestion,
      });
      expect(response.headers["x-request-id"]).toMatch(/^[0-9a-f-]{36}$/i);
      expect(response.status).toBe(500);
      const diagnosticOutput = log.mock.calls.map(([entry]) => String(entry)).join("\n");
      expect(diagnosticOutput).toContain(response.headers["x-request-id"]);
      expect(diagnosticOutput).toContain('"duration_ms"');
      expect(diagnosticOutput).toContain('"failure_category"');
      expect(diagnosticOutput).toContain('"phase"');
      expect(diagnosticOutput).not.toContain(privateSource);
      expect(diagnosticOutput).not.toContain(privateQuestion);
    } finally {
      log.mockRestore();
    }
  });

  it("rejects additional explanations while the concurrency limit is reached", async () => {
    const pendingDrafts: Array<(value: unknown) => void> = [];
    const runner: CodexRunner = async (prompt) => {
      if (String(prompt).includes("You are the second-pass quality reviewer")) return passedReview;
      if (pendingDrafts.length < 2) {
        return new Promise((resolve) => pendingDrafts.push(resolve));
      }
      return explanation;
    };
    const isolatedRequest = createApiHarness(runner, { maxConcurrentModelRequests: 2 });
    const first = isolatedRequest("/api/explain", { source_text: "const first = true;" });
    const second = isolatedRequest("/api/explain", { source_text: "const second = true;" });

    await vi.waitFor(() => expect(pendingDrafts).toHaveLength(2));
    const rejected = await isolatedRequest("/api/explain", { source_text: "const third = true;" });
    expect(rejected.status).toBe(503);
    expect(rejected.json()).toEqual({ error: "The local model service is busy. Wait for a request to finish, then try again." });

    for (const resolveDraft of pendingDrafts) resolveDraft(explanation);
    const completed = await Promise.all([first, second]);
    expect(completed.map((response) => response.status)).toEqual([200, 200]);
  });

  it("expires terminal progress records after their retention period", async () => {
    const progressId = "8c0ab8d0-2d67-4aaf-90cd-b1802fc363aa";
    const isolatedRequest = createApiHarness(runCodex, { terminalProgressTtlMs: 20 });
    const response = await isolatedRequest("/api/explain", { source_text: "const answer = 42;", progress_id: progressId });
    expect(response.status).toBe(200);

    const progressPath = `/api/explain-progress?id=${progressId}`;
    const available = await isolatedRequest(progressPath, {}, undefined, "GET");
    expect(available.status).toBe(200);
    expect(available.json()).toEqual({ phase: "complete" });

    await new Promise((resolve) => setTimeout(resolve, 35));
    const expired = await isolatedRequest(progressPath, {}, undefined, "GET");
    expect(expired.status).toBe(404);
  });
});
