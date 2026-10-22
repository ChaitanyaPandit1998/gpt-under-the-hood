import { Readable } from "node:stream";
import type { IncomingMessage, ServerResponse } from "node:http";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ViteDevServer } from "vite";
import { createLocalExplanationApi, type CodexRunner } from "./local-api";

type Middleware = (request: IncomingMessage, response: ServerResponse, next: () => void) => unknown;
type RegisteredMiddleware = { path: string; handler: Middleware };

const explanation = {
  title: "Retry helper",
  summary: "Retries a rejected operation.",
  concepts: [],
  walkthrough: [],
  assumptions: [],
  visual: null,
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

function createApiHarness(runCodex: CodexRunner) {
  const stack: RegisteredMiddleware[] = [];
  const plugin = createLocalExplanationApi({ runCodex });
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

  return async function request(path: string, body: unknown, origin = "http://127.0.0.1:5173") {
    const incoming = Readable.from([Buffer.from(JSON.stringify(body))]) as Readable & Partial<IncomingMessage>;
    incoming.method = "POST";
    incoming.url = path;
    incoming.headers = { host: "127.0.0.1:5173", origin, "content-type": "application/json" };
    let responseBody = "";
    const responseState = {
      statusCode: 200,
      headersSent: false,
      headers: {} as Record<string, string>,
      setHeader(name: string, value: string) { responseState.headers[name.toLowerCase()] = value; },
      end(value?: string) { responseBody = value ?? ""; responseState.headersSent = true; },
    };
    const response = responseState as unknown as ServerResponse;

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
      json: () => JSON.parse(responseBody) as unknown,
    };
  };
}

describe("local explanation API integration", () => {
  const runCodex = vi.fn<CodexRunner>(async (rawPrompt) => {
    const prompt = String(rawPrompt);
    if (prompt.includes("# Knowledge Check Skill")) return responsesBySchemaKey.questions;
    if (prompt.includes("# Answer a Follow-up Question")) return responsesBySchemaKey.answer;
    if (prompt.includes("# Transcribe a Screenshot")) return responsesBySchemaKey.transcription;
    return responsesBySchemaKey.title;
  });
  const request = createApiHarness(runCodex);

  beforeEach(() => runCodex.mockClear());

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
    expect(response.json()).toEqual(explanation);
    expect(runCodex).toHaveBeenCalledOnce();
    expect(runCodex.mock.calls[0]?.[0]).toContain("explanation_depth: overview");
    expect(runCodex.mock.calls[0]?.[0]).toContain("learner_level: intermediate");
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
});
