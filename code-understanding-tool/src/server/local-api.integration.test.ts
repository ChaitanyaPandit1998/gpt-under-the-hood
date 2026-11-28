import { Readable } from "node:stream";
import { EventEmitter } from "node:events";
import type { IncomingMessage, ServerResponse } from "node:http";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ViteDevServer } from "vite";
import { createLocalExplanationApi, findCodexCliForRequest, type CodexRunner } from "./local-api";
import type { PullRequestFetch } from "./github-pull-request";
import { defaultVisualPreferences, sceneVisualGuidance } from "../domain/visual-brief";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { visualCheckStages } from "../domain/visual-evaluation";
import { makeVisualPng } from "../test-fixtures/visual-png";

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

function createApiHarness(runCodex: CodexRunner, limits?: NonNullable<Parameters<typeof createLocalExplanationApi>[0]>["limits"], fetchPullRequest?: PullRequestFetch, visualOptions?: Pick<NonNullable<Parameters<typeof createLocalExplanationApi>[0]>, "imageCredentials" | "fetchImages" | "diagramExporter" | "compactVisualPlanning">) {
  const stack: RegisteredMiddleware[] = [];
  const plugin = createLocalExplanationApi({ runCodex, limits, fetchPullRequest, ...visualOptions });
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
  const visualReviewExplanation = "The limit is fixed.";
  const visualReviewBody = {
    confirmed: true, explanation: visualReviewExplanation, caption: "A fixed limit.", brief: "Show the limit.", resolution: "standard",
    pngBase64: makeVisualPng().toString("base64"),
    plan: JSON.stringify({ background: "#ffffff", items: [{ kind: "text", text: "Limit", x: 50, y: 80, size: 24, color: "#000000", align: "left" }] }),
    contract: { version: 1, id: "review-test", explanationSha256: createHash("sha256").update(visualReviewExplanation).digest("hex"),
      audience: "beginner", purpose: "factual-diagram", teachingGoal: "Show the limit.", audienceObjectives: ["Understand the limit."],
      requirements: [{ id: "limit", kind: "claim", statement: "The limit is fixed.", evidence: "background", evidenceReference: "Explanation" }] },
  };
  it("reviews an explicitly confirmed image through the guarded local API", async () => {
    const runner = vi.fn<CodexRunner>().mockResolvedValue({ imageInspected: true, checks: visualCheckStages.map((stage) => ({ stage,
      status: "pass", summary: "Mock review only.", requirementIds: ["limit"], findings: [] })) });
    const request = createApiHarness(runner);
    const response = await request("/api/visual-review", visualReviewBody);
    expect(response.status).toBe(200);
    expect(response.headers["cache-control"]).toBe("no-store");
    expect(response.json()).toMatchObject({ report: { contractId: "review-test", artifacts: {
      captionSha256: createHash("sha256").update(visualReviewBody.caption).digest("hex"),
    } } });
    expect(runner.mock.calls[0][4]).toMatch(/visual\.png$/);
  });
  it("rejects missing consent, wrong origin and non-POST visual review without model calls", async () => {
    const runner = vi.fn<CodexRunner>(); const request = createApiHarness(runner);
    expect((await request("/api/visual-review", { ...visualReviewBody, confirmed: false })).status).toBe(400);
    expect((await request("/api/visual-review", visualReviewBody, "https://untrusted.example")).status).toBe(403);
    expect((await request("/api/visual-review", visualReviewBody, undefined, "GET")).status).toBe(405);
    expect(runner).not.toHaveBeenCalled();
  });
  it("returns blocking diagnostics for corrupt PNGs without invoking the model", async () => {
    const runner = vi.fn<CodexRunner>(); const request = createApiHarness(runner);
    const response = await request("/api/visual-review", { ...visualReviewBody, pngBase64: "YWJj" });
    expect(response.status).toBe(200);
    expect(response.json()).toMatchObject({ report: { checks: expect.arrayContaining([expect.objectContaining({ stage: "raster", status: "fail" })]) } });
    expect(runner).not.toHaveBeenCalled();
  });

  it("keeps scene guidance in the brief and the render plan even with an edited prompt", async () => {
    const brief = {
      disposition: "render", visualType: "concept-map", takeaway: "A limit controls the result.", composition: "Show the source relationship.",
      caption: "A generated schematic.", altText: "The result depends on the limit.",
      content: [{ label: "Limit", role: "background", statement: "The limit controls the result.", chapterLines: [] }],
      guardrails: Array.from({ length: 10 }, (_, i) => `Guardrail ${i + 1}`),
    };
    const scene = { background: "#ffffff", items: [{ kind: "text", text: "Limit", x: 40, y: 60, size: 24, color: "#202033", align: "left" }] };
    const runner = vi.fn<CodexRunner>().mockResolvedValueOnce(brief).mockResolvedValueOnce(scene);
    const request = createApiHarness(runner, undefined, undefined, {
      diagramExporter: { available: async () => true, export: vi.fn(), renderScene: vi.fn() },
    });
    const body = { context: "The result depends on the limit.", preferences: defaultVisualPreferences, designPrompt: "Use the scene's character as a label." };
    expect((await request("/api/visual-brief", { ...body, purpose: "invalid" })).status).toBe(400);
    expect(runner).not.toHaveBeenCalled();
    const response = await request("/api/visual-brief", { ...body, purpose: "factual-diagram" });
    expect(response.status).toBe(200);
    const expected = { ...brief, guardrails: [sceneVisualGuidance["factual-diagram"], ...brief.guardrails.slice(0, 9)] };
    expect(response.json()).toMatchObject({ brief: expected, generation: { artifactSha256: createHash("sha256").update(JSON.stringify(expected)).digest("hex") } });
    expect(runner.mock.calls[0][0]).toContain(sceneVisualGuidance["factual-diagram"]);
    expect((await request("/api/visual-plan", expected)).status).toBe(200);
    expect(runner.mock.calls[1][0]).toContain(sceneVisualGuidance["factual-diagram"]);
  });

  it("expands reusable Story heads before returning the plan and keeps factual diagrams on the ordinary schema", async () => {
    const brief = {
      disposition: "render", visualType: "conceptual-illustration", takeaway: "Match a supplied token.", composition: "Fictional counter.",
      caption: "A fictional example.", altText: "A fictional counter.", guardrails: [],
      content: [{ label: "Token", role: "illustrative", statement: "The guest shows the token.", chapterLines: [] }],
    };
    const raw = { background: "#ffffff", items: [
      { kind: "text", text: "Fictional analogy", x: 60, y: 60, size: 24, color: "#202033", align: "left" },
      { kind: "rectangle", x: 240, y: 260, width: 80, height: 100, color: "#443333" },
      { kind: "story-head", asset: "short-hair", torsoIndex: 1, collarX: 280, scale: 1.5, mirror: false },
    ] };
    const runner = vi.fn<CodexRunner>().mockResolvedValue(raw);
    const renderScene = vi.fn().mockResolvedValue(Buffer.from("png"));
    const request = createApiHarness(runner, undefined, undefined, {
      diagramExporter: { available: async () => true, export: vi.fn(), renderScene },
    });
    expect((await request("/api/visual-plan", { brief, purpose: null })).status).toBe(400);
    expect(runner).not.toHaveBeenCalled();
    const plan = await request("/api/visual-plan", { brief, purpose: "story-illustration" });
    expect(plan.status).toBe(200);
    const { scene, generation } = plan.json() as { scene: { items: { kind: string }[] }; generation: { instructions: unknown[] } };
    expect(scene.items.map((item) => item.kind)).toEqual(["text", "rectangle", ...Array(9).fill("path")]);
    expect(generation.instructions).toContainEqual(expect.objectContaining({ id: "visual-composition-story-cast" }));
    expect(runner.mock.calls[0][0]).toContain("short-hair");
    expect((await request("/api/visual-render", { scene: raw })).status).toBe(400);
    expect((await request("/api/visual-render", { brief, purpose: "story-illustration", resolution: "hd" })).status).toBe(200);
    expect(renderScene).toHaveBeenCalledWith(scene, expect.any(AbortSignal), "hd");
    expect((await request("/api/visual-plan", { brief, purpose: "factual-diagram" })).status).toBe(502);
    expect(runner.mock.calls.at(-1)?.[0]).not.toContain("short-hair");
    const cast = { "short-hair": "Clerk" };
    const before = runner.mock.calls.length;
    expect((await request("/api/visual-plan", { brief, purpose: "factual-diagram", cast })).status).toBe(400);
    expect((await request("/api/visual-plan", { brief, purpose: "story-illustration", cast: { ...cast, "long-hair": "clerk" } })).status).toBe(400);
    expect(runner).toHaveBeenCalledTimes(before);
    runner.mockResolvedValueOnce({ ...raw, items: [...raw.items.slice(0, 2), { ...raw.items[2], asset: "cast-1" }] });
    const locked = await request("/api/visual-plan", { brief, purpose: "story-illustration", cast });
    expect(locked.status).toBe(200);
    expect(locked.json()).toMatchObject({ scene });
    expect(runner.mock.calls.at(-1)?.[0]).toContain('"role":"Clerk"');
  });

  it("blocks missing teaching labels and Story qualifiers before rendering", async () => {
    const brief = {
      disposition: "render", visualType: "conceptual-illustration", takeaway: "A teaching assumption.", composition: "Illustrate the assumption.",
      caption: "An analogy.", altText: "A fictional example.", guardrails: [], requiredLabels: ["Specified limit"],
      content: [{ label: "Limit", role: "illustrative", statement: "A toy limit.", chapterLines: [] }],
    };
    const scene = { background: "#ffffff", items: [{ kind: "text", text: "Specified limit", x: 60, y: 60, size: 24, color: "#202033", align: "left" }] };
    const runner = vi.fn<CodexRunner>().mockResolvedValue(scene);
    const renderScene = vi.fn().mockResolvedValue(Buffer.from("png"));
    const request = createApiHarness(runner, undefined, undefined, {
      diagramExporter: { available: async () => true, export: vi.fn(), renderScene },
    });
    const input = { brief, purpose: "story-illustration" };
    const planned = await request("/api/visual-plan", input);
    expect(planned.status).toBe(422);
    expect(planned.json()).toMatchObject({ error: expect.stringContaining("Fictional analogy") });
    expect(runner).toHaveBeenCalledTimes(2);
    expect((await request("/api/visual-render", { ...input, resolution: "standard" })).status).toBe(422);
    expect((await request("/api/visual-render", { scene, resolution: "standard", requiredLabels: ["Unproved"] })).status).toBe(422);
    expect((await request("/api/visual-render", { scene, resolution: "standard", purpose: "story-illustration" })).status).toBe(422);
    expect(renderScene).not.toHaveBeenCalled();
    expect((await request("/api/visual-render", { scene, resolution: "standard", purpose: "factual-diagram", requiredLabels: ["Specified limit"] })).status).toBe(200);
    expect(renderScene).toHaveBeenCalledTimes(1);
  });

  const storyInput = { source: "const limit = 3;", context: "Overview of the limit.", mode: "code", depth: "balanced", learnerLevel: "beginner" };
  const story = {
    title: "The final ticket", setup: "A clerk has three tickets.",
    scenes: [{ heading: "A fixed supply", narrative: "The clerk counts three tickets.", meaning: "The constant is set to three.", source_ranges: [{ start_line: 1, end_line: 1 }] }],
    takeaway: "The limit is three.",
  };

  it.each(["beginner", "intermediate", "advanced"])("preserves the %s audience through story generation and review", async (learnerLevel) => {
    const runner = vi.fn<CodexRunner>().mockResolvedValueOnce(story).mockResolvedValueOnce(passedReview);
    const request = createApiHarness(runner);
    const response = await request("/api/chapter-story", { ...storyInput, learnerLevel });
    expect(response.status).toBe(200);
    expect(response.json()).toEqual({ story });
    expect(runner).toHaveBeenCalledTimes(2);
    for (const [prompt] of runner.mock.calls) {
      expect(prompt).toContain(`Audience: ${learnerLevel}.`);
    }
  });

  it("reviews and at most once repairs a source-linked story on request", async () => {
    const revised = { ...story, takeaway: "Three is the value shown in the source." };
    const runner = vi.fn<CodexRunner>().mockResolvedValueOnce(story).mockResolvedValueOnce(reviseReview)
      .mockResolvedValueOnce(revised).mockResolvedValueOnce(passedReview);
    const request = createApiHarness(runner);
    expect((await request("/api/chapter-story", storyInput, "https://other.example")).status).toBe(403);
    expect((await request("/api/chapter-story", { ...storyInput, depth: "unknown" })).status).toBe(400);
    expect(runner).not.toHaveBeenCalled();
    const response = await request("/api/chapter-story", storyInput);
    expect(response.status).toBe(200);
    expect(response.headers["cache-control"]).toBe("no-store");
    expect(response.json()).toEqual({ story: revised });
    expect(runner).toHaveBeenCalledTimes(4);
    expect(runner.mock.calls[0][0]).toContain("1 | const limit = 3;");
  });

  it("does not return a story with invalid citations or an uncertain review", async () => {
    const runner = vi.fn<CodexRunner>().mockResolvedValueOnce({
      ...story, scenes: [{ ...story.scenes[0], source_ranges: [{ start_line: 2, end_line: 2 }] }],
    }).mockResolvedValueOnce(story).mockResolvedValueOnce({ status: "uncertain", findings: [] });
    const request = createApiHarness(runner);
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const response = await request("/api/chapter-story", storyInput);
      expect(response.status).toBe(502);
      expect(response.json()).toEqual({ error: expect.stringContaining("direct chapter is still available") });
    }
    expect(runner).toHaveBeenCalledTimes(3);
  });

  it("renders only an approved brief with validated scene data", async () => {
    const brief = { disposition: "render", visualType: "comparison", takeaway: "Compare two outcomes.", composition: "Use two panels.",
      caption: "Comparison.", altText: "Two outcomes side by side.", guardrails: [], content: [{ label: "Compare", role: "illustrative", statement: "Example only.", chapterLines: [] }] };
    const scene = { background: "#f7f7f7", items: [{ kind: "text", text: "Comparison", x: 50, y: 70, size: 28, color: "#002233", align: "left" }] };
    const runner = vi.fn<CodexRunner>().mockResolvedValueOnce(scene).mockResolvedValueOnce({ ...scene, items: [{ ...scene.items[0], x: -10 }] });
    const renderScene = vi.fn(async () => Buffer.from("fixture-png"));
    const request = createApiHarness(runner, undefined, undefined, { diagramExporter: { available: async () => true, export: vi.fn(), renderScene } });
    expect((await request("/api/visual-render", { ...brief, disposition: "text-only", visualType: "none" })).status).toBe(400);
    expect(runner).not.toHaveBeenCalled();
    expect((await request("/api/visual-render", { brief, resolution: "arbitrary-size" })).status).toBe(400);
    expect(runner).not.toHaveBeenCalled();
    const reply = await request("/api/visual-render", { brief, resolution: "full-hd" });
    expect(reply.status).toBe(200);
    expect(reply.headers["content-type"]).toBe("image/png");
    expect(renderScene).toHaveBeenCalledWith(scene, expect.any(AbortSignal), "full-hd");
    expect((await request("/api/visual-render", brief)).status).toBe(502);
    expect(renderScene).toHaveBeenCalledTimes(1);
    runner.mockResolvedValueOnce(scene);
    const planned = await request("/api/visual-plan", brief);
    expect(planned.status).toBe(200);
    expect(planned.json()).toMatchObject({ scene, generation: { artifactSha256: createHash("sha256").update(JSON.stringify(scene)).digest("hex") } });
    expect((await request("/api/visual-render", { scene, resolution: "hd" })).status).toBe(200);
    expect((await request("/api/visual-render", { scene, resolution: "full-hd" })).status).toBe(200);
    expect((await request("/api/visual-render", { scene: { ...scene, items: [] } })).status).toBe(400);
    expect(runner).toHaveBeenCalledTimes(4);
    expect(renderScene).toHaveBeenCalledTimes(3);
  });
  it("only expands compact drawings in opted-in planning, not in client render requests", async () => {
    const brief = { disposition: "render", visualType: "comparison", takeaway: "Compare two choices.", composition: "Show two labels.",
      caption: "Choices.", altText: "Two choices.", guardrails: [], content: [{ label: "Choices", role: "illustrative", statement: "Two labels.", chapterLines: [] }] };
    const compact = { background: "#ffffff", items: [
      { kind: "text-run", x: 50, y: 70, dx: 100, dy: 0, size: 24, color: "#123456", align: "left", labels: ["One", "Two"] },
    ] };
    const runner = vi.fn<CodexRunner>(async () => compact);
    const renderScene = vi.fn(async () => Buffer.from("fixture-png"));
    const diagramExporter = { available: async () => true, export: vi.fn(), renderScene };
    const defaultRequest = createApiHarness(runner, undefined, undefined, { diagramExporter });
    const enabledRequest = createApiHarness(runner, undefined, undefined, { diagramExporter, compactVisualPlanning: true });
    expect((await defaultRequest("/api/visual-plan", brief)).status).toBe(502);
    const result = await enabledRequest("/api/visual-plan", brief);
    expect(result.status).toBe(200);
    expect(runner.mock.calls[0][0]).not.toContain("Application instruction: visual-composition-compact;");
    expect(runner.mock.calls[2][0]).toContain("Application instruction: visual-composition-compact;");
    const scene = { background: "#ffffff", items: ["One", "Two"].map((text, index) => ({
      kind: "text", text, x: 50 + index * 100, y: 70, size: 24, color: "#123456", align: "left",
    })) };
    const payload = result.json() as { scene: unknown; generation: { artifactSha256: string } };
    expect(payload.scene).toEqual(scene);
    expect(payload.generation.artifactSha256).toBe(createHash("sha256").update(JSON.stringify(payload.scene)).digest("hex"));
    expect((await enabledRequest("/api/visual-render", { scene: compact, resolution: "standard" })).status).toBe(400);
    expect((await enabledRequest("/api/visual-render", { scene, resolution: "standard" })).status).toBe(200);
    expect(renderScene).toHaveBeenCalledTimes(1);
    expect(renderScene).toHaveBeenCalledWith(scene, expect.any(AbortSignal), "standard");
  });
  it("generates a bounded visual brief with opted-in source only", async () => {
    const brief = { disposition: "render", visualType: "decision-flow", takeaway: "Show what retries.", composition: "A short flow.",
      caption: "Retry on rejection.", altText: "An operation repeats.", guardrails: ["Not a runtime trace."],
      content: [{ label: "Retry", role: "source", statement: "The source repeats the operation.", chapterLines: [2] }] };
    const runner = vi.fn<CodexRunner>(async () => brief);
    const request = createApiHarness(runner);
    const body = { context: "A retry explanation.", source: "try();\nretry();", preferences: { ...defaultVisualPreferences, style: "expressive", includeSource: true, additionalContext: "A learner-provided variant with a waiting period." } };
    expect((await request("/api/visual-brief", { ...body, preferences: defaultVisualPreferences })).status).toBe(400);
    expect((await request("/api/visual-brief", body, "http://example.test")).status).toBe(403);
    expect(runner).not.toHaveBeenCalled();
    const result = await request("/api/visual-brief", body);
    expect(result.status).toBe(200);
    expect(result.headers["cache-control"]).toBe("no-store");
    expect(result.json()).toMatchObject({ brief, generation: { artifactSha256: createHash("sha256").update(JSON.stringify(brief)).digest("hex") } });
    expect(runner.mock.calls[0][0]).toContain('2: retry();');
    expect(runner.mock.calls[0][0]).toContain("Visual style: Expressive illustration");
    expect(runner.mock.calls[0][0]).toContain("A learner-provided variant with a waiting period.");
    expect(runner.mock.calls[0][0]).not.toMatch(/Riemann Hypothesis|Attention walkthrough|Retrying worker/);
    const invalid = await request("/api/visual-brief", { context: body.context, preferences: defaultVisualPreferences });
    expect(invalid.status).toBe(502);
    expect(runner.mock.calls[1][0]).not.toContain("retry();");
    expect((await request("/api/visual-brief", { ...body, designPrompt: " " })).status).toBe(400);
    expect((await request("/api/visual-brief", { ...body, designPrompt: "x".repeat(8001) })).status).toBe(400);
    expect(runner).toHaveBeenCalledTimes(2);
    expect((await request("/api/visual-brief", { ...body, designPrompt: "Use a single annotated comparison." })).status).toBe(200);
    expect(runner.mock.calls[2][0]).toMatch(/BEGIN_UNTRUSTED_CONTEXT_[^\n]+\n\n\{"designPrompt":"Use a single annotated comparison\."/);
    expect(runner.mock.calls[2][0]).not.toContain("A learner-provided variant");
  });
  it("generates a generic opt-in cover without sending chapter material or invoking Codex", async () => {
    const image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=";
    const runner = vi.fn<CodexRunner>();
    const fetchImages = vi.fn<typeof fetch>(async () => Response.json({ success: true, result: { image } }));
    const request = createApiHarness(runner, { maxChapterCovers: 1 }, undefined, { imageCredentials: { accountId: "a".repeat(32), apiToken: "test-token" }, fetchImages });
    expect((await request("/api/chapter-cover", { theme: "review", approved: false })).status).toBe(400);
    expect((await request("/api/chapter-cover", { theme: "review", approved: true, source: "PRIVATE_CODE" })).status).toBe(400);
    expect(fetchImages).not.toHaveBeenCalled();
    const response = await request("/api/chapter-cover", { theme: "review", approved: true });
    expect(response.status).toBe(200);
    expect(response.json()).toMatchObject({ theme: "review", dataUrl: `data:image/png;base64,${image}` });
    const input = JSON.parse(String(fetchImages.mock.calls[0][1]?.body));
    expect(Object.keys(input).sort()).toEqual(["prompt", "steps"]);
    expect(input.prompt).toContain("Decorative metaphor only");
    expect(input.prompt).not.toContain("PRIVATE_CODE");
    expect(runner).not.toHaveBeenCalled();
    expect((await request("/api/chapter-cover", { theme: "review", approved: true })).status).toBe(429);
    expect(fetchImages).toHaveBeenCalledTimes(1);
    expect((await request("/api/chapter-cover", { theme: "review", approved: true }, "https://evil.example")).status).toBe(403);
    expect((await request("/api/chapter-cover", {}, undefined, "GET")).status).toBe(405);
  });

  it("shares model request capacity between visual rendering and explanations", async () => {
    const scene = { background: "#f7f7f7", items: [{ kind: "text", text: "Comparison", x: 50, y: 70, size: 28, color: "#002233", align: "left" }] };
    let markRenderStarted!: () => void;
    const renderStarted = new Promise<void>((resolve) => { markRenderStarted = resolve; });
    let finishRender!: (image: Buffer) => void;
    const renderedImage = new Promise<Buffer>((resolve) => { finishRender = resolve; });
    const runner = vi.fn<CodexRunner>().mockResolvedValueOnce(explanation).mockResolvedValueOnce(passedReview);
    const request = createApiHarness(runner, { maxConcurrentModelRequests: 1 }, undefined, {
      diagramExporter: {
        available: async () => true,
        export: vi.fn(),
        renderScene() {
          markRenderStarted();
          return renderedImage;
        },
      },
    });
    const rendering = request("/api/visual-render", { scene, resolution: "standard" });
    try {
      await Promise.race([renderStarted, rendering.then(() => {
        throw new Error("The visual render finished before acquiring a request slot.");
      })]);
      expect((await request("/api/explain", { source_text: "const answer = 42;" })).status).toBe(503);
      expect(runner).not.toHaveBeenCalled();
    } finally {
      finishRender(Buffer.from("fixture-png"));
      await rendering;
    }
    expect((await rendering).status).toBe(200);
    expect((await request("/api/explain", { source_text: "const answer = 42;" })).status).toBe(200);
  });

  it("returns a clear JSON error when covers are unconfigured or the provider fails", async () => {
    const empty = createApiHarness(vi.fn<CodexRunner>());
    expect((await empty("/api/chapter-cover", { theme: "review", approved: true })).status).toBe(503);
    const fetchImages = vi.fn<typeof fetch>(async () => new Response("PRIVATE_PROVIDER_BODY", { status: 401 }));
    const request = createApiHarness(vi.fn<CodexRunner>(), undefined, undefined, { imageCredentials: { accountId: "a".repeat(32), apiToken: "test-token" }, fetchImages });
    const response = await request("/api/chapter-cover", { theme: "review", approved: true });
    expect(response.status).toBe(500);
    expect(JSON.stringify(response.json())).toContain("HTTP 401");
    expect(JSON.stringify(response.json())).not.toContain("PRIVATE_PROVIDER_BODY");
  });

  it("loads on-demand evidence behind the local guard without invoking a model", async () => {
    const headSha = "a".repeat(40);
    const runner = vi.fn<CodexRunner>();
    const fetcher: PullRequestFetch = async (input) => String(input).includes("/files?")
      ? Response.json([{ filename: "src/app.ts", patch: "+validate(input);" }])
      : Response.json({ title: "Validate", head: { sha: headSha }, changed_files: 1 });
    const request = createApiHarness(runner, undefined, fetcher);
    const body = { url: "https://github.com/acme/demo/pull/7", headSha, filename: "src/app.ts", page: 1, offset: 0 };
    const response = await request("/api/pull-request-evidence", body);
    expect(response.status).toBe(200);
    expect(response.json()).toMatchObject({ filename: "src/app.ts", nextOffset: null, chapter: { files: ["src/app.ts"] } });
    expect(runner).not.toHaveBeenCalled();
    expect((await request("/api/pull-request-evidence", body, "https://evil.example")).status).toBe(403);
    expect((await request("/api/pull-request-evidence", {}, undefined, "GET")).status).toBe(405);
    expect((await request("/api/pull-request-evidence", { ...body, page: 6 })).status).toBe(400);
  });

  it("builds a coverage-aware chapter outline without invoking the model", async () => {
    const runner = vi.fn<CodexRunner>();
    const fetcher: PullRequestFetch = async (input) => String(input).includes("/files?")
      ? Response.json([{ filename: "src/app.ts", status: "modified", additions: 1, deletions: 0, patch: "+const enabled = true;" }])
      : Response.json({ title: "Add feature", head: { sha: "deadbeef" }, changed_files: 1 });
    const request = createApiHarness(runner, undefined, fetcher);
    const response = await request("/api/pull-request-plan", { url: "https://github.com/acme/demo/pull/7" });
    expect(response.status).toBe(200);
    expect(response.json()).toMatchObject({ plan: { filesComplete: true, chapters: [{ id: "overview" }, { id: "chapter-1", files: ["src/app.ts"] }] } });
    expect(runner).not.toHaveBeenCalled();
    expect((await request("/api/pull-request-plan", { url: "https://github.com/acme/demo/pull/7" }, "https://evil.example")).status).toBe(403);
    expect((await request("/api/pull-request-plan", {}, undefined, "GET")).status).toBe(405);
  });

  it("loads a public GitHub pull request through the guarded local API", async () => {
    const fetchPullRequest: PullRequestFetch = vi.fn(async (input) => String(input).endsWith("/files?per_page=100")
      ? Response.json([{ filename: "src/app.ts", status: "modified", additions: 1, deletions: 0, patch: "+const enabled = true;" }])
      : Response.json({ title: "Add feature", body: "A short description", html_url: "https://github.com/acme/demo/pull/7", state: "open", base: { ref: "main" }, head: { ref: "feature", sha: "deadbeef" } }));
    const isolatedRequest = createApiHarness(vi.fn<CodexRunner>(), undefined, fetchPullRequest);
    const response = await isolatedRequest("/api/pull-request", { url: "https://github.com/acme/demo/pull/7" });

    expect(response.status).toBe(200);
    const payload = response.json() as { source_text: string };
    expect(payload.source_text).toContain("Pull request: Add feature");
    expect(payload.source_text).toContain("File: src/app.ts");
  });

  it("rejects non-GitHub pull request URLs without making an outbound request", async () => {
    const fetchPullRequest: PullRequestFetch = vi.fn();
    const isolatedRequest = createApiHarness(vi.fn<CodexRunner>(), undefined, fetchPullRequest);
    const response = await isolatedRequest("/api/pull-request", { url: "https://attacker.example/collect" });

    expect(response.status).toBe(400);
    expect(fetchPullRequest).not.toHaveBeenCalled();
  });

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

  it("passes chapter tone and focus through drafting and review", async () => {
    const response = await request("/api/explain", {
      source_text: "const value = 42;", explanation_tone: "technical", explanation_focus: "behaviour",
      learner_question: "Explain only the assignment section.",
    });
    expect(response.status).toBe(200);
    expect(runCodex).toHaveBeenCalledTimes(2);
    for (const call of runCodex.mock.calls) {
      expect(call[0]).toContain("tone=technical");
      expect(call[0]).toContain("focus=behaviour");
      expect(call[0]).toContain("Explain only the assignment section.");
    }
  });

  it("keeps drafts and repairs text-only without skipping factual review", async () => {
    const strayVisual = { visuals: [{ title: "Unrequested diagram" }], visual_note: "Unrequested note" };
    const writerSchemas: unknown[] = [];
    const runner = vi.fn<CodexRunner>()
      .mockImplementationOnce(async (prompt, schemaPath) => {
        expect(prompt).toContain("Visuals are disabled. Write only prose");
        expect(prompt).not.toContain("return an Overview visual plus focused visual(s)");
        writerSchemas.push(JSON.parse(await readFile(schemaPath, "utf8")));
        return { ...explanation, ...strayVisual };
      })
      .mockResolvedValueOnce(reviseReview)
      .mockImplementationOnce(async (prompt, schemaPath) => {
        expect(prompt).toContain("Visuals are disabled. Write only prose");
        writerSchemas.push(JSON.parse(await readFile(schemaPath, "utf8")));
        return { ...explanation, ...strayVisual, summary: "Explains when retries stop." };
      })
      .mockResolvedValueOnce(passedReview);
    const response = await createApiHarness(runner)("/api/explain", {
      source_text: "if (attempt >= limit) return false;\nreturn retry();",
      visual_preference: "no",
    });

    expect(response.status).toBe(200);
    expect(response.json()).toMatchObject({
      summary: "Explains when retries stop.", visuals: [], visual_note: "", quality_review_status: "revised",
    });
    expect(writerSchemas).toHaveLength(2);
    for (const schema of writerSchemas) {
      expect(schema).not.toHaveProperty("properties.visuals");
      expect(schema).not.toHaveProperty("properties.visual_note");
    }
    expect(runner).toHaveBeenCalledTimes(4);
    for (const [prompt] of [runner.mock.calls[1]!, runner.mock.calls[3]!]) {
      expect(prompt).toContain("Review prose, factual accuracy, and citations only");
      expect(prompt).not.toContain("Unrequested");
    }
  });

  it("returns an unreviewed draft in one call and reviews that exact draft on request", async () => {
    const runner = vi.fn<CodexRunner>()
      .mockResolvedValueOnce(explanation)
      .mockResolvedValueOnce(passedReview);
    const api = createApiHarness(runner);
    const input = { source_text: "const attempts = 3;", visual_preference: "no" };
    const response = await api("/api/explain", { ...input, review_mode: "draft" });
    expect(response.status).toBe(200);
    expect(response.json()).toMatchObject({ title: explanation.title, summary: explanation.summary, quality_review_status: "unreviewed" });
    expect(runner).toHaveBeenCalledTimes(1);

    const invalid = await api("/api/explain", { ...input, review_mode: "review", draft: { ...explanation, concepts: [
      { term: "Attempts", explanation: "A limit.", source_ranges: [{ start_line: 2, end_line: 2 }] },
    ] } });
    expect(invalid.status).toBe(400);
    expect(runner).toHaveBeenCalledTimes(1);

    const reviewed = await api("/api/explain", { ...input, review_mode: "review", draft: response.json() });
    expect(reviewed.status).toBe(200);
    expect(reviewed.json()).toMatchObject({ title: explanation.title, summary: explanation.summary, quality_review_status: "passed" });
    expect(runner).toHaveBeenCalledTimes(2);
    expect(runner.mock.calls[1]?.[0]).toContain(explanation.summary);
    expect(runner.mock.calls[1]?.[0]).toContain("You are the second-pass quality reviewer");
  });

  it("rejects unsupported chapter preferences before model work", async () => {
    for (const preferences of [{ explanation_tone: "persuasive" }, { explanation_focus: "invented" }, { explanation_tone: ["plain"] }]) {
      const response = await request("/api/explain", { source_text: "const value = 42;", ...preferences });
      expect(response.status).toBe(400);
    }
    expect(runCodex).not.toHaveBeenCalled();
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

  it("reports TODO comments and empty conditional branches in code results", async () => {
    const source = [
      "async function fetchUserProfile(userId) {",
      "  if (!userId) return { error: 'Missing user ID' };",
      "  const response = await fetch(`/api/users/${userId}`);",
      "  if (!response.ok) {",
      "    // TODO: handle the error",
      "  }",
      "  const profile = await response.json();",
      "  // TODO: check whether the user is allowed to view this profile",
      "  return { name: profile.name, email: profile.email };",
      "}",
    ].join("\n");
    const response = await request("/api/explain", { source_text: source });
    const result = response.json() as typeof explanation & { incomplete_items: { title: string; explanation: string; source_ranges: { start_line: number; end_line: number }[] }[] };

    expect(response.status).toBe(200);
    expect(result.incomplete_items).toHaveLength(2);
    expect(result.incomplete_items[0]?.title).toBe("Unfinished branch");
    expect(result.incomplete_items[0]?.explanation).toContain("Execution falls through to the next statement");
    expect(result.incomplete_items[0]?.source_ranges).toEqual([{ start_line: 5, end_line: 6 }]);
    expect(result.incomplete_items[1]?.title).toBe("Unfinished-work marker");
    expect(result.incomplete_items[1]?.source_ranges).toEqual([{ start_line: 8, end_line: 8 }]);
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
