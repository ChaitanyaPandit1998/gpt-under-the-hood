import type { ViteDevServer } from "vite";
import { parseChapterStoryRequest } from "../domain/chapter-story.ts";
import { createReviewedChapterStory } from "./chapter-story-service.ts";
import type { CodexRunner } from "./explanation-service.ts";
import { readJsonRequest, sendJson } from "./local-api-http.ts";
import type { RunModelRequest } from "./model-request.ts";

export function registerChapterStoryApi(middlewares: ViteDevServer["middlewares"], {
  executeCodex, runModelRequest, maxSourceCharacters, timeoutMs, root,
}: { executeCodex: CodexRunner; runModelRequest: RunModelRequest; maxSourceCharacters: number; timeoutMs: number; root: string }) {
  middlewares.use("/api/chapter-story", async (request, response) => {
    response.setHeader("Cache-Control", "no-store");
    if (request.method !== "POST") {
      response.setHeader("Allow", "POST");
      sendJson(response, 405, { error: "Method not allowed." });
      return;
    }
    let input: ReturnType<typeof parseChapterStoryRequest>;
    try {
      input = parseChapterStoryRequest(await readJsonRequest(request, 750_000), maxSourceCharacters);
    } catch (error) {
      const tooLarge = error instanceof Error && error.message === "REQUEST_TOO_LARGE";
      sendJson(response, tooLarge ? 413 : 400, { error: tooLarge ? "Chapter story request is too large." : "A valid chapter, source, and reading preferences are required." });
      return;
    }
    try {
      const result = await runModelRequest(request, response, "chapter-story", (signal, markPhase) =>
        createReviewedChapterStory(input, executeCodex, signal, markPhase, root), timeoutMs);
      if (result.status === "completed") sendJson(response, 200, { story: result.value });
    } catch {
      if (!response.destroyed && !response.headersSent) {
        sendJson(response, 502, { error: "Could not prepare a reviewed story. The direct chapter is still available; try again." });
      }
    }
  });
}
