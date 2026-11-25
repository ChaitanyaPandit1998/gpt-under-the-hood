import type { ViteDevServer } from "vite";
import { chapterArtworkPrompt, parseArtworkRequest } from "../domain/chapter-illustrations.ts";
import { parseVisualBriefRequest, validateVisualBrief } from "../domain/visual-brief.ts";
import { parseVisualResolution, type VisualResolution } from "../domain/visual-resolution.ts";
import { validateVisualScene, type VisualScene } from "../domain/visual-scene.ts";
import { generateCloudflareImage, CLOUDFLARE_IMAGE_MODEL, type CloudflareImageCredentials } from "./cloudflare-image-client.ts";
import type { CodexRunner } from "./explanation-service.ts";
import { readJsonRequest, sendJson } from "./local-api-http.ts";
import type { RunModelRequest } from "./model-request.ts";
import type { DiagramExporter } from "./vectorcraft-export.ts";
import { createVisualBrief, createVisualScene } from "./visual-brief-service.ts";
import { parseVisualReviewRequest, visualReviewRequestLimit } from "./visual-review-request.ts";
import { reviewVisualImage } from "./visual-review-service.ts";
import { generationStage, type ApplicationInstruction } from "./application-instructions.ts";

type VisualApiOptions = {
  root: string;
  maxSourceLines: number;
  executeCodex: CodexRunner;
  runModelRequest: RunModelRequest;
  diagrams: DiagramExporter;
  compactVisualPlanning?: boolean;
  imageCredentials?: CloudflareImageCredentials;
  fetchImages?: typeof fetch;
  maxChapterCovers: number;
};

export function registerVisualApi(middlewares: ViteDevServer["middlewares"], options: VisualApiOptions) {
  const { root, maxSourceLines, executeCodex, runModelRequest, diagrams } = options;
  let chapterCoverRequests = 0;

  middlewares.use("/api/visual-review", async (request, response) => {
    response.setHeader("Cache-Control", "no-store");
    if (request.method !== "POST") { response.setHeader("Allow", "POST"); sendJson(response, 405, { error: "Method not allowed." }); return; }
    let input: ReturnType<typeof parseVisualReviewRequest>;
    try { input = parseVisualReviewRequest(await readJsonRequest(request, visualReviewRequestLimit)); }
    catch (error) {
      const oversized = error instanceof Error && error.message === "REQUEST_TOO_LARGE";
      sendJson(response, oversized ? 413 : 400, { error: oversized ? "Visual review request is too large." : "Confirm a valid checklist, explanation, caption and PNG before reviewing." });
      return;
    }
    try {
      const result = await runModelRequest(request, response, "visual-review", (signal, markPhase) => {
        markPhase("reviewing_image");
        return reviewVisualImage(input, { run: executeCodex, signal, root,
          reviewer: { name: "Configured Codex vision reviewer", version: "visual-review-rubric-1; provider model version unverified" } });
      }, 180_000);
      if (result.status === "completed") sendJson(response, 200, { report: result.value });
    } catch {
      if (!response.destroyed && !response.headersSent) sendJson(response, 502, { error: "Could not complete visual review. Your image is unchanged." });
    }
  });

  middlewares.use("/api/visual-brief", async (request, response) => {
    response.setHeader("Cache-Control", "no-store");
    if (request.method !== "POST") { response.setHeader("Allow", "POST"); sendJson(response, 405, { error: "Method not allowed." }); return; }
    let input: ReturnType<typeof parseVisualBriefRequest>;
    try { input = parseVisualBriefRequest(await readJsonRequest(request, 1_000_000)); }
    catch (error) {
      const oversized = error instanceof Error && error.message === "REQUEST_TOO_LARGE";
      sendJson(response, oversized ? 413 : 400, { error: oversized ? "Visual brief request is too large." : "Invalid visual brief preferences or context." });
      return;
    }
    try {
      let instructions: ApplicationInstruction[] = [];
      const result = await runModelRequest(request, response, "visual-brief", (signal, markPhase) => {
        markPhase("designing_visual");
        return createVisualBrief(input, root, executeCodex, signal, loaded => { instructions = loaded; });
      });
      if (result.status === "completed") sendJson(response, 200, { brief: result.value, generation: generationStage(result.value, instructions) });
    } catch {
      if (!response.destroyed && !response.headersSent) sendJson(response, 502, { error: "Could not prepare a visual brief. Try again or narrow the question." });
    }
  });
  middlewares.use("/api/visual-plan", async (request, response) => {
    response.setHeader("Cache-Control", "no-store");
    if (request.method !== "POST") { response.setHeader("Allow", "POST"); sendJson(response, 405, { error: "Method not allowed." }); return; }
    let brief: ReturnType<typeof validateVisualBrief>;
    try {
      brief = validateVisualBrief(await readJsonRequest(request, 120_000), maxSourceLines);
      if (brief.disposition !== "render") throw new Error("Text-only brief.");
    } catch { sendJson(response, 400, { error: "A renderable visual brief is required." }); return; }
    if (!diagrams.renderScene || !await diagrams.available()) {
      sendJson(response, 503, { error: "VectorCraft is unavailable. Enable the approved local CLI to render the image." });
      return;
    }
    try {
      let instructions: ApplicationInstruction[] = [];
      const result = await runModelRequest(request, response, "visual-plan", (signal, markPhase) => {
        markPhase("planning_scene");
        return createVisualScene(brief, root, executeCodex, signal, options.compactVisualPlanning, loaded => { instructions = loaded; });
      });
      if (result.status === "completed") sendJson(response, 200, { scene: result.value, generation: generationStage(result.value, instructions) });
    } catch {
      if (!response.destroyed && !response.headersSent) sendJson(response, 502, { error: "Could not plan the drawing. Try again or simplify the brief." });
    }
  });
  middlewares.use("/api/visual-render", async (request, response) => {
    response.setHeader("Cache-Control", "no-store");
    if (request.method !== "POST") { response.setHeader("Allow", "POST"); sendJson(response, 405, { error: "Method not allowed." }); return; }
    let brief: ReturnType<typeof validateVisualBrief> | undefined;
    let scene: VisualScene | undefined;
    let resolution: VisualResolution = "standard";
    try {
      const data = await readJsonRequest(request, 500_000);
      if (data && typeof data === "object" && "scene" in data) {
        scene = validateVisualScene(data.scene);
        resolution = parseVisualResolution("resolution" in data ? data.resolution : undefined);
      } else if (data && typeof data === "object" && "brief" in data) {
        brief = validateVisualBrief(data.brief, maxSourceLines);
        resolution = parseVisualResolution("resolution" in data ? data.resolution : undefined);
      } else {
        brief = validateVisualBrief(data, maxSourceLines);
      }
      if (brief && brief.disposition !== "render") throw new Error("Text-only brief.");
    } catch { sendJson(response, 400, { error: "A valid drawing or renderable brief and a supported image size are required." }); return; }
    const renderScene = diagrams.renderScene;
    if (!renderScene || !await diagrams.available()) {
      sendJson(response, 503, { error: "VectorCraft is unavailable. Enable the approved local CLI to render the image." });
      return;
    }
    try {
      const result = await runModelRequest(request, response, "visual-render", async (signal, markPhase) => {
        if (brief) {
          markPhase("planning_scene");
          scene = await createVisualScene(brief, root, executeCodex, signal, options.compactVisualPlanning);
        }
        if (!scene) throw new Error("Missing validated drawing.");
        markPhase("rendering_scene");
        return renderScene(scene, signal, resolution);
      });
      if (result.status === "completed") {
        response.setHeader("Content-Type", "image/png");
        response.setHeader("X-Content-Type-Options", "nosniff");
        response.end(result.value);
      }
    } catch {
      if (!response.destroyed && !response.headersSent) sendJson(response, 502, { error: "Could not render a complete image. Try again or simplify the brief." });
    }
  });

  middlewares.use("/api/chapter-cover", async (request, response) => {
    response.setHeader("Cache-Control", "no-store");
    if (request.method !== "POST") {
      response.setHeader("Allow", "POST");
      sendJson(response, 405, { error: "Method not allowed" });
      return;
    }
    let theme;
    try { theme = parseArtworkRequest(await readJsonRequest(request, 4_000)); }
    catch (error) {
      sendJson(response, error instanceof Error && error.message === "REQUEST_TOO_LARGE" ? 413 : 400,
        { error: error instanceof Error && error.message !== "REQUEST_TOO_LARGE" ? error.message : "Artwork request is too large." });
      return;
    }
    const credentials = options.imageCredentials;
    if (!credentials?.accountId || !credentials.apiToken) {
      sendJson(response, 503, { error: "Cloudflare image credentials are not configured on the local server. Configure them locally and restart the server." });
      return;
    }
    if (chapterCoverRequests >= options.maxChapterCovers) {
      sendJson(response, 429, { error: "The server-session chapter-cover request limit has been reached. No image request was sent." });
      return;
    }
    try {
      const result = await runModelRequest(request, response, "chapter-cover", async (signal, markPhase) => {
        chapterCoverRequests += 1;
        markPhase("generating_artwork");
        const image = await generateCloudflareImage(chapterArtworkPrompt(theme), credentials, options.fetchImages ?? fetch, signal);
        return { theme, model: CLOUDFLARE_IMAGE_MODEL, latencyMs: image.latencyMs, dataUrl: `data:image/${image.extension === "jpg" ? "jpeg" : "png"};base64,${image.bytes.toString("base64")}` };
      }, 95_000);
      if (result.status === "completed") sendJson(response, 200, result.value);
    } catch (error) {
      if (!response.headersSent && !response.destroyed) sendJson(response, 500, { error: error instanceof Error ? error.message : "Artwork generation failed." });
    }
  });
}
