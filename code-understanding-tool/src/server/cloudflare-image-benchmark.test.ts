import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { generateBenchmarkImage, imageTrials, IMAGE_MODEL, type ImageCase } from "./cloudflare-image-benchmark";

const dataset = JSON.parse(readFileSync(new URL("../../evals/image-cases.json", import.meta.url), "utf8")) as { cases: ImageCase[] };
const credentials = { accountId: "a".repeat(32), apiToken: "test-only-token" };
const image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=";

describe("Cloudflare image benchmark", () => {
  it("prepares exactly six bounded prompts from curated synthetic cases", () => {
    const trials = imageTrials(dataset.cases);
    expect(trials).toHaveLength(6);
    expect(new Set(trials.map((trial) => trial.id)).size).toBe(6);
    for (const trial of trials) {
      expect(trial.prompt.length).toBeLessThanOrEqual(2048);
      expect(trial.prompt).toContain("not a verified architecture or execution diagram");
      expect(trial.steps).toBe(4);
      expect(trial.seed).toBeNull();
    }
  });

  it("rejects too many cases, unsafe IDs, and identifying or secret content", () => {
    expect(() => imageTrials([...dataset.cases, dataset.cases[0]])).toThrow("one to three");
    expect(() => imageTrials([{ ...dataset.cases[0], id: "../unsafe" }])).toThrow("safe ID");
    for (const summary of ["Visit https://private.example", "Email user@example.com", "Use ghp_secretvalue"]) {
      expect(() => imageTrials([{ ...dataset.cases[0], summary }])).toThrow("Review the curated case manually");
    }
    expect(() => imageTrials([{ ...dataset.cases[0], summary: "x".repeat(2048) }])).toThrow("2048");
  });

  it("keeps decorative artwork prompts distinct from execution diagrams", () => {
    const cases = JSON.parse(readFileSync(new URL("../../evals/image-illustration-cases.json", import.meta.url), "utf8")).cases as ImageCase[];
    const trials = imageTrials(cases, "decorative");
    expect(trials.map((trial) => trial.style)).toEqual(["editorial", "isometric"]);
    for (const trial of trials) {
      expect(trial.prompt).toContain("Decorative metaphor only");
      expect(trial.prompt).not.toContain("three-panel");
      expect(trial.prompt.length).toBeLessThanOrEqual(2048);
    }
  });

  it("makes a single authenticated model request with fixed benchmark settings", async () => {
    const fetcher = vi.fn<typeof fetch>(async () => Response.json({ success: true, result: { image } }, { headers: { "cf-ray": "request-example" } }));
    const trial = imageTrials(dataset.cases)[0];
    const result = await generateBenchmarkImage(trial, credentials, fetcher);
    expect(fetcher).toHaveBeenCalledTimes(1);
    const [url, options] = fetcher.mock.calls[0];
    expect(url).toBe(`https://api.cloudflare.com/client/v4/accounts/${credentials.accountId}/ai/run/${IMAGE_MODEL}`);
    expect(options?.headers).toMatchObject({ Authorization: `Bearer ${credentials.apiToken}` });
    expect(options?.redirect).toBe("error");
    expect(JSON.parse(String(options?.body))).toEqual({ prompt: trial.prompt, steps: 4 });
    expect(result.extension).toBe("png");
    expect(result.bytes.equals(Buffer.from(image, "base64"))).toBe(true);
    expect(result.latencyMs).toBeGreaterThanOrEqual(0);
    expect(result.requestId).toBe("request-example");
  });

  it("does not make requests without credentials", async () => {
    const fetcher = vi.fn<typeof fetch>();
    await expect(generateBenchmarkImage(imageTrials(dataset.cases)[0], { accountId: "", apiToken: "" }, fetcher)).rejects.toThrow("Configure");
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("does not leak provider error bodies or automatically retry", async () => {
    const fetcher = vi.fn<typeof fetch>(async () => new Response(credentials.apiToken, { status: 401 }));
    await expect(generateBenchmarkImage(imageTrials(dataset.cases)[0], credentials, fetcher)).rejects.toThrow("HTTP 401");
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("redacts transport errors", async () => {
    const fetcher = vi.fn<typeof fetch>(async () => { throw new Error(credentials.apiToken); });
    await expect(generateBenchmarkImage(imageTrials(dataset.cases)[0], credentials, fetcher)).rejects.toThrow("credentials and response details were not logged");
  });

  it.each([
    { success: false, errors: [{ message: "Do not persist this message" }] },
    { success: true, result: { image: "not base64" } },
    { success: true, result: { image: Buffer.from("not an image").toString("base64") } },
  ])("rejects unsuccessful or malformed image payloads", async (payload) => {
    await expect(generateBenchmarkImage(imageTrials(dataset.cases)[0], credentials, async () => Response.json(payload))).rejects.toThrow();
  });

  it("enforces response-size limits", async () => {
    await expect(generateBenchmarkImage(imageTrials(dataset.cases)[0], credentials,
      async () => new Response("{}", { headers: { "content-length": "10000001" } }))).rejects.toThrow("oversized");
  });
});
