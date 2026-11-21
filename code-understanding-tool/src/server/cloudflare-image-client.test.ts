import { describe, expect, it, vi } from "vitest";
import { generateCloudflareImage } from "./cloudflare-image-client";

describe("Cloudflare image cancellation", () => {
  it("never calls the provider for an already cancelled request", async () => {
    const controller = new AbortController();
    controller.abort();
    const fetcher = vi.fn<typeof fetch>();
    await expect(generateCloudflareImage("Artwork", { accountId: "a".repeat(32), apiToken: "test-token" }, fetcher, controller.signal)).rejects.toThrow();
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("propagates cancellation to the image provider without logging its body", async () => {
    const controller = new AbortController();
    const fetcher = vi.fn<typeof fetch>(async (_input, init) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new Error("PRIVATE_DETAILS")), { once: true });
    }));
    const pending = generateCloudflareImage("Artwork", { accountId: "a".repeat(32), apiToken: "test-token" }, fetcher, controller.signal);
    controller.abort();
    await expect(pending).rejects.toThrow("cancelled or timed out");
    expect(fetcher.mock.calls[0][1]?.signal?.aborted).toBe(true);
  });
});
