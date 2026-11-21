import { describe, expect, it, vi } from "vitest";
import { loadPullRequestEvidence } from "./pull-request-evidence";

const request = { url: "https://github.com/acme/demo/pull/7", headSha: "a".repeat(40), filename: "src/large.ts", page: 2, offset: 0 };
function fixture(patch: string) {
  return vi.fn(async (input: string | URL) => String(input).includes("/files?")
    ? Response.json([{ filename: request.filename, patch }])
    : Response.json({ title: "Update code", head: { sha: request.headSha }, changed_files: 101 }));
}

describe("on-demand PR patch evidence", () => {
  it("retrieves a selected file from its manifest page without invoking arbitrary URLs", async () => {
    const fetcher = fixture("@@ -1 +1 @@\n-old\n+new");
    const result = await loadPullRequestEvidence(request, fetcher);
    expect(fetcher.mock.calls[1][0]).toBe("https://api.github.com/repos/acme/demo/pulls/7/files?per_page=100&page=2");
    expect(result.nextOffset).toBeNull();
    expect(result.chapter.sourceText).toContain("Other sections, other chapters, and full repository files are not supplied");
    expect(result.chapter.sourceText).toContain("+new");
    expect(result.chapter.files).toEqual([request.filename]);
  });

  it("retrieves every part of oversized patches without gaps or duplicate characters", async () => {
    const patch = "+long line\n".repeat(6_000);
    let offset: number | null = 0;
    let combined = "";
    const ids = new Set<string>();
    while (offset !== null) {
      const result = await loadPullRequestEvidence({ ...request, offset }, fixture(patch));
      const section = patch.slice(result.offset, result.endOffset);
      expect(result.chapter.sourceText.endsWith(section)).toBe(true);
      expect(result.endOffset - result.offset).toBeLessThanOrEqual(24_000);
      expect(result.endOffset).toBeGreaterThan(result.offset);
      expect(ids.has(result.chapter.id)).toBe(false);
      ids.add(result.chapter.id);
      combined += section;
      offset = result.nextOffset;
    }
    expect(ids.size).toBeGreaterThan(1);
    expect(combined).toBe(patch);
  });

  it("does not split surrogate pairs in unusually long lines", async () => {
    const patch = "+".repeat(23_999) + "\u{1F680}" + "tail";
    const first = await loadPullRequestEvidence(request, fixture(patch));
    expect(first.endOffset).toBe(23_999);
    const second = await loadPullRequestEvidence({ ...request, offset: first.nextOffset }, fixture(patch));
    expect(second.chapter.sourceText).toContain("\u{1F680}tail");
  });

  it.each([{ page: 0 }, { page: 6 }, { offset: -1 }, { offset: 0.5 }, { headSha: "bad" }, { baseSha: "bad" }, { filename: "" }, { url: "https://evil.example/pull/1" }])("rejects invalid requests before fetching: %j", async (invalid) => {
    const fetcher = fixture("+safe");
    await expect(loadPullRequestEvidence({ ...request, ...invalid }, fetcher)).rejects.toThrow();
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("rejects a revision change before reading patches", async () => {
    const fetcher = vi.fn(async () => Response.json({ head: { sha: "b".repeat(40) } }));
    await expect(loadPullRequestEvidence(request, fetcher)).rejects.toThrow("revision changed");
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("rejects a revision change during retrieval", async () => {
    let reads = 0;
    const fetcher = async (input: string | URL) => String(input).includes("/files?") ? Response.json([{ filename: request.filename, patch: "+new" }])
      : Response.json({ head: { sha: ++reads === 1 ? request.headSha : "b".repeat(40) } });
    await expect(loadPullRequestEvidence(request, fetcher)).rejects.toThrow("changed while loading");
  });

  it("rejects base changes even when the PR head stays the same", async () => {
    const fetcher = vi.fn(async () => Response.json({ head: { sha: request.headSha }, base: { sha: "c".repeat(40) } }));
    await expect(loadPullRequestEvidence({ ...request, baseSha: "b".repeat(40) }, fetcher)).rejects.toThrow("base revision changed");
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("does not invent missing patches or accept out-of-range offsets", async () => {
    await expect(loadPullRequestEvidence(request, fixture(""))).rejects.toThrow("does not provide a patch");
    await expect(loadPullRequestEvidence({ ...request, offset: 10 }, fixture("+new"))).rejects.toThrow("out of range");
    await expect(loadPullRequestEvidence({ ...request, filename: "other.ts" }, fixture("+new"))).rejects.toThrow("not uniquely present");
  });

  it("surfaces rate limits and response-size limits", async () => {
    await expect(loadPullRequestEvidence(request, async () => new Response("", { status: 429 }))).rejects.toThrow("rate-limiting");
    await expect(loadPullRequestEvidence(request, async () => new Response("{}", { headers: { "content-length": "1000001" } }))).rejects.toThrow("unusually large");
  });
});
