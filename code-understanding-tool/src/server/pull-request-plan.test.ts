import { describe, expect, it, vi } from "vitest";
import { loadPullRequestPlan } from "./pull-request-plan";
import { fileTheme } from "../domain/pull-request-plan";

const url = "https://github.com/acme/demo/pull/7";
const file = (name: string, patch = "@@ -1 +1 @@\n-old\n+new") => ({ filename: name, status: "modified", additions: 1, deletions: 1, patch });
function fixture(files: ReturnType<typeof file>[], reportedCount = files.length) {
  return vi.fn(async (input: string | URL) => {
    const path = String(input);
    if (!path.includes("/files?")) return Response.json({ title: "A migration", body: "Update behavior", head: { sha: "abc" }, changed_files: reportedCount });
    const page = Number(new URL(path).searchParams.get("page"));
    return Response.json(files.slice((page - 1) * 100, page * 100), { headers: page * 100 < files.length ? { link: '<https://untrusted.example>; rel="next"' } : {} });
  });
}

describe("coverage-aware PR planning", () => {
  it("groups implementation, tests, documentation, dependencies and CI predictably", () => {
    expect(fileTheme("src/server/router.ts")).toBe("Implementation: src/server");
    expect(fileTheme("src/router.test.ts")).toBe("Tests and verification");
    expect(fileTheme("docs/setup.md")).toBe("Documentation and examples");
    expect(fileTheme("package-lock.json")).toBe("Dependencies and configuration");
    expect(fileTheme(".github/workflows/check.yml")).toBe("Build and automation");
    expect(fileTheme(".github/workflows/test.yml")).toBe("Build and automation");
  });

  it("fetches subsequent pages from GitHub, never the supplied next URL", async () => {
    const fetcher = fixture(Array.from({ length: 105 }, (_, index) => file(`src/file${index}.ts`)));
    const plan = await loadPullRequestPlan(url, fetcher);
    expect(plan.files).toHaveLength(105);
    expect(plan.filesComplete).toBe(true);
    expect(fetcher.mock.calls.every(([input]) => String(input).startsWith("https://api.github.com/repos/acme/demo/pulls/7"))).toBe(true);
    expect(plan.chapters[0].sourceText).toContain("No patches are provided for this overview");
    expect(plan.chapters[0].sourceText).not.toContain("+new");
    const chapterPaths = plan.chapters.flatMap((chapter) => chapter.files);
    expect(new Set(chapterPaths).size).toBe(105);
    expect(chapterPaths).toHaveLength(105);
    expect(plan.chapters.every((chapter) => chapter.sourceText.length < 200_000)).toBe(true);
  });

  it("retains unavailable files in the manifest without inventing evidence chapters", async () => {
    const plan = await loadPullRequestPlan(url, fixture([file("image.png", "")]));
    expect(plan.files[0].coverage).toBe("unavailable");
    expect(plan.chapters).toHaveLength(1);
    expect(plan.warnings.join(" ")).toContain("patch unavailable");
  });

  it("marks per-file excerpts and evidence-budget omissions explicitly", async () => {
    const plan = await loadPullRequestPlan(url, fixture(Array.from({ length: 28 }, (_, index) => file(`src/${index}.ts`, "+line\n".repeat(5_000)))));
    expect(plan.files.some((entry) => entry.coverage === "partial")).toBe(true);
    expect(plan.files.some((entry) => entry.coverage === "omitted_budget")).toBe(true);
    expect(plan.chapters.some((chapter) => chapter.sourceText.includes("per-file size limit"))).toBe(true);
    expect(plan.warnings.join(" ")).toContain("evidence budget");
  });

  it("caps pagination and distinguishes an incomplete manifest from patch coverage", async () => {
    const plan = await loadPullRequestPlan(url, fixture(Array.from({ length: 501 }, (_, index) => file(`src/${index}.ts`))));
    expect(plan.files).toHaveLength(500);
    expect(plan.totalFiles).toBe(501);
    expect(plan.filesComplete).toBe(false);
    expect(plan.warnings.join(" ")).toContain("first 500");
  });

  it("rejects a PR whose head changes while pages are being fetched", async () => {
    let metadataReads = 0;
    const fetcher = async (input: string | URL) => String(input).includes("/files?") ? Response.json([]) : Response.json({ head: { sha: String(++metadataReads) } });
    await expect(loadPullRequestPlan(url, fetcher)).rejects.toThrow("changed during import");
  });

  it("marks reported-count mismatches as incomplete", async () => {
    const plan = await loadPullRequestPlan(url, fixture([file("a.ts")], 10));
    expect(plan.filesComplete).toBe(false);
    expect(plan.warnings.join(" ")).toContain("incomplete");
  });

  it("rejects unsafe URLs before making requests and surfaces rate limits", async () => {
    const fetcher = vi.fn(async () => new Response("", { status: 429 }));
    await expect(loadPullRequestPlan("https://evil.example/pull/1", fetcher)).rejects.toThrow("public GitHub");
    expect(fetcher).not.toHaveBeenCalled();
    await expect(loadPullRequestPlan(url, fetcher)).rejects.toThrow("rate-limiting");
  });
});
