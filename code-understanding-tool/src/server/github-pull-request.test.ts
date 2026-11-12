import { describe, expect, it, vi } from "vitest";
import { loadPublicPullRequest, parsePublicPullRequestUrl } from "./github-pull-request";

describe("GitHub pull request import", () => {
  it("accepts canonical public GitHub pull request URLs only", () => {
    expect(parsePublicPullRequestUrl("https://github.com/acme/widget/pull/42")).toEqual({ owner: "acme", repo: "widget", number: 42 });
    expect(() => parsePublicPullRequestUrl("http://github.com/acme/widget/pull/42")).toThrow("Use a public GitHub");
    expect(() => parsePublicPullRequestUrl("https://github.com.example/acme/widget/pull/42")).toThrow("Use a public GitHub");
    expect(() => parsePublicPullRequestUrl("https://github.com/acme/widget/issues/42")).toThrow("Use a public GitHub");
  });

  it("combines pull request metadata with changed-file patches", async () => {
    const fetcher = vi.fn(async (input: string | URL) => {
      if (String(input).endsWith("/files?per_page=100")) {
        return Response.json([{ filename: "src/math.ts", status: "modified", additions: 2, deletions: 1, patch: "@@ -1 +1,2 @@\n-return 1;\n+return 2;" }]);
      }
      return Response.json({ title: "Fix math", body: "Correct the result.", html_url: "https://github.com/acme/widget/pull/42", state: "open", base: { ref: "main" }, head: { ref: "fix/math", sha: "abc123" } });
    });

    const source = await loadPublicPullRequest("https://github.com/acme/widget/pull/42", fetcher);
    expect(source).toContain("Pull request: Fix math");
    expect(source).toContain("Correct the result.");
    expect(source).toContain("File: src/math.ts (modified; +2/-1)");
    expect(source).toContain("+return 2;");
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("explains that inaccessible and private pull requests are unsupported", async () => {
    await expect(loadPublicPullRequest("https://github.com/acme/private/pull/1", async () => new Response("", { status: 404 })))
      .rejects.toThrow("public GitHub pull requests only");
  });
});
