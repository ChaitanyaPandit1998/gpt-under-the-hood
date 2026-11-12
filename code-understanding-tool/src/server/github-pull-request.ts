const MAX_GITHUB_RESPONSE_BYTES = 1_000_000;
const MAX_PULL_REQUEST_SOURCE_CHARACTERS = 180_000;
const MAX_PULL_REQUEST_FILES = 100;

export type PullRequestFetch = (input: string | URL, init?: RequestInit) => Promise<Response>;

export function parsePublicPullRequestUrl(value: unknown): { owner: string; repo: string; number: number } {
  if (typeof value !== "string" || value.length > 2_000) {
    throw new Error("Enter a public GitHub pull request link.");
  }
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error("Enter a valid GitHub pull request link.");
  }
  const match = url.pathname.match(/^\/([^/]+)\/([^/]+)\/pull\/(\d+)\/?$/);
  if (url.protocol !== "https:" || url.hostname !== "github.com" || url.port || url.username || url.password || !match) {
    throw new Error("Use a public GitHub pull request link like https://github.com/owner/repo/pull/123.");
  }
  const number = Number(match[3]);
  if (!Number.isSafeInteger(number) || number < 1) throw new Error("The pull request number is invalid.");
  return { owner: match[1]!, repo: match[2]!, number };
}

export async function readBoundedJson(response: Response): Promise<unknown> {
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > MAX_GITHUB_RESPONSE_BYTES) {
    throw new Error("GitHub returned an unusually large pull request response.");
  }
  if (!response.body) throw new Error("GitHub returned an empty response.");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_GITHUB_RESPONSE_BYTES) {
        await reader.cancel();
        throw new Error("GitHub returned an unusually large pull request response.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return JSON.parse(new TextDecoder().decode(bytes));
}

export async function loadPublicPullRequest(value: unknown, fetcher: PullRequestFetch = fetch): Promise<string> {
  const { owner, repo, number } = parsePublicPullRequestUrl(value);
  const base = `https://api.github.com/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/pulls/${number}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    const headers = { Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "code-understanding-tool" };
    const pullResponse = await fetcher(base, { headers, signal: controller.signal });
    if (pullResponse.status === 404) throw new Error("Pull request not found. This importer supports public GitHub pull requests only.");
    if (pullResponse.status === 403 || pullResponse.status === 429) throw new Error("GitHub is rate-limiting requests. Wait a little and try again.");
    if (!pullResponse.ok) throw new Error(`GitHub could not load this pull request (HTTP ${pullResponse.status}).`);
    const pull = await readBoundedJson(pullResponse) as {
      title?: unknown; body?: unknown; html_url?: unknown; state?: unknown;
      base?: { ref?: unknown }; head?: { ref?: unknown; sha?: unknown };
    };
    const filesResponse = await fetcher(`${base}/files?per_page=${MAX_PULL_REQUEST_FILES}`, { headers, signal: controller.signal });
    if (filesResponse.status === 403 || filesResponse.status === 429) throw new Error("GitHub is rate-limiting requests. Wait a little and try again.");
    if (!filesResponse.ok) throw new Error(`GitHub could not load the pull request files (HTTP ${filesResponse.status}).`);
    const files = await readBoundedJson(filesResponse);
    if (!Array.isArray(files)) throw new Error("GitHub returned an invalid pull request file list.");
    const lines = [
      `Pull request: ${String(pull.title ?? "Untitled")}`,
      `Link: ${String(pull.html_url ?? value)}`,
      `State: ${String(pull.state ?? "unknown")}`,
      `Branch: ${String(pull.head?.ref ?? "unknown")} -> ${String(pull.base?.ref ?? "unknown")}`,
      `Head commit: ${String(pull.head?.sha ?? "unknown")}`,
      "",
      "Pull request description:",
      typeof pull.body === "string" && pull.body.trim() ? pull.body : "(No description provided.)",
      "",
      "Changed files and patches:",
    ];
    if (filesResponse.headers.get("link")?.includes('rel="next"')) {
      lines.push(`[Only the first ${MAX_PULL_REQUEST_FILES} changed files were fetched.]`);
    }
    let omittedFiles = 0;
    for (const file of files.slice(0, MAX_PULL_REQUEST_FILES) as { filename?: unknown; status?: unknown; additions?: unknown; deletions?: unknown; patch?: unknown }[]) {
      const section = [
        `\n--- File: ${String(file.filename ?? "unknown")} (${String(file.status ?? "changed")}; +${Number(file.additions) || 0}/-${Number(file.deletions) || 0}) ---`,
        typeof file.patch === "string" ? file.patch : "[Patch unavailable for this file, often because it is binary or too large.]",
      ].join("\n");
      if (lines.join("\n").length + section.length > MAX_PULL_REQUEST_SOURCE_CHARACTERS) { omittedFiles += 1; continue; }
      lines.push(section);
    }
    if (omittedFiles) lines.push(`\n[${omittedFiles} file patch(es) omitted to stay within the local explanation size limit.]`);
    return lines.join("\n").slice(0, MAX_PULL_REQUEST_SOURCE_CHARACTERS);
  } catch (error) {
    if (controller.signal.aborted) throw new Error("Loading the pull request timed out. Try again.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
