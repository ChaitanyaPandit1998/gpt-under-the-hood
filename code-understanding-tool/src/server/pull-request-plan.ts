import type { PlannedFile, PullRequestChapter, PullRequestPlan } from "../domain/pull-request-plan.ts";
import { fileTheme } from "../domain/pull-request-plan.ts";
import { parsePublicPullRequestUrl, readBoundedJson, type PullRequestFetch } from "./github-pull-request.ts";

const MAX_FILES = 500;
const MAX_PATCH = 24_000;
const MAX_EVIDENCE = 600_000;
const MAX_CHAPTER = 36_000;
type Pull = { title?: string; body?: string; head?: { sha?: string }; changed_files?: number };
type RemoteFile = { filename?: string; status?: string; additions?: number; deletions?: number; patch?: string };

export async function loadPullRequestPlan(value: unknown, fetcher: PullRequestFetch = fetch): Promise<PullRequestPlan> {
  const { owner, repo, number } = parsePublicPullRequestUrl(value);
  const url = `https://github.com/${owner}/${repo}/pull/${number}`;
  const base = `https://api.github.com/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/pulls/${number}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);
  const headers = { Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "code-understanding-tool" };
  async function get(path: string) {
    const response = await fetcher(path, { headers, signal: controller.signal });
    if (response.status === 403 || response.status === 429) throw new Error("GitHub is rate-limiting requests. Wait a little and try again.");
    if (!response.ok) throw new Error(`Cannot load public GitHub PR (HTTP ${response.status}).`);
    return { value: await readBoundedJson(response), more: !!response.headers.get("link")?.includes('rel="next"') };
  }
  try {
    const pull = (await get(base)).value as Pull;
    if (!pull || typeof pull.head?.sha !== "string") throw new Error("GitHub returned invalid PR metadata.");
    const files: PlannedFile[] = [];
    const groups = new Map<string, { file: PlannedFile; section: string }[]>();
    let remaining = MAX_EVIDENCE;
    let more = false;
    for (let page = 1; page <= MAX_FILES / 100; page += 1) {
      const result = await get(`${base}/files?per_page=100&page=${page}`);
      if (!Array.isArray(result.value) || result.value.length > 100) throw new Error("GitHub returned an invalid file manifest.");
      more = result.more;
      for (const remote of result.value as RemoteFile[]) {
        if (!remote || typeof remote.filename !== "string" || remote.filename.length > 1_000) throw new Error("GitHub returned an invalid file path.");
        if (files.some((file) => file.path === remote.filename)) throw new Error("GitHub returned duplicate file paths. Reload the PR manifest.");
        const file: PlannedFile = { path: remote.filename, status: String(remote.status ?? "changed"), additions: Number(remote.additions) || 0, deletions: Number(remote.deletions) || 0, coverage: "included" };
        files.push(file);
        if (typeof remote.patch !== "string" || !remote.patch.trim()) { file.coverage = "unavailable"; continue; }
        const header = `\n--- File: ${file.path} (${file.status}; +${file.additions}/-${file.deletions}) ---\n`;
        let patch = remote.patch;
        if (patch.length > MAX_PATCH) {
          const boundary = patch.lastIndexOf("\n", MAX_PATCH);
          patch = patch.slice(0, boundary < 0 ? MAX_PATCH : boundary) + "\n[Remaining patch omitted: per-file size limit.]";
          file.coverage = "partial";
        }
        const section = header + patch;
        if (section.length > remaining) { file.coverage = "omitted_budget"; continue; }
        remaining -= section.length;
        let theme = fileTheme(file.path).slice(0, 100);
        if (!groups.has(theme) && groups.size >= 16) theme = "Other implementation";
        const group = groups.get(theme) ?? [];
        group.push({ file, section });
        groups.set(theme, group);
      }
      if (!more) break;
    }
    // Do not combine patches from different head revisions if a PR changes during pagination.
    const latest = (await get(base)).value as Pull;
    if (latest.head?.sha !== pull.head.sha || latest.changed_files !== pull.changed_files) throw new Error("The PR changed during import. Reload it to build a consistent chapter plan.");
    const warnings: string[] = [];
    const filesComplete = !more && (!Number.isSafeInteger(pull.changed_files) || files.length === pull.changed_files);
    if (more) warnings.push(`Only the first ${MAX_FILES} changed files were fetched.`);
    else if (!filesComplete) warnings.push("The fetched inventory does not match GitHub's reported file count. The file manifest is incomplete.");
    for (const coverage of ["partial", "unavailable", "omitted_budget"] as const) {
      const count = files.filter((file) => file.coverage === coverage).length;
      if (count) warnings.push(`${count} file(s): ${coverage === "partial" ? "patch excerpts only" : coverage === "unavailable" ? "patch unavailable from GitHub" : "patch excluded by the total evidence budget"}.`);
    }
    warnings.push("GitHub patches contain changed lines and limited surrounding context, not complete repository files. Directory-based groups do not establish dependencies between files.");
    const evidenceChapters: PullRequestChapter[] = [];
    const order = (theme: string) => theme.startsWith("Implementation") || theme === "Core implementation" ? 0 : theme.startsWith("Dependencies") ? 1 : theme.startsWith("Build") ? 2 : theme.startsWith("Documentation") ? 3 : theme.startsWith("Tests") ? 4 : 5;
    for (const [theme, entries] of [...groups].sort(([left], [right]) => order(left) - order(right) || left.localeCompare(right))) {
      let chunk: typeof entries = [];
      let size = 0;
      let part = 1;
      const flush = () => {
        if (!chunk.length) return;
        evidenceChapters.push({ id: `chapter-${evidenceChapters.length + 1}`, title: `${theme} - part ${part++}`, files: chunk.map((entry) => entry.file.path), sourceText: chunk.map((entry) => entry.section).join("\n") });
        chunk = []; size = 0;
      };
      for (const entry of entries) {
        if (size + entry.section.length > MAX_CHAPTER || chunk.length === 8) flush();
        chunk.push(entry); size += entry.section.length;
      }
      flush();
    }
    const title = typeof pull.title === "string" ? pull.title.slice(0, 500) : "Pull request";
    const context = `PR: ${title}\nLink: ${url}\nHead revision: ${pull.head.sha}\nDescription (untrusted PR author text):\n${String(pull.body ?? "No description.").slice(0, 8_000)}\n\nCoverage limits:\n${warnings.join("\n")}\n\nShared outline:\n${evidenceChapters.map((chapter) => `${chapter.id}: ${chapter.title}`).join("\n")}`;
    const fullManifest = files.map((file) => `${file.path} (${file.status}; +${file.additions}/-${file.deletions}; ${file.coverage})`).join("\n");
    const manifest = fullManifest.length <= 80_000 ? fullManifest : fullManifest.slice(0, 80_000) + "\n[Overview inventory excerpt only; use the structured file manifest for the complete fetched list.]";
    const chapters = [{ id: "overview", title: "Overview and change inventory", files: [], sourceText: `${context}\n\nFile inventory (not implementation evidence):\n${manifest}\n\nNo patches are provided for this overview. Describe scope and the reading plan; do not infer implementation behavior from filenames.` }, ...evidenceChapters.map((chapter) => ({ ...chapter, sourceText: `${context}\n\nCurrent chapter: ${chapter.title}\nOnly the following patches are evidence for this chapter. Other chapters have not been supplied.\n${chapter.sourceText}` }))];
    return { title, url, headSha: pull.head.sha, totalFiles: Number.isSafeInteger(pull.changed_files) ? pull.changed_files! : null, filesComplete, files, chapters, warnings };
  } catch (error) {
    if (controller.signal.aborted) throw new Error("Loading the PR manifest timed out. Try again.");
    throw error;
  } finally { clearTimeout(timeout); }
}
