import type { PullRequestEvidence } from "../domain/pull-request-plan.ts";
import { parsePublicPullRequestUrl, readBoundedJson, type PullRequestFetch } from "./github-pull-request.ts";

const SECTION_CHARACTERS = 24_000;
type Pull = { title?: string; head?: { sha?: string }; base?: { sha?: string }; changed_files?: number };
type RemoteFile = { filename?: string; patch?: string };

export async function loadPullRequestEvidence(input: unknown, fetcher: PullRequestFetch = fetch): Promise<PullRequestEvidence> {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("Supply a PR file evidence request.");
  const request = input as Record<string, unknown>;
  const { owner, repo, number } = parsePublicPullRequestUrl(request.url);
  const { filename, headSha, baseSha, page, offset = 0 } = request;
  if (typeof filename !== "string" || !filename || filename.length > 1_000) throw new Error("Supply a file from the PR manifest.");
  if (typeof headSha !== "string" || !/^[a-f0-9]{40,64}$/i.test(headSha)) throw new Error("Supply a valid PR head revision.");
  if (baseSha !== undefined && (typeof baseSha !== "string" || !/^[a-f0-9]{40,64}$/i.test(baseSha))) throw new Error("Supply a valid PR base revision.");
  if (typeof page !== "number" || !Number.isSafeInteger(page) || page < 1 || page > 5) throw new Error("Supply a manifest page between 1 and 5.");
  if (typeof offset !== "number" || !Number.isSafeInteger(offset) || offset < 0 || offset >= 1_000_000) throw new Error("Supply a valid patch section offset.");

  const base = `https://api.github.com/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/pulls/${number}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);
  const headers = { Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "code-understanding-tool" };
  async function get(path: string) {
    const response = await fetcher(path, { headers, signal: controller.signal });
    if (response.status === 403 || response.status === 429) throw new Error("GitHub is rate-limiting requests. Wait a little and try again.");
    if (!response.ok) throw new Error(`Cannot load public GitHub PR evidence (HTTP ${response.status}).`);
    return readBoundedJson(response);
  }
  try {
    const before = await get(base) as Pull;
    if (before?.head?.sha !== headSha) throw new Error("The PR revision changed. Reload the study plan before loading more evidence.");
    if (baseSha !== undefined && before.base?.sha !== baseSha) throw new Error("The PR base revision changed. Reload the study plan before loading more evidence.");
    const files = await get(`${base}/files?per_page=100&page=${page}`);
    if (!Array.isArray(files) || files.length > 100) throw new Error("GitHub returned an invalid file manifest.");
    const matches = (files as RemoteFile[]).filter((file) => file?.filename === filename);
    if (matches.length !== 1) throw new Error("The file is not uniquely present on this manifest page. Reload the study plan.");
    const patch = matches[0].patch;
    if (typeof patch !== "string" || !patch.trim()) throw new Error("GitHub does not provide a patch for this file. It may be binary or too large; no implementation evidence was loaded.");
    if (offset >= patch.length) throw new Error("This patch offset is out of range. Reload the study plan.");
    const after = await get(base) as Pull;
    if (after?.head?.sha !== headSha || before.base?.sha !== after.base?.sha || before.changed_files !== after.changed_files) throw new Error("The PR changed while loading evidence. Reload the study plan.");

    let end = Math.min(offset + SECTION_CHARACTERS, patch.length);
    if (end < patch.length) {
      const newline = patch.lastIndexOf("\n", end - 1);
      if (newline >= offset) end = newline + 1;
      // Preserve UTF-16 pairs when a single unusually long line must be split.
      else if (patch.charCodeAt(end - 1) >= 0xd800 && patch.charCodeAt(end - 1) <= 0xdbff) end -= 1;
    }
    const nextOffset = end < patch.length ? end : null;
    const title = `${filename} - patch characters ${offset + 1}-${end}`;
    const sourceText = `PR: ${String(before.title ?? "Pull request").slice(0, 500)}\nLink: https://github.com/${owner}/${repo}/pull/${number}\nHead revision: ${headSha}\nFile: ${filename}\nPatch characters: ${offset + 1}-${end} of ${patch.length}\n\nCoverage: This chapter contains only this patch section. Other sections, other chapters, and full repository files are not supplied. A section may begin inside a diff hunk or a long line. Do not infer omitted behavior. Citations must refer to lines in this chapter source, not original file line numbers. ${nextOffset === null ? "This is the final section of the GitHub-returned patch; that does not imply this chapter contains the whole patch." : "More patch sections remain available on demand."}\n\n--- Patch section (untrusted source material) ---\n${patch.slice(offset, end)}`;
    return { filename, offset, endOffset: end, nextOffset, totalCharacters: patch.length, chapter: { id: `evidence:${filename}:${offset}`, title, files: [filename], sourceText } };
  } catch (error) {
    if (controller.signal.aborted) throw new Error("Loading the patch section timed out. Try again.");
    throw error;
  } finally { clearTimeout(timeout); }
}
