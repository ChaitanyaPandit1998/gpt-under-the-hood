export type PlannedFile = {
  path: string;
  status: string;
  additions: number;
  deletions: number;
  coverage: "included" | "partial" | "unavailable" | "omitted_budget";
};

export type PullRequestChapter = { id: string; title: string; files: string[]; sourceText: string };

export type PullRequestPlan = {
  title: string;
  url: string;
  headSha: string;
  totalFiles: number | null;
  filesComplete: boolean;
  files: PlannedFile[];
  chapters: PullRequestChapter[];
  warnings: string[];
};

export function fileTheme(path: string): string {
  if (/^(\.github|\.gitlab|scripts)\/|(^|\/)(Dockerfile|.*\.config\.[^/]+)$/.test(path)) return "Build and automation";
  if (/(^|\/)(__tests__|tests?|specs?)(\/|\.)|\.(test|spec)\./i.test(path)) return "Tests and verification";
  if (/(^|\/)(docs?|examples?|docs_src)\/|\.(md|rst|mdx)$/i.test(path)) return "Documentation and examples";
  if (/^(package(-lock)?\.json|.*lock|requirements.*|pyproject\.toml|go\.(mod|sum))$/.test(path)) return "Dependencies and configuration";
  const parts = path.split("/");
  return parts.length > 1 ? `Implementation: ${parts.slice(0, parts[0] === "src" && parts.length > 2 ? 2 : 1).join("/")}` : "Core implementation";
}
