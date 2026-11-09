import { EventEmitter } from "node:events";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { Readable } from "node:stream";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createLocalExplanationApi } from "../src/server/local-api.ts";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const casesPath = resolve(root, "evals/cases.json");
const cliArgs = process.argv.slice(2);
const caseFilter = cliArgs.includes("--case") ? cliArgs[cliArgs.indexOf("--case") + 1] : undefined;
const outputArgument = cliArgs.includes("--output") ? cliArgs[cliArgs.indexOf("--output") + 1] : undefined;
const runStamp = new Date().toISOString().replace(/[:.]/g, "-");
const outputPath = outputArgument
  ? resolve(outputArgument)
  : resolve(root, `evals/results/baseline-${runStamp}.json`);
const allCases = JSON.parse(await readFile(casesPath, "utf8")).cases;
const cases = caseFilter ? allCases.filter((testCase) => testCase.id === caseFilter) : allCases;
if (cases.length === 0) throw new Error(`No evaluation case found for ${caseFilter ?? "the selected filters"}.`);
const stack = [];
const plugin = createLocalExplanationApi();
plugin.configureServer({
  middlewares: {
    use(pathOrHandler, handler) {
      stack.push(typeof pathOrHandler === "string"
        ? { path: pathOrHandler, handler }
        : { path: "/", handler: pathOrHandler });
    },
  },
});

async function requestExplain(testCase) {
  const path = "/api/explain";
  const body = {
    source_text: testCase.source_text,
    material_kind: testCase.material_kind,
    teaching_mode: testCase.teaching_mode,
    explanation_depth: "balanced",
    learner_level: "beginner",
    visual_preference: testCase.visual_expected ? "control_flow" : "auto",
  };
  const incoming = Readable.from([Buffer.from(JSON.stringify(body))]);
  incoming.method = "POST";
  incoming.url = path;
  incoming.headers = { host: "127.0.0.1:5173", origin: "http://127.0.0.1:5173", "content-type": "application/json" };
  let responseBody = "";
  const events = new EventEmitter();
  const response = {
    statusCode: 200,
    headersSent: false,
    writableEnded: false,
    destroyed: false,
    setHeader() {},
    end(value = "") {
      responseBody = value;
      response.headersSent = true;
      response.writableEnded = true;
    },
    once: events.once.bind(events),
    removeListener: events.removeListener.bind(events),
  };

  let cursor = 0;
  while (cursor < stack.length && !response.headersSent) {
    const entry = stack[cursor++];
    if (!path.startsWith(entry.path)) continue;
    const originalUrl = incoming.url;
    if (entry.path !== "/") incoming.url = path.slice(entry.path.length) || "/";
    let shouldContinue = false;
    await entry.handler(incoming, response, () => { shouldContinue = true; });
    incoming.url = originalUrl;
    if (!shouldContinue) break;
  }
  return { status: response.statusCode, body: JSON.parse(responseBody) };
}

function collectRanges(explanation, sourceText) {
  const ranges = [];
  for (const concept of explanation.concepts ?? []) ranges.push(...(concept.source_ranges ?? []));
  for (const step of explanation.walkthrough ?? []) ranges.push(...(step.source_ranges ?? []));
  for (const visual of explanation.visuals ?? []) {
    for (const node of visual.nodes ?? []) ranges.push(...(node.source_ranges ?? []));
  }
  for (const range of ranges) {
    if (!Number.isInteger(range.start_line) || !Number.isInteger(range.end_line) || range.start_line < 1 || range.end_line < range.start_line || range.end_line > sourceText.split(/\r\n|\r|\n/).length) return false;
  }
  return true;
}

const results = [];
for (const testCase of cases) {
  const startedAt = Date.now();
  try {
    const response = await requestExplain(testCase);
    const explanation = response.body;
    results.push({
      id: testCase.id,
      status: response.status,
      api_error: response.status === 200 ? undefined : explanation.error ?? "Request failed without an error message.",
      duration_ms: Date.now() - startedAt,
      checks: {
        api_success: response.status === 200,
        visual_present: (explanation.visuals?.length ?? 0) > 0,
        visual_requirement_met: !testCase.visual_expected || (explanation.visuals?.length ?? 0) > 0,
        incomplete_detection_matches: response.status !== 200 ? null : testCase.incomplete_expected
          ? (explanation.incomplete_items?.length ?? 0) > 0 && testCase.incomplete_kinds.every((kind) => explanation.incomplete_items.some((item) => item.kind === kind))
          : (explanation.incomplete_items?.length ?? 0) === 0,
        source_ranges_well_formed: response.status === 200 ? collectRanges(explanation, testCase.source_text) : null,
        review_status: explanation.quality_review_status ?? "missing",
      },
      expectations: testCase.expectations,
      manual_scores: null,
      explanation,
    });
  } catch (error) {
    results.push({
      id: testCase.id,
      status: 500,
      duration_ms: Date.now() - startedAt,
      checks: { api_success: false },
      expectations: testCase.expectations,
      manual_scores: null,
      error: error instanceof Error ? error.message : "Unknown evaluation failure",
    });
    console.error(`${testCase.id}: request failed; continuing with the remaining cases.`);
  }
}

// `visual_present` is descriptive. `visual_requirement_met` is the assertion:
// a missing diagram is fine when a case does not need one.
const autoChecks = results.flatMap((result) => Object.entries(result.checks)
  .filter(([key, value]) => key !== "review_status" && key !== "visual_present" && typeof value === "boolean")
  .map(([, value]) => value));
const output = {
  evaluation_version: 1,
  generated_at: new Date().toISOString(),
  git_revision: execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root, encoding: "utf8" }).trim(),
  working_tree_dirty: execFileSync("git", ["status", "--porcelain"], { cwd: root, encoding: "utf8" }).trim().length > 0,
  source_corpus: "evals/cases.json",
  note: "Automated checks are recorded per case. Human rubric scores remain null until reviewed against the written expectations.",
  results,
  summary: {
    api_successes: results.filter((result) => result.status === 200).length,
    total_cases: results.length,
    api_success_rate_percent: Math.round(results.filter((result) => result.status === 200).length / results.length * 100),
    completed_reviews: results.filter((result) => ["passed", "revised"].includes(result.checks.review_status)).length,
    boolean_automated_checks_passed: autoChecks.filter(Boolean).length,
    boolean_automated_checks_total: autoChecks.length,
    manual_factual_score: null,
  },
};
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");

const passed = results.filter((result) => result.status === 200).length;
console.log(`Saved ${results.length} case results to ${outputPath}`);
console.log(`API success: ${passed}/${results.length}; automated checks: ${autoChecks.filter(Boolean).length}/${autoChecks.length}`);
console.log("Human factual/grounding/diagram scores are intentionally not guessed; see evals/README.md.");
