import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { generateBenchmarkImage, imageTrials, IMAGE_MODEL } from "../src/server/cloudflare-image-benchmark.ts";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
for (const filename of [".env", ".env.local"]) {
  try { process.loadEnvFile(resolve(root, filename)); }
  catch (error) { if (error.code !== "ENOENT") throw new Error(`Could not load ${filename}; contents were not logged.`); }
}
const args = process.argv.slice(2);
const live = args.includes("--live");
let selectedCase;
let suite = "technical";
for (let i = 0; i < args.length; i += 1) {
  if (args[i] === "--live") continue;
  if (args[i] === "--case" && args[i + 1] && !selectedCase) { selectedCase = args[++i]; continue; }
  if (args[i] === "--suite" && ["technical", "decorative"].includes(args[i + 1])) { suite = args[++i]; continue; }
  throw new Error("Usage: npm run eval:images -- [--live] [--case CASE_ID] [--suite technical|decorative]. Default: dry run, no API calls.");
}
const dataset = JSON.parse(await readFile(resolve(root, suite === "decorative" ? "evals/image-illustration-cases.json" : "evals/image-cases.json"), "utf8"));
const cases = selectedCase ? dataset.cases.filter((item) => item.id === selectedCase) : dataset.cases;
const trials = imageTrials(cases, suite);
const credentials = {
  accountId: process.env.CLOUDFLARE_ACCOUNT_ID || process.env.CF_ACCOUNT_ID || "",
  apiToken: process.env.CLOUDFLARE_API_TOKEN || process.env.CF_API_TOKEN || "",
};
const configured = Boolean(credentials.accountId && credentials.apiToken);
const runId = new Date().toISOString().replace(/[:.]/g, "-");
const output = resolve(root, "evals/image-results", runId);
await mkdir(output, { recursive: true });
const report = {
  runId, model: IMAGE_MODEL, suite, mode: live ? "live" : "dry-run",
  status: live ? configured ? "running" : "blocked_missing_credentials" : "prepared",
  sanitization: dataset.sanitization,
  limitations: ["Six-image maximum pilot; two styles, unseeded generation, no statistical quality claim. The live REST input schema does not support seed.", "Images are conceptual illustrations, not verified technical diagrams.", "Actual charges and Neuron usage must be checked in the Cloudflare dashboard; no free-quota availability is assumed."],
  references: { model: "https://developers.cloudflare.com/workers-ai/models/flux-1-schnell/", pricing: "https://developers.cloudflare.com/workers-ai/platform/pricing/" },
  cases, trials: trials.map((trial) => ({ ...trial, status: "not_run", imageFile: null, latencyMs: null, byteLength: null, requestId: null, billedUsd: null, billedNeurons: null, review: { relevance: null, semanticAccuracy: null, clarity: null, teachingUsefulness: null, misleadingElements: [], notes: "", decision: "not_reviewed" } })),
};
const escape = (value) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
async function saveReport() {
  await writeFile(resolve(output, "report.json"), JSON.stringify(report, null, 2));
  await writeFile(resolve(output, "gallery.html"), `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cloudflare image pilot</title><style>body{font:16px/1.6 Georgia,serif;background:#f6f2e9;color:#213b3b;max-width:1200px;margin:auto;padding:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}article{background:white;padding:20px;border:1px solid #d6ddd6;border-radius:12px}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}</style><h1>Cloudflare image pilot</h1><p>${escape(report.status)} · ${escape(report.model)}</p><p>Only curated synthetic summaries are used. No image quality scores have been assigned automatically.</p><div class="grid">${report.trials.map((trial) => `<article><h2>${escape(trial.id)}</h2><p>${escape(trial.status)}${trial.latencyMs !== null ? ` · ${trial.latencyMs} ms` : ""}</p>${trial.imageFile ? `<img src="${escape(trial.imageFile)}" alt="Unreviewed generated illustration for ${escape(trial.caseId)}">` : "<p>No image generated.</p>"}<details><summary>Exact prompt</summary><pre>${escape(trial.prompt)}</pre></details><p>Review: ${escape(trial.review.decision)}</p></article>`).join("")}</div></html>`);
}
await saveReport();
if (live && !configured) {
  console.error("Live benchmark blocked: configure CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN locally. Do not paste credentials into chat.");
  process.exitCode = 2;
} else if (live) {
  for (const trial of report.trials) {
    try {
      const image = await generateBenchmarkImage(trial, credentials);
      trial.imageFile = `${trial.id}.${image.extension}`;
      await writeFile(resolve(output, trial.imageFile), image.bytes);
      Object.assign(trial, { status: "generated_unreviewed", latencyMs: image.latencyMs, byteLength: image.byteLength, requestId: image.requestId });
      await saveReport();
    } catch (error) {
      trial.status = "failed";
      trial.error = error.message;
      report.status = "stopped_on_error";
      process.exitCode = 1;
      break;
    }
  }
  if (report.status === "running") report.status = "generated_pending_review";
  await saveReport();
}
console.log(JSON.stringify({ status: report.status, requestsPlanned: report.trials.length, imagesGenerated: report.trials.filter((trial) => trial.imageFile).length, report: resolve(output, "report.json"), gallery: resolve(output, "gallery.html") }, null, 2));
