export const visualCheckStages = ["brief", "plan", "raster", "alignment", "audience"] as const;
export type VisualCheckStage = typeof visualCheckStages[number];
export type VisualVerdict = "pass" | "revise" | "fail" | "uncertain";

export type VisualContract = {
  version: 1;
  id: string;
  explanationSha256: string;
  purpose: "story-illustration" | "factual-diagram";
  audience: "beginner" | "intermediate" | "advanced";
  teachingGoal: string;
  audienceObjectives: string[];
  requirements: {
    id: string;
    kind: "claim" | "element" | "qualifier" | "prohibition";
    statement: string;
    evidence: "source" | "background" | "illustrative";
    evidenceReference: string;
  }[];
};

export type VisualEvaluationReport = {
  version: 1;
  contractId: string;
  explanationSha256: string;
  artifacts: { briefSha256: string; planSha256: string; pngSha256: string; captionSha256?: string };
  repairAttempt: 0 | 1;
  checks: {
    stage: VisualCheckStage;
    status: VisualVerdict | "not-run";
    reviewer: { kind: "deterministic" | "model" | "human"; name: string; version: string };
    durationMs: number;
    summary: string;
    requirementIds: string[];
    findings: {
      severity: "critical" | "major" | "minor";
      message: string;
      evidence: string;
      requirementIds: string[];
    }[];
  }[];
};

const invalid = () => { throw new Error("Invalid visual evaluation data."); };
function object(value: unknown, keys: string[]): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return invalid();
  const result = value as Record<string, unknown>;
  if (Object.keys(result).length !== keys.length || keys.some((key) => !Object.hasOwn(result, key))) return invalid();
  return result;
}
function string(value: unknown, max = 2_000): string {
  if (typeof value !== "string" || !value.trim() || value.length > max) return invalid();
  return value;
}
function member(value: unknown, choices: readonly string[]) {
  if (typeof value !== "string" || !choices.includes(value)) invalid();
}
function list(value: unknown, max: number): unknown[] {
  if (!Array.isArray(value) || value.length > max) return invalid();
  return value;
}
function hash(value: unknown) {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) invalid();
}
function identifiers(value: unknown, allowed: Set<string>): string[] {
  const ids = list(value, 40).map((id) => string(id, 100));
  if (new Set(ids).size !== ids.length || ids.some((id) => !allowed.has(id))) invalid();
  return ids;
}

export function validateVisualContract(value: unknown): VisualContract {
  const contract = object(value, ["version", "id", "explanationSha256", "purpose", "audience", "teachingGoal", "audienceObjectives", "requirements"]);
  if (contract.version !== 1) invalid();
  string(contract.id, 100);
  hash(contract.explanationSha256);
  member(contract.purpose, ["story-illustration", "factual-diagram"]);
  member(contract.audience, ["beginner", "intermediate", "advanced"]);
  string(contract.teachingGoal);
  const objectives = list(contract.audienceObjectives, 10);
  if (!objectives.length) invalid();
  objectives.forEach((item) => string(item));
  const requirements = list(contract.requirements, 40);
  if (!requirements.length) invalid();
  const ids = new Set<string>();
  for (const value of requirements) {
    const requirement = object(value, ["id", "kind", "statement", "evidence", "evidenceReference"]);
    const id = string(requirement.id, 100);
    if (ids.has(id)) invalid();
    ids.add(id);
    member(requirement.kind, ["claim", "element", "qualifier", "prohibition"]);
    member(requirement.evidence, ["source", "background", "illustrative"]);
    string(requirement.statement);
    string(requirement.evidenceReference);
  }
  return contract as VisualContract;
}

export function validateVisualEvaluationReport(value: unknown, input: VisualContract): VisualEvaluationReport {
  const contract = validateVisualContract(input);
  const report = object(value, ["version", "contractId", "explanationSha256", "artifacts", "repairAttempt", "checks"]);
  if (report.version !== 1 || report.contractId !== contract.id || report.explanationSha256 !== contract.explanationSha256 ||
      (report.repairAttempt !== 0 && report.repairAttempt !== 1)) invalid();
  const hasCaption = report.artifacts !== null && typeof report.artifacts === "object" && Object.hasOwn(report.artifacts, "captionSha256");
  const artifacts = object(report.artifacts, ["briefSha256", "planSha256", "pngSha256", ...(hasCaption ? ["captionSha256"] : [])]);
  Object.values(artifacts).forEach(hash);
  const allowed = new Set(contract.requirements.map(({ id }) => id));
  const stages = new Set<string>();
  for (const value of list(report.checks, visualCheckStages.length)) {
    const check = object(value, ["stage", "status", "reviewer", "durationMs", "summary", "requirementIds", "findings"]);
    member(check.stage, visualCheckStages);
    if (stages.has(check.stage as string)) invalid();
    stages.add(check.stage as string);
    member(check.status, ["pass", "revise", "fail", "uncertain", "not-run"]);
    const reviewer = object(check.reviewer, ["kind", "name", "version"]);
    member(reviewer.kind, ["deterministic", "model", "human"]);
    string(reviewer.name, 200);
    string(reviewer.version, 200);
    if (typeof check.durationMs !== "number" || !Number.isFinite(check.durationMs) || check.durationMs < 0) invalid();
    string(check.summary);
    identifiers(check.requirementIds, allowed);
    for (const value of list(check.findings, 40)) {
      const finding = object(value, ["severity", "message", "evidence", "requirementIds"]);
      member(finding.severity, ["critical", "major", "minor"]);
      string(finding.message);
      string(finding.evidence);
      identifiers(finding.requirementIds, allowed);
    }
  }
  return report as VisualEvaluationReport;
}

// A successful render or aesthetic score cannot override missing review or factual failures.
export function visualEvaluationVerdict(input: unknown, contract: VisualContract): VisualVerdict {
  const report = validateVisualEvaluationReport(input, contract);
  if (report.checks.some((check) => check.status === "fail" || check.findings.some((finding) => finding.severity === "critical"))) return "fail";
  const covered = new Set(report.checks.filter((check) => check.status === "pass").flatMap((check) => check.requirementIds));
  const uncoveredPassingReport = report.checks.every((check) => check.status === "pass") &&
    contract.requirements.some(({ id }) => !covered.has(id));
  if (report.checks.length !== visualCheckStages.length || report.checks.some((check) => check.status === "not-run" || check.status === "uncertain") ||
      uncoveredPassingReport) return "uncertain";
  if (report.checks.some((check) => check.status === "revise" || check.findings.some((finding) => finding.severity === "major"))) return "revise";
  return "pass";
}
