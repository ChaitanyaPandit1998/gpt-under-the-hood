import ts from "typescript";
import type { SourceRange } from "./source-ranges";

type CoverageNode = { id: string; label: string; source_ranges?: SourceRange[] };
type CoverageEdge = { from: string; to: string; label: string };
type CoverageVisual = { kind: string; nodes: CoverageNode[]; edges: CoverageEdge[] };
type SourceDecision = { startLine: number; endLine: number; terms: string[]; outcomes: string[] };
export type ControlFlowCoverageStatus = "matched" | "incomplete" | "unverifiable";
export type ControlFlowCoverage = { status: ControlFlowCoverageStatus; reason?: string };

const genericBranchLabels = new Set([
  "continue",
  "continues",
  "continue calculation",
  "either outcome",
  "next",
  "proceed",
  "then continue",
]);

/** Check supported JavaScript/TypeScript decisions; never treat unsupported syntax as verified. */
export function analyzeControlFlowCoverage(sourceText: string, visual: CoverageVisual): ControlFlowCoverage {
  if (visual.kind !== "control_flow") return { status: "matched" };

  const parsed = collectSourceDecisions(sourceText);
  if (!parsed.supported) {
    return { status: "unverifiable", reason: "The source uses syntax outside the coverage checker's supported subset." };
  }
  if (parsed.decisions.length === 0) return { status: "matched" };
  if (parsed.decisions.some((decision) => !visual.nodes.some((node) => rangesOverlapDecision(node.source_ranges, decision)))) {
    return { status: "unverifiable", reason: "Diagram nodes do not include source ranges that identify every decision." };
  }

  const nodesById = new Map(visual.nodes.map((node) => [node.id, node]));
  const outgoing = new Map<string, CoverageEdge[]>();
  for (const edge of visual.edges) {
    if (!nodesById.has(edge.from) || !nodesById.has(edge.to)) continue;
    outgoing.set(edge.from, [...(outgoing.get(edge.from) ?? []), edge]);
  }

  const decisionNodes = [...outgoing.entries()].filter(([, edges]) => edges.length > 1);
  if (!decisionNodes.every(([, edges]) => hasDistinctOutcomes(edges, nodesById))) {
    return { status: "incomplete", reason: "At least one diagram branch has duplicate or unclear outcomes." };
  }
  if (decisionNodes.length < parsed.decisions.length) {
    return { status: "incomplete", reason: "The diagram has fewer explicit branch points than the supported source decisions." };
  }

  const candidates = parsed.decisions.map((decision) => decisionNodes.flatMap(([nodeId, edges], index) => {
    if (edges.length < decision.outcomes.length) return [];
    const node = nodesById.get(nodeId)!;
    if (!rangesOverlapDecision(node.source_ranges, decision)) return [];
    const text = [node.label, ...edges.map((edge) => edge.label), ...edges.map((edge) => nodesById.get(edge.to)?.label)]
      .filter(Boolean).join(" ").toLocaleLowerCase();
    if (decision.terms.length > 0 && !decision.terms.every((term) => containsConditionTerm(text, term))) {
      return [];
    }
    if (decision.outcomes.length > 2 && !decision.outcomes.every((outcome) => outcomeIsRepresented(outcome, edges, nodesById))) {
      return [];
    }
    return [index];
  }));
  if (candidates.some((matches) => matches.length === 0)) {
    const hasMissingRanges = parsed.decisions.some((decision) => !visual.nodes.some((node) => rangesOverlapDecision(node.source_ranges, decision)));
    return hasMissingRanges
      ? { status: "unverifiable", reason: "Diagram nodes do not include source ranges that identify every decision." }
      : { status: "incomplete", reason: "A source decision is not linked to a diagram branch with the expected outcomes." };
  }

  return hasDistinctDecisionMatches(candidates)
    ? { status: "matched" }
    : { status: "unverifiable", reason: "The diagram's source links do not distinguish each decision unambiguously." };
}

/** Backwards-compatible boolean helper for callers that only need a match result. */
export function hasCompleteControlFlowCoverage(sourceText: string, visual: CoverageVisual): boolean {
  return analyzeControlFlowCoverage(sourceText, visual).status === "matched";
}

/** Preserve generated diagrams; flag uncertain coverage instead of discarding useful visuals. */
export function annotateControlFlowCoverage(sourceText: string, explanation: unknown): unknown {
  if (!isRecord(explanation) || !Array.isArray(explanation.visuals)) return explanation;
  const visuals = explanation.visuals.filter(isCoverageVisual);
  const controlFlow = visuals.filter((visual) => visual.kind === "control_flow");
  if (controlFlow.length === 0) return explanation;

  // Prefix IDs so independently authored panels can safely be checked as one graph.
  const combined: CoverageVisual = {
    kind: "control_flow",
    nodes: controlFlow.flatMap((visual, index) => visual.nodes.map((node) => ({ ...node, id: `${index}:${node.id}` }))),
    edges: controlFlow.flatMap((visual, index) => visual.edges.map((edge) => ({ ...edge, from: `${index}:${edge.from}`, to: `${index}:${edge.to}` }))),
  };
  const coverage = analyzeControlFlowCoverage(sourceText, combined);
  if (coverage.status === "matched") return explanation;
  const note = coverage.status === "unverifiable"
    ? "Automated checks could not verify every source decision. The diagram is shown; compare it with the walkthrough and linked source lines."
    : "Automated checks found source decisions without distinct diagram paths. The diagram is shown; compare it with the walkthrough and linked source lines.";
  return {
    ...explanation,
    visual_note: [typeof explanation.visual_note === "string" ? explanation.visual_note.trim() : "", note].filter(Boolean).join(" "),
  };
}

function isCoverageVisual(value: unknown): value is CoverageVisual {
  return isRecord(value) && typeof value.kind === "string" && Array.isArray(value.nodes) && Array.isArray(value.edges);
}

/** Count control-flow constructs and ternaries while ignoring optional TS markers. */
export function countSourceDecisions(sourceText: string): number {
  return Math.max(collectSourceDecisions(sourceText).decisions.length, countFallbackDecisions(sourceText));
}

function collectSourceDecisions(sourceText: string): { decisions: SourceDecision[]; supported: boolean } {
  const sourceFile = ts.createSourceFile("submitted-source.ts", sourceText, ts.ScriptTarget.Latest, true);
  const decisions: SourceDecision[] = [];
  let supported = ((sourceFile as ts.SourceFile & { parseDiagnostics?: readonly ts.Diagnostic[] }).parseDiagnostics?.length ?? 0) === 0;
  const visit = (node: ts.Node) => {
    let condition: ts.Expression | undefined;
    let outcomes: string[] = [];
    if (ts.isIfStatement(node)) {
      condition = node.expression;
      outcomes = ["true", "false"];
    } else if (ts.isConditionalExpression(node)) {
      condition = node.condition;
      outcomes = ["true", "false"];
    } else if (ts.isSwitchStatement(node)) {
      condition = node.expression;
      outcomes = node.caseBlock.clauses.map((clause) => ts.isDefaultClause(clause)
        ? "default"
        : clause.expression.getText(sourceFile).replace(/^['"`]|['"`]$/g, ""));
      if (!node.caseBlock.clauses.some(ts.isDefaultClause)) outcomes.push("no matching case");
      if (hasSwitchFallthrough(node)) supported = false;
    } else if (ts.isForStatement(node) || ts.isForInStatement(node) || ts.isForOfStatement(node) || ts.isWhileStatement(node) || ts.isDoStatement(node)) {
      // Loop repetition and exit behavior needs a dedicated model; avoid claiming coverage here.
      supported = false;
    }
    if (condition) {
      const start = sourceFile.getLineAndCharacterOfPosition(condition.getStart(sourceFile)).line + 1;
      const end = sourceFile.getLineAndCharacterOfPosition(condition.getEnd()).line + 1;
      decisions.push({ startLine: start, endLine: end, terms: extractConditionTerms(condition), outcomes });
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);
  if (countFallbackDecisions(sourceText) > decisions.length) supported = false;
  return { decisions, supported };
}

function hasDistinctOutcomes(edges: CoverageEdge[], nodesById: Map<string, CoverageNode>): boolean {
  const labels = edges.map((edge) => edge.label.trim().toLocaleLowerCase());
  if (labels.some((label) => genericBranchLabels.has(label))) return false;

  if (labels.every((label) => !label)) {
    const targets = edges.map((edge) => nodesById.get(edge.to)?.label.trim().toLocaleLowerCase() ?? "");
    return targets.every(Boolean) && new Set(targets).size === targets.length;
  }

  return labels.every(Boolean) && new Set(labels).size === labels.length;
}

function hasDistinctDecisionMatches(candidates: number[][]): boolean {
  const match = (conditionIndex: number, used: Set<number>): boolean => {
    if (conditionIndex === candidates.length) return true;
    return candidates[conditionIndex].some((candidate) => {
      if (used.has(candidate)) return false;
      used.add(candidate);
      if (match(conditionIndex + 1, used)) return true;
      used.delete(candidate);
      return false;
    });
  };
  return match(0, new Set());
}

function containsTerm(text: string, term: string): boolean {
  return new RegExp(`(^|[^\\p{L}\\p{N}_])${escapeRegExp(term)}([^\\p{L}\\p{N}_]|$)`, "u").test(text);
}

function containsConditionTerm(text: string, term: string): boolean {
  const numericAliases: Record<string, string[]> = {
    "0": ["zero", "no", "none", "empty"],
    "1": ["one"],
    "2": ["two"],
    "3": ["three"],
  };
  return containsTerm(text, term) || (numericAliases[term] ?? []).some((alias) => containsTerm(text, alias));
}

function extractConditionTerms(condition: ts.Expression): string[] {
  const terms = new Set<string>();
  const visit = (node: ts.Node) => {
    if (ts.isIdentifier(node)) {
      const parent = node.parent;
      if (ts.isPropertyAccessExpression(parent) && parent.expression === node) return;
      const accessors = new Set(["length", "size", "keys", "values", "entries"]);
      if (accessors.has(node.text.toLocaleLowerCase())) return;
      normalizeConditionIdentifier(node.text).forEach((term) => terms.add(term));
    } else if (ts.isNumericLiteral(node)) {
      terms.add(node.text.toLocaleLowerCase());
    }
    ts.forEachChild(node, visit);
  };
  visit(condition);
  return [...terms];
}

function normalizeConditionIdentifier(identifier: string): string[] {
  const words = identifier
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .toLocaleLowerCase()
    .split(/[^\p{L}\p{N}_]+/u)
    .filter(Boolean);
  const booleanPrefixes = new Set(["is", "has", "can", "should", "does", "did", "will", "was", "were"]);
  if (words.length > 1 && booleanPrefixes.has(words[0])) words.shift();
  return words;
}

function rangesOverlapDecision(ranges: SourceRange[] | undefined, decision: SourceDecision): boolean {
  return ranges?.some((range) => range.start_line <= decision.endLine && range.end_line >= decision.startLine) ?? false;
}

function outcomeIsRepresented(outcome: string, edges: CoverageEdge[], nodesById: Map<string, CoverageNode>): boolean {
  if (outcome === "default" || outcome === "no matching case") {
    return edges.some((edge) => /default|otherwise|no match|no other case|remaining case/i.test(`${edge.label} ${nodesById.get(edge.to)?.label ?? ""}`));
  }
  const normalized = outcome.toLocaleLowerCase().replace(/^['"`]|['"`]$/g, "");
  return edges.some((edge) => containsTerm(`${edge.label} ${nodesById.get(edge.to)?.label ?? ""}`.toLocaleLowerCase(), normalized));
}

function hasSwitchFallthrough(node: ts.SwitchStatement): boolean {
  return node.caseBlock.clauses.some((clause, index) => {
    if (ts.isDefaultClause(clause) && index === node.caseBlock.clauses.length - 1) return false;
    if (clause.statements.length === 0) return false;
    const last = clause.statements.at(-1);
    return !(last && (ts.isBreakStatement(last) || ts.isReturnStatement(last) || ts.isThrowStatement(last) || ts.isContinueStatement(last)));
  });
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function countFallbackDecisions(sourceText: string): number {
  const code = sourceText
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/\/\/[^\n]*/g, " ")
    .replace(/#[^\n]*/g, " ")
    .replace(/("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)/g, " ")
    .replace(/([A-Za-z_$][\w$]*)\?(?=\s*[:;,])/g, "$1 ");
  const cStyleIfs = code.match(/\bif\s*\(/g)?.length ?? 0;
  const pythonIfs = code.match(/^\s*(?:if|elif)\s+.+:/gm)?.length ?? 0;
  const pythonTernaries = code.match(/^\s*.+\s+if\s+.+\s+else\s+.+$/gm)?.length ?? 0;
  const ternaries = code.match(/\?(?![.?])/g)?.length ?? 0;
  const loops = code.match(/\b(?:for|while)\s*\(/g)?.length ?? 0;
  const pythonLoops = code.match(/^\s*(?:for|while)\s+.+:/gm)?.length ?? 0;
  return cStyleIfs + pythonIfs + pythonTernaries + ternaries + loops + pythonLoops;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
