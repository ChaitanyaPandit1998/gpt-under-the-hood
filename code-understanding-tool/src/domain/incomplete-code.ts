import ts from "typescript";
import type { SourceRange } from "./source-ranges.ts";

export type IncompleteCodeItem = {
  kind: "unfinished_marker" | "empty_branch";
  title: string;
  explanation: string;
  source_ranges: SourceRange[];
};

type CommentMarker = { start: number; end: number; line: number };

const unfinishedMarker = /\b(?:TODO|FIXME|XXX|HACK)\b/i;
const maxFindings = 20;

/** Detect explicit unfinished-work comments and empty if/else blocks without guessing the intended behavior. */
export function findIncompleteCode(sourceText: string): IncompleteCodeItem[] {
  const sourceFile = ts.createSourceFile("submitted-source.ts", sourceText, ts.ScriptTarget.Latest, true);
  const comments = findUnfinishedComments(sourceText);
  const items: IncompleteCodeItem[] = [];
  const coveredComments = new Set<CommentMarker>();
  const lineAt = (position: number) => sourceFile.getLineAndCharacterOfPosition(position).line + 1;
  const addEmptyBranch = (branch: ts.Statement) => {
    if (!(ts.isBlock(branch) ? branch.statements.length === 0 : ts.isEmptyStatement(branch))) return;
    const blockStart = branch.getStart(sourceFile);
    const blockEnd = branch.getEnd();
    const marker = comments.find((comment) => comment.start >= blockStart && comment.start <= blockEnd);
    if (marker) coveredComments.add(marker);
    items.push({
      kind: "empty_branch",
      title: marker ? "Unfinished branch" : "Empty conditional branch",
      explanation: marker
        ? "This branch has no executable statements and contains an unfinished-work marker. Execution falls through to the next statement; the intended handling is not specified."
        : "This branch has no executable statements. Execution falls through to the next statement; the intended handling is not specified.",
      // Anchor the finding to the branch body. Using the enclosing `if` line
      // can also tag the implemented condition as incomplete when both share
      // a source range in the generated diagram.
      source_ranges: [{ start_line: marker?.line ?? lineAt(branch.getEnd()), end_line: lineAt(branch.getEnd()) }],
    });
  };

  const visit = (node: ts.Node) => {
    if (ts.isIfStatement(node)) {
      addEmptyBranch(node.thenStatement);
      if (node.elseStatement) addEmptyBranch(node.elseStatement);
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);

  for (const comment of comments) {
    if (coveredComments.has(comment)) continue;
    items.push({
      kind: "unfinished_marker",
      title: "Unfinished-work marker",
      explanation: "The source marks work as unfinished. It does not specify the intended behavior.",
      source_ranges: [{ start_line: comment.line, end_line: comment.line }],
    });
  }

  // Support common hash-comment languages as well as JS/TS comments from the lexer.
  const lines = sourceText.split(/\r\n|\r|\n/);
  lines.forEach((line, index) => {
    if (/^\s*#\s*/.test(line) && unfinishedMarker.test(line) && !items.some((item) => item.kind === "unfinished_marker" && item.source_ranges[0]?.start_line === index + 1)) {
      items.push({
        kind: "unfinished_marker",
        title: "Unfinished-work marker",
        explanation: "The source marks work as unfinished. It does not specify the intended behavior.",
        source_ranges: [{ start_line: index + 1, end_line: index + 1 }],
      });
    }
  });

  return items.slice(0, maxFindings);
}

export function annotateIncompleteCode(sourceText: string, explanation: unknown): unknown {
  const incompleteItems = findIncompleteCode(sourceText);
  if (incompleteItems.length === 0 || !isRecord(explanation)) return explanation;
  return { ...explanation, incomplete_items: incompleteItems };
}

function findUnfinishedComments(sourceText: string): CommentMarker[] {
  const scanner = ts.createScanner(ts.ScriptTarget.Latest, false, ts.LanguageVariant.Standard, sourceText);
  const markers: CommentMarker[] = [];
  const templateBraceDepths: number[] = [];
  let token = scanner.scan();
  while (token !== ts.SyntaxKind.EndOfFileToken) {
    if (token === ts.SyntaxKind.SingleLineCommentTrivia || token === ts.SyntaxKind.MultiLineCommentTrivia) {
      const text = scanner.getTokenText();
      if (unfinishedMarker.test(text)) {
        markers.push({
          start: scanner.getTokenPos(),
          end: scanner.getTextPos(),
          line: sourceText.slice(0, scanner.getTokenPos()).split(/\r\n|\r|\n/).length,
        });
      }
    }
    if (token === ts.SyntaxKind.TemplateHead) templateBraceDepths.push(0);
    else if (token === ts.SyntaxKind.TemplateMiddle && templateBraceDepths.length > 0) templateBraceDepths[templateBraceDepths.length - 1] = 0;
    else if (token === ts.SyntaxKind.TemplateTail && templateBraceDepths.length > 0) templateBraceDepths.pop();
    else if (templateBraceDepths.length > 0 && token === ts.SyntaxKind.OpenBraceToken) templateBraceDepths[templateBraceDepths.length - 1] += 1;
    else if (templateBraceDepths.length > 0 && token === ts.SyntaxKind.CloseBraceToken) {
      const currentDepth = templateBraceDepths.length - 1;
      if (templateBraceDepths[currentDepth] === 0) {
        token = scanner.reScanTemplateToken(false);
        if (token === ts.SyntaxKind.TemplateTail) templateBraceDepths.pop();
        continue;
      }
      templateBraceDepths[currentDepth] -= 1;
    }
    token = scanner.scan();
  }
  return markers;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
