export type SourceRange = { start_line: number; end_line: number };

export function isValidRange(range: SourceRange, lineCount: number): boolean {
  return Number.isInteger(range.start_line) &&
    Number.isInteger(range.end_line) &&
    range.start_line >= 1 &&
    range.start_line <= range.end_line &&
    range.end_line <= lineCount;
}

export function sourceOffsetsForRange(source: string, range: SourceRange): { start: number; end: number } | null {
  const lines = source.split("\n");
  if (!isValidRange(range, lines.length)) return null;
  const lineStart = (line: number) => lines.slice(0, line - 1).reduce((offset, text) => offset + text.length + 1, 0);
  return {
    start: lineStart(range.start_line),
    end: lineStart(range.end_line) + lines[range.end_line - 1].length,
  };
}
