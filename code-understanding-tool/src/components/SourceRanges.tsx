import { type SourceRange } from "../domain/source-ranges";

export function SourceRanges({ ranges, lineCount, onSelectRange }: {
  ranges: SourceRange[];
  lineCount: number;
  onSelectRange: (range: SourceRange) => void;
}) {
  const validRanges = ranges.filter((range) =>
    Number.isInteger(range.start_line) &&
    Number.isInteger(range.end_line) &&
    range.start_line >= 1 &&
    range.start_line <= range.end_line &&
    range.end_line <= lineCount,
  );
  if (validRanges.length === 0) return null;
  return <div className="source-range-list">{validRanges.map((range, index) => <button className="source-range" type="button" key={`${range.start_line}-${range.end_line}-${index}`} onClick={() => onSelectRange(range)} aria-label={`Select source lines ${range.start_line} through ${range.end_line}`}>Lines {range.start_line}{range.end_line !== range.start_line ? `–${range.end_line}` : ""}</button>)}</div>;
}
