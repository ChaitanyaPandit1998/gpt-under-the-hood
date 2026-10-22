import { describe, expect, it } from "vitest";
import { isValidRange, sourceOffsetsForRange } from "./source-ranges";

describe("source ranges", () => {
  it("accepts valid one-based inclusive ranges", () => {
    expect(isValidRange({ start_line: 2, end_line: 3 }, 3)).toBe(true);
  });

  it("rejects reversed and out-of-bounds ranges", () => {
    expect(isValidRange({ start_line: 3, end_line: 2 }, 3)).toBe(false);
    expect(isValidRange({ start_line: 1, end_line: 4 }, 3)).toBe(false);
    expect(isValidRange({ start_line: 0, end_line: 1 }, 3)).toBe(false);
  });

  it("maps line ranges to textarea selection offsets", () => {
    expect(sourceOffsetsForRange("one\ntwo\nthree", { start_line: 2, end_line: 3 })).toEqual({ start: 4, end: 13 });
  });

  it("returns no offsets for an invalid range", () => {
    expect(sourceOffsetsForRange("one\ntwo", { start_line: 1, end_line: 3 })).toBeNull();
  });
});
