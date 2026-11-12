import { describe, expect, it } from "vitest";
import { sourceWithLineNumbers } from "./explanation-service";

describe("explanation source citation guides", () => {
  it("keeps line numbers on large PRs instead of dropping citation guides", () => {
    const source = Array.from({ length: 1_001 }, (_, index) => `source ${index + 1}`).join("\n");
    const prompt = sourceWithLineNumbers(source);
    expect(prompt).toContain("Source line count: 1001.\n");
    expect(prompt).toContain("   1 | source 1\n");
    expect(prompt).toContain("1000 | source 1000\n1001 | source 1001");
  });

  it("counts blank lines and mixed newline styles consistently", () => {
    expect(sourceWithLineNumbers("first\r\n\r\nthird\rfourth\n")).toBe("Source line count: 5.\n1 | first\n2 | \n3 | third\n4 | fourth\n5 | ");
  });
});
