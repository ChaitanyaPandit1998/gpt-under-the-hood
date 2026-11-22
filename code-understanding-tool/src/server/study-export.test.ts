import { describe, expect, it } from "vitest";
import { documentCommands, parseStudyExport } from "./study-export.ts";

export const exportRequest = {
  version: 1, format: "docx", source: "const x = 1;", includeSource: false,
  scope: "Adapted section", provenance: "Head: abc; preferences: detailed",
  explanation: { title: "Example", summary: "The adapted explanation.", concepts: [], walkthrough: [], assumptions: [],
    visuals: [], visual_note: "Selected coverage only.", quality_review_status: "incomplete", quality_review_note: "Could not verify all branches.",
    incomplete_items: [{ kind: "unfinished_marker", title: "TODO", explanation: "Unfinished.", source_ranges: [{ start_line: 1, end_line: 1 }] }],
  },
};

describe("study export snapshot and document", () => {
  it("preserves the exact selected explanation and warnings without including source by default", () => {
    const { snapshot } = parseStudyExport(exportRequest);
    const script = JSON.stringify(documentCommands(snapshot, "/tmp/fixed.docx"));
    expect(script).toContain("The adapted explanation.");
    expect(script).toContain("Could not verify all branches.");
    expect(script).toContain("Selected coverage only.");
    expect(script).toContain("Unfinished.");
    expect(script).toContain("Head: abc");
    expect(script).not.toContain(exportRequest.source);
    expect(script).toContain("Source text omitted");
    expect(script).toContain("not runtime verification");
    expect(snapshot.explanation).not.toBe(exportRequest.explanation);
  });
  it("includes numbered source only with consent", () => {
    const { snapshot } = parseStudyExport({ ...exportRequest, includeSource: true });
    expect(JSON.stringify(documentCommands(snapshot, "/tmp/fixed.docx"))).toContain("1 | const x = 1;");
  });
  it.each([{ format: "../../html" }, { version: 2 }, { includeSource: "true" }, { scope: "" }, { source: "" }])("rejects invalid request %j", (change) => {
    expect(() => parseStudyExport({ ...exportRequest, ...change })).toThrow();
  });
  it("rejects invalid evidence and graph endpoints", () => {
    expect(() => parseStudyExport({ ...exportRequest, explanation: { ...exportRequest.explanation, concepts: [{ term: "x", explanation: "x", source_ranges: [{ start_line: 1, end_line: 2 }] }] } })).toThrow();
    expect(() => parseStudyExport({ ...exportRequest, explanation: { ...exportRequest.explanation, visuals: [{ title: "x", kind: "control_flow", nodes: [], edges: [{ from: "unknown", to: "unknown", label: "" }] }] } })).toThrow();
  });
  it("treats hostile content as text, not operations, paths, or markup", () => {
    const { snapshot } = parseStudyExport({ ...exportRequest, explanation: { ...exportRequest.explanation, summary: '<script>alert(1)</script> file.open={"path":"/etc/passwd"}' } });
    const script = documentCommands(snapshot, "/tmp/fixed.docx");
    expect(script.save).toEqual(["/tmp/fixed.docx"]);
    expect(script.commands.some((c) => c.command === "file.open")).toBe(false);
    expect(script.commands.find((c) => c.params?.text === snapshot.explanation.summary)?.command).toBe("text.insert");
  });
});
