import { createHash } from "node:crypto";
import { validateExplanation } from "./response-validation.ts";
import type { Explanation } from "../domain/explanation-types.ts";
import { stripInlineEmphasis } from "../domain/inline-emphasis.ts";

export type StudyExportSnapshot = {
  version: 1;
  explanation: Explanation;
  source: string;
  includeSource: boolean;
  scope: string;
  provenance: string;
};
export type DocumentFormat = "docx" | "pdf";

export function parseStudyExport(value: unknown): { snapshot: StudyExportSnapshot; format: DocumentFormat } {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid export request.");
  const body = value as Record<string, unknown>;
  if (body.version !== 1 || !["docx", "pdf"].includes(String(body.format))
    || typeof body.source !== "string" || !body.source.trim() || body.source.length > 200_000
    || typeof body.includeSource !== "boolean"
    || typeof body.scope !== "string" || !body.scope.trim() || body.scope.length > 500
    || typeof body.provenance !== "string" || body.provenance.length > 4_000) throw new Error("Invalid export scope, format, or source.");
  const explanation = structuredClone(validateExplanation(body.explanation, body.source));
  if (explanation.quality_review_status !== undefined && !["passed", "revised", "incomplete"].includes(explanation.quality_review_status)) throw new Error("Invalid review status.");
  if (explanation.quality_review_note !== undefined && (typeof explanation.quality_review_note !== "string" || explanation.quality_review_note.length > 8_000)) throw new Error("Invalid review note.");
  if (explanation.incomplete_items !== undefined) {
    if (!Array.isArray(explanation.incomplete_items) || explanation.incomplete_items.length > 20) throw new Error("Invalid incomplete-code findings.");
    for (const item of explanation.incomplete_items) {
      if (!item || !["unfinished_marker", "empty_branch"].includes(item.kind)
        || typeof item.title !== "string" || item.title.length > 300
        || typeof item.explanation !== "string" || item.explanation.length > 8_000) throw new Error("Invalid incomplete-code finding.");
      validateExplanation({ ...explanation, concepts: [{ term: "Finding", explanation: "", source_ranges: item.source_ranges }] }, body.source);
    }
  }
  return { format: body.format as DocumentFormat, snapshot: {
    version: 1, explanation, source: body.source, includeSource: body.includeSource,
    scope: body.scope, provenance: body.provenance,
  } };
}

type Command = { command: string; params?: Record<string, unknown> };

/** Content is inserted only as plain text, never parsed as commands or markup. */
export function documentCommands(snapshot: StudyExportSnapshot, docxPath: string, image?: { path: string; visualIndex: number; width: number }): { commands: Command[]; save: string[] } {
  const e = snapshot.explanation;
  const commands: Command[] = [];
  const add = (command: string, params: Record<string, unknown>) => commands.push({ command, params });
  let first = true;
  function paragraph(text: string, style = "Normal") {
    for (const line of text.split(/\r\n|\r|\n/)) {
      if (!first) commands.push({ command: "text.newParagraph" });
      first = false;
      add("styles.apply", { style });
      add("para.pageBreakBefore", { value: false });
      add("para.keepNext", { value: style !== "Normal" });
      add("para.borders", { kind: "none" });
      add("text.insert", { text: line.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, "") });
    }
  }
  function cite(ranges: { start_line: number; end_line: number }[]) {
    if (ranges.length) paragraph(`Chapter source lines: ${ranges.map((r) => `${r.start_line}-${r.end_line}`).join(", ")}.${snapshot.includeSource ? " See Source appendix." : " Source omitted; coordinates require the original chapter source."}`);
  }
  add("file.properties", { title: e.title, author: "Code Understanding Tool" });
  add("layout.size", { name: "A4" });
  add("layout.margins", { top: 56, bottom: 56, left: 56, right: 56 });
  for (const [style, size] of [["Normal", 11], ["Title", 24], ["Heading1", 17], ["Heading2", 13]] as const) {
    add("styles.modify", { style, chr: { font: "Liberation Sans", size, bold: style !== "Normal", color: { Rgb: [0, 0, 0] } }, para: { spaceAfter: 8, widowControl: true } });
  }
  paragraph(e.title, "Title");
  paragraph(`Scope: ${snapshot.scope}. Only this existing result is exported; other chapters and variants are not included.`);
  paragraph(`Experimental study export. Model review is not runtime verification. ${image ? "Includes one selected diagram image; diagram text and citations are retained." : "Diagrams are represented as text, not rendered images."} Follow-ups and conceptual artwork are not included.`);
  paragraph(`Source SHA256: ${createHash("sha256").update(snapshot.source).digest("hex")}`);
  if (snapshot.provenance) paragraph(snapshot.provenance);
  paragraph(`Review status: ${e.quality_review_status ?? "not provided"}`);
  paragraph(e.quality_review_note ?? "No model review note provided.");
  paragraph(snapshot.includeSource ? "Source inclusion approved for this download." : "Source text omitted from the appendix by choice; explanation text may still quote code. Evidence coordinates are chapter-local and cannot be checked from this document alone.");
  paragraph("Overview", "Heading1"); paragraph(stripInlineEmphasis(e.summary));
  for (const item of e.incomplete_items ?? []) { paragraph(`Incomplete code ${item.title}`, "Heading2"); paragraph(item.explanation); cite(item.source_ranges); }
  if (e.concepts.length) paragraph("Key ideas", "Heading1");
  for (const item of e.concepts) { paragraph(item.term, "Heading2"); paragraph(stripInlineEmphasis(item.explanation)); cite(item.source_ranges); }
  if (e.walkthrough.length) paragraph("Walkthrough", "Heading1");
  for (const item of e.walkthrough) { paragraph(item.heading, "Heading2"); paragraph(stripInlineEmphasis(item.explanation)); cite(item.source_ranges); }
  paragraph("Diagram coverage and limits", "Heading1"); paragraph(e.visual_note || "No diagram coverage note provided.");
  for (const visual of e.visuals) {
    paragraph(visual.title, "Heading2");
    for (const node of visual.nodes) { paragraph(`[${node.id}] ${node.label}: ${node.description}`); cite(node.source_ranges); }
    for (const edge of visual.edges) paragraph(`${edge.from} -> ${edge.to}: ${edge.label}`);
  }
  paragraph("Assumptions and unknowns", "Heading1");
  for (const item of e.assumptions) paragraph(`${item.statement} ${item.reason}`);
  if (snapshot.includeSource) { paragraph("Source appendix", "Heading1"); snapshot.source.split(/\r\n|\r|\n/).forEach((line, index) => paragraph(`${index + 1} | ${line}`)); }
  if (image) {
    const visual = e.visuals[image.visualIndex];
    paragraph("Selected diagram image", "Heading1");
    add("para.pageBreakBefore", { value: true });
    paragraph("VectorCraft rendering of the selected explanation diagram, not a runtime trace. See Diagram coverage and limits for selectable text and citations.");
    commands.push({ command: "text.newParagraph" });
    add("styles.apply", { style: "Normal" });
    add("para.pageBreakBefore", { value: false });
    add("para.keepNext", { value: false });
    add("insert.picture", { path: image.path, width: image.width, alt: `${visual.title}. See Diagram coverage and limits for the text alternative.` });
  }
  return { commands, save: [docxPath] };
}
