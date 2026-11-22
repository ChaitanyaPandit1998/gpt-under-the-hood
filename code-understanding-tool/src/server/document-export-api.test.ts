import { Readable } from "node:stream";
import { EventEmitter } from "node:events";
import type { IncomingMessage, ServerResponse } from "node:http";
import { describe, expect, it, vi } from "vitest";
import { documentExportApi } from "./document-export-api.ts";
import type { DocumentExporter } from "./wordcraft-export.ts";

const body = { version: 1, format: "pdf", source: "x", includeSource: false, scope: "Selected", provenance: "",
  explanation: { title: "Study", summary: "Content", concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "" } };

function request(method: string, value: unknown = body) {
  const req = Readable.from([Buffer.from(typeof value === "string" ? value : JSON.stringify(value))]) as unknown as IncomingMessage;
  req.method = method;
  const headers: Record<string, unknown> = {};
  const res = Object.assign(new EventEmitter(), { statusCode: 200, destroyed: false, headersSent: false,
    setHeader: (key: string, value: unknown) => { headers[key] = value; },
    end: vi.fn(),
  }) as unknown as ServerResponse;
  return { req, res, headers };
}

describe("document export API", () => {
  it("reports availability without generating anything", async () => {
    const exporter = { available: async () => false, export: vi.fn() };
    const { req, res } = request("GET");
    await documentExportApi(exporter)(req, res);
    expect(JSON.parse(vi.mocked(res.end).mock.calls[0][0] as string).available).toBe(false);
    expect(exporter.export).not.toHaveBeenCalled();
  });
  it("returns a fixed attachment filename, no cache and no paths", async () => {
    const exporter = { available: async () => true, export: vi.fn(async () => Buffer.from("%PDF-test")) };
    const { req, res, headers } = request("POST");
    await documentExportApi(exporter)(req, res);
    expect(headers["Content-Disposition"]).toBe('attachment; filename="study-guide.pdf"');
    expect(headers["Cache-Control"]).toBe("no-store");
    expect(exporter.export.mock.calls).toHaveLength(1);
  });
  const illustrated = { ...body, diagramIndex: 0, explanation: { ...body.explanation, visuals: [
    { title: "Map", kind: "concept_map", nodes: [{ id: "a", label: "A", description: "Node", source_ranges: [] }], edges: [] },
  ] } };
  it("renders only the selected existing diagram and passes its bytes to PDF export", async () => {
    const png = Buffer.from("server-generated PNG fixture");
    const diagrams = { available: async () => true, export: vi.fn(async () => png) };
    const exporter = { available: async () => true, export: vi.fn(async () => Buffer.from("%PDF-test")) };
    const handler = documentExportApi(exporter, diagrams);
    const capabilities = request("GET");
    await handler(capabilities.req, capabilities.res);
    expect(vi.mocked(capabilities.res.end).mock.calls[0][0]).toContain('"diagramImages":true');
    expect(diagrams.export).not.toHaveBeenCalled();
    const { req, res } = request("POST", illustrated);
    await handler(req, res);
    expect(res.statusCode).toBe(200);
    expect(diagrams.export).toHaveBeenCalledWith(expect.objectContaining({ visualIndex: 0 }), expect.any(AbortSignal));
    expect(exporter.export).toHaveBeenCalledWith(expect.objectContaining({ explanation: illustrated.explanation }), "pdf", expect.any(AbortSignal), { visualIndex: 0, png });
  });
  it.each([[{ diagramIndex: 1 }, 400], [{ format: "docx" }, 400], [{}, 503]] as const)("rejects invalid or unavailable image selection %#", async (override, status) => {
    const diagrams = { available: async () => false, export: vi.fn() };
    const exporter = { available: async () => true, export: vi.fn() };
    const { req, res } = request("POST", { ...illustrated, ...override });
    await documentExportApi(exporter, diagrams)(req, res);
    expect(res.statusCode).toBe(status);
    expect(diagrams.export).not.toHaveBeenCalled();
    expect(exporter.export).not.toHaveBeenCalled();
  });
  it.each([["POST", "bad-json", 400], ["POST", "x".repeat(2 * 1024 * 1024 + 1), 413], ["PUT", body, 405]] as const)("rejects invalid input or method case %#", async (method, value, status) => {
    const exporter = { available: async () => true, export: vi.fn() };
    const { req, res } = request(method, value);
    await documentExportApi(exporter)(req, res);
    expect(res.statusCode).toBe(status);
    expect(exporter.export).not.toHaveBeenCalled();
  });
  it("allows one export and propagates disconnect cancellation", async () => {
    let start!: () => void;
    const started = new Promise<void>((resolve) => { start = resolve; });
    const exporter: DocumentExporter = { available: async () => true, export: async (_snapshot, _format, signal) => {
      start();
      return new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(new Error("private content")), { once: true }));
    } };
    const handler = documentExportApi(exporter);
    const first = request("POST");
    const pending = handler(first.req, first.res);
    await started;
    const second = request("POST");
    await handler(second.req, second.res);
    expect(second.res.statusCode).toBe(429);
    first.res.emit("close");
    await pending;
    expect(first.res.statusCode).toBe(408);
    expect(vi.mocked(first.res.end).mock.calls[0][0]).not.toContain("private content");
  });
});
