import type { IncomingMessage, ServerResponse } from "node:http";
import { parseStudyExport } from "./study-export.ts";
import type { DocumentExporter } from "./wordcraft-export.ts";
import { validateDiagramSelection, type DiagramExporter } from "./vectorcraft-export.ts";

function json(response: ServerResponse, status: number, payload: unknown) {
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify(payload));
}

/** Immediate download avoids retaining private artifacts or introducing a job store. */
export function documentExportApi(exporter: DocumentExporter, diagrams?: DiagramExporter) {
  return artifactExportApi({
    available: exporter.available,
    formats: ["docx", "pdf"],
    capabilities: async () => ({ diagramImages: await diagrams?.available() === true }),
    parse(value) {
      const result = parseStudyExport(value);
      const { diagramIndex } = value as { diagramIndex?: unknown };
      if (diagramIndex === undefined) return { ...result, diagramIndex: undefined };
      if (result.format !== "pdf") throw new Error("Diagram images are supported only in PDF export.");
      return { ...result, diagramIndex: validateDiagramSelection(result.snapshot, diagramIndex) };
    },
    async render({ snapshot, format, diagramIndex }, signal) {
      let figure;
      if (diagramIndex !== undefined) {
        if (!diagrams || !(await diagrams.available())) throw new ExportUnavailable("VectorCraft is unavailable. Choose No image to export a text-only PDF.");
        figure = { visualIndex: diagramIndex, png: await diagrams.export({ study: snapshot, visualIndex: diagramIndex }, signal) };
      }
      const bytes = figure ? await exporter.export(snapshot, format, signal, figure) : await exporter.export(snapshot, format, signal);
      return { bytes, filename: `study-guide.${format}`,
        contentType: format === "pdf" ? "application/pdf" : "application/vnd.openxmlformats-officedocument.wordprocessingml.document" };
    },
  });
}

class ExportUnavailable extends Error {}

export function artifactExportApi<T>(options: {
  available: () => Promise<boolean>;
  formats: string[];
  capabilities?: () => Promise<Record<string, boolean>>;
  parse: (value: unknown) => T;
  render: (snapshot: T, signal: AbortSignal) => Promise<{ bytes: Buffer; filename: string; contentType: string }>;
}) {
  let active = false;
  return async (request: IncomingMessage, response: ServerResponse) => {
    response.setHeader("Cache-Control", "no-store");
    response.setHeader("X-Content-Type-Options", "nosniff");
    if (request.method === "GET") {
      const available = await options.available();
      json(response, 200, { available, formats: options.formats, experimental: true, ...(available && await options.capabilities?.()) });
      return;
    }
    if (request.method !== "POST") { response.setHeader("Allow", "GET, POST"); json(response, 405, { error: "Method not allowed." }); return; }
    if (active) { json(response, 429, { error: "Another export of this type is running. Try again shortly." }); return; }
    active = true;
    const controller = new AbortController();
    const cancel = () => controller.abort();
    request.on("aborted", cancel);
    response.on("close", cancel);
    let exporting = false;
    const deadline = setTimeout(() => {
      cancel();
      if (!exporting && !response.headersSent && !response.destroyed) {
        json(response, 408, { error: "Export request timed out." });
        request.destroy();
      }
    }, 120_000);
    try {
      const chunks: Buffer[] = [];
      let bytes = 0;
      for await (const chunk of request) {
        const buffer = Buffer.from(chunk);
        bytes += buffer.length;
        if (bytes > 2 * 1024 * 1024) { json(response, 413, { error: "Export requests must be under 2 MB." }); return; }
        chunks.push(buffer);
      }
      const snapshot = options.parse(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      if (!(await options.available())) { json(response, 503, { error: "Experimental export is disabled or the approved renderer executable is unavailable." }); return; }
      exporting = true;
      const artifact = await options.render(snapshot, controller.signal);
      if (controller.signal.aborted || response.destroyed) return;
      response.setHeader("Content-Type", artifact.contentType);
      response.setHeader("Content-Disposition", `attachment; filename="${artifact.filename}"`);
      response.setHeader("Content-Length", artifact.bytes.length);
      response.end(artifact.bytes);
    } catch (error) {
      if (!response.destroyed && !response.headersSent) json(response, controller.signal.aborted ? 408 : error instanceof ExportUnavailable ? 503 : exporting ? 500 : 400, {
        error: controller.signal.aborted ? "Export cancelled or timed out." : error instanceof ExportUnavailable ? error.message : exporting ? "Export failed. Check the approved CLI setup or try a smaller explanation." : "Invalid export request. Check scope, size limits and source citations.",
      });
    } finally {
      clearTimeout(deadline);
      request.removeListener("aborted", cancel);
      response.removeListener("close", cancel);
      active = false;
    }
  };
}
