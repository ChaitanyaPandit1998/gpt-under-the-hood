import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { access, lstat, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join } from "node:path";
import { documentCommands, type DocumentFormat, type StudyExportSnapshot } from "./study-export.ts";

export type CraftRunner = (cli: string, args: string[], directory: string, signal: AbortSignal) => Promise<string>;
export type DiagramFigure = { visualIndex: number; png: Buffer };
export type DocumentExporter = {
  available: () => Promise<boolean>;
  export: (snapshot: StudyExportSnapshot, format: DocumentFormat, signal: AbortSignal, figure?: DiagramFigure) => Promise<Buffer>;
};
export const WORDCRAFT_SHA256 = "9d110decf7f99871cdc981798f11fa06d2daaa6f73621db71177bc5f5aee565c";

export const runCraft: CraftRunner = (cli, args, directory, signal) => new Promise((resolve, reject) => {
  if (signal.aborted) { reject(new Error("Export cancelled.")); return; }
  const child = spawn(cli, args, { cwd: directory, shell: false, windowsHide: true,
    env: { HOME: directory, TMPDIR: directory, LANG: "en_GB.UTF-8", WORDCRAFT_REGION: "en-GB", VECTORCRAFT_NO_PREFS: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let output = "";
  let bytes = 0;
  let overflow = false;
  let timedOut = false;
  let killTimer: NodeJS.Timeout | undefined;
  const terminate = () => {
    child.kill("SIGTERM");
    killTimer ??= setTimeout(() => child.kill("SIGKILL"), 1_000);
    killTimer.unref();
  };
  for (const stream of [child.stdout, child.stderr]) stream.on("data", (chunk: Buffer) => {
    bytes += chunk.length;
    if (bytes > 256_000) { overflow = true; terminate(); }
    else if (stream === child.stdout) output += chunk.toString("utf8");
  });
  const timeout = setTimeout(() => { timedOut = true; terminate(); }, 120_000);
  signal.addEventListener("abort", terminate, { once: true });
  if (signal.aborted) terminate();
  const cleanup = () => { clearTimeout(timeout); if (killTimer) clearTimeout(killTimer); signal.removeEventListener("abort", terminate); };
  child.once("error", () => { cleanup(); reject(new Error("Local renderer could not start.")); });
  child.once("close", (code) => {
    cleanup();
    if (code !== 0 || signal.aborted || overflow || timedOut) reject(new Error("Local render failed or was cancelled."));
    else resolve(output);
  });
});

export function createWordcraftExporter(options: { enabled?: boolean; cli?: string; run?: CraftRunner } = {}): DocumentExporter {
  const run = options.run ?? runCraft;
  // Match the tested macOS universal executable (not the archive) checksum/version.
  async function approved() {
    if (!options.enabled || !options.cli || !isAbsolute(options.cli)) return false;
    try {
      await access(options.cli, constants.X_OK);
      return createHash("sha256").update(await readFile(options.cli)).digest("hex") === WORDCRAFT_SHA256;
    } catch { return false; }
  }
  return {
    available: approved,
    async export(snapshot, format, signal, figure) {
      if (!(await approved())) throw new Error("Configure the tested WordCraft 0.18.1 executable to enable experimental export.");
      return renderWordcraftDocument(options.cli!, snapshot, format, signal, run, figure);
    },
  };
}

export async function renderWordcraftDocument(cli: string, snapshot: StudyExportSnapshot, format: DocumentFormat, signal: AbortSignal, run: CraftRunner = runCraft, figure?: DiagramFigure) {
  const directory = await mkdtemp(join(tmpdir(), "understand-export-"));
  try {
    const version = await run(cli, ["--version"], directory, signal);
    if (!/\b0\.18\.1\b/.test(version)) throw new Error("Unsupported WordCraft version.");
    const docx = join(directory, "study.docx");
    const output = join(directory, `study.${format}`);
    const script = join(directory, "commands.json");
    let image;
    if (figure) {
      signal.throwIfAborted();
      const path = join(directory, "diagram.png");
      const { png } = figure;
      if (png.length < 24 || !png.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) throw new Error("Invalid diagram image.");
      const width = png.readUInt32BE(16), height = png.readUInt32BE(20);
      if (width !== 1000 || height < 1 || height > 8000) throw new Error("Invalid diagram dimensions.");
      await writeFile(path, png, { mode: 0o600 });
      image = { path, visualIndex: figure.visualIndex, width: Math.min(480, 620 * width / height) };
    }
    await writeFile(script, JSON.stringify(documentCommands(snapshot, docx, image)), { mode: 0o600 });
    await run(cli, ["run", "--script", script, "--author", "Code Understanding Tool"], directory, signal);
    if (format === "pdf") await run(cli, ["convert", docx, output, "--strict"], directory, signal);
    const stats = await lstat(output);
    if (!stats.isFile() || stats.size < 4 || stats.size > 25 * 1024 * 1024) throw new Error("Invalid or oversized export artifact.");
    const bytes = await readFile(output);
    if (signal.aborted) throw new Error("Export cancelled.");
    if (format === "pdf" ? !bytes.subarray(0, 5).equals(Buffer.from("%PDF-")) : !bytes.subarray(0, 4).equals(Buffer.from([80, 75, 3, 4]))) throw new Error("WordCraft returned an invalid document.");
    return bytes;
  } finally { await rm(directory, { recursive: true, force: true }); }
}
