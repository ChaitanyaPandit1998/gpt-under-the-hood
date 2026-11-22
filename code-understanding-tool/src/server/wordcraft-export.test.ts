import { access, readFile, writeFile, symlink } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createWordcraftExporter, renderWordcraftDocument, runCraft, type CraftRunner } from "./wordcraft-export.ts";
import { parseStudyExport } from "./study-export.ts";

const snapshot = parseStudyExport({ version: 1, format: "pdf", source: "x", includeSource: false, scope: "Original", provenance: "",
  explanation: { title: "Study", summary: "Content", concepts: [], walkthrough: [], assumptions: [], visuals: [], visual_note: "" },
}).snapshot;

describe("WordCraft adapter", () => {
  it("is off by default and rejects unapproved executables", async () => {
    expect(await createWordcraftExporter().available()).toBe(false);
    const exporter = createWordcraftExporter({ enabled: true, cli: process.execPath });
    expect(await exporter.available()).toBe(false);
    await expect(exporter.export(snapshot, "pdf", new AbortController().signal)).rejects.toThrow("Configure");
  });
  it("uses fixed args and removes all temporary files after success", async () => {
    let directory = "";
    const calls: string[][] = [];
    const runner: CraftRunner = async (_cli, args, cwd) => {
      directory = cwd; calls.push(args);
      if (args[0] === "--version") return "wordcraft-cli 0.18.1";
      if (args[0] === "run") {
        const script = JSON.parse(await readFile(args[2], "utf8"));
        expect(script.save).toEqual([join(cwd, "study.docx")]);
        await writeFile(script.save[0], Buffer.from([80, 75, 3, 4]));
      }
      if (args[0] === "convert") { expect(args.at(-1)).toBe("--strict"); await writeFile(args[2], "%PDF-1.7"); }
      return "";
    };
    expect((await renderWordcraftDocument("/approved/cli", snapshot, "pdf", new AbortController().signal, runner)).toString()).toBe("%PDF-1.7");
    expect(calls.map((c) => c[0])).toEqual(["--version", "run", "convert"]);
    await expect(access(directory)).rejects.toThrow();
  });
  it.each(["wrong version", "invalid signature", "symlink", "failure"])("fails closed and cleans up on %s", async (reason) => {
    let directory = "";
    const runner: CraftRunner = async (_cli, args, cwd) => {
      directory = cwd;
      if (args[0] === "--version") return reason === "wrong version" ? "0.99.0" : "0.18.1";
      if (reason === "failure") throw new Error("Private renderer diagnostic");
      if (reason === "symlink") await symlink(join(cwd, "commands.json"), join(cwd, "study.docx"));
      else await writeFile(join(cwd, "study.docx"), "not a docx");
      return "";
    };
    await expect(renderWordcraftDocument("/approved/cli", snapshot, "docx", new AbortController().signal, runner)).rejects.toThrow();
    await expect(access(directory)).rejects.toThrow();
  });
  it("does not inherit provider tokens, proxies, or user HOME", async () => {
    const output = await runCraft(process.execPath, ["-e", "process.stdout.write(JSON.stringify(process.env))"], "/tmp", new AbortController().signal);
    const env = JSON.parse(output);
    // macOS inserts this locale variable when launching native processes.
    delete env.__CF_USER_TEXT_ENCODING;
    expect(Object.keys(env).sort()).toEqual(["HOME", "LANG", "TMPDIR", "WORDCRAFT_REGION", "VECTORCRAFT_NO_PREFS"].sort());
    expect(env.HOME).toBe("/tmp");
  });
  it("cancels a running renderer and bounds its output", async () => {
    const controller = new AbortController();
    const pending = runCraft(process.execPath, ["-e", "setInterval(()=>{},1000)"], "/tmp", controller.signal);
    controller.abort();
    await expect(pending).rejects.toThrow();
    await expect(runCraft(process.execPath, ["-e", "process.stdout.write('x'.repeat(300000))"], "/tmp", new AbortController().signal)).rejects.toThrow();
  });
});
