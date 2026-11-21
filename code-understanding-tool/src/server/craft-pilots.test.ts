import { mkdtempSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";

describe("case-specific Craft pilot guards", () => {
  for (const name of ["deckcraft", "wordcraft", "vectorcraft"]) {
    it(`${name} requires explicit arguments before running any executable`, () => {
      const script = fileURLToPath(new URL(`../../evals/${name}-pilot.mjs`, import.meta.url));
      const result = spawnSync(process.execPath, [script], { encoding: "utf8", timeout: 10_000 });
      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain("Supply --cli");
    });

    it(`${name} rejects another corpus before invoking a CLI or creating outputs`, () => {
      const folder = mkdtempSync(join(tmpdir(), "craft-pilot-guard-"));
      try {
        const input = join(folder, "input.json"), output = join(folder, "output");
        writeFileSync(input, JSON.stringify({ url: "https://github.com/example/private/pull/1" }));
        const script = fileURLToPath(new URL(`../../evals/${name}-pilot.mjs`, import.meta.url));
        const result = spawnSync(process.execPath, [script, "--cli", join(folder, "must-not-run"), "--input", input, "--output", output], { encoding: "utf8", timeout: 10_000 });
        expect(result.status).not.toBe(0);
        expect(result.stderr).toContain("AssertionError");
        expect(result.stderr).not.toContain("ENOENT");
        expect(existsSync(output)).toBe(false);
      } finally { rmSync(folder, { recursive: true, force: true }); }
    });
  }
});
