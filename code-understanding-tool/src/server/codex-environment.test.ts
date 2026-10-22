import { describe, expect, it } from "vitest";
import { createCodexEnvironment } from "./local-api";

describe("createCodexEnvironment", () => {
  it("passes required Codex, authentication, and runtime variables while excluding unrelated secrets", () => {
    const environment = createCodexEnvironment({
      PATH: "/bin", HOME: "/home/learner", CODEX_HOME: "/home/learner/.codex",
      CODEX_ACCESS_TOKEN: "codex-token", OPENAI_API_KEY: "openai-key", HTTP_PROXY: "http://proxy",
      DATABASE_PASSWORD: "do-not-pass", RANDOM_SECRET: "do-not-pass-either",
    }, "linux");

    expect(environment).toEqual({
      PATH: "/bin", HOME: "/home/learner", CODEX_HOME: "/home/learner/.codex",
      CODEX_ACCESS_TOKEN: "codex-token", OPENAI_API_KEY: "openai-key", HTTP_PROXY: "http://proxy",
    });
  });

  it("keeps Windows process and user profile variables on Windows", () => {
    const environment = createCodexEnvironment({
      PATH: "C:\\Windows\\System32", SystemRoot: "C:\\Windows", APPDATA: "C:\\Users\\Learner\\AppData\\Roaming",
      DATABASE_PASSWORD: "excluded",
    }, "win32");

    expect(environment).toMatchObject({ PATH: "C:\\Windows\\System32", SystemRoot: "C:\\Windows", APPDATA: "C:\\Users\\Learner\\AppData\\Roaming" });
    expect(environment).not.toHaveProperty("DATABASE_PASSWORD");
  });

  it("supplies a home directory when no profile variable is present", () => {
    expect(createCodexEnvironment({}, "linux").HOME).toBeTruthy();
  });
});
