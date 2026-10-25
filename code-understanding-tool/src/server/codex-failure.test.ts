import { describe, expect, it } from "vitest";
import { describeCodexFailure, safeCodexDiagnostic } from "./local-api";

describe("describeCodexFailure", () => {
  it("suggests signing in when the CLI reports an authentication failure", () => {
    expect(describeCodexFailure("ERROR: not authenticated; run codex login", 1)).toContain("codex login");
  });

  it("explains local permission failures", () => {
    expect(describeCodexFailure("Operation not permitted", 1)).toContain("operating system blocked");
  });

  it("points to the local app terminal when network access is blocked", () => {
    expect(describeCodexFailure("getaddrinfo ENOTFOUND api.openai.com", 1)).toContain("terminal running this local app");
  });

  it("includes a safe network error code and service host when available", () => {
    const error = describeCodexFailure("getaddrinfo ENOTFOUND api.openai.com", 1);
    expect(error).toContain("ENOTFOUND api.openai.com");
  });

  it("includes a sanitized network diagnostic when no known socket code is present", () => {
    const error = describeCodexFailure("request failed: connection reset by peer", 1);
    expect(error).toContain("Diagnostic: request failed: connection reset by peer");
  });

  it("reports structured-output schema errors as schema errors instead of network failures", () => {
    const error = describeCodexFailure(
      `ERROR: ${JSON.stringify({ error: { message: "Invalid schema for response_format: schema.required is missing visual_note." } })}`,
      1,
    );
    expect(error).toContain("Codex rejected the app's response schema");
    expect(error).toContain("missing visual_note");
    expect(error).not.toContain("network access");
  });

  it("redacts credentials and URL query strings from CLI diagnostics", () => {
    const diagnostic = safeCodexDiagnostic("error: https://api.example.test/path?token=secret API_KEY=private-value");
    expect(diagnostic).toBe("error: https://api.example.test/path API_KEY=[redacted]");
  });

  it("identifies outdated CLI options", () => {
    expect(describeCodexFailure("error: unknown feature: code_mode", 2)).toContain("does not support the options");
  });

  it("does not expose raw CLI output in the fallback message", () => {
    const error = describeCodexFailure("private prompt or token value", 7);
    expect(error).toContain("status 7");
    expect(error).not.toContain("private prompt");
  });
});
