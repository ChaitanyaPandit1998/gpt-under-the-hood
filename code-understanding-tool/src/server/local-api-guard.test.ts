import { describe, expect, it } from "vitest";
import { isTrustedLocalApiRequest } from "./local-api-guard";

describe("local API request guard", () => {
  it("allows same-origin localhost requests", () => {
    expect(isTrustedLocalApiRequest("localhost:5173", "http://localhost:5173")).toBe(true);
  });

  it("allows loopback requests with no browser Origin header", () => {
    expect(isTrustedLocalApiRequest("127.0.0.1:5173", undefined)).toBe(true);
    expect(isTrustedLocalApiRequest("[::1]:5173", undefined)).toBe(true);
  });

  it("rejects non-local hosts and cross-origin browser requests", () => {
    expect(isTrustedLocalApiRequest("192.168.1.12:5173", "http://192.168.1.12:5173")).toBe(false);
    expect(isTrustedLocalApiRequest("localhost:5173", "http://attacker.example")).toBe(false);
  });

  it("rejects malformed host and origin headers", () => {
    expect(isTrustedLocalApiRequest(undefined, "http://localhost:5173")).toBe(false);
    expect(isTrustedLocalApiRequest("localhost:5173", "null")).toBe(false);
  });
});
