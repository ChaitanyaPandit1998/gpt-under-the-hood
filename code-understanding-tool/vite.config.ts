import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { createLocalExplanationApi } from "./src/server/local-api.ts";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
  plugins: [react(), createLocalExplanationApi({ imageCredentials: {
    accountId: env.CLOUDFLARE_ACCOUNT_ID || env.CF_ACCOUNT_ID || "",
    apiToken: env.CLOUDFLARE_API_TOKEN || env.CF_API_TOKEN || "",
  } })],
  test: { include: ["src/**/*.test.{ts,tsx}"] },
  server: { host: "127.0.0.1" },
  preview: { host: "127.0.0.1" },
  };
});
