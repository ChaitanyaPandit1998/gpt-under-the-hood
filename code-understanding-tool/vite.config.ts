import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { createLocalExplanationApi } from "./src/server/local-api.ts";
import { createWordcraftExporter } from "./src/server/wordcraft-export.ts";
import { createVectorcraftExporter } from "./src/server/vectorcraft-export.ts";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
  plugins: [react(), createLocalExplanationApi({
    compactVisualPlanning: env.ENABLE_EXPERIMENTAL_COMPACT_VISUAL_PLANNING === "1",
    documentExporter: createWordcraftExporter({ enabled: env.ENABLE_EXPERIMENTAL_DOCUMENT_EXPORT === "1", cli: env.WORDCRAFT_CLI_PATH }),
    diagramExporter: createVectorcraftExporter({ enabled: env.ENABLE_EXPERIMENTAL_DIAGRAM_EXPORT === "1", cli: env.VECTORCRAFT_CLI_PATH }),
    imageCredentials: {
    accountId: env.CLOUDFLARE_ACCOUNT_ID || env.CF_ACCOUNT_ID || "",
    apiToken: env.CLOUDFLARE_API_TOKEN || env.CF_API_TOKEN || "",
  } })],
  test: { include: ["src/**/*.test.{ts,tsx}"] },
  server: { host: "127.0.0.1" },
  preview: { host: "127.0.0.1" },
  };
});
