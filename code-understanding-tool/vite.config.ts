import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { createLocalExplanationApi } from "./src/server/local-api.ts";

export default defineConfig({
  plugins: [react(), createLocalExplanationApi()],
});
