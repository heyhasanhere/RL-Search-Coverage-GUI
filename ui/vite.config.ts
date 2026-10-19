import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { chunkSizeWarningLimit: 2000 },
  worker: { format: "es" },
  server: {
    proxy: {
      "/api": backend,
      "/ws": { target: backend, ws: true },
    },
  },
});
