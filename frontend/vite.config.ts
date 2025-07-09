import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import themePlugin from "@replit/vite-plugin-shadcn-theme-json";
import path from "path";
import runtimeErrorOverlay from "@replit/vite-plugin-runtime-error-modal";

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api/process-voice': {
        target: 'http://0.0.0.0:8082',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/process-voice/, '/process-voice'),
        ws: true,
      },
      '/api/analyze-video': {
        target: 'http://0.0.0.0:8083',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/analyze-video/, '/analyze-video'),
        ws: true,
      },
      '/api/qr-scan': {
        target: 'http://0.0.0.0:8081',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/qr-scan/, '/qr-scan'),
        ws: true,
      },
      '/api': 'http://localhost:6900',
    },
  },
  plugins: [
    react(),
    runtimeErrorOverlay(),
    themePlugin(),
    ...(process.env.NODE_ENV !== "production" &&
    process.env.REPL_ID !== undefined
      ? [
          await import("@replit/vite-plugin-cartographer").then((m) =>
            m.cartographer(),
          ),
        ]
      : []),
  ],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "client", "src"),
      "@shared": path.resolve(import.meta.dirname, "shared"),
      "@assets": path.resolve(import.meta.dirname, "attached_assets"),
    },
  },
  root: path.resolve(import.meta.dirname, "client"),
  build: {
    outDir: path.resolve(import.meta.dirname, "dist/public"),
    emptyOutDir: true,
  },
});
