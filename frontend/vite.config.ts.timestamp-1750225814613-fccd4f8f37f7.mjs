// vite.config.ts
import { defineConfig } from "file:///E:/safepayfinal-main%20with%20new%20model/frontend/node_modules/vite/dist/node/index.js";
import react from "file:///E:/safepayfinal-main%20with%20new%20model/frontend/node_modules/@vitejs/plugin-react/dist/index.mjs";
import themePlugin from "file:///E:/safepayfinal-main%20with%20new%20model/frontend/node_modules/@replit/vite-plugin-shadcn-theme-json/dist/index.mjs";
import path from "path";
import runtimeErrorOverlay from "file:///E:/safepayfinal-main%20with%20new%20model/frontend/node_modules/@replit/vite-plugin-runtime-error-modal/dist/index.mjs";
var __vite_injected_original_dirname = "E:\\safepayfinal-main with new model\\frontend";
var vite_config_default = defineConfig({
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api/process-voice": {
        target: "http://0.0.0.0:8082",
        changeOrigin: true,
        rewrite: (path2) => path2.replace(/^\/api\/process-voice/, "/process-voice"),
        ws: true
      },
      "/api/analyze-video": {
        target: "http://0.0.0.0:8083",
        changeOrigin: true,
        rewrite: (path2) => path2.replace(/^\/api\/analyze-video/, "/analyze-video"),
        ws: true
      },
      "/api/qr-scan": {
        target: "http://0.0.0.0:8081",
        changeOrigin: true,
        rewrite: (path2) => path2.replace(/^\/api\/qr-scan/, "/qr-scan"),
        ws: true
      },
      "/api": "http://localhost:6900"
    }
  },
  plugins: [
    react(),
    runtimeErrorOverlay(),
    themePlugin(),
    ...process.env.NODE_ENV !== "production" && process.env.REPL_ID !== void 0 ? [
      await import("file:///E:/safepayfinal-main%20with%20new%20model/frontend/node_modules/@replit/vite-plugin-cartographer/dist/index.mjs").then(
        (m) => m.cartographer()
      )
    ] : []
  ],
  resolve: {
    alias: {
      "@": path.resolve(__vite_injected_original_dirname, "client", "src"),
      "@shared": path.resolve(__vite_injected_original_dirname, "shared"),
      "@assets": path.resolve(__vite_injected_original_dirname, "attached_assets")
    }
  },
  root: path.resolve(__vite_injected_original_dirname, "client"),
  build: {
    outDir: path.resolve(__vite_injected_original_dirname, "dist/public"),
    emptyOutDir: true
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJFOlxcXFxzYWZlcGF5ZmluYWwtbWFpbiB3aXRoIG5ldyBtb2RlbFxcXFxmcm9udGVuZFwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiRTpcXFxcc2FmZXBheWZpbmFsLW1haW4gd2l0aCBuZXcgbW9kZWxcXFxcZnJvbnRlbmRcXFxcdml0ZS5jb25maWcudHNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0U6L3NhZmVwYXlmaW5hbC1tYWluJTIwd2l0aCUyMG5ldyUyMG1vZGVsL2Zyb250ZW5kL3ZpdGUuY29uZmlnLnRzXCI7aW1wb3J0IHsgZGVmaW5lQ29uZmlnIH0gZnJvbSBcInZpdGVcIjtcbmltcG9ydCByZWFjdCBmcm9tIFwiQHZpdGVqcy9wbHVnaW4tcmVhY3RcIjtcbmltcG9ydCB0aGVtZVBsdWdpbiBmcm9tIFwiQHJlcGxpdC92aXRlLXBsdWdpbi1zaGFkY24tdGhlbWUtanNvblwiO1xuaW1wb3J0IHBhdGggZnJvbSBcInBhdGhcIjtcbmltcG9ydCBydW50aW1lRXJyb3JPdmVybGF5IGZyb20gXCJAcmVwbGl0L3ZpdGUtcGx1Z2luLXJ1bnRpbWUtZXJyb3ItbW9kYWxcIjtcblxuZXhwb3J0IGRlZmF1bHQgZGVmaW5lQ29uZmlnKHtcbiAgc2VydmVyOiB7XG4gICAgaG9zdDogJzAuMC4wLjAnLFxuICAgIHBvcnQ6IDUxNzMsXG4gICAgc3RyaWN0UG9ydDogdHJ1ZSxcbiAgICBwcm94eToge1xuICAgICAgJy9hcGkvcHJvY2Vzcy12b2ljZSc6IHtcbiAgICAgICAgdGFyZ2V0OiAnaHR0cDovLzAuMC4wLjA6ODA4MicsXG4gICAgICAgIGNoYW5nZU9yaWdpbjogdHJ1ZSxcbiAgICAgICAgcmV3cml0ZTogKHBhdGgpID0+IHBhdGgucmVwbGFjZSgvXlxcL2FwaVxcL3Byb2Nlc3Mtdm9pY2UvLCAnL3Byb2Nlc3Mtdm9pY2UnKSxcbiAgICAgICAgd3M6IHRydWUsXG4gICAgICB9LFxuICAgICAgJy9hcGkvYW5hbHl6ZS12aWRlbyc6IHtcbiAgICAgICAgdGFyZ2V0OiAnaHR0cDovLzAuMC4wLjA6ODA4MycsXG4gICAgICAgIGNoYW5nZU9yaWdpbjogdHJ1ZSxcbiAgICAgICAgcmV3cml0ZTogKHBhdGgpID0+IHBhdGgucmVwbGFjZSgvXlxcL2FwaVxcL2FuYWx5emUtdmlkZW8vLCAnL2FuYWx5emUtdmlkZW8nKSxcbiAgICAgICAgd3M6IHRydWUsXG4gICAgICB9LFxuICAgICAgJy9hcGkvcXItc2Nhbic6IHtcbiAgICAgICAgdGFyZ2V0OiAnaHR0cDovLzAuMC4wLjA6ODA4MScsXG4gICAgICAgIGNoYW5nZU9yaWdpbjogdHJ1ZSxcbiAgICAgICAgcmV3cml0ZTogKHBhdGgpID0+IHBhdGgucmVwbGFjZSgvXlxcL2FwaVxcL3FyLXNjYW4vLCAnL3FyLXNjYW4nKSxcbiAgICAgICAgd3M6IHRydWUsXG4gICAgICB9LFxuICAgICAgJy9hcGknOiAnaHR0cDovL2xvY2FsaG9zdDo2OTAwJyxcbiAgICB9LFxuICB9LFxuICBwbHVnaW5zOiBbXG4gICAgcmVhY3QoKSxcbiAgICBydW50aW1lRXJyb3JPdmVybGF5KCksXG4gICAgdGhlbWVQbHVnaW4oKSxcbiAgICAuLi4ocHJvY2Vzcy5lbnYuTk9ERV9FTlYgIT09IFwicHJvZHVjdGlvblwiICYmXG4gICAgcHJvY2Vzcy5lbnYuUkVQTF9JRCAhPT0gdW5kZWZpbmVkXG4gICAgICA/IFtcbiAgICAgICAgICBhd2FpdCBpbXBvcnQoXCJAcmVwbGl0L3ZpdGUtcGx1Z2luLWNhcnRvZ3JhcGhlclwiKS50aGVuKChtKSA9PlxuICAgICAgICAgICAgbS5jYXJ0b2dyYXBoZXIoKSxcbiAgICAgICAgICApLFxuICAgICAgICBdXG4gICAgICA6IFtdKSxcbiAgXSxcbiAgcmVzb2x2ZToge1xuICAgIGFsaWFzOiB7XG4gICAgICBcIkBcIjogcGF0aC5yZXNvbHZlKGltcG9ydC5tZXRhLmRpcm5hbWUsIFwiY2xpZW50XCIsIFwic3JjXCIpLFxuICAgICAgXCJAc2hhcmVkXCI6IHBhdGgucmVzb2x2ZShpbXBvcnQubWV0YS5kaXJuYW1lLCBcInNoYXJlZFwiKSxcbiAgICAgIFwiQGFzc2V0c1wiOiBwYXRoLnJlc29sdmUoaW1wb3J0Lm1ldGEuZGlybmFtZSwgXCJhdHRhY2hlZF9hc3NldHNcIiksXG4gICAgfSxcbiAgfSxcbiAgcm9vdDogcGF0aC5yZXNvbHZlKGltcG9ydC5tZXRhLmRpcm5hbWUsIFwiY2xpZW50XCIpLFxuICBidWlsZDoge1xuICAgIG91dERpcjogcGF0aC5yZXNvbHZlKGltcG9ydC5tZXRhLmRpcm5hbWUsIFwiZGlzdC9wdWJsaWNcIiksXG4gICAgZW1wdHlPdXREaXI6IHRydWUsXG4gIH0sXG59KTtcbiJdLAogICJtYXBwaW5ncyI6ICI7QUFBa1UsU0FBUyxvQkFBb0I7QUFDL1YsT0FBTyxXQUFXO0FBQ2xCLE9BQU8saUJBQWlCO0FBQ3hCLE9BQU8sVUFBVTtBQUNqQixPQUFPLHlCQUF5QjtBQUpoQyxJQUFNLG1DQUFtQztBQU16QyxJQUFPLHNCQUFRLGFBQWE7QUFBQSxFQUMxQixRQUFRO0FBQUEsSUFDTixNQUFNO0FBQUEsSUFDTixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsSUFDWixPQUFPO0FBQUEsTUFDTCxzQkFBc0I7QUFBQSxRQUNwQixRQUFRO0FBQUEsUUFDUixjQUFjO0FBQUEsUUFDZCxTQUFTLENBQUNBLFVBQVNBLE1BQUssUUFBUSx5QkFBeUIsZ0JBQWdCO0FBQUEsUUFDekUsSUFBSTtBQUFBLE1BQ047QUFBQSxNQUNBLHNCQUFzQjtBQUFBLFFBQ3BCLFFBQVE7QUFBQSxRQUNSLGNBQWM7QUFBQSxRQUNkLFNBQVMsQ0FBQ0EsVUFBU0EsTUFBSyxRQUFRLHlCQUF5QixnQkFBZ0I7QUFBQSxRQUN6RSxJQUFJO0FBQUEsTUFDTjtBQUFBLE1BQ0EsZ0JBQWdCO0FBQUEsUUFDZCxRQUFRO0FBQUEsUUFDUixjQUFjO0FBQUEsUUFDZCxTQUFTLENBQUNBLFVBQVNBLE1BQUssUUFBUSxtQkFBbUIsVUFBVTtBQUFBLFFBQzdELElBQUk7QUFBQSxNQUNOO0FBQUEsTUFDQSxRQUFRO0FBQUEsSUFDVjtBQUFBLEVBQ0Y7QUFBQSxFQUNBLFNBQVM7QUFBQSxJQUNQLE1BQU07QUFBQSxJQUNOLG9CQUFvQjtBQUFBLElBQ3BCLFlBQVk7QUFBQSxJQUNaLEdBQUksUUFBUSxJQUFJLGFBQWEsZ0JBQzdCLFFBQVEsSUFBSSxZQUFZLFNBQ3BCO0FBQUEsTUFDRSxNQUFNLE9BQU8seUhBQWtDLEVBQUU7QUFBQSxRQUFLLENBQUMsTUFDckQsRUFBRSxhQUFhO0FBQUEsTUFDakI7QUFBQSxJQUNGLElBQ0EsQ0FBQztBQUFBLEVBQ1A7QUFBQSxFQUNBLFNBQVM7QUFBQSxJQUNQLE9BQU87QUFBQSxNQUNMLEtBQUssS0FBSyxRQUFRLGtDQUFxQixVQUFVLEtBQUs7QUFBQSxNQUN0RCxXQUFXLEtBQUssUUFBUSxrQ0FBcUIsUUFBUTtBQUFBLE1BQ3JELFdBQVcsS0FBSyxRQUFRLGtDQUFxQixpQkFBaUI7QUFBQSxJQUNoRTtBQUFBLEVBQ0Y7QUFBQSxFQUNBLE1BQU0sS0FBSyxRQUFRLGtDQUFxQixRQUFRO0FBQUEsRUFDaEQsT0FBTztBQUFBLElBQ0wsUUFBUSxLQUFLLFFBQVEsa0NBQXFCLGFBQWE7QUFBQSxJQUN2RCxhQUFhO0FBQUEsRUFDZjtBQUNGLENBQUM7IiwKICAibmFtZXMiOiBbInBhdGgiXQp9Cg==
