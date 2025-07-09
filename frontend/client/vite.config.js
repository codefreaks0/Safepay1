import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    proxy: {
      '/predict-audio': {
        target: 'http://localhost:8083',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/predict-audio/, '/predict-audio'),
      },
    },
  },
}); 