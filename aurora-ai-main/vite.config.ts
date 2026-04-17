import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8000,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        ws: true, // Enable WebSocket proxying
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('Proxy error:', err);
          });
        },
      },
      '/uploads': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        ws: true, // Enable WebSocket proxying
      },
      '/chat': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        ws: true,
      },
      '/upload': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
