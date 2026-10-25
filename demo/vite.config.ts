import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  root: dirname(fileURLToPath(import.meta.url)),
  base: './',
  // Serve @omernach/o3d as published so `new URL('engine_bg.wasm', import.meta.url)` resolves.
  optimizeDeps: { exclude: ['@omernach/o3d'] },
  build: { target: 'es2022', outDir: 'dist', emptyOutDir: true },
});
