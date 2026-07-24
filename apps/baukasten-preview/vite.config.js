import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' so the built app works mounted at any sub-path
// (deployed under https://zoomylab.github.io/Zoomy/baukasten-preview/).
export default defineConfig({
  base: './',
  plugins: [react()],
})
