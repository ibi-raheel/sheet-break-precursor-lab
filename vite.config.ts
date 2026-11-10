import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// Hosted under a sub-path on ibiraheel.com; every asset URL is resolved against this base.
export default defineConfig({ base: '/demos/sheet-break-precursor-lab/', plugins: [react()] })
