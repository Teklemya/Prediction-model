import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/odds': 'http://127.0.0.1:5000', // Proxy API for fetching odds
      '/predict': 'http://127.0.0.1:5000' // Proxy API for predictions
    }
  }
});
