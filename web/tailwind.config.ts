import type { Config } from 'tailwindcss'

export default {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f7ff',
          100: '#dff0ff',
          200: '#b9deff',
          300: '#86c6ff',
          400: '#4fa6ff',
          500: '#1f86ff',
          600: '#0f6ce6',
          700: '#0a56b4',
          800: '#0a478f',
          900: '#0c3c74',
        }
      },
      boxShadow: {
        card: '0 8px 30px rgba(0,0,0,0.06)'
      }
    },
  },
  plugins: [],
} satisfies Config

