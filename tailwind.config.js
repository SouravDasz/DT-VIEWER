/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/templates/**/*.html",
    "./app/static/**/*.js",
  ],
  theme: {
    extend: {
      colors: {
        'primary-green': '#10B981',
        'leaf-green': '#059669',
        'background-light': '#020617',
        'card-bg': '#1E293B',
        'text-dark': '#F3F4F6',
        'text-secondary': '#D1D5DB',
        'feature-bg': '#0F172A',
      },
      fontFamily: {
        'sans': ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
