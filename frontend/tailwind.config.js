/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
        display: ['Space Grotesk', 'Inter', 'ui-sans-serif']
      },
      colors: {
        ink: "#0B0F13",
        neon: { green: "#2af598", pink: "#ff6ec7", blue: "#009EFD" }
      }
    }
  },
  plugins: []
}
