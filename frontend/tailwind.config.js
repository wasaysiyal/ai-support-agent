/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f4f6f8",
          100: "#e6eaef",
          600: "#3d4f66",
          700: "#2c3b4d",
          900: "#161e29",
        },
      },
    },
  },
  plugins: [],
};
