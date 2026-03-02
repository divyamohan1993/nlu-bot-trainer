import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/lib/index.ts"],
  format: ["cjs", "esm"],
  dts: true,
  clean: true,
  splitting: false,
  sourcemap: true,
  minify: false,
  target: "es2017",
  outDir: "dist",
  external: ["react", "react-dom", "next"],
  esbuildOptions(options) {
    options.platform = "neutral"; // Works in both browser and Node.js
  },
});
