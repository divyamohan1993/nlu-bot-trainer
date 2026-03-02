import { describe, it, expect } from "vitest";
import { normalizeText, tokenizeV2, DEFAULT_TOKENIZER_CONFIG } from "@/lib/engine/tokenizer-v2";

describe("Tokenizer V2", () => {
  describe("normalizeText", () => {
    it("lowercases input", () => {
      expect(normalizeText("HELLO WORLD")).toContain("hello world");
    });

    it("trims whitespace", () => {
      expect(normalizeText("  hello  ")).toBe(normalizeText("hello"));
    });

    it("expands contractions", () => {
      const result = normalizeText("I can't do this");
      expect(result).toContain("can not");
    });

    it("normalizes unicode apostrophes before contractions", () => {
      // Curly apostrophe → straight, then contraction expansion
      const result = normalizeText("I can\u2019t do this");
      expect(result).toContain("can not");
    });

    it("normalizes unicode dashes", () => {
      const result = normalizeText("hello\u2014world");
      expect(result).toContain("hello-world");
    });

    it("normalizes unicode quotes", () => {
      const result = normalizeText("\u201Chello\u201D");
      expect(result).toContain('"hello"');
    });
  });

  describe("tokenizeV2", () => {
    it("produces features from text", () => {
      const features = tokenizeV2("track my order please", DEFAULT_TOKENIZER_CONFIG);
      expect(features.length).toBeGreaterThan(0);
    });

    it("produces word unigrams", () => {
      const features = tokenizeV2("hello world", DEFAULT_TOKENIZER_CONFIG);
      expect(features.some((f) => f.includes("hello") || f.includes("hel"))).toBe(true);
    });

    it("handles empty string", () => {
      const features = tokenizeV2("", DEFAULT_TOKENIZER_CONFIG);
      expect(Array.isArray(features)).toBe(true);
    });

    it("handles single word", () => {
      const features = tokenizeV2("hello", DEFAULT_TOKENIZER_CONFIG);
      expect(features.length).toBeGreaterThan(0);
    });

    it("different texts produce different features", () => {
      const f1 = tokenizeV2("track my order", DEFAULT_TOKENIZER_CONFIG);
      const f2 = tokenizeV2("cancel my subscription", DEFAULT_TOKENIZER_CONFIG);
      const f1Set = new Set(f1);
      const f2Set = new Set(f2);
      // They should have some different features
      const unique1 = [...f1Set].filter((f) => !f2Set.has(f));
      expect(unique1.length).toBeGreaterThan(0);
    });

    it("produces signal features for domain text", () => {
      const features = tokenizeV2("I want to track my delivery", DEFAULT_TOKENIZER_CONFIG);
      expect(features.some((f) => f.startsWith("__sig_"))).toBe(true);
    });

    it("produces character n-grams", () => {
      const features = tokenizeV2("hello", DEFAULT_TOKENIZER_CONFIG);
      // Char n-grams use _c3_ and _c4_ prefix format
      expect(features.some((f) => f.startsWith("_c3_") || f.startsWith("_c4_"))).toBe(true);
    });
  });
});
