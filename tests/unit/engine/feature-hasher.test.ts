import { describe, it, expect } from "vitest";
import {
  hashFeature,
  hashFeaturesDense,
  hashFeaturesTF,
  hashFeaturesSparse,
  dotProduct,
  quantizeToInt8,
  dequantizeFromInt8,
  DEFAULT_HASH_DIM,
} from "@/lib/engine/feature-hasher";

describe("Feature Hasher", () => {
  describe("hashFeature", () => {
    it("returns deterministic index and sign", () => {
      const [idx1, sign1] = hashFeature("hello");
      const [idx2, sign2] = hashFeature("hello");
      expect(idx1).toBe(idx2);
      expect(sign1).toBe(sign2);
    });

    it("index is within bounds", () => {
      const words = ["hello", "world", "test", "machine", "learning", "intent", "classify"];
      for (const word of words) {
        const [idx] = hashFeature(word, DEFAULT_HASH_DIM);
        expect(idx).toBeGreaterThanOrEqual(0);
        expect(idx).toBeLessThan(DEFAULT_HASH_DIM);
      }
    });

    it("sign is +1 or -1", () => {
      const [, sign] = hashFeature("test");
      expect([1, -1]).toContain(sign);
    });

    it("different features hash to different indices (mostly)", () => {
      const indices = new Set<number>();
      const words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];
      for (const w of words) {
        indices.add(hashFeature(w)[0]);
      }
      // With 1024 buckets and 8 words, collision probability is low
      expect(indices.size).toBeGreaterThanOrEqual(6);
    });

    it("respects custom dimensions", () => {
      const [idx] = hashFeature("test", 64);
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(64);
    });
  });

  describe("hashFeaturesDense", () => {
    it("returns Float32Array of correct length", () => {
      const vec = hashFeaturesDense(["hello", "world"]);
      expect(vec).toBeInstanceOf(Float32Array);
      expect(vec.length).toBe(DEFAULT_HASH_DIM);
    });

    it("accumulates repeated features", () => {
      const vec = hashFeaturesDense(["hello", "hello", "hello"]);
      const nonZero = Array.from(vec).filter((v) => v !== 0);
      expect(nonZero.length).toBeGreaterThanOrEqual(1);
    });

    it("empty features produce zero vector", () => {
      const vec = hashFeaturesDense([]);
      expect(Array.from(vec).every((v) => v === 0)).toBe(true);
    });
  });

  describe("hashFeaturesTF (L2 normalized)", () => {
    it("produces L2-normalized vectors", () => {
      const vec = hashFeaturesTF(["hello", "world", "test"]);
      let norm = 0;
      for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
      expect(Math.abs(norm - 1.0)).toBeLessThan(0.001);
    });

    it("empty features produce zero vector", () => {
      const vec = hashFeaturesTF([]);
      expect(Array.from(vec).every((v) => v === 0)).toBe(true);
    });
  });

  describe("hashFeaturesSparse", () => {
    it("returns Map with entries", () => {
      const sparse = hashFeaturesSparse(["hello", "world"]);
      expect(sparse).toBeInstanceOf(Map);
      expect(sparse.size).toBeGreaterThanOrEqual(1);
    });
  });

  describe("dotProduct", () => {
    it("computes correct dot product", () => {
      const a = new Float32Array([1, 2, 3, 4]);
      const b = new Float32Array([4, 3, 2, 1]);
      expect(dotProduct(a, b)).toBe(20); // 4+6+6+4
    });

    it("zero vector produces zero", () => {
      const a = new Float32Array([1, 2, 3]);
      const zero = new Float32Array([0, 0, 0]);
      expect(dotProduct(a, zero)).toBe(0);
    });

    it("handles large vectors efficiently", () => {
      const n = 1024;
      const a = new Float32Array(n).fill(1);
      const b = new Float32Array(n).fill(1);
      expect(dotProduct(a, b)).toBe(n);
    });
  });

  describe("quantization", () => {
    it("roundtrip preserves approximate values", () => {
      const original = new Float32Array([0.1, -0.5, 0.8, 0.0, -0.3]);
      const { data, scale, zeroPoint } = quantizeToInt8(original);
      const restored = dequantizeFromInt8(data, scale, zeroPoint);

      for (let i = 0; i < original.length; i++) {
        expect(Math.abs(original[i] - restored[i])).toBeLessThan(0.02);
      }
    });

    it("quantized values are in int8 range", () => {
      const original = new Float32Array([0.1, -0.5, 0.8, 0.0, -0.3]);
      const { data } = quantizeToInt8(original);
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(-128);
        expect(data[i]).toBeLessThanOrEqual(127);
      }
    });
  });
});
