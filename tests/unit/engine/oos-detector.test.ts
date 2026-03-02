import { describe, it, expect, beforeAll } from "vitest";
import { trainEnsemble, type EnsembleModel } from "@/lib/engine/ensemble";
import { detectOOS, calibrateOOS, evaluateOOS, DEFAULT_OOS_CONFIG } from "@/lib/engine/oos-detector";

const TRAIN_DATA = [
  { text: "hello", intent: "greet" },
  { text: "hi there", intent: "greet" },
  { text: "good morning", intent: "greet" },
  { text: "hey how are you", intent: "greet" },
  { text: "howdy partner", intent: "greet" },
  { text: "goodbye", intent: "farewell" },
  { text: "bye bye", intent: "farewell" },
  { text: "see you later", intent: "farewell" },
  { text: "take care friend", intent: "farewell" },
  { text: "catch you later", intent: "farewell" },
  { text: "track my order", intent: "order_status" },
  { text: "where is my package", intent: "order_status" },
  { text: "order tracking info", intent: "order_status" },
  { text: "when will my order arrive", intent: "order_status" },
  { text: "shipping status update", intent: "order_status" },
];

let model: EnsembleModel;

beforeAll(() => {
  model = trainEnsemble(TRAIN_DATA, { learnWeights: false });
});

describe("OOS Detector", () => {
  describe("detectOOS", () => {
    it("returns OOS result for in-scope text", () => {
      const result = detectOOS("hello", model);
      expect(result).toHaveProperty("isOOS");
      expect(result).toHaveProperty("oosScore");
      expect(result).toHaveProperty("signals");
      expect(result).toHaveProperty("fallbackIntent");
    });

    it("computes all signal values", () => {
      const result = detectOOS("hello", model);
      expect(result.signals.entropy).toBeGreaterThanOrEqual(0);
      expect(result.signals.normalizedEntropy).toBeGreaterThanOrEqual(0);
      expect(result.signals.normalizedEntropy).toBeLessThanOrEqual(1);
      expect(result.signals.maxProbability).toBeGreaterThan(0);
      expect(result.signals.maxProbability).toBeLessThanOrEqual(1);
    });

    it("oosScore is between 0 and 1", () => {
      const result = detectOOS("hello", model);
      expect(result.oosScore).toBeGreaterThanOrEqual(0);
      expect(result.oosScore).toBeLessThanOrEqual(1);
    });

    it("gibberish has higher OOS score than in-scope", () => {
      const inScope = detectOOS("track my order", model);
      const gibberish = detectOOS("xyzzy flurbo quantum spatula", model);
      expect(gibberish.oosScore).toBeGreaterThan(inScope.oosScore);
    });

    it("provides fallback intent even when OOS", () => {
      const result = detectOOS("something totally random and weird", model);
      expect(result.fallbackIntent).toBeTruthy();
    });
  });

  describe("calibrateOOS", () => {
    it("returns calibrated config", () => {
      const config = calibrateOOS(model, TRAIN_DATA, 0.95);
      expect(config.fusedThreshold).toBeGreaterThan(0);
      expect(config.entropyThreshold).toBeGreaterThan(0);
      expect(config.maxProbThreshold).toBeGreaterThan(0);
    });

    it("handles small datasets gracefully", () => {
      const config = calibrateOOS(model, TRAIN_DATA.slice(0, 3), 0.95);
      expect(config).toBeDefined();
    });
  });

  describe("evaluateOOS", () => {
    it("returns evaluation metrics", () => {
      const inScope = [{ text: "hello" }, { text: "goodbye" }, { text: "track order" }];
      const oos = [{ text: "quantum physics is interesting" }, { text: "the mitochondria is the powerhouse" }];

      const result = evaluateOOS(model, inScope, oos);
      expect(result.auroc).toBeGreaterThanOrEqual(0);
      expect(result.auroc).toBeLessThanOrEqual(1);
      expect(result.f1).toBeGreaterThanOrEqual(0);
      expect(result.inScopeAcceptRate).toBeGreaterThanOrEqual(0);
      expect(result.oosRejectRate).toBeGreaterThanOrEqual(0);
    });
  });
});
