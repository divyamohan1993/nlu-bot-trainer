import { describe, it, expect, beforeAll } from "vitest";
import {
  trainEnsemble,
  predictEnsemble,
  crossValidateEnsemble,
  serializeEnsemble,
  deserializeEnsemble,
  type EnsembleModel,
} from "@/lib/engine/ensemble";

// Minimal training data for fast tests
const TRAIN_DATA = [
  { text: "hello", intent: "greet" },
  { text: "hi there", intent: "greet" },
  { text: "good morning", intent: "greet" },
  { text: "hey how are you", intent: "greet" },
  { text: "howdy", intent: "greet" },
  { text: "goodbye", intent: "farewell" },
  { text: "bye bye", intent: "farewell" },
  { text: "see you later", intent: "farewell" },
  { text: "take care", intent: "farewell" },
  { text: "see ya", intent: "farewell" },
  { text: "track my order", intent: "order_status" },
  { text: "where is my package", intent: "order_status" },
  { text: "order tracking info", intent: "order_status" },
  { text: "when will my order arrive", intent: "order_status" },
  { text: "delivery status please", intent: "order_status" },
];

let model: EnsembleModel;

beforeAll(() => {
  model = trainEnsemble(TRAIN_DATA, { learnWeights: false });
});

describe("Ensemble Model", () => {
  describe("trainEnsemble", () => {
    it("trains successfully with valid data", () => {
      expect(model).toBeDefined();
      expect(model.type).toBe("ensemble_v2");
    });

    it("identifies all classes", () => {
      expect(model.classes.sort()).toEqual(["farewell", "greet", "order_status"]);
    });

    it("has 5 meta-weights", () => {
      expect(model.metaWeights).toHaveLength(5);
    });

    it("meta-weights sum to ~1", () => {
      const sum = model.metaWeights.reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1.0)).toBeLessThan(0.01);
    });

    it("records training metrics", () => {
      expect(model.metrics.accuracy).toBeGreaterThan(0);
      expect(model.metrics.totalExamples).toBe(TRAIN_DATA.length);
      expect(model.metrics.trainingTimeMs).toBeGreaterThan(0);
      expect(model.metrics.vocabularySize).toBeGreaterThan(0);
    });

    it("throws on insufficient data", () => {
      expect(() => trainEnsemble([{ text: "hi", intent: "greet" }])).toThrow();
    });

    it("throws on single class", () => {
      expect(() =>
        trainEnsemble([
          { text: "hi", intent: "greet" },
          { text: "hello", intent: "greet" },
        ])
      ).toThrow();
    });
  });

  describe("predictEnsemble", () => {
    it("returns intent and confidence", () => {
      const result = predictEnsemble("hello", model);
      expect(result.intent).toBeTruthy();
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    });

    it("returns ranked results for all classes", () => {
      const result = predictEnsemble("hi", model);
      expect(result.ranking).toHaveLength(3);
    });

    it("ranking is sorted by confidence (descending)", () => {
      const result = predictEnsemble("track my package", model);
      for (let i = 1; i < result.ranking.length; i++) {
        expect(result.ranking[i - 1].confidence).toBeGreaterThanOrEqual(
          result.ranking[i].confidence
        );
      }
    });

    it("returns per-model scores", () => {
      const result = predictEnsemble("hello", model);
      expect(result.perModelScores.logReg).toBeDefined();
      expect(result.perModelScores.naiveBayes).toBeDefined();
      expect(result.perModelScores.svm).toBeDefined();
      expect(result.perModelScores.mlp).toBeDefined();
      expect(result.perModelScores.gradBoost).toBeDefined();
    });

    it("records inference time", () => {
      const result = predictEnsemble("hello", model);
      expect(result.inferenceTimeUs).toBeGreaterThan(0);
    });

    it("greet examples classify as greet", () => {
      const result = predictEnsemble("hello", model);
      expect(result.intent).toBe("greet");
    });

    it("order examples classify as order_status", () => {
      const result = predictEnsemble("where is my order", model);
      expect(result.intent).toBe("order_status");
    });
  });

  describe("serialization roundtrip", () => {
    it("serializes and deserializes correctly", () => {
      const serialized = serializeEnsemble(model);
      expect(typeof serialized).toBe("string");

      const restored = deserializeEnsemble(serialized);
      expect(restored.type).toBe(model.type);
      expect(restored.classes).toEqual(model.classes);
      expect(restored.metaWeights).toEqual(model.metaWeights);
    });

    it("restored model produces same predictions", () => {
      const serialized = serializeEnsemble(model);
      const restored = deserializeEnsemble(serialized);

      const original = predictEnsemble("hello", model);
      const fromRestored = predictEnsemble("hello", restored);

      expect(fromRestored.intent).toBe(original.intent);
      expect(Math.abs(fromRestored.confidence - original.confidence)).toBeLessThan(0.001);
    });
  });

  describe("crossValidateEnsemble", () => {
    it("returns accuracy and per-class F1", () => {
      const result = crossValidateEnsemble(TRAIN_DATA, 3);
      expect(result.accuracy).toBeGreaterThanOrEqual(0);
      expect(result.accuracy).toBeLessThanOrEqual(1);
      expect(Object.keys(result.perClassF1).length).toBeGreaterThan(0);
    });
  });
});
