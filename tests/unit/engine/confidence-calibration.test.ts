import { describe, it, expect, beforeAll } from "vitest";
import { trainEnsemble, type EnsembleModel } from "@/lib/engine/ensemble";
import {
  applyTemperature,
  learnTemperature,
  calibrateConfidence,
} from "@/lib/engine/confidence-calibration";

const TRAIN_DATA = [
  { text: "hello", intent: "greet" }, { text: "hi there", intent: "greet" },
  { text: "good morning", intent: "greet" }, { text: "hey", intent: "greet" },
  { text: "howdy", intent: "greet" },
  { text: "goodbye", intent: "farewell" }, { text: "bye bye", intent: "farewell" },
  { text: "see you later", intent: "farewell" }, { text: "take care", intent: "farewell" },
  { text: "see ya", intent: "farewell" },
  { text: "track my order", intent: "order_status" }, { text: "where is my package", intent: "order_status" },
  { text: "order tracking", intent: "order_status" }, { text: "delivery status", intent: "order_status" },
  { text: "shipping update", intent: "order_status" },
];

let model: EnsembleModel;

beforeAll(() => {
  model = trainEnsemble(TRAIN_DATA, { learnWeights: false });
});

describe("Confidence Calibration", () => {
  describe("applyTemperature", () => {
    it("T=1 returns identical scores", () => {
      const scores = [
        { intent: "a", score: 0.7 },
        { intent: "b", score: 0.2 },
        { intent: "c", score: 0.1 },
      ];
      const result = applyTemperature(scores, 1.0);
      for (let i = 0; i < scores.length; i++) {
        expect(Math.abs(result[i].score - scores[i].score)).toBeLessThan(0.01);
      }
    });

    it("T>1 makes distribution more uniform", () => {
      const scores = [
        { intent: "a", score: 0.9 },
        { intent: "b", score: 0.08 },
        { intent: "c", score: 0.02 },
      ];
      const result = applyTemperature(scores, 3.0);
      // Max should be lower, min should be higher
      expect(result[0].score).toBeLessThan(0.9);
    });

    it("T<1 makes distribution more peaked", () => {
      const scores = [
        { intent: "a", score: 0.5 },
        { intent: "b", score: 0.3 },
        { intent: "c", score: 0.2 },
      ];
      const result = applyTemperature(scores, 0.5);
      expect(result[0].score).toBeGreaterThan(0.5);
    });

    it("output sums to ~1", () => {
      const scores = [
        { intent: "a", score: 0.6 },
        { intent: "b", score: 0.3 },
        { intent: "c", score: 0.1 },
      ];
      for (const t of [0.5, 1.0, 2.0, 5.0]) {
        const result = applyTemperature(scores, t);
        const sum = result.reduce((s, r) => s + r.score, 0);
        expect(Math.abs(sum - 1.0)).toBeLessThan(0.01);
      }
    });

    it("handles empty scores", () => {
      const result = applyTemperature([], 2.0);
      expect(result).toHaveLength(0);
    });
  });

  describe("learnTemperature", () => {
    it("returns valid calibration result", () => {
      const result = learnTemperature(model, TRAIN_DATA);
      expect(result.temperature).toBeGreaterThan(0);
      expect(result.eceBefore).toBeGreaterThanOrEqual(0);
      expect(result.eceAfter).toBeGreaterThanOrEqual(0);
      expect(result.numExamples).toBe(TRAIN_DATA.length);
    });

    it("temperature is positive", () => {
      const result = learnTemperature(model, TRAIN_DATA);
      expect(result.temperature).toBeGreaterThan(0);
    });

    it("produces reliability diagram bins", () => {
      const result = learnTemperature(model, TRAIN_DATA);
      expect(result.reliabilityBefore.length).toBe(10);
      expect(result.reliabilityAfter.length).toBe(10);
    });

    it("handles small dataset", () => {
      const result = learnTemperature(model, TRAIN_DATA.slice(0, 3));
      expect(result.temperature).toBe(1.0); // fallback
    });
  });

  describe("calibrateConfidence", () => {
    it("adjusts ranking confidences", () => {
      const ranking = [
        { name: "a", confidence: 0.8 },
        { name: "b", confidence: 0.15 },
        { name: "c", confidence: 0.05 },
      ];
      const result = calibrateConfidence(ranking, 2.0);
      expect(result).toHaveLength(3);
      const sum = result.reduce((s, r) => s + r.confidence, 0);
      expect(Math.abs(sum - 1.0)).toBeLessThan(0.01);
    });
  });
});
