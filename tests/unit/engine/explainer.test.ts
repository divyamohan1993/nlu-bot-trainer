import { describe, it, expect, beforeAll } from "vitest";
import { trainEnsemble, type EnsembleModel } from "@/lib/engine/ensemble";
import { explainPrediction, explainMisclassification } from "@/lib/engine/explainer";

const TRAIN_DATA = [
  { text: "hello", intent: "greet" }, { text: "hi there", intent: "greet" },
  { text: "good morning", intent: "greet" }, { text: "hey", intent: "greet" },
  { text: "howdy", intent: "greet" },
  { text: "goodbye", intent: "farewell" }, { text: "bye bye", intent: "farewell" },
  { text: "see you later", intent: "farewell" }, { text: "take care", intent: "farewell" },
  { text: "farewell friend", intent: "farewell" },
  { text: "track my order", intent: "order_status" }, { text: "where is my package", intent: "order_status" },
  { text: "order tracking", intent: "order_status" }, { text: "delivery status", intent: "order_status" },
  { text: "shipping update", intent: "order_status" },
];

let model: EnsembleModel;

beforeAll(() => {
  model = trainEnsemble(TRAIN_DATA, { learnWeights: false });
});

describe("Explainer", () => {
  describe("explainPrediction", () => {
    it("returns explanation for text", () => {
      const result = explainPrediction("hello there", model);
      expect(result.text).toBe("hello there");
      expect(result.predictedIntent).toBeTruthy();
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.tokenExplanations.length).toBeGreaterThan(0);
    });

    it("identifies supporting tokens", () => {
      const result = explainPrediction("track my order", model);
      expect(result.supportingTokens.length).toBeGreaterThan(0);
    });

    it("generates human-readable summary", () => {
      const result = explainPrediction("hello", model);
      expect(result.summary).toContain("Classified as");
    });

    it("respects topK parameter", () => {
      const result = explainPrediction("track my order please now", model, 2);
      expect(result.tokenExplanations.length).toBeLessThanOrEqual(2);
    });

    it("handles empty string", () => {
      const result = explainPrediction("", model);
      expect(result.tokenExplanations).toHaveLength(0);
    });

    it("normalized contributions are in [-1, 1]", () => {
      const result = explainPrediction("track my order", model);
      for (const t of result.tokenExplanations) {
        expect(t.normalizedContribution).toBeGreaterThanOrEqual(-1);
        expect(t.normalizedContribution).toBeLessThanOrEqual(1);
      }
    });

    it("records computation time", () => {
      const result = explainPrediction("hello", model);
      expect(result.computeTimeMs).toBeGreaterThanOrEqual(0);
    });
  });

  describe("explainMisclassification", () => {
    it("explains a potential misclassification", () => {
      const result = explainMisclassification("hello", "farewell", model);
      expect(result.explanation).toBeDefined();
      expect(result.trueIntentRank).toBeGreaterThan(0);
      expect(result.trueIntentConfidence).toBeGreaterThanOrEqual(0);
    });
  });
});
