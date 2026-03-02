import { describe, it, expect } from "vitest";
import {
  trainEnsemble,
  predictEnsemble,
  serializeEnsemble,
  deserializeEnsemble,
} from "@/lib/engine/ensemble";
import { detectOOS } from "@/lib/engine/oos-detector";
import { learnTemperature, calibrateConfidence } from "@/lib/engine/confidence-calibration";
import { explainPrediction } from "@/lib/engine/explainer";
import { validateDataQuality } from "@/lib/engine/data-quality";

// Realistic customer support training data
const TRAIN_DATA = [
  // greet
  { text: "hello", intent: "greet" },
  { text: "hi there", intent: "greet" },
  { text: "good morning", intent: "greet" },
  { text: "hey how are you", intent: "greet" },
  { text: "howdy partner", intent: "greet" },
  { text: "hi I need help", intent: "greet" },
  { text: "good afternoon", intent: "greet" },
  // farewell
  { text: "goodbye", intent: "farewell" },
  { text: "bye bye", intent: "farewell" },
  { text: "see you later", intent: "farewell" },
  { text: "take care", intent: "farewell" },
  { text: "thanks bye", intent: "farewell" },
  { text: "have a nice day", intent: "farewell" },
  { text: "catch you later", intent: "farewell" },
  // order_status
  { text: "track my order", intent: "order_status" },
  { text: "where is my package", intent: "order_status" },
  { text: "order tracking info", intent: "order_status" },
  { text: "when will my order arrive", intent: "order_status" },
  { text: "delivery status please", intent: "order_status" },
  { text: "has my order shipped", intent: "order_status" },
  { text: "shipping update", intent: "order_status" },
];

describe("Full Pipeline Integration", () => {
  it("train → predict → serialize → deserialize → predict roundtrip", { timeout: 30000 }, () => {
    // Train
    const model = trainEnsemble(TRAIN_DATA);
    expect(model.metrics.accuracy).toBeGreaterThan(0.5);

    // Predict
    const result = predictEnsemble("hello there", model);
    expect(result.intent).toBe("greet");
    expect(result.confidence).toBeGreaterThan(0);

    // Serialize → Deserialize
    const serialized = serializeEnsemble(model);
    const restored = deserializeEnsemble(serialized);

    // Predict with restored model
    const result2 = predictEnsemble("hello there", restored);
    expect(result2.intent).toBe(result.intent);
    expect(Math.abs(result2.confidence - result.confidence)).toBeLessThan(0.001);
  });

  it("train → OOS detection pipeline", () => {
    const model = trainEnsemble(TRAIN_DATA, { learnWeights: false });

    // In-scope should not be flagged
    const inScope = detectOOS("track my order", model);
    expect(inScope.fallbackIntent).toBe("order_status");

    // Out-of-scope should have higher score
    const oos = detectOOS("the mitochondria is the powerhouse of the cell", model);
    expect(oos.oosScore).toBeGreaterThan(inScope.oosScore);
  });

  it("train → calibration pipeline", () => {
    const model = trainEnsemble(TRAIN_DATA, { learnWeights: false });

    const cal = learnTemperature(model, TRAIN_DATA);
    expect(cal.temperature).toBeGreaterThan(0);
    expect(cal.eceBefore).toBeGreaterThanOrEqual(0);

    // Apply calibration to a prediction
    const result = predictEnsemble("hello", model);
    const calibrated = calibrateConfidence(result.ranking, cal.temperature);
    const sum = calibrated.reduce((s, r) => s + r.confidence, 0);
    expect(Math.abs(sum - 1.0)).toBeLessThan(0.02);
  });

  it("train → explanation pipeline", () => {
    const model = trainEnsemble(TRAIN_DATA, { learnWeights: false });

    const explanation = explainPrediction("track my order", model);
    expect(explanation.predictedIntent).toBeTruthy();
    expect(explanation.tokenExplanations.length).toBe(3); // 3 words
    expect(explanation.summary).toContain("Classified as");
  });

  it("data quality → train → evaluate pipeline", () => {
    // Quality check first
    const quality = validateDataQuality(TRAIN_DATA);
    expect(quality.score).toBeGreaterThan(0);
    expect(quality.stats.totalExamples).toBe(TRAIN_DATA.length);

    // Then train
    const model = trainEnsemble(TRAIN_DATA, { learnWeights: false });

    // Quality check with model (enables mislabel detection)
    const qualityWithModel = validateDataQuality(TRAIN_DATA, model);
    expect(qualityWithModel.stats.totalIntents).toBe(3);
  });
});
