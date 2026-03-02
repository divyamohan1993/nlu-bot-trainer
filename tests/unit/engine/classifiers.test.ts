import { describe, it, expect } from "vitest";
import { hashFeaturesTF } from "@/lib/engine/feature-hasher";
import { tokenizeV2, DEFAULT_TOKENIZER_CONFIG } from "@/lib/engine/tokenizer-v2";
import { trainLogisticRegression, predictLogReg } from "@/lib/engine/classifiers/logistic-regression";
import { trainNaiveBayesV2, predictNBV2 } from "@/lib/engine/classifiers/naive-bayes-v2";
import { trainSVM, predictSVM } from "@/lib/engine/classifiers/svm";
import { trainMLP, predictMLP } from "@/lib/engine/classifiers/mlp";
import { trainGradientBoost, predictGradientBoost } from "@/lib/engine/classifiers/gradient-boost";

// Shared training data
const texts = [
  "hello", "hi there", "good morning", "hey", "howdy",
  "goodbye", "bye bye", "see you later", "take care", "farewell",
];
const labels = [
  "greet", "greet", "greet", "greet", "greet",
  "farewell", "farewell", "farewell", "farewell", "farewell",
];
const dim = 1024;

const featureSets = texts.map((t) => tokenizeV2(t, DEFAULT_TOKENIZER_CONFIG));
const vectors = featureSets.map((f) => hashFeaturesTF(f, dim));

describe("Individual Classifiers", () => {
  describe("Logistic Regression", () => {
    it("trains and predicts", () => {
      const model = trainLogisticRegression(vectors, labels, dim, { epochs: 10 });
      const result = predictLogReg(vectors[0], model);
      expect(result.length).toBe(2);
      expect(result[0].intent).toBeTruthy();
      expect(result[0].score).toBeGreaterThan(0);
    });

    it("scores sum to ~1", () => {
      const model = trainLogisticRegression(vectors, labels, dim);
      const result = predictLogReg(vectors[0], model);
      const sum = result.reduce((s, r) => s + r.score, 0);
      expect(Math.abs(sum - 1.0)).toBeLessThan(0.01);
    });
  });

  describe("Naive Bayes V2", () => {
    it("trains and predicts", () => {
      const model = trainNaiveBayesV2(featureSets, labels);
      const result = predictNBV2(featureSets[0], model);
      expect(result.length).toBe(2);
      expect(result[0].score).toBeGreaterThan(0);
    });
  });

  describe("SVM (Pegasos)", () => {
    it("trains and predicts", () => {
      const model = trainSVM(vectors, labels, dim, { epochs: 5 });
      const result = predictSVM(vectors[0], model);
      expect(result.length).toBe(2);
      expect(result[0].score).toBeGreaterThan(0);
    });

    it("softmax scores sum to ~1", () => {
      const model = trainSVM(vectors, labels, dim);
      const result = predictSVM(vectors[0], model);
      const sum = result.reduce((s, r) => s + r.score, 0);
      expect(Math.abs(sum - 1.0)).toBeLessThan(0.01);
    });

    it("does not produce NaN scores (overflow protection)", () => {
      const model = trainSVM(vectors, labels, dim, { epochs: 50 }); // more epochs = larger margins
      const result = predictSVM(vectors[0], model);
      for (const r of result) {
        expect(isNaN(r.score)).toBe(false);
        expect(isFinite(r.score)).toBe(true);
      }
    });
  });

  describe("MLP Neural Network", () => {
    it("trains and predicts", () => {
      const model = trainMLP(vectors, labels, dim, { hiddenDim: 32, epochs: 10 });
      const result = predictMLP(vectors[0], model);
      expect(result.length).toBe(2);
      expect(result[0].score).toBeGreaterThan(0);
    });
  });

  describe("Gradient Boosted Stumps", () => {
    it("trains and predicts", () => {
      const model = trainGradientBoost(vectors, labels, dim, { rounds: 20 });
      const result = predictGradientBoost(vectors[0], model);
      expect(result.length).toBe(2);
      expect(result[0].score).toBeGreaterThan(0);
    });
  });
});
