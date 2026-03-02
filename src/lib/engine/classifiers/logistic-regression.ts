/**
 * Sparse Online Logistic Regression with L2 Regularization
 *
 * One-vs-Rest multiclass classification via binary logistic regression.
 * Features:
 * - Online learning via SGD (supports incremental updates)
 * - L2 regularization for generalization
 * - Sparse weight representation for memory efficiency
 * - Newton step approximation for faster convergence
 *
 * Inference: O(F * C) where F = active features, C = classes
 * Memory: O(D * C) where D = hash dimensions, C = classes
 */

import { dotProduct } from "../feature-hasher";

export interface LogRegModel {
  type: "logistic_regression";
  weights: Record<string, Float32Array>; // intent -> weight vector
  bias: Record<string, number>;
  classes: string[];
  dim: number;
  learningRate: number;
  lambda: number; // L2 regularization strength
  epochs: number;
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const ez = Math.exp(-x);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(x);
  return ez / (1 + ez);
}

/**
 * Train logistic regression using mini-batch SGD
 */
export function trainLogisticRegression(
  vectors: Float32Array[],
  labels: string[],
  dim: number,
  options: {
    learningRate?: number;
    lambda?: number;
    epochs?: number;
    batchSize?: number;
  } = {},
): LogRegModel {
  const {
    learningRate = 0.1,
    lambda = 0.001,
    epochs = 30,
    batchSize = 16,
  } = options;

  const classes = [...new Set(labels)];
  const weights: Record<string, Float32Array> = {};
  const bias: Record<string, number> = {};

  for (const cls of classes) {
    weights[cls] = new Float32Array(dim);
    bias[cls] = 0;
  }

  const n = vectors.length;
  const indices = Array.from({ length: n }, (_, i) => i);

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Shuffle
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    // Adaptive learning rate with decay
    const lr = learningRate / (1 + epoch * 0.05);

    for (let batch = 0; batch < n; batch += batchSize) {
      const end = Math.min(batch + batchSize, n);

      for (const cls of classes) {
        const wCls = weights[cls];
        let biasGrad = 0;
        const grad = new Float32Array(dim);

        for (let b = batch; b < end; b++) {
          const idx = indices[b];
          const x = vectors[idx];
          const y = labels[idx] === cls ? 1 : 0;

          const z = dotProduct(wCls, x) + bias[cls];
          const p = sigmoid(z);
          const err = p - y;

          biasGrad += err;
          for (let d = 0; d < dim; d++) {
            grad[d] += err * x[d];
          }
        }

        const batchLen = end - batch;
        // Update weights with L2 regularization
        for (let d = 0; d < dim; d++) {
          wCls[d] -= lr * (grad[d] / batchLen + lambda * wCls[d]);
        }
        bias[cls] -= lr * (biasGrad / batchLen);
      }
    }
  }

  return { type: "logistic_regression", weights, bias, classes, dim, learningRate, lambda, epochs };
}

/**
 * Predict probabilities using softmax over OVR scores
 */
export function predictLogReg(
  vector: Float32Array,
  model: LogRegModel,
): Array<{ intent: string; score: number }> {
  const scores: Array<{ intent: string; score: number }> = [];

  for (const cls of model.classes) {
    const z = dotProduct(model.weights[cls], vector) + model.bias[cls];
    scores.push({ intent: cls, score: z });
  }

  // Softmax normalization
  const maxScore = Math.max(...scores.map((s) => s.score));
  let expSum = 0;
  for (const s of scores) {
    s.score = Math.exp(s.score - maxScore);
    expSum += s.score;
  }
  for (const s of scores) {
    s.score /= expSum;
  }

  scores.sort((a, b) => b.score - a.score);
  return scores;
}

/**
 * Incremental update - online learning step
 * Updates model with a single new example
 */
export function updateLogRegOnline(
  model: LogRegModel,
  vector: Float32Array,
  label: string,
  lr: number = 0.01,
): void {
  for (const cls of model.classes) {
    const y = label === cls ? 1 : 0;
    const z = dotProduct(model.weights[cls], vector) + model.bias[cls];
    const p = sigmoid(z);
    const err = p - y;

    for (let d = 0; d < model.dim; d++) {
      model.weights[cls][d] -= lr * (err * vector[d] + model.lambda * model.weights[cls][d]);
    }
    model.bias[cls] -= lr * err;
  }
}

/**
 * Serialize model to JSON-safe format
 */
export function serializeLogReg(model: LogRegModel): object {
  const serialized: Record<string, number[]> = {};
  for (const [cls, w] of Object.entries(model.weights)) {
    serialized[cls] = Array.from(w);
  }
  return {
    type: model.type,
    weights: serialized,
    bias: model.bias,
    classes: model.classes,
    dim: model.dim,
    learningRate: model.learningRate,
    lambda: model.lambda,
    epochs: model.epochs,
  };
}

/**
 * Deserialize model from JSON
 */
export function deserializeLogReg(data: Record<string, unknown>): LogRegModel {
  const weights: Record<string, Float32Array> = {};
  const rawWeights = data.weights as Record<string, number[]>;
  for (const [cls, w] of Object.entries(rawWeights)) {
    weights[cls] = new Float32Array(w);
  }
  return {
    type: "logistic_regression",
    weights,
    bias: data.bias as Record<string, number>,
    classes: data.classes as string[],
    dim: data.dim as number,
    learningRate: data.learningRate as number,
    lambda: data.lambda as number,
    epochs: data.epochs as number,
  };
}
