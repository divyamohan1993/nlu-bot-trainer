/**
 * Linear SVM (Support Vector Machine) via Pegasos SGD
 *
 * Pegasos: Primal Estimated sub-GrAdient SOlver for SVM (Shalev-Shwartz et al. 2007)
 * - One-vs-Rest multiclass via binary SVMs
 * - O(1/t) convergence rate (optimal for SVMs)
 * - Sparse-friendly: only updates active support vectors
 * - Projection step keeps weights bounded (regularization)
 *
 * Inference: O(D * C) where D = dimensions, C = classes
 * Training: O(T * D * C) where T = epochs * n
 */

import { dotProduct } from "../feature-hasher";

export interface SVMModel {
  type: "svm";
  weights: Record<string, Float32Array>;
  bias: Record<string, number>;
  classes: string[];
  dim: number;
  lambda: number;
  epochs: number;
}

/**
 * Train linear SVM using Pegasos algorithm
 */
export function trainSVM(
  vectors: Float32Array[],
  labels: string[],
  dim: number,
  options: {
    lambda?: number;
    epochs?: number;
  } = {},
): SVMModel {
  const { lambda = 0.001, epochs = 20 } = options;
  const classes = [...new Set(labels)];
  const n = vectors.length;
  const weights: Record<string, Float32Array> = {};
  const bias: Record<string, number> = {};

  for (const cls of classes) {
    weights[cls] = new Float32Array(dim);
    bias[cls] = 0;
  }

  const indices = Array.from({ length: n }, (_, i) => i);
  let t = 1;

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Shuffle
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    for (let b = 0; b < n; b++) {
      const idx = indices[b];
      const x = vectors[idx];
      const eta = 1 / (lambda * t); // Pegasos learning rate

      for (const cls of classes) {
        const y = labels[idx] === cls ? 1 : -1;
        const w = weights[cls];
        const margin = y * (dotProduct(w, x) + bias[cls]);

        // Hinge loss: max(0, 1 - y * w^T x)
        if (margin < 1) {
          // Misclassified or in margin - update
          for (let d = 0; d < dim; d++) {
            w[d] = (1 - eta * lambda) * w[d] + eta * y * x[d];
          }
          bias[cls] += eta * y * 0.01; // Small bias learning rate
        } else {
          // Correctly classified - just regularize
          for (let d = 0; d < dim; d++) {
            w[d] *= (1 - eta * lambda);
          }
        }

        // Pegasos projection step: ||w|| <= 1/sqrt(lambda)
        let normSq = 0;
        for (let d = 0; d < dim; d++) normSq += w[d] * w[d];
        const maxNorm = 1 / Math.sqrt(lambda);
        if (normSq > maxNorm * maxNorm) {
          const scale = maxNorm / Math.sqrt(normSq);
          for (let d = 0; d < dim; d++) w[d] *= scale;
        }
      }
      t++;
    }
  }

  return { type: "svm", weights, bias, classes, dim, lambda, epochs };
}

/**
 * Predict using SVM with Platt scaling for probability estimates
 */
export function predictSVM(
  vector: Float32Array,
  model: SVMModel,
): Array<{ intent: string; score: number }> {
  const rawScores: Array<{ intent: string; score: number }> = [];

  for (const cls of model.classes) {
    const score = dotProduct(model.weights[cls], vector) + model.bias[cls];
    rawScores.push({ intent: cls, score });
  }

  // Numerically stable softmax (subtract max to prevent overflow)
  let maxScore = -Infinity;
  for (const s of rawScores) {
    if (s.score > maxScore) maxScore = s.score;
  }
  let expSum = 0;
  for (const s of rawScores) {
    s.score = Math.exp(s.score - maxScore);
    expSum += s.score;
  }
  for (const s of rawScores) {
    s.score /= expSum;
  }

  rawScores.sort((a, b) => b.score - a.score);
  return rawScores;
}

/**
 * Serialize SVM model
 */
export function serializeSVM(model: SVMModel): object {
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
    lambda: model.lambda,
    epochs: model.epochs,
  };
}

export function deserializeSVM(data: Record<string, unknown>): SVMModel {
  const weights: Record<string, Float32Array> = {};
  const raw = data.weights as Record<string, number[]>;
  for (const [cls, w] of Object.entries(raw)) {
    weights[cls] = new Float32Array(w);
  }
  return {
    type: "svm",
    weights,
    bias: data.bias as Record<string, number>,
    classes: data.classes as string[],
    dim: data.dim as number,
    lambda: data.lambda as number,
    epochs: data.epochs as number,
  };
}
