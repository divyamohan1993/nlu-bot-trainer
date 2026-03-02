/**
 * Gradient Boosted Decision Stumps for NLU
 *
 * Lightweight gradient boosting with:
 * - Decision stumps (depth-1 trees) as weak learners
 * - Feature-hashed input for O(1) feature access
 * - One-vs-Rest multiclass via binary boosting
 * - Shrinkage regularization
 * - Feature sampling per round (random subspace)
 *
 * Extremely fast inference: just N threshold comparisons per class
 * N = number of boosting rounds (typically 50-200)
 *
 * Inference: O(R * C) where R = rounds, C = classes
 * Memory: O(R * C * 3) - each stump stores (feature_idx, threshold, weight)
 */

export interface Stump {
  featureIdx: number;
  threshold: number;
  leftWeight: number;  // weight if feature <= threshold
  rightWeight: number; // weight if feature > threshold
}

export interface GradientBoostModel {
  type: "gradient_boost";
  stumps: Record<string, Stump[]>; // class -> array of stumps
  classes: string[];
  dim: number;
  rounds: number;
  shrinkage: number;
}

/**
 * Find best stump for a set of weighted residuals
 */
function findBestStump(
  vectors: Float32Array[],
  residuals: Float64Array,
  sampleIndices: number[],
  featureSubset: number[],
): Stump {
  let bestGain = -Infinity;
  let bestStump: Stump = { featureIdx: 0, threshold: 0, leftWeight: 0, rightWeight: 0 };

  for (const featIdx of featureSubset) {
    // Collect feature values
    const values: Array<{ val: number; idx: number }> = [];
    for (const i of sampleIndices) {
      values.push({ val: vectors[i][featIdx], idx: i });
    }
    values.sort((a, b) => a.val - b.val);

    // Scan thresholds
    let leftSum = 0, leftCount = 0;
    let rightSum = 0, rightCount = sampleIndices.length;
    for (const { idx } of values) {
      rightSum += residuals[idx];
    }

    for (let v = 0; v < values.length - 1; v++) {
      const { val, idx } = values[v];
      leftSum += residuals[idx];
      leftCount++;
      rightSum -= residuals[idx];
      rightCount--;

      if (values[v + 1].val === val) continue; // skip ties

      // Gain = left_sum^2/left_count + right_sum^2/right_count
      const gain = (leftSum * leftSum) / leftCount + (rightSum * rightSum) / rightCount;

      if (gain > bestGain) {
        bestGain = gain;
        const threshold = (val + values[v + 1].val) / 2;
        bestStump = {
          featureIdx: featIdx,
          threshold,
          leftWeight: leftSum / leftCount,
          rightWeight: rightSum / rightCount,
        };
      }
    }
  }

  return bestStump;
}

/**
 * Train gradient boosted stumps
 */
export function trainGradientBoost(
  vectors: Float32Array[],
  labels: string[],
  dim: number,
  options: {
    rounds?: number;
    shrinkage?: number;
    featureSampleRatio?: number;
    subsampleRatio?: number;
  } = {},
): GradientBoostModel {
  const {
    rounds = 100,
    shrinkage = 0.1,
    featureSampleRatio = 0.5,
    subsampleRatio = 0.8,
  } = options;

  const classes = [...new Set(labels)];
  const n = vectors.length;
  const stumps: Record<string, Stump[]> = {};
  const featureCount = Math.max(1, Math.floor(dim * featureSampleRatio));
  const sampleCount = Math.max(1, Math.floor(n * subsampleRatio));

  for (const cls of classes) {
    stumps[cls] = [];
    const predictions = new Float64Array(n);
    const residuals = new Float64Array(n);

    for (let round = 0; round < rounds; round++) {
      // Compute residuals (negative gradient of log-loss)
      for (let i = 0; i < n; i++) {
        const y = labels[i] === cls ? 1 : 0;
        const p = 1 / (1 + Math.exp(-predictions[i]));
        residuals[i] = y - p;
      }

      // Random feature subset
      const featureSubset: number[] = [];
      const allFeatures = Array.from({ length: dim }, (_, i) => i);
      for (let i = allFeatures.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [allFeatures[i], allFeatures[j]] = [allFeatures[j], allFeatures[i]];
      }
      for (let i = 0; i < featureCount; i++) featureSubset.push(allFeatures[i]);

      // Random sample subset
      const sampleIndices: number[] = [];
      const allIndices = Array.from({ length: n }, (_, i) => i);
      for (let i = allIndices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [allIndices[i], allIndices[j]] = [allIndices[j], allIndices[i]];
      }
      for (let i = 0; i < sampleCount; i++) sampleIndices.push(allIndices[i]);

      const stump = findBestStump(vectors, residuals, sampleIndices, featureSubset);
      stump.leftWeight *= shrinkage;
      stump.rightWeight *= shrinkage;
      stumps[cls].push(stump);

      // Update predictions
      for (let i = 0; i < n; i++) {
        if (vectors[i][stump.featureIdx] <= stump.threshold) {
          predictions[i] += stump.leftWeight;
        } else {
          predictions[i] += stump.rightWeight;
        }
      }
    }
  }

  return { type: "gradient_boost", stumps, classes, dim, rounds, shrinkage };
}

/**
 * Predict with gradient boosted stumps
 * Extremely fast: just threshold comparisons
 */
export function predictGradientBoost(
  vector: Float32Array,
  model: GradientBoostModel,
): Array<{ intent: string; score: number }> {
  const rawScores: Array<{ intent: string; score: number }> = [];

  for (const cls of model.classes) {
    let score = 0;
    const clsStumps = model.stumps[cls];
    for (let i = 0; i < clsStumps.length; i++) {
      const s = clsStumps[i];
      score += vector[s.featureIdx] <= s.threshold ? s.leftWeight : s.rightWeight;
    }
    rawScores.push({ intent: cls, score });
  }

  // Softmax normalization on raw log-odds scores
  const maxScore = Math.max(...rawScores.map((s) => s.score));
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
 * Serialize/deserialize
 */
export function serializeGBoost(model: GradientBoostModel): object {
  return { ...model };
}

export function deserializeGBoost(data: Record<string, unknown>): GradientBoostModel {
  return data as unknown as GradientBoostModel;
}
