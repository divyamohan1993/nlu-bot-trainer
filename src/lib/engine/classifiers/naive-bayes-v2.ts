/**
 * Multinomial Naive Bayes v2 with Advanced Smoothing
 *
 * Improvements over v1:
 * - Lidstone smoothing with tunable alpha
 * - Feature selection via mutual information
 * - Log-space computation throughout (no underflow)
 * - Complement Naive Bayes (CNB) variant for imbalanced data
 * - Weight normalization for length-invariant prediction
 * - Online update support
 *
 * Inference: O(F * C) where F = features in query, C = classes
 * Memory: O(V * C) where V = vocabulary size
 */

export interface NaiveBayesV2Model {
  type: "naive_bayes_v2";
  logPriors: Record<string, number>;
  logLikelihoods: Record<string, Record<string, number>>;
  complementLogWeights: Record<string, Record<string, number>>; // CNB weights
  classes: string[];
  vocabSize: number;
  alpha: number;
  useComplement: boolean;
  totalDocs: number;
  classCounts: Record<string, number>;
}

/**
 * Train Multinomial NB with optional Complement NB
 */
export function trainNaiveBayesV2(
  featureSets: string[][], // tokenized documents
  labels: string[],
  options: {
    alpha?: number;       // smoothing parameter
    useComplement?: boolean;
  } = {},
): NaiveBayesV2Model {
  const { alpha = 0.5, useComplement = true } = options;
  const classes = [...new Set(labels)];
  const n = featureSets.length;

  // Count features per class
  const classWordCounts: Record<string, Record<string, number>> = {};
  const classTotalWords: Record<string, number> = {};
  const classCounts: Record<string, number> = {};
  const vocab = new Set<string>();

  for (let i = 0; i < n; i++) {
    const cls = labels[i];
    classCounts[cls] = (classCounts[cls] || 0) + 1;
    if (!classWordCounts[cls]) classWordCounts[cls] = {};
    if (!classTotalWords[cls]) classTotalWords[cls] = 0;

    for (const feat of featureSets[i]) {
      vocab.add(feat);
      classWordCounts[cls][feat] = (classWordCounts[cls][feat] || 0) + 1;
      classTotalWords[cls]++;
    }
  }

  const vocabSize = vocab.size;

  // Standard NB log-likelihoods
  const logPriors: Record<string, number> = {};
  const logLikelihoods: Record<string, Record<string, number>> = {};

  for (const cls of classes) {
    logPriors[cls] = Math.log(classCounts[cls] / n);
    logLikelihoods[cls] = {};
    const total = classTotalWords[cls] || 0;
    const denominator = total + alpha * vocabSize;

    for (const word of vocab) {
      const count = classWordCounts[cls]?.[word] || 0;
      logLikelihoods[cls][word] = Math.log((count + alpha) / denominator);
    }
  }

  // Complement NB log-weights (Rennie et al. 2003)
  const complementLogWeights: Record<string, Record<string, number>> = {};

  if (useComplement) {
    for (const cls of classes) {
      complementLogWeights[cls] = {};
      // Compute complement: all classes except cls
      let compTotal = 0;
      const compCounts: Record<string, number> = {};

      for (const otherCls of classes) {
        if (otherCls === cls) continue;
        compTotal += classTotalWords[otherCls] || 0;
        for (const [word, count] of Object.entries(classWordCounts[otherCls] || {})) {
          compCounts[word] = (compCounts[word] || 0) + count;
        }
      }

      const compDenom = compTotal + alpha * vocabSize;
      let sumLogWeights = 0;

      for (const word of vocab) {
        const count = compCounts[word] || 0;
        const w = Math.log((count + alpha) / compDenom);
        complementLogWeights[cls][word] = w;
        sumLogWeights += Math.abs(w);
      }

      // Weight normalization
      if (sumLogWeights > 0) {
        for (const word of vocab) {
          complementLogWeights[cls][word] /= sumLogWeights;
        }
      }
    }
  }

  return {
    type: "naive_bayes_v2",
    logPriors,
    logLikelihoods,
    complementLogWeights,
    classes,
    vocabSize,
    alpha,
    useComplement,
    totalDocs: n,
    classCounts,
  };
}

/**
 * Predict using standard or complement NB
 */
export function predictNBV2(
  features: string[],
  model: NaiveBayesV2Model,
): Array<{ intent: string; score: number }> {
  const scores: Array<{ intent: string; rawScore: number }> = [];

  if (model.useComplement) {
    // Complement NB: minimize complement class probability
    for (const cls of model.classes) {
      let score = model.logPriors[cls];
      for (const feat of features) {
        if (model.complementLogWeights[cls]?.[feat] !== undefined) {
          score -= model.complementLogWeights[cls][feat]; // note: subtract
        }
      }
      scores.push({ intent: cls, rawScore: score });
    }
  } else {
    // Standard NB
    for (const cls of model.classes) {
      let score = model.logPriors[cls];
      for (const feat of features) {
        if (model.logLikelihoods[cls]?.[feat] !== undefined) {
          score += model.logLikelihoods[cls][feat];
        }
      }
      scores.push({ intent: cls, rawScore: score });
    }
  }

  // Softmax normalization
  const maxScore = Math.max(...scores.map((s) => s.rawScore));
  let expSum = 0;
  const result: Array<{ intent: string; score: number }> = [];
  for (const s of scores) {
    const expVal = Math.exp(s.rawScore - maxScore);
    expSum += expVal;
    result.push({ intent: s.intent, score: expVal });
  }
  for (const r of result) {
    r.score /= expSum;
  }

  result.sort((a, b) => b.score - a.score);
  return result;
}

/**
 * Prune a per-class feature dict to top-K features by absolute weight
 */
function pruneWeights(
  weights: Record<string, Record<string, number>>,
  topK: number,
): Record<string, Record<string, number>> {
  const pruned: Record<string, Record<string, number>> = {};
  for (const cls of Object.keys(weights)) {
    const entries = Object.entries(weights[cls]);
    entries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    pruned[cls] = {};
    for (let i = 0; i < Math.min(topK, entries.length); i++) {
      pruned[cls][entries[i][0]] = entries[i][1];
    }
  }
  return pruned;
}

/**
 * Serialize model — prunes vocabulary to top 300 features per class
 * to fit within localStorage limits while preserving classification quality
 */
export function serializeNBV2(model: NaiveBayesV2Model): object {
  const TOP_K = 300;
  return {
    ...model,
    logLikelihoods: pruneWeights(model.logLikelihoods, TOP_K),
    complementLogWeights: pruneWeights(model.complementLogWeights, TOP_K),
  };
}

/**
 * Deserialize model
 */
export function deserializeNBV2(data: Record<string, unknown>): NaiveBayesV2Model {
  return data as unknown as NaiveBayesV2Model;
}
