/**
 * Stacking Ensemble with Learned Meta-Weights
 *
 * Combines multiple diverse classifiers through learned stacking:
 * 1. Logistic Regression (linear decision boundaries)
 * 2. Complement Naive Bayes (probability-based, great for NLU)
 * 3. Linear SVM (maximum margin)
 * 4. Multi-Layer Perceptron (non-linear neural network)
 * 5. Gradient Boosted Stumps (non-linear interactions)
 *
 * Total: ~171K parameters across 5 classifiers
 *
 * Meta-learner: Weighted average with cross-validated weights
 * Fallback: Majority voting if meta-learning fails
 *
 * Diversity principle: ensemble members should make different errors
 * We achieve this by:
 * - Different model families (linear, probabilistic, margin, neural, tree ensemble)
 * - Different feature views (dense hashed vs raw features)
 * - Different regularization strategies
 */

import { hashFeaturesTF, DEFAULT_HASH_DIM, type dotProduct } from "./feature-hasher";
import { tokenizeV2, DEFAULT_TOKENIZER_CONFIG, type TokenizerConfig } from "./tokenizer-v2";
import {
  trainLogisticRegression, predictLogReg,
  serializeLogReg, deserializeLogReg,
  type LogRegModel,
} from "./classifiers/logistic-regression";
import {
  trainNaiveBayesV2, predictNBV2,
  serializeNBV2, deserializeNBV2,
  type NaiveBayesV2Model,
} from "./classifiers/naive-bayes-v2";
import {
  trainSVM, predictSVM,
  serializeSVM, deserializeSVM,
  type SVMModel,
} from "./classifiers/svm";
import {
  trainGradientBoost, predictGradientBoost,
  serializeGBoost, deserializeGBoost,
  type GradientBoostModel,
} from "./classifiers/gradient-boost";
import {
  trainMLP, predictMLP,
  serializeMLP, deserializeMLP,
  type MLPModel,
} from "./classifiers/mlp";

export interface EnsembleModel {
  type: "ensemble_v2";
  logReg: LogRegModel;
  naiveBayes: NaiveBayesV2Model;
  svm: SVMModel;
  mlp: MLPModel;
  gradBoost: GradientBoostModel;
  metaWeights: number[]; // [logReg, nb, svm, mlp, gradBoost]
  classes: string[];
  hashDim: number;
  tokenizerConfig: TokenizerConfig;
  trainedAt: string;
  version: number;
  metrics: {
    accuracy: number;
    perClassF1: Record<string, number>;
    confusionMatrix: Record<string, Record<string, number>>;
    trainingTimeMs: number;
    totalExamples: number;
    vocabularySize: number;
  };
}

interface TrainingItem {
  text: string;
  intent: string;
}

/**
 * Compute per-class F1 scores
 */
function computeF1Scores(
  predictions: string[],
  actuals: string[],
  classes: string[],
): Record<string, number> {
  const f1Scores: Record<string, number> = {};

  for (const cls of classes) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] === cls && actuals[i] === cls) tp++;
      else if (predictions[i] === cls && actuals[i] !== cls) fp++;
      else if (predictions[i] !== cls && actuals[i] === cls) fn++;
    }
    const precision = tp / (tp + fp || 1);
    const recall = tp / (tp + fn || 1);
    f1Scores[cls] = precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : 0;
  }

  return f1Scores;
}

/**
 * Build confusion matrix
 */
function buildConfusionMatrix(
  predictions: string[],
  actuals: string[],
  classes: string[],
): Record<string, Record<string, number>> {
  const matrix: Record<string, Record<string, number>> = {};
  for (const actual of classes) {
    matrix[actual] = {};
    for (const pred of classes) {
      matrix[actual][pred] = 0;
    }
  }
  for (let i = 0; i < predictions.length; i++) {
    if (matrix[actuals[i]]) {
      matrix[actuals[i]][predictions[i]] = (matrix[actuals[i]][predictions[i]] || 0) + 1;
    }
  }
  return matrix;
}

/**
 * Learn meta-weights via cross-validation
 * Uses coordinate descent on validation predictions
 */
function learnMetaWeights(
  data: TrainingItem[],
  hashDim: number,
  tokConfig: TokenizerConfig,
  folds: number = 3,
): number[] {
  const n = data.length;
  if (n < folds * 2) return [0.2, 0.2, 0.2, 0.2, 0.2]; // uniform fallback

  // Stratified fold assignment
  const byClass: Record<string, number[]> = {};
  for (let i = 0; i < n; i++) {
    const cls = data[i].intent;
    if (!byClass[cls]) byClass[cls] = [];
    byClass[cls].push(i);
  }

  const foldAssign = new Int32Array(n);
  for (const indices of Object.values(byClass)) {
    // Shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    indices.forEach((idx, i) => { foldAssign[idx] = i % folds; });
  }

  // Collect out-of-fold predictions from each classifier
  const modelPredictions: Array<Array<Array<{ intent: string; score: number }>>> = [[], [], [], [], []];
  const trueLabels: string[] = [];
  const valIndices: number[] = [];

  for (let fold = 0; fold < folds; fold++) {
    const trainIdx: number[] = [];
    const valIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if (foldAssign[i] === fold) valIdx.push(i);
      else trainIdx.push(i);
    }

    if (trainIdx.length < 2 || valIdx.length < 1) continue;

    const trainTexts = trainIdx.map((i) => data[i].text);
    const trainLabels = trainIdx.map((i) => data[i].intent);
    const trainFeatures = trainTexts.map((t) => tokenizeV2(t, tokConfig));
    const trainVectors = trainFeatures.map((f) => hashFeaturesTF(f, hashDim));

    const uniqueLabels = [...new Set(trainLabels)];
    if (uniqueLabels.length < 2) continue;

    // Train all 5 classifiers on this fold's training data
    const lr = trainLogisticRegression(trainVectors, trainLabels, hashDim, { epochs: 15 });
    const nb = trainNaiveBayesV2(trainFeatures, trainLabels);
    const svm = trainSVM(trainVectors, trainLabels, hashDim, { epochs: 10 });
    const mlpFold = trainMLP(trainVectors, trainLabels, hashDim, { hiddenDim: 128, epochs: 50 });
    const gb = trainGradientBoost(trainVectors, trainLabels, hashDim, { rounds: 150 });

    // Predict on validation
    for (const vi of valIdx) {
      const features = tokenizeV2(data[vi].text, tokConfig);
      const vector = hashFeaturesTF(features, hashDim);

      modelPredictions[0].push(predictLogReg(vector, lr));
      modelPredictions[1].push(predictNBV2(features, nb));
      modelPredictions[2].push(predictSVM(vector, svm));
      modelPredictions[3].push(predictMLP(vector, mlpFold));
      modelPredictions[4].push(predictGradientBoost(vector, gb));

      trueLabels.push(data[vi].intent);
      valIndices.push(vi);
    }
  }

  if (trueLabels.length === 0) return [0.2, 0.2, 0.2, 0.2, 0.2];

  // Grid search for optimal weights using log-likelihood as primary objective
  // Log-likelihood naturally rewards confident correct predictions and penalizes
  // confident wrong predictions — the proper scoring rule for probabilistic classifiers
  // 5 classifiers: LR, NB, SVM, MLP, GBM
  let bestWeights = [0.2, 0.2, 0.2, 0.2, 0.2];
  let bestLogLik = -Infinity;
  const steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

  for (const w0 of steps) {
    for (const w1 of steps) {
      if (1 - w0 - w1 < -0.01) continue;
      for (const w2 of steps) {
        if (1 - w0 - w1 - w2 < -0.01) continue;
        for (const w3 of steps) {
          const w4 = 1 - w0 - w1 - w2 - w3;
          if (w4 < -0.01 || w4 > 1.01) continue;
          const weights = [w0, w1, w2, w3, Math.max(0, w4)];

          let logLik = 0;
          for (let i = 0; i < trueLabels.length; i++) {
            const combined: Record<string, number> = {};
            for (let m = 0; m < 5; m++) {
              for (const pred of modelPredictions[m][i]) {
                combined[pred.intent] = (combined[pred.intent] || 0) + weights[m] * pred.score;
              }
            }
            const trueScore = combined[trueLabels[i]] || 1e-10;
            logLik += Math.log(Math.max(trueScore, 1e-10));
          }

          if (logLik > bestLogLik) {
            bestLogLik = logLik;
            bestWeights = [...weights];
          }
        }
      }
    }
  }

  // Normalize
  const sum = bestWeights.reduce((a, b) => a + b, 0);
  return bestWeights.map((w) => w / (sum || 1));
}

/**
 * Train the full ensemble
 */
export function trainEnsemble(
  data: TrainingItem[],
  options: {
    hashDim?: number;
    tokenizerConfig?: TokenizerConfig;
    learnWeights?: boolean;
  } = {},
): EnsembleModel {
  const startTime = performance.now();
  const {
    hashDim = DEFAULT_HASH_DIM,
    tokenizerConfig = DEFAULT_TOKENIZER_CONFIG,
    learnWeights = true,
  } = options;

  if (data.length < 2) throw new Error("Need at least 2 training examples");
  const classes = [...new Set(data.map((d) => d.intent))];
  if (classes.length < 2) throw new Error("Need at least 2 distinct intents");

  // Feature extraction
  const featureSets = data.map((d) => tokenizeV2(d.text, tokenizerConfig));
  const vectors = featureSets.map((f) => hashFeaturesTF(f, hashDim));
  const labels = data.map((d) => d.intent);

  // Train all classifiers (they're independent)
  const logReg = trainLogisticRegression(vectors, labels, hashDim, { epochs: 15 });
  const naiveBayes = trainNaiveBayesV2(featureSets, labels, { alpha: 0.5, useComplement: true });
  const svm = trainSVM(vectors, labels, hashDim, { epochs: 10, lambda: 0.001 });
  const mlp = trainMLP(vectors, labels, hashDim, { hiddenDim: 128, epochs: 50 });
  const gradBoost = trainGradientBoost(vectors, labels, hashDim, { rounds: 150, shrinkage: 0.1 });

  // Learn meta-weights
  const metaWeights = learnWeights
    ? learnMetaWeights(data, hashDim, tokenizerConfig)
    : [0.2, 0.2, 0.2, 0.2, 0.2];

  // Compute training metrics
  const predictions: string[] = [];
  for (let i = 0; i < data.length; i++) {
    const combined = combineScores(
      predictLogReg(vectors[i], logReg),
      predictNBV2(featureSets[i], naiveBayes),
      predictSVM(vectors[i], svm),
      predictMLP(vectors[i], mlp),
      predictGradientBoost(vectors[i], gradBoost),
      metaWeights,
    );
    predictions.push(combined[0]?.intent || "unknown");
  }

  const correct = predictions.filter((p, i) => p === labels[i]).length;
  const accuracy = correct / labels.length;
  const perClassF1 = computeF1Scores(predictions, labels, classes);
  const confusionMatrix = buildConfusionMatrix(predictions, labels, classes);
  const trainingTimeMs = performance.now() - startTime;

  const vocabSet = new Set<string>();
  for (const fs of featureSets) for (const f of fs) vocabSet.add(f);

  return {
    type: "ensemble_v2",
    logReg,
    naiveBayes,
    svm,
    mlp,
    gradBoost,
    metaWeights,
    classes,
    hashDim,
    tokenizerConfig,
    trainedAt: new Date().toISOString(),
    version: 1,
    metrics: {
      accuracy,
      perClassF1,
      confusionMatrix,
      trainingTimeMs,
      totalExamples: data.length,
      vocabularySize: vocabSet.size,
    },
  };
}

/**
 * Combine scores from multiple classifiers using meta-weights
 */
function combineScores(
  lrScores: Array<{ intent: string; score: number }>,
  nbScores: Array<{ intent: string; score: number }>,
  svmScores: Array<{ intent: string; score: number }>,
  mlpScores: Array<{ intent: string; score: number }>,
  gbScores: Array<{ intent: string; score: number }>,
  weights: number[],
): Array<{ intent: string; score: number }> {
  const combined: Record<string, number> = {};

  for (const s of lrScores) combined[s.intent] = (combined[s.intent] || 0) + weights[0] * s.score;
  for (const s of nbScores) combined[s.intent] = (combined[s.intent] || 0) + weights[1] * s.score;
  for (const s of svmScores) combined[s.intent] = (combined[s.intent] || 0) + weights[2] * s.score;
  for (const s of mlpScores) combined[s.intent] = (combined[s.intent] || 0) + weights[3] * s.score;
  for (const s of gbScores) combined[s.intent] = (combined[s.intent] || 0) + weights[4] * s.score;

  return Object.entries(combined)
    .map(([intent, score]) => ({ intent, score }))
    .sort((a, b) => b.score - a.score);
}

/**
 * Predict with the full ensemble
 * Returns ranked intents with calibrated confidence scores
 */
export function predictEnsemble(
  text: string,
  model: EnsembleModel,
): {
  intent: string;
  confidence: number;
  ranking: Array<{ name: string; confidence: number }>;
  perModelScores: {
    logReg: Array<{ intent: string; score: number }>;
    naiveBayes: Array<{ intent: string; score: number }>;
    svm: Array<{ intent: string; score: number }>;
    mlp: Array<{ intent: string; score: number }>;
    gradBoost: Array<{ intent: string; score: number }>;
  };
  inferenceTimeUs: number;
} {
  const start = performance.now();

  const features = tokenizeV2(text, model.tokenizerConfig);
  const vector = hashFeaturesTF(features, model.hashDim);

  const lrScores = predictLogReg(vector, model.logReg);
  const nbScores = predictNBV2(features, model.naiveBayes);
  const svmScores = predictSVM(vector, model.svm);
  const mlpScores = predictMLP(vector, model.mlp);
  const gbScores = predictGradientBoost(vector, model.gradBoost);

  const combined = combineScores(lrScores, nbScores, svmScores, mlpScores, gbScores, model.metaWeights);

  // Combined scores are already weighted probabilities — use directly
  const ranking = combined.map((s) => ({
    name: s.intent,
    confidence: s.score,
  }));

  const inferenceTimeUs = (performance.now() - start) * 1000; // microseconds

  return {
    intent: ranking[0]?.name || "unknown",
    confidence: ranking[0]?.confidence || 0,
    ranking,
    perModelScores: { logReg: lrScores, naiveBayes: nbScores, svm: svmScores, mlp: mlpScores, gradBoost: gbScores },
    inferenceTimeUs,
  };
}

/**
 * Cross-validate ensemble with stratified folds
 */
export function crossValidateEnsemble(
  data: TrainingItem[],
  folds: number = 3,
  hashDim: number = DEFAULT_HASH_DIM,
): {
  accuracy: number;
  perClassF1: Record<string, number>;
  confusionMatrix: Record<string, Record<string, number>>;
} {
  const n = data.length;
  const classes = [...new Set(data.map((d) => d.intent))];
  if (n < folds) return { accuracy: 0, perClassF1: {}, confusionMatrix: {} };

  // Stratified splits
  const byClass: Record<string, TrainingItem[]> = {};
  for (const item of data) {
    if (!byClass[item.intent]) byClass[item.intent] = [];
    byClass[item.intent].push(item);
  }

  for (const items of Object.values(byClass)) {
    for (let i = items.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
  }

  const foldData: TrainingItem[][] = Array.from({ length: folds }, () => []);
  for (const items of Object.values(byClass)) {
    items.forEach((item, i) => { foldData[i % folds].push(item); });
  }

  const allPredictions: string[] = [];
  const allActuals: string[] = [];

  for (let i = 0; i < folds; i++) {
    const testSet = foldData[i];
    const trainSet = foldData.filter((_, idx) => idx !== i).flat();
    if (trainSet.length < 2 || testSet.length < 1) continue;

    const trainClasses = new Set(trainSet.map((t) => t.intent));
    if (trainClasses.size < 2) continue;

    try {
      const model = trainEnsemble(trainSet, { hashDim, learnWeights: false });
      for (const item of testSet) {
        const result = predictEnsemble(item.text, model);
        allPredictions.push(result.intent);
        allActuals.push(item.intent);
      }
    } catch {
      continue;
    }
  }

  if (allPredictions.length === 0) return { accuracy: 0, perClassF1: {}, confusionMatrix: {} };

  const correct = allPredictions.filter((p, i) => p === allActuals[i]).length;
  return {
    accuracy: correct / allPredictions.length,
    perClassF1: computeF1Scores(allPredictions, allActuals, classes),
    confusionMatrix: buildConfusionMatrix(allPredictions, allActuals, classes),
  };
}

/**
 * Serialize ensemble model
 */
export function serializeEnsemble(model: EnsembleModel): string {
  return JSON.stringify({
    ...model,
    logReg: serializeLogReg(model.logReg),
    naiveBayes: serializeNBV2(model.naiveBayes),
    svm: serializeSVM(model.svm),
    mlp: serializeMLP(model.mlp),
    gradBoost: serializeGBoost(model.gradBoost),
  });
}

/**
 * Deserialize ensemble model
 */
export function deserializeEnsemble(json: string): EnsembleModel {
  const data = JSON.parse(json);
  return {
    ...data,
    logReg: deserializeLogReg(data.logReg),
    naiveBayes: deserializeNBV2(data.naiveBayes),
    svm: deserializeSVM(data.svm),
    mlp: deserializeMLP(data.mlp),
    gradBoost: deserializeGBoost(data.gradBoost),
  };
}
