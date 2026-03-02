/**
 * Out-of-Scope (OOS) Detection
 *
 * Detects inputs that don't belong to any known intent class.
 * Critical for production chatbots that receive gibberish, adversarial probes,
 * or genuinely novel queries.
 *
 * Research basis:
 * - Hendrycks & Gimpel (2017) "A Baseline for Detecting Misclassified and
 *   Out-of-Distribution Examples in Neural Networks" — entropy-based OOD detection
 * - Larson et al. (2019) "An Evaluation Dataset for Intent Classification and
 *   Out-of-Scope Prediction" — CLINC150 OOS benchmark
 *
 * Approach: Multi-signal fusion
 * 1. Prediction entropy (Shannon entropy of class distribution)
 * 2. Max-probability threshold (softmax confidence ceiling)
 * 3. Margin gap (distance between top-1 and top-2 predictions)
 * 4. Committee disagreement (ensemble member vote spread)
 *
 * Calibration: Threshold learned on held-out in-scope data via percentile method
 */

import { type EnsembleModel, predictEnsemble } from "./ensemble";

export interface OOSConfig {
  /** Entropy threshold — above this, input is OOS. Default auto-calibrated. */
  entropyThreshold: number;
  /** Max-probability threshold — below this, input is OOS */
  maxProbThreshold: number;
  /** Margin threshold — below this, model is too confused */
  marginThreshold: number;
  /** Weight for each signal in fusion. [entropy, maxProb, margin, committee] */
  signalWeights: [number, number, number, number];
  /** Final fused score threshold — above this, input is OOS */
  fusedThreshold: number;
}

export const DEFAULT_OOS_CONFIG: OOSConfig = {
  entropyThreshold: 2.0,
  maxProbThreshold: 0.3,
  marginThreshold: 0.1,
  fusedThreshold: 0.5,
  signalWeights: [0.35, 0.30, 0.20, 0.15],
};

export interface OOSResult {
  /** Whether the input is detected as out-of-scope */
  isOOS: boolean;
  /** Fused OOS score (0 = definitely in-scope, 1 = definitely OOS) */
  oosScore: number;
  /** Individual signal values */
  signals: {
    entropy: number;
    normalizedEntropy: number;
    maxProbability: number;
    marginGap: number;
    committeeDisagreement: number;
  };
  /** The prediction (even if OOS — useful for fallback) */
  fallbackIntent: string;
  fallbackConfidence: number;
}

/**
 * Compute Shannon entropy of a probability distribution
 * H(p) = -sum(p_i * log2(p_i))
 */
function shannonEntropy(probs: number[]): number {
  let h = 0;
  for (const p of probs) {
    if (p > 1e-10) h -= p * Math.log2(p);
  }
  return h;
}

/**
 * Compute committee disagreement from per-model predictions
 * Uses vote entropy: H(votes) where votes are counts of top-1 picks
 */
function computeCommitteeDisagreement(
  perModelScores: {
    logReg: Array<{ intent: string; score: number }>;
    naiveBayes: Array<{ intent: string; score: number }>;
    svm: Array<{ intent: string; score: number }>;
    mlp: Array<{ intent: string; score: number }>;
    gradBoost: Array<{ intent: string; score: number }>;
  },
): number {
  const topPicks = [
    perModelScores.logReg[0]?.intent,
    perModelScores.naiveBayes[0]?.intent,
    perModelScores.svm[0]?.intent,
    perModelScores.mlp[0]?.intent,
    perModelScores.gradBoost[0]?.intent,
  ].filter(Boolean);

  const votes: Record<string, number> = {};
  for (const intent of topPicks) {
    votes[intent] = (votes[intent] || 0) + 1;
  }

  // Vote entropy (normalized to [0, 1])
  const maxEntropy = Math.log2(topPicks.length); // max when all disagree
  if (maxEntropy === 0) return 0;

  let entropy = 0;
  for (const count of Object.values(votes)) {
    const p = count / topPicks.length;
    if (p > 0) entropy -= p * Math.log2(p);
  }

  return entropy / maxEntropy; // normalize to [0, 1]
}

/**
 * Detect if an input is out-of-scope
 *
 * @param text - Input text to classify
 * @param model - Trained ensemble model
 * @param config - OOS detection configuration
 * @returns OOS detection result with fused score and individual signals
 */
export function detectOOS(
  text: string,
  model: EnsembleModel,
  config: OOSConfig = DEFAULT_OOS_CONFIG,
): OOSResult {
  const prediction = predictEnsemble(text, model);
  const numClasses = model.classes.length;
  const maxEntropy = Math.log2(numClasses); // uniform distribution entropy

  // Signal 1: Prediction entropy
  const probs = prediction.ranking.map((r) => r.confidence);
  const entropy = shannonEntropy(probs);
  const normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0;

  // Signal 2: Max probability
  const maxProb = prediction.confidence;

  // Signal 3: Margin gap (top1 - top2)
  const marginGap = prediction.ranking.length >= 2
    ? prediction.ranking[0].confidence - prediction.ranking[1].confidence
    : prediction.confidence;

  // Signal 4: Committee disagreement
  const committeeDisagreement = computeCommitteeDisagreement(prediction.perModelScores);

  // Fuse signals into OOS score
  // Each signal contributes: higher = more likely OOS
  const entropySignal = normalizedEntropy;
  const maxProbSignal = 1 - maxProb; // invert: low prob = high OOS
  const marginSignal = 1 - marginGap; // invert: small margin = high OOS
  const committeeSignal = committeeDisagreement;

  const [wE, wP, wM, wC] = config.signalWeights;
  const oosScore = wE * entropySignal + wP * maxProbSignal + wM * marginSignal + wC * committeeSignal;

  // Multi-gate OOS decision
  const isOOS = oosScore >= config.fusedThreshold
    || entropy >= config.entropyThreshold
    || maxProb < config.maxProbThreshold;

  return {
    isOOS,
    oosScore,
    signals: {
      entropy,
      normalizedEntropy,
      maxProbability: maxProb,
      marginGap,
      committeeDisagreement,
    },
    fallbackIntent: prediction.intent,
    fallbackConfidence: prediction.confidence,
  };
}

/**
 * Calibrate OOS thresholds on in-scope validation data
 *
 * Uses percentile-based calibration (Hendrycks & Gimpel 2017):
 * Set threshold at the (1 - target_recall)th percentile of in-scope OOS scores,
 * ensuring that target_recall% of in-scope data is correctly classified.
 *
 * @param model - Trained ensemble model
 * @param inScopeData - Held-out in-scope validation examples
 * @param targetRecall - What fraction of in-scope should be correctly kept (default 0.95)
 * @returns Calibrated OOS configuration
 */
export function calibrateOOS(
  model: EnsembleModel,
  inScopeData: Array<{ text: string; intent: string }>,
  targetRecall: number = 0.95,
): OOSConfig {
  const config = { ...DEFAULT_OOS_CONFIG };

  if (inScopeData.length < 10) return config;

  // Compute OOS scores for all in-scope examples
  const scores: number[] = [];
  const entropies: number[] = [];
  const maxProbs: number[] = [];

  for (const item of inScopeData) {
    const result = detectOOS(item.text, model, config);
    scores.push(result.oosScore);
    entropies.push(result.signals.entropy);
    maxProbs.push(result.signals.maxProbability);
  }

  // Sort and pick percentile thresholds
  scores.sort((a, b) => a - b);
  entropies.sort((a, b) => a - b);
  maxProbs.sort((a, b) => a - b);

  const percentileIdx = Math.floor(scores.length * targetRecall);

  // Set thresholds so that targetRecall% of in-scope is kept
  config.fusedThreshold = scores[Math.min(percentileIdx, scores.length - 1)] + 0.01;
  config.entropyThreshold = entropies[Math.min(percentileIdx, entropies.length - 1)] + 0.1;
  config.maxProbThreshold = maxProbs[Math.max(0, scores.length - percentileIdx - 1)] - 0.01;

  // Clamp to reasonable ranges
  config.fusedThreshold = Math.max(0.3, Math.min(0.9, config.fusedThreshold));
  config.entropyThreshold = Math.max(1.0, Math.min(4.0, config.entropyThreshold));
  config.maxProbThreshold = Math.max(0.05, Math.min(0.5, config.maxProbThreshold));

  return config;
}

/**
 * Evaluate OOS detection performance
 * Computes AUROC approximation and F1 for OOS class
 */
export function evaluateOOS(
  model: EnsembleModel,
  inScopeData: Array<{ text: string }>,
  oosData: Array<{ text: string }>,
  config: OOSConfig = DEFAULT_OOS_CONFIG,
): {
  auroc: number;
  f1: number;
  precision: number;
  recall: number;
  inScopeAcceptRate: number;
  oosRejectRate: number;
} {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  const inScoreList: number[] = [];
  const oosScoreList: number[] = [];

  for (const item of inScopeData) {
    const result = detectOOS(item.text, model, config);
    inScoreList.push(result.oosScore);
    if (result.isOOS) fp++; // false positive: in-scope flagged as OOS
    else tn++; // true negative: in-scope correctly kept
  }

  for (const item of oosData) {
    const result = detectOOS(item.text, model, config);
    oosScoreList.push(result.oosScore);
    if (result.isOOS) tp++; // true positive: OOS correctly rejected
    else fn++; // false negative: OOS missed
  }

  const precision = tp / (tp + fp || 1);
  const recall = tp / (tp + fn || 1);
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  // AUROC approximation via Mann-Whitney U statistic
  let u = 0;
  for (const os of oosScoreList) {
    for (const is_ of inScoreList) {
      if (os > is_) u++;
      else if (os === is_) u += 0.5;
    }
  }
  const auroc = (oosScoreList.length > 0 && inScoreList.length > 0)
    ? u / (oosScoreList.length * inScoreList.length)
    : 0;

  return {
    auroc,
    f1,
    precision,
    recall,
    inScopeAcceptRate: tn / (tn + fp || 1),
    oosRejectRate: tp / (tp + fn || 1),
  };
}
