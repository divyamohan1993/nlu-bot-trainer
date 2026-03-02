/**
 * Confidence Calibration via Temperature Scaling
 *
 * Raw neural network / ensemble confidences are typically poorly calibrated:
 * a prediction of "90% confident" might only be correct 70% of the time.
 * Temperature scaling is a simple, effective post-hoc calibration method.
 *
 * Research basis:
 * - Guo et al. (2017) "On Calibration of Modern Neural Networks"
 *   Showed that modern networks are miscalibrated and temperature scaling
 *   is a single-parameter fix that preserves accuracy while improving calibration.
 * - Niculescu-Mizil & Caruana (2005) "Predicting Good Probabilities with Supervised Learning"
 *   Platt scaling and isotonic regression baselines.
 *
 * Method:
 * 1. Collect predicted probabilities on a held-out validation set
 * 2. Optimize a single temperature T to minimize NLL on validation set
 * 3. At inference, divide logits by T before softmax
 *    - T > 1: softens probabilities (more uniform → less overconfident)
 *    - T < 1: sharpens probabilities (more peaked → more confident)
 *    - T = 1: no change (uncalibrated baseline)
 *
 * Evaluation:
 * - ECE (Expected Calibration Error): weighted average of |accuracy - confidence| per bin
 * - MCE (Maximum Calibration Error): worst-case bin miscalibration
 * - Reliability diagram: visual tool showing calibration curve vs diagonal
 */

import { type EnsembleModel, predictEnsemble } from "./ensemble";

export interface CalibrationConfig {
  /** Temperature parameter (T > 1 softens, T < 1 sharpens) */
  temperature: number;
  /** Number of bins for ECE/reliability diagram */
  numBins: number;
}

export const DEFAULT_CALIBRATION_CONFIG: CalibrationConfig = {
  temperature: 1.0,
  numBins: 10,
};

export interface CalibrationResult {
  /** Optimized temperature */
  temperature: number;
  /** Expected Calibration Error before calibration */
  eceBefore: number;
  /** Expected Calibration Error after calibration */
  eceAfter: number;
  /** Maximum Calibration Error before */
  mceBefore: number;
  /** Maximum Calibration Error after */
  mceAfter: number;
  /** Reliability diagram data (for visualization) */
  reliabilityBefore: ReliabilityBin[];
  reliabilityAfter: ReliabilityBin[];
  /** Number of validation examples used */
  numExamples: number;
}

export interface ReliabilityBin {
  /** Bin center (e.g., 0.05, 0.15, ..., 0.95) */
  binCenter: number;
  /** Average confidence in this bin */
  avgConfidence: number;
  /** Actual accuracy in this bin */
  avgAccuracy: number;
  /** Number of predictions in this bin */
  count: number;
}

/**
 * Apply temperature scaling to a set of scores
 * Divides log-probabilities by T, then re-normalizes via softmax
 */
export function applyTemperature(
  scores: Array<{ intent: string; score: number }>,
  temperature: number,
): Array<{ intent: string; score: number }> {
  if (temperature === 1.0 || scores.length === 0) return scores;

  // Convert probabilities to logits (inverse softmax)
  const logits = scores.map((s) => ({
    intent: s.intent,
    logit: Math.log(Math.max(s.score, 1e-10)),
  }));

  // Scale by temperature
  const scaledLogits = logits.map((l) => l.logit / temperature);

  // Numerically stable softmax
  const maxLogit = Math.max(...scaledLogits);
  const exps = scaledLogits.map((l) => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);

  return logits.map((l, i) => ({
    intent: l.intent,
    score: exps[i] / sumExps,
  }));
}

/**
 * Compute reliability diagram bins
 */
function computeReliabilityBins(
  predictions: Array<{ confidence: number; correct: boolean }>,
  numBins: number,
): ReliabilityBin[] {
  const bins: ReliabilityBin[] = [];

  for (let b = 0; b < numBins; b++) {
    const lower = b / numBins;
    const upper = (b + 1) / numBins;
    const binCenter = (lower + upper) / 2;

    const inBin = predictions.filter(
      (p) => p.confidence >= lower && p.confidence < upper
    );

    if (inBin.length === 0) {
      bins.push({ binCenter, avgConfidence: binCenter, avgAccuracy: binCenter, count: 0 });
      continue;
    }

    const avgConfidence = inBin.reduce((s, p) => s + p.confidence, 0) / inBin.length;
    const avgAccuracy = inBin.filter((p) => p.correct).length / inBin.length;

    bins.push({ binCenter, avgConfidence, avgAccuracy, count: inBin.length });
  }

  return bins;
}

/**
 * Compute Expected Calibration Error (ECE)
 * ECE = sum over bins of (bin_weight * |accuracy - confidence|)
 */
function computeECE(bins: ReliabilityBin[], totalPredictions: number): number {
  let ece = 0;
  for (const bin of bins) {
    if (bin.count === 0) continue;
    const weight = bin.count / totalPredictions;
    ece += weight * Math.abs(bin.avgAccuracy - bin.avgConfidence);
  }
  return ece;
}

/**
 * Compute Maximum Calibration Error (MCE)
 */
function computeMCE(bins: ReliabilityBin[]): number {
  let mce = 0;
  for (const bin of bins) {
    if (bin.count === 0) continue;
    mce = Math.max(mce, Math.abs(bin.avgAccuracy - bin.avgConfidence));
  }
  return mce;
}

/**
 * Compute Negative Log-Likelihood (NLL) — the objective for temperature optimization
 */
function computeNLL(
  predictions: Array<{ scores: Array<{ intent: string; score: number }>; trueLabel: string }>,
  temperature: number,
): number {
  let nll = 0;
  for (const pred of predictions) {
    const calibrated = applyTemperature(pred.scores, temperature);
    const trueScore = calibrated.find((s) => s.intent === pred.trueLabel)?.score || 1e-10;
    nll -= Math.log(Math.max(trueScore, 1e-10));
  }
  return nll / predictions.length;
}

/**
 * Learn optimal temperature via grid search on validation NLL
 *
 * Grid search is used instead of gradient descent because:
 * 1. Single parameter — grid search is exact and simple
 * 2. NLL is convex in T for a single parameter (Guo et al. 2017)
 * 3. No learning rate tuning needed
 *
 * @param model - Trained ensemble model
 * @param validationData - Held-out validation examples with true labels
 * @returns Calibrated configuration with optimal temperature
 */
export function learnTemperature(
  model: EnsembleModel,
  validationData: Array<{ text: string; intent: string }>,
): CalibrationResult {
  const numBins = 10;

  if (validationData.length < 5) {
    return {
      temperature: 1.0,
      eceBefore: 0,
      eceAfter: 0,
      mceBefore: 0,
      mceAfter: 0,
      reliabilityBefore: [],
      reliabilityAfter: [],
      numExamples: 0,
    };
  }

  // Collect raw predictions
  const rawPredictions: Array<{
    scores: Array<{ intent: string; score: number }>;
    trueLabel: string;
    confidence: number;
    correct: boolean;
  }> = [];

  for (const item of validationData) {
    const result = predictEnsemble(item.text, model);
    rawPredictions.push({
      scores: result.ranking.map((r) => ({ intent: r.name, score: r.confidence })),
      trueLabel: item.intent,
      confidence: result.confidence,
      correct: result.intent === item.intent,
    });
  }

  // Before calibration metrics
  const reliabilityBefore = computeReliabilityBins(rawPredictions, numBins);
  const eceBefore = computeECE(reliabilityBefore, rawPredictions.length);
  const mceBefore = computeMCE(reliabilityBefore);

  // Grid search for optimal temperature
  // Search range [0.1, 5.0] with fine granularity
  let bestT = 1.0;
  let bestNLL = computeNLL(rawPredictions, 1.0);

  // Coarse search
  for (let t = 0.1; t <= 5.0; t += 0.1) {
    const nll = computeNLL(rawPredictions, t);
    if (nll < bestNLL) {
      bestNLL = nll;
      bestT = t;
    }
  }

  // Fine search around best
  const fineStart = Math.max(0.05, bestT - 0.15);
  const fineEnd = bestT + 0.15;
  for (let t = fineStart; t <= fineEnd; t += 0.01) {
    const nll = computeNLL(rawPredictions, t);
    if (nll < bestNLL) {
      bestNLL = nll;
      bestT = t;
    }
  }

  bestT = Math.round(bestT * 100) / 100; // round to 2 decimal places

  // After calibration metrics
  const calibratedPredictions = rawPredictions.map((pred) => {
    const calibrated = applyTemperature(pred.scores, bestT);
    const topScore = calibrated[0]?.score || 0;
    return {
      confidence: topScore,
      correct: pred.correct,
    };
  });

  const reliabilityAfter = computeReliabilityBins(calibratedPredictions, numBins);
  const eceAfter = computeECE(reliabilityAfter, calibratedPredictions.length);
  const mceAfter = computeMCE(reliabilityAfter);

  return {
    temperature: bestT,
    eceBefore,
    eceAfter,
    mceBefore,
    mceAfter,
    reliabilityBefore,
    reliabilityAfter,
    numExamples: validationData.length,
  };
}

/**
 * Apply calibration to a prediction result
 * Returns calibrated ranking with adjusted confidence values
 */
export function calibrateConfidence(
  ranking: Array<{ name: string; confidence: number }>,
  temperature: number,
): Array<{ name: string; confidence: number }> {
  const scores = ranking.map((r) => ({ intent: r.name, score: r.confidence }));
  const calibrated = applyTemperature(scores, temperature);
  return calibrated.map((s) => ({ name: s.intent, confidence: s.score }));
}
