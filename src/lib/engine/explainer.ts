/**
 * Prediction Explainer — LIME-lite for NLU
 *
 * Explains why the model made a particular classification by identifying
 * which words/features contributed most (positively and negatively).
 *
 * Research basis:
 * - Ribeiro et al. (2016) "Why Should I Trust You? Explaining the Predictions
 *   of Any Classifier" — LIME (Local Interpretable Model-agnostic Explanations)
 * - Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
 *   — SHAP values (we use a simplified leave-one-out approximation)
 *
 * Approach: Leave-One-Out (LOO) feature importance
 * For each word in the input:
 *   1. Remove the word
 *   2. Re-predict on the perturbed input
 *   3. Measure the change in predicted class probability
 *   4. Words that cause the biggest drop are most important
 *
 * This is a simplified LIME that avoids the complexity of:
 * - Random perturbation sampling
 * - Fitting a local linear model
 * - Kernel-based weighting
 *
 * Advantages of LOO over full LIME:
 * - Deterministic (no random sampling)
 * - Exact attribution (no approximation error)
 * - Faster for short texts (<20 words, typical for NLU)
 * - No hyperparameters to tune
 *
 * Disadvantage: O(n) predictions where n = number of words (acceptable for NLU)
 */

import { type EnsembleModel, predictEnsemble } from "./ensemble";

export interface TokenExplanation {
  /** The token/word */
  token: string;
  /** Position in original text */
  position: number;
  /** How much this token contributes to the predicted class (positive = supports) */
  contribution: number;
  /** Normalized contribution [-1, 1] */
  normalizedContribution: number;
  /** What the predicted class would be without this token */
  classWithout: string;
  /** Confidence without this token */
  confidenceWithout: number;
}

export interface ExplanationResult {
  /** Original text */
  text: string;
  /** Predicted intent */
  predictedIntent: string;
  /** Original confidence */
  confidence: number;
  /** Per-token explanations sorted by absolute contribution (highest first) */
  tokenExplanations: TokenExplanation[];
  /** Top contributing tokens (positive influence) */
  supportingTokens: TokenExplanation[];
  /** Top opposing tokens (negative influence or supporting other classes) */
  opposingTokens: TokenExplanation[];
  /** Human-readable explanation string */
  summary: string;
  /** Explanation computation time (ms) */
  computeTimeMs: number;
}

/**
 * Explain a prediction using Leave-One-Out feature importance
 *
 * @param text - Input text to explain
 * @param model - Trained ensemble model
 * @param topK - Number of top features to return (default: all)
 * @returns Explanation with per-token contributions
 */
export function explainPrediction(
  text: string,
  model: EnsembleModel,
  topK?: number,
): ExplanationResult {
  const startTime = performance.now();

  // Get baseline prediction
  const baseline = predictEnsemble(text, model);
  const predictedIntent = baseline.intent;
  const baselineConfidence = baseline.confidence;

  // Tokenize into words (preserving positions)
  const words = text.split(/\s+/).filter((w) => w.length > 0);

  if (words.length === 0) {
    return {
      text,
      predictedIntent,
      confidence: baselineConfidence,
      tokenExplanations: [],
      supportingTokens: [],
      opposingTokens: [],
      summary: "Empty input — no features to explain.",
      computeTimeMs: performance.now() - startTime,
    };
  }

  // Leave-One-Out: remove each word and re-predict
  const explanations: TokenExplanation[] = [];

  for (let i = 0; i < words.length; i++) {
    // Create perturbed text without word i
    const perturbed = [...words.slice(0, i), ...words.slice(i + 1)].join(" ");

    if (perturbed.trim().length === 0) {
      // Removing this word leaves nothing — it's maximally important
      explanations.push({
        token: words[i],
        position: i,
        contribution: baselineConfidence,
        normalizedContribution: 1.0,
        classWithout: "unknown",
        confidenceWithout: 0,
      });
      continue;
    }

    const perturbedResult = predictEnsemble(perturbed, model);

    // Contribution = how much confidence drops when we remove this word
    // For the predicted intent specifically
    const perturbedConfForPredicted = perturbedResult.ranking.find(
      (r) => r.name === predictedIntent
    )?.confidence || 0;

    const contribution = baselineConfidence - perturbedConfForPredicted;

    explanations.push({
      token: words[i],
      position: i,
      contribution,
      normalizedContribution: 0, // computed after all words
      classWithout: perturbedResult.intent,
      confidenceWithout: perturbedResult.confidence,
    });
  }

  // Normalize contributions to [-1, 1]
  const maxAbs = Math.max(...explanations.map((e) => Math.abs(e.contribution)), 1e-10);
  for (const exp of explanations) {
    exp.normalizedContribution = exp.contribution / maxAbs;
  }

  // Sort by absolute contribution
  explanations.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  // Limit to topK if specified
  const limited = topK ? explanations.slice(0, topK) : explanations;

  // Separate supporting vs opposing
  const supportingTokens = limited
    .filter((e) => e.contribution > 0.001)
    .sort((a, b) => b.contribution - a.contribution);

  const opposingTokens = limited
    .filter((e) => e.contribution < -0.001)
    .sort((a, b) => a.contribution - b.contribution);

  // Generate human-readable summary
  const topSupporting = supportingTokens.slice(0, 3).map((t) => `"${t.token}"`);
  const topOpposing = opposingTokens.slice(0, 2).map((t) => `"${t.token}"`);

  let summary = `Classified as "${predictedIntent}" (${(baselineConfidence * 100).toFixed(1)}%).`;
  if (topSupporting.length > 0) {
    summary += ` Key signals: ${topSupporting.join(", ")}.`;
  }
  if (topOpposing.length > 0) {
    summary += ` Opposing signals: ${topOpposing.join(", ")}.`;
  }

  return {
    text,
    predictedIntent,
    confidence: baselineConfidence,
    tokenExplanations: limited,
    supportingTokens,
    opposingTokens,
    summary,
    computeTimeMs: performance.now() - startTime,
  };
}

/**
 * Batch explain multiple predictions
 * Useful for analyzing misclassifications
 */
export function batchExplain(
  texts: string[],
  model: EnsembleModel,
  topK: number = 5,
): ExplanationResult[] {
  return texts.map((text) => explainPrediction(text, model, topK));
}

/**
 * Explain a misclassification
 * Shows which words pushed the prediction away from the true label
 */
export function explainMisclassification(
  text: string,
  trueIntent: string,
  model: EnsembleModel,
): {
  explanation: ExplanationResult;
  trueIntentConfidence: number;
  trueIntentRank: number;
  confusingWords: TokenExplanation[];
} {
  const explanation = explainPrediction(text, model);

  const trueIntentRanking = explanation.confidence > 0
    ? predictEnsemble(text, model).ranking
    : [];

  const trueIntentRank = trueIntentRanking.findIndex((r) => r.name === trueIntent) + 1;
  const trueIntentConfidence = trueIntentRanking.find((r) => r.name === trueIntent)?.confidence || 0;

  // Find words that, when removed, would make the model predict the true intent
  const confusingWords = explanation.tokenExplanations.filter(
    (t) => t.classWithout === trueIntent
  );

  return {
    explanation,
    trueIntentConfidence,
    trueIntentRank,
    confusingWords,
  };
}
