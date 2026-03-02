/**
 * Active Learning Engine
 *
 * Selects the most informative unlabeled examples for human annotation.
 * This minimizes the number of examples needed to reach target accuracy.
 *
 * Query Strategies:
 * 1. Least Confidence - pick examples where model is least sure
 * 2. Margin Sampling - pick examples where top-2 predictions are closest
 * 3. Entropy Sampling - pick examples with highest prediction entropy
 * 4. Query-by-Committee (QBC) - pick examples where ensemble disagrees most
 * 5. Expected Model Change - pick examples that would change model most
 * 6. Information Density - combine uncertainty with representativeness
 */

import { type EnsembleModel, predictEnsemble } from "../engine/ensemble";

export interface ActiveLearningCandidate {
  text: string;
  uncertaintyScore: number;
  strategy: string;
  topPrediction: string;
  topConfidence: number;
  marginGap: number;
  entropy: number;
  committeeDisagreement: number;
}

/**
 * Least confidence: 1 - P(most likely class)
 */
function leastConfidence(ranking: Array<{ confidence: number }>): number {
  return 1 - (ranking[0]?.confidence || 0);
}

/**
 * Margin sampling: difference between top-2 predictions
 * Small margin = model is confused between top choices
 */
function marginSampling(ranking: Array<{ confidence: number }>): number {
  if (ranking.length < 2) return 1;
  return 1 - (ranking[0].confidence - ranking[1].confidence);
}

/**
 * Entropy: -sum(p * log(p)) — information-theoretic uncertainty
 */
function entropySampling(ranking: Array<{ confidence: number }>): number {
  let entropy = 0;
  for (const r of ranking) {
    if (r.confidence > 0) {
      entropy -= r.confidence * Math.log2(r.confidence);
    }
  }
  return entropy;
}

/**
 * Query-by-Committee: measure disagreement between ensemble members
 * Uses vote entropy over per-model top predictions
 */
function committeeDisagreement(
  perModelScores: {
    logReg: Array<{ intent: string; score: number }>;
    naiveBayes: Array<{ intent: string; score: number }>;
    svm: Array<{ intent: string; score: number }>;
    gradBoost: Array<{ intent: string; score: number }>;
  },
): number {
  const models = [
    perModelScores.logReg,
    perModelScores.naiveBayes,
    perModelScores.svm,
    perModelScores.gradBoost,
  ];

  // Count votes for each class
  const votes: Record<string, number> = {};
  for (const model of models) {
    if (model.length > 0) {
      const topIntent = model[0].intent;
      votes[topIntent] = (votes[topIntent] || 0) + 1;
    }
  }

  // Vote entropy
  const numModels = models.length;
  let entropy = 0;
  for (const count of Object.values(votes)) {
    const p = count / numModels;
    if (p > 0) entropy -= p * Math.log2(p);
  }

  return entropy;
}

/**
 * Information density weighting
 * Combines uncertainty with how representative the sample is
 * Dense regions are preferred (more informative)
 */
function informationDensity(
  uncertaintyScore: number,
  text: string,
  allTexts: string[],
  beta: number = 1.0,
): number {
  // Simple cosine similarity approximation via character overlap
  const textChars = new Set(text.toLowerCase().split(""));
  let totalSimilarity = 0;
  for (const other of allTexts) {
    const otherChars = new Set(other.toLowerCase().split(""));
    let intersection = 0;
    for (const c of textChars) if (otherChars.has(c)) intersection++;
    totalSimilarity += intersection / Math.max(textChars.size, otherChars.size);
  }
  const avgSimilarity = totalSimilarity / allTexts.length;

  return uncertaintyScore * Math.pow(avgSimilarity, beta);
}

/**
 * Score a batch of unlabeled texts for active learning
 */
export function scoreForActiveLearning(
  texts: string[],
  model: EnsembleModel,
  strategy: "least_confidence" | "margin" | "entropy" | "committee" | "density" = "committee",
): ActiveLearningCandidate[] {
  const candidates: ActiveLearningCandidate[] = [];

  for (const text of texts) {
    const prediction = predictEnsemble(text, model);
    const ranking = prediction.ranking;

    const lc = leastConfidence(ranking);
    const ms = marginSampling(ranking);
    const ent = entropySampling(ranking);
    const cd = committeeDisagreement(prediction.perModelScores);

    let primaryScore: number;
    switch (strategy) {
      case "least_confidence": primaryScore = lc; break;
      case "margin": primaryScore = ms; break;
      case "entropy": primaryScore = ent; break;
      case "committee": primaryScore = cd; break;
      case "density": primaryScore = informationDensity(ent, text, texts); break;
    }

    candidates.push({
      text,
      uncertaintyScore: primaryScore,
      strategy,
      topPrediction: prediction.intent,
      topConfidence: prediction.confidence,
      marginGap: ranking.length >= 2 ? ranking[0].confidence - ranking[1].confidence : 1,
      entropy: ent,
      committeeDisagreement: cd,
    });
  }

  // Sort by uncertainty (highest first = most informative)
  candidates.sort((a, b) => b.uncertaintyScore - a.uncertaintyScore);
  return candidates;
}

/**
 * Select diverse batch for annotation
 * Uses maximal marginal relevance to avoid redundant queries
 */
export function selectDiverseBatch(
  candidates: ActiveLearningCandidate[],
  batchSize: number,
  diversityWeight: number = 0.3,
): ActiveLearningCandidate[] {
  if (candidates.length <= batchSize) return candidates;

  const selected: ActiveLearningCandidate[] = [];
  const remaining = [...candidates];

  // Always pick the most uncertain first
  selected.push(remaining.shift()!);

  while (selected.length < batchSize && remaining.length > 0) {
    let bestIdx = 0;
    let bestScore = -Infinity;

    for (let i = 0; i < remaining.length; i++) {
      const candidate = remaining[i];
      const relevance = candidate.uncertaintyScore;

      // Diversity: minimum "distance" to already selected items
      let minDist = Infinity;
      for (const sel of selected) {
        // Simple text distance: different first word = more diverse
        const cWords = new Set(candidate.text.toLowerCase().split(/\s+/));
        const sWords = new Set(sel.text.toLowerCase().split(/\s+/));
        let overlap = 0;
        for (const w of cWords) if (sWords.has(w)) overlap++;
        const dist = 1 - overlap / Math.max(cWords.size, sWords.size);
        minDist = Math.min(minDist, dist);
      }

      const score = (1 - diversityWeight) * relevance + diversityWeight * minDist;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    selected.push(remaining.splice(bestIdx, 1)[0]);
  }

  return selected;
}

/**
 * Generate candidate texts for active learning from unlabeled pool
 * Creates synthetic candidates by combining existing patterns
 */
export function generateCandidatePool(
  existingExamples: Array<{ text: string; intent: string }>,
  poolSize: number = 100,
): string[] {
  const pool: Set<string> = new Set();
  const existingTexts = new Set(existingExamples.map((e) => e.text.toLowerCase()));

  // Strategy 1: Combine parts from different intents
  const byIntent: Record<string, string[]> = {};
  for (const ex of existingExamples) {
    if (!byIntent[ex.intent]) byIntent[ex.intent] = [];
    byIntent[ex.intent].push(ex.text);
  }

  const intents = Object.keys(byIntent);
  for (let attempt = 0; attempt < poolSize * 3 && pool.size < poolSize; attempt++) {
    const intent1 = intents[Math.floor(Math.random() * intents.length)];
    const intent2 = intents[Math.floor(Math.random() * intents.length)];
    const text1 = byIntent[intent1][Math.floor(Math.random() * byIntent[intent1].length)];
    const text2 = byIntent[intent2][Math.floor(Math.random() * byIntent[intent2].length)];

    const words1 = text1.split(/\s+/);
    const words2 = text2.split(/\s+/);

    // Take first half from one, second half from another
    const midPoint = Math.floor(words1.length / 2);
    const combined = [...words1.slice(0, midPoint), ...words2.slice(Math.floor(words2.length / 2))].join(" ");

    if (!existingTexts.has(combined.toLowerCase()) && combined.split(/\s+/).length >= 2) {
      pool.add(combined);
    }
  }

  // Strategy 2: Partial sentences (truncated)
  for (const ex of existingExamples) {
    if (pool.size >= poolSize) break;
    const words = ex.text.split(/\s+/);
    if (words.length >= 4) {
      const partial = words.slice(0, Math.ceil(words.length * 0.6)).join(" ");
      if (!existingTexts.has(partial.toLowerCase())) {
        pool.add(partial);
      }
    }
  }

  return [...pool].slice(0, poolSize);
}
