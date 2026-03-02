import {
  tokenizeWithNgrams,
  buildVocabulary,
  computeIDF,
  computeTFIDF,
  cosineSimilarity,
} from "./tokenizer";

export interface TrainedModel {
  vocabulary: string[];
  idfWeights: Record<string, number>;
  trainingVectors: Array<{ vector: number[]; intent: string }>;
  nbLogPriors: Record<string, number>;
  nbLogLikelihoods: Record<string, Record<string, number>>;
  trainedAt: string;
  version?: number;
}

interface TrainingItem {
  text: string;
  intent: string;
}

/** Train a kNN + Naive Bayes hybrid classifier */
export function trainClassifier(data: TrainingItem[]): TrainedModel {
  if (data.length === 0) throw new Error("No training data provided");

  const tokenizedDocs = data.map((item) => tokenizeWithNgrams(item.text));
  const vocabulary = buildVocabulary(tokenizedDocs);
  const idfWeights = computeIDF(tokenizedDocs, vocabulary);

  const trainingVectors: Array<{ vector: number[]; intent: string }> = [];
  for (let i = 0; i < data.length; i++) {
    const vector = computeTFIDF(tokenizedDocs[i], vocabulary, idfWeights);
    trainingVectors.push({ vector, intent: data[i].intent });
  }

  // Naive Bayes with Lidstone smoothing
  const intentWordCounts: Record<string, Record<string, number>> = {};
  const intentTotalWords: Record<string, number> = {};
  const intentDocCounts: Record<string, number> = {};

  for (let i = 0; i < data.length; i++) {
    const intent = data[i].intent;
    intentDocCounts[intent] = (intentDocCounts[intent] || 0) + 1;
    if (!intentWordCounts[intent]) intentWordCounts[intent] = {};
    if (!intentTotalWords[intent]) intentTotalWords[intent] = 0;
    for (const token of tokenizedDocs[i]) {
      intentWordCounts[intent][token] = (intentWordCounts[intent][token] || 0) + 1;
      intentTotalWords[intent]++;
    }
  }

  const nbLogPriors: Record<string, number> = {};
  const nbLogLikelihoods: Record<string, Record<string, number>> = {};
  const vocabSize = vocabulary.length;

  for (const intent of Object.keys(intentDocCounts)) {
    nbLogPriors[intent] = Math.log(intentDocCounts[intent] / data.length);
    nbLogLikelihoods[intent] = {};
    const totalWords = intentTotalWords[intent] || 0;
    for (const word of vocabulary) {
      const wordCount = intentWordCounts[intent]?.[word] || 0;
      nbLogLikelihoods[intent][word] = Math.log(
        (wordCount + 0.5) / (totalWords + vocabSize * 0.5)
      );
    }
  }

  return {
    vocabulary, idfWeights, trainingVectors,
    nbLogPriors, nbLogLikelihoods,
    trainedAt: new Date().toISOString(),
    version: 6,
  };
}

/** Predict intent using kNN + Naive Bayes ensemble */
export function predict(
  text: string,
  model: TrainedModel
): { intent: string; confidence: number; ranking: Array<{ name: string; confidence: number }> } {
  const tokens = tokenizeWithNgrams(text);
  const vector = computeTFIDF(tokens, model.vocabulary, model.idfWeights);

  // kNN with similarity-weighted voting
  const K = 5;
  const similarities: Array<{ intent: string; sim: number }> = [];
  for (const tv of model.trainingVectors) {
    similarities.push({ intent: tv.intent, sim: cosineSimilarity(vector, tv.vector) });
  }
  similarities.sort((a, b) => b.sim - a.sim);

  const topK = similarities.slice(0, K);
  const knnVotes: Record<string, number> = {};
  for (const { intent, sim } of topK) {
    knnVotes[intent] = (knnVotes[intent] || 0) + Math.max(sim, 0.001);
  }
  const knnTotal = Object.values(knnVotes).reduce((a, b) => a + b, 0);
  const allIntents = Array.from(new Set(model.trainingVectors.map((tv) => tv.intent)));
  const knnNorm: Record<string, number> = {};
  for (const intent of allIntents) {
    knnNorm[intent] = knnTotal > 0 ? (knnVotes[intent] || 0) / knnTotal : 0;
  }

  // Naive Bayes
  const nbScores: Record<string, number> = {};
  for (const intent of Object.keys(model.nbLogPriors)) {
    let logProb = model.nbLogPriors[intent];
    for (const token of tokens) {
      if (model.nbLogLikelihoods[intent]?.[token] !== undefined) {
        logProb += model.nbLogLikelihoods[intent][token];
      }
    }
    nbScores[intent] = logProb;
  }

  const maxNb = Math.max(...Object.values(nbScores));
  let nbExpSum = 0;
  const nbExp: Record<string, number> = {};
  for (const intent of Object.keys(nbScores)) {
    nbExp[intent] = Math.exp(nbScores[intent] - maxNb);
    nbExpSum += nbExp[intent];
  }
  const nbNorm: Record<string, number> = {};
  for (const intent of Object.keys(nbScores)) {
    nbNorm[intent] = nbExp[intent] / nbExpSum;
  }

  // Ensemble: 50% kNN + 50% NB
  const combined: Array<{ name: string; score: number }> = [];
  for (const intent of allIntents) {
    combined.push({
      name: intent,
      score: 0.5 * (knnNorm[intent] || 0) + 0.5 * (nbNorm[intent] || 0),
    });
  }
  combined.sort((a, b) => b.score - a.score);

  const maxScore = combined[0]?.score || 0;
  const expScores = combined.map((s) => Math.exp((s.score - maxScore) * 12));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const ranking = combined.map((s, i) => ({
    name: s.name,
    confidence: sumExp > 0 ? expScores[i] / sumExp : 0,
  }));

  return {
    intent: ranking[0]?.name || "unknown",
    confidence: ranking[0]?.confidence || 0,
    ranking,
  };
}

/** Cross-validate with stratified folds, averaged over multiple runs */
export function crossValidate(data: TrainingItem[], folds: number = 5): number {
  if (data.length < folds) return 0;
  const RUNS = 3;
  let totalCorrect = 0;
  let totalSamples = 0;

  for (let run = 0; run < RUNS; run++) {
    const byIntent: Record<string, TrainingItem[]> = {};
    for (const item of data) {
      if (!byIntent[item.intent]) byIntent[item.intent] = [];
      byIntent[item.intent].push(item);
    }
    for (const items of Object.values(byIntent)) {
      for (let i = items.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [items[i], items[j]] = [items[j], items[i]];
      }
    }
    const foldData: TrainingItem[][] = Array.from({ length: folds }, () => []);
    for (const items of Object.values(byIntent)) {
      items.forEach((item, i) => { foldData[i % folds].push(item); });
    }
    for (let i = 0; i < folds; i++) {
      const testSet = foldData[i];
      const trainSet = foldData.filter((_, idx) => idx !== i).flat();
      if (trainSet.length === 0 || testSet.length === 0) continue;
      const trainIntents = new Set(trainSet.map((t) => t.intent));
      if (trainIntents.size < 2) continue;
      try {
        const model = trainClassifier(trainSet);
        for (const item of testSet) {
          const result = predict(item.text, model);
          if (result.intent === item.intent) totalCorrect++;
          totalSamples++;
        }
      } catch { continue; }
    }
  }
  return totalSamples > 0 ? totalCorrect / totalSamples : 0;
}
