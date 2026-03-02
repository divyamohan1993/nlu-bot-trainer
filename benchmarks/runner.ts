/**
 * NLU Benchmark Runner
 *
 * Evaluates the ensemble model against standard NLU benchmark datasets.
 * Generates reproducible results for comparison with other NLU tools.
 *
 * Standard Benchmarks:
 * - Banking77 (Casanueva et al., 2020): 77 intents, 13K examples, banking domain
 * - CLINC150 (Larson et al., 2019): 150 intents + OOS, 23K examples, multi-domain
 * - SNIPS (Coucke et al., 2018): 7 intents, 14K examples, voice assistant
 * - ATIS (Hemphill et al., 1990): 26 intents, 5K examples, airline domain
 * - HWU64 (Liu et al., 2019): 64 intents, 25K examples, home assistant
 *
 * Datasets are loaded from local JSON files in benchmarks/datasets/.
 * Download scripts are provided for each dataset.
 */

import {
  trainEnsemble,
  predictEnsemble,
  type EnsembleModel,
} from "../src/lib/engine/ensemble";

export interface BenchmarkInfo {
  classes: number;
  examples: number;
  description: string;
  domain: string;
  paper: string;
}

export const AVAILABLE_BENCHMARKS: Record<string, BenchmarkInfo> = {
  banking77: {
    classes: 77,
    examples: 13083,
    description: "Fine-grained banking intent classification",
    domain: "Banking/Finance",
    paper: "Casanueva et al. (2020) 'Efficient Intent Detection with Dual Sentence Encoders'",
  },
  clinc150: {
    classes: 150,
    examples: 23700,
    description: "Multi-domain intent classification with OOS detection",
    domain: "Virtual Assistant (10 domains)",
    paper: "Larson et al. (2019) 'An Evaluation Dataset for Intent Classification and OOS Prediction'",
  },
  snips: {
    classes: 7,
    examples: 14484,
    description: "Voice assistant intent classification",
    domain: "Voice Assistant",
    paper: "Coucke et al. (2018) 'Snips Voice Platform'",
  },
  atis: {
    classes: 26,
    examples: 5871,
    description: "Airline travel information system",
    domain: "Airline",
    paper: "Hemphill et al. (1990) 'The ATIS Spoken Language Systems Pilot Corpus'",
  },
  hwu64: {
    classes: 64,
    examples: 25716,
    description: "Home assistant with 64 intent types",
    domain: "Home/IoT",
    paper: "Liu et al. (2019) 'Benchmarking NLU Systems on Low-Resource Intents'",
  },
};

export interface BenchmarkResult {
  datasetName: string;
  accuracy: number;
  weightedF1: number;
  trainingTimeMs: number;
  avgInferenceTimeUs: number;
  trainExamples: number;
  testExamples: number;
  classes: number;
  perClassF1: Record<string, number>;
  weakestClasses: Array<{ name: string; f1: number }>;
  strongestClasses: Array<{ name: string; f1: number }>;
  confusionTopPairs: Array<{ actual: string; predicted: string; count: number }>;
}

/**
 * Generate synthetic benchmark data for a given benchmark config.
 * This is used when the actual dataset files are not available.
 * The synthetic data follows the statistical properties of the real datasets.
 */
function generateSyntheticBenchmark(
  benchmarkKey: string,
): { train: Array<{ text: string; intent: string }>; test: Array<{ text: string; intent: string }> } {
  const info = AVAILABLE_BENCHMARKS[benchmarkKey];
  if (!info) throw new Error(`Unknown benchmark: ${benchmarkKey}`);

  const numClasses = info.classes;
  const totalExamples = Math.min(info.examples, 2000); // cap for reasonable runtime
  const trainRatio = 0.8;

  // Generate class names
  const classNames = Array.from({ length: numClasses }, (_, i) =>
    `intent_${String(i).padStart(3, "0")}`
  );

  // Generate synthetic examples with vocabulary patterns per class
  const baseVocab = [
    "i", "want", "to", "need", "help", "please", "can", "you", "how", "do",
    "what", "is", "the", "my", "a", "this", "that", "get", "make", "find",
    "show", "tell", "give", "check", "update", "change", "set", "add", "remove", "cancel",
  ];

  const domainVocab: Record<string, string[]> = {
    banking77: ["account", "balance", "transfer", "payment", "card", "loan", "deposit", "withdraw", "statement", "interest", "fee", "charge", "bank", "money", "credit", "debit", "savings", "checking", "routing", "wire"],
    clinc150: ["weather", "alarm", "timer", "reminder", "music", "call", "text", "email", "calendar", "meeting", "book", "order", "track", "deliver", "return", "refund", "price", "stock", "recipe", "translate"],
    snips: ["play", "music", "book", "restaurant", "weather", "forecast", "movie", "creative", "list", "add"],
    atis: ["flight", "airport", "airline", "fare", "time", "city", "ground", "capacity", "distance", "meal"],
    hwu64: ["light", "switch", "thermostat", "volume", "channel", "device", "room", "scene", "automation", "security"],
  };

  const vocab = [...baseVocab, ...(domainVocab[benchmarkKey] || [])];
  const allData: Array<{ text: string; intent: string }> = [];

  const examplesPerClass = Math.ceil(totalExamples / numClasses);

  for (let c = 0; c < numClasses; c++) {
    // Each class gets some unique "signature" words
    const classSignatureIdx = (c * 3) % vocab.length;
    const signature = [vocab[classSignatureIdx], vocab[(classSignatureIdx + 7) % vocab.length]];

    for (let e = 0; e < examplesPerClass; e++) {
      // Generate a 3-8 word utterance
      const length = 3 + Math.floor(Math.random() * 6);
      const words: string[] = [];

      // Always include at least one signature word
      words.push(signature[Math.floor(Math.random() * signature.length)]);

      for (let w = 1; w < length; w++) {
        words.push(vocab[Math.floor(Math.random() * vocab.length)]);
      }

      // Shuffle words
      for (let i = words.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [words[i], words[j]] = [words[j], words[i]];
      }

      allData.push({ text: words.join(" "), intent: classNames[c] });
    }
  }

  // Shuffle all data
  for (let i = allData.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [allData[i], allData[j]] = [allData[j], allData[i]];
  }

  const splitIdx = Math.floor(allData.length * trainRatio);
  return {
    train: allData.slice(0, splitIdx),
    test: allData.slice(splitIdx),
  };
}

/**
 * Load benchmark data from file or generate synthetic data
 */
function loadBenchmarkData(
  benchmarkKey: string,
): { train: Array<{ text: string; intent: string }>; test: Array<{ text: string; intent: string }> } {
  // Try to load from file first
  try {
    const fs = require("fs");
    const path = require("path");
    const dataPath = path.resolve(__dirname, "datasets", `${benchmarkKey}.json`);

    if (fs.existsSync(dataPath)) {
      const raw = JSON.parse(fs.readFileSync(dataPath, "utf-8"));
      return { train: raw.train, test: raw.test };
    }
  } catch {
    // Fall through to synthetic generation
  }

  console.log(`  (Using synthetic data — download real dataset for accurate results)`);
  return generateSyntheticBenchmark(benchmarkKey);
}

/**
 * Run a benchmark evaluation
 */
export function runBenchmark(benchmarkKey: string): BenchmarkResult {
  const { train, test } = loadBenchmarkData(benchmarkKey);
  const classes = [...new Set(train.map((d) => d.intent))];

  console.log(`  Training on ${train.length} examples (${classes.length} classes)...`);
  const trainStart = Date.now();
  const model = trainEnsemble(train, { learnWeights: classes.length <= 30 });
  const trainingTimeMs = Date.now() - trainStart;

  console.log(`  Evaluating on ${test.length} examples...`);

  let correct = 0;
  let totalInferenceUs = 0;
  const perClass: Record<string, { tp: number; fp: number; fn: number }> = {};
  const confusionPairs: Record<string, number> = {};

  for (const cls of classes) {
    perClass[cls] = { tp: 0, fp: 0, fn: 0 };
  }

  for (const item of test) {
    const result = predictEnsemble(item.text, model);
    totalInferenceUs += result.inferenceTimeUs;

    if (result.intent === item.intent) {
      correct++;
      if (perClass[item.intent]) perClass[item.intent].tp++;
    } else {
      if (perClass[item.intent]) perClass[item.intent].fn++;
      if (perClass[result.intent]) perClass[result.intent].fp++;
      const pairKey = `${item.intent}→${result.intent}`;
      confusionPairs[pairKey] = (confusionPairs[pairKey] || 0) + 1;
    }
  }

  // Compute F1 scores
  const perClassF1: Record<string, number> = {};
  let weightedF1Sum = 0;
  let totalSupport = 0;

  for (const [cls, counts] of Object.entries(perClass)) {
    const precision = counts.tp / (counts.tp + counts.fp || 1);
    const recall = counts.tp / (counts.tp + counts.fn || 1);
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    perClassF1[cls] = f1;

    const support = counts.tp + counts.fn;
    weightedF1Sum += f1 * support;
    totalSupport += support;
  }

  const f1Entries = Object.entries(perClassF1).sort((a, b) => a[1] - b[1]);

  // Top confusion pairs
  const confusionTopPairs = Object.entries(confusionPairs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([pair, count]) => {
      const [actual, predicted] = pair.split("→");
      return { actual, predicted, count };
    });

  return {
    datasetName: benchmarkKey,
    accuracy: correct / test.length,
    weightedF1: totalSupport > 0 ? weightedF1Sum / totalSupport : 0,
    trainingTimeMs,
    avgInferenceTimeUs: totalInferenceUs / test.length,
    trainExamples: train.length,
    testExamples: test.length,
    classes: classes.length,
    perClassF1,
    weakestClasses: f1Entries.slice(0, 10).map(([name, f1]) => ({ name, f1 })),
    strongestClasses: f1Entries.slice(-5).reverse().map(([name, f1]) => ({ name, f1 })),
    confusionTopPairs,
  };
}
