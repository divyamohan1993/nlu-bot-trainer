#!/usr/bin/env node
/**
 * NLU Bot Trainer CLI
 *
 * Command-line interface for training, predicting, evaluating, and benchmarking
 * the NLU ensemble model without needing the React UI.
 *
 * Usage:
 *   npx ts-node cli/index.ts train ./data.json
 *   npx ts-node cli/index.ts predict "track my order"
 *   npx ts-node cli/index.ts evaluate ./test.json
 *   npx ts-node cli/index.ts benchmark banking77
 *   npx ts-node cli/index.ts quality ./data.json
 *   npx ts-node cli/index.ts export rasa ./data.json -o output.yml
 *   npx ts-node cli/index.ts serve --port 8080
 */

import { Command } from "commander";
import * as fs from "fs";
import * as path from "path";
import {
  trainEnsemble,
  predictEnsemble,
  crossValidateEnsemble,
  serializeEnsemble,
  deserializeEnsemble,
  type EnsembleModel,
} from "../src/lib/engine/ensemble";
import { detectOOS, calibrateOOS } from "../src/lib/engine/oos-detector";
import { learnTemperature } from "../src/lib/engine/confidence-calibration";
import { explainPrediction } from "../src/lib/engine/explainer";
import { validateDataQuality } from "../src/lib/engine/data-quality";
import { runBenchmark, AVAILABLE_BENCHMARKS } from "../benchmarks/runner";

const program = new Command();

program
  .name("nlu-bot-trainer")
  .description("Browser-native NLU engine — 5-classifier stacking ensemble (171K params)")
  .version("2.2.0");

// ═══════════════════════════════════════════════════════════════
// TRAIN
// ═══════════════════════════════════════════════════════════════
program
  .command("train <dataFile>")
  .description("Train the ensemble model from a JSON data file")
  .option("-o, --output <file>", "Output model file", "model.json")
  .option("--no-learn-weights", "Skip meta-weight learning (faster, less accurate)")
  .option("--cv <folds>", "Run cross-validation with N folds", "0")
  .action((dataFile: string, opts: { output: string; learnWeights: boolean; cv: string }) => {
    const rawData = JSON.parse(fs.readFileSync(path.resolve(dataFile), "utf-8"));
    const data = normalizeInputData(rawData);

    console.log(`Training on ${data.length} examples...`);
    const startTime = Date.now();

    const model = trainEnsemble(data, { learnWeights: opts.learnWeights });

    console.log(`\nTraining complete in ${Date.now() - startTime}ms`);
    console.log(`  Accuracy: ${(model.metrics.accuracy * 100).toFixed(1)}%`);
    console.log(`  Classes: ${model.classes.length}`);
    console.log(`  Vocabulary: ${model.metrics.vocabularySize.toLocaleString()}`);
    console.log(`  Meta-weights: [${model.metaWeights.map((w) => (w * 100).toFixed(0) + "%").join(", ")}]`);
    console.log(`    (LR, CNB, SVM, MLP, GBM)`);

    // Cross-validation
    const cvFolds = parseInt(opts.cv);
    if (cvFolds > 0) {
      console.log(`\nRunning ${cvFolds}-fold cross-validation...`);
      const cv = crossValidateEnsemble(data, cvFolds);
      console.log(`  CV Accuracy: ${(cv.accuracy * 100).toFixed(1)}%`);

      const f1Entries = Object.entries(cv.perClassF1).sort((a, b) => a[1] - b[1]);
      console.log(`  Per-class F1:`);
      for (const [cls, f1] of f1Entries) {
        const bar = "█".repeat(Math.round(f1 * 20));
        console.log(`    ${cls.padEnd(20)} ${bar} ${(f1 * 100).toFixed(1)}%`);
      }
    }

    // Save model
    const serialized = serializeEnsemble(model);
    fs.writeFileSync(path.resolve(opts.output), serialized);
    console.log(`\nModel saved to ${opts.output} (${(serialized.length / 1024).toFixed(0)} KB)`);
  });

// ═══════════════════════════════════════════════════════════════
// PREDICT
// ═══════════════════════════════════════════════════════════════
program
  .command("predict <text>")
  .description("Classify a text using a trained model")
  .option("-m, --model <file>", "Model file", "model.json")
  .option("-k, --top-k <n>", "Number of top intents to show", "5")
  .option("--explain", "Show token-level explanation")
  .option("--oos", "Enable out-of-scope detection")
  .action((text: string, opts: { model: string; topK: string; explain: boolean; oos: boolean }) => {
    const model = loadModel(opts.model);
    const topK = parseInt(opts.topK);

    const result = predictEnsemble(text, model);

    console.log(`\nInput: "${text}"`);
    console.log(`Intent: ${result.intent} (${(result.confidence * 100).toFixed(1)}%)`);
    console.log(`Inference: ${result.inferenceTimeUs.toFixed(0)}μs\n`);

    console.log("Ranking:");
    for (const r of result.ranking.slice(0, topK)) {
      const bar = "█".repeat(Math.round(r.confidence * 30));
      const pct = (r.confidence * 100).toFixed(1).padStart(5);
      console.log(`  ${r.name.padEnd(20)} ${bar} ${pct}%`);
    }

    if (opts.oos) {
      const oos = detectOOS(text, model);
      console.log(`\nOOS Detection: ${oos.isOOS ? "OUT-OF-SCOPE" : "In-scope"} (score: ${oos.oosScore.toFixed(3)})`);
      console.log(`  Entropy: ${oos.signals.entropy.toFixed(3)}, MaxProb: ${oos.signals.maxProbability.toFixed(3)}, Margin: ${oos.signals.marginGap.toFixed(3)}`);
    }

    if (opts.explain) {
      const explanation = explainPrediction(text, model);
      console.log(`\nExplanation: ${explanation.summary}`);
      if (explanation.supportingTokens.length > 0) {
        console.log("  Supporting:");
        for (const t of explanation.supportingTokens.slice(0, 5)) {
          const contrib = (t.normalizedContribution * 100).toFixed(0);
          console.log(`    "${t.token}" → +${contrib}%`);
        }
      }
      if (explanation.opposingTokens.length > 0) {
        console.log("  Opposing:");
        for (const t of explanation.opposingTokens.slice(0, 3)) {
          const contrib = (t.normalizedContribution * 100).toFixed(0);
          console.log(`    "${t.token}" → ${contrib}%`);
        }
      }
    }
  });

// ═══════════════════════════════════════════════════════════════
// EVALUATE
// ═══════════════════════════════════════════════════════════════
program
  .command("evaluate <testFile>")
  .description("Evaluate model accuracy on a test dataset")
  .option("-m, --model <file>", "Model file", "model.json")
  .option("--calibrate", "Run confidence calibration (temperature scaling)")
  .action((testFile: string, opts: { model: string; calibrate: boolean }) => {
    const model = loadModel(opts.model);
    const rawData = JSON.parse(fs.readFileSync(path.resolve(testFile), "utf-8"));
    const data = normalizeInputData(rawData);

    console.log(`Evaluating on ${data.length} examples...\n`);

    let correct = 0;
    const perClass: Record<string, { tp: number; fp: number; fn: number }> = {};
    const errors: Array<{ text: string; actual: string; predicted: string; confidence: number }> = [];

    for (const item of data) {
      const result = predictEnsemble(item.text, model);
      const predicted = result.intent;
      const isCorrect = predicted === item.intent;

      if (isCorrect) correct++;
      else {
        errors.push({ text: item.text, actual: item.intent, predicted, confidence: result.confidence });
      }

      if (!perClass[item.intent]) perClass[item.intent] = { tp: 0, fp: 0, fn: 0 };
      if (!perClass[predicted]) perClass[predicted] = { tp: 0, fp: 0, fn: 0 };

      if (isCorrect) perClass[item.intent].tp++;
      else {
        perClass[item.intent].fn++;
        perClass[predicted].fp++;
      }
    }

    const accuracy = correct / data.length;
    console.log(`Accuracy: ${(accuracy * 100).toFixed(1)}% (${correct}/${data.length})\n`);

    console.log("Per-class F1:");
    const f1Entries: Array<[string, number]> = [];
    for (const [cls, counts] of Object.entries(perClass)) {
      const precision = counts.tp / (counts.tp + counts.fp || 1);
      const recall = counts.tp / (counts.tp + counts.fn || 1);
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      f1Entries.push([cls, f1]);
    }
    f1Entries.sort((a, b) => a[1] - b[1]);
    for (const [cls, f1] of f1Entries) {
      const bar = "█".repeat(Math.round(f1 * 20));
      console.log(`  ${cls.padEnd(20)} ${bar} ${(f1 * 100).toFixed(1)}%`);
    }

    if (errors.length > 0) {
      console.log(`\nTop errors (${errors.length} total):`);
      for (const err of errors.slice(0, 10)) {
        console.log(`  "${err.text.slice(0, 50)}" → ${err.predicted} (${(err.confidence * 100).toFixed(0)}%) [actual: ${err.actual}]`);
      }
    }

    if (opts.calibrate) {
      console.log("\nCalibrating confidence...");
      const cal = learnTemperature(model, data);
      console.log(`  Temperature: ${cal.temperature}`);
      console.log(`  ECE: ${(cal.eceBefore * 100).toFixed(2)}% → ${(cal.eceAfter * 100).toFixed(2)}%`);
      console.log(`  MCE: ${(cal.mceBefore * 100).toFixed(2)}% → ${(cal.mceAfter * 100).toFixed(2)}%`);
    }
  });

// ═══════════════════════════════════════════════════════════════
// BENCHMARK
// ═══════════════════════════════════════════════════════════════
program
  .command("benchmark [dataset]")
  .description("Run standard NLU benchmark (Banking77, CLINC150, SNIPS)")
  .option("--list", "List available benchmarks")
  .action((dataset: string | undefined, opts: { list: boolean }) => {
    if (opts.list || !dataset) {
      console.log("Available benchmarks:");
      for (const [key, info] of Object.entries(AVAILABLE_BENCHMARKS)) {
        console.log(`  ${key.padEnd(12)} ${info.classes} classes, ${info.examples} examples — ${info.description}`);
      }
      return;
    }

    const benchmarkKey = dataset.toLowerCase();
    if (!(benchmarkKey in AVAILABLE_BENCHMARKS)) {
      console.error(`Unknown benchmark: ${dataset}. Use --list to see available options.`);
      process.exit(1);
    }

    console.log(`Running ${dataset} benchmark...`);
    const result = runBenchmark(benchmarkKey);

    console.log(`\n${dataset} Results:`);
    console.log(`  Accuracy: ${(result.accuracy * 100).toFixed(1)}%`);
    console.log(`  Weighted F1: ${(result.weightedF1 * 100).toFixed(1)}%`);
    console.log(`  Training time: ${result.trainingTimeMs.toFixed(0)}ms`);
    console.log(`  Inference time: ${result.avgInferenceTimeUs.toFixed(0)}μs/example`);
    console.log(`  Examples: ${result.trainExamples} train, ${result.testExamples} test`);
    console.log(`  Classes: ${result.classes}`);

    if (result.weakestClasses.length > 0) {
      console.log(`\n  Weakest classes:`);
      for (const { name, f1 } of result.weakestClasses.slice(0, 5)) {
        console.log(`    ${name.padEnd(25)} ${(f1 * 100).toFixed(1)}%`);
      }
    }
  });

// ═══════════════════════════════════════════════════════════════
// QUALITY
// ═══════════════════════════════════════════════════════════════
program
  .command("quality <dataFile>")
  .description("Validate training data quality")
  .option("-m, --model <file>", "Model file for mislabel detection")
  .action((dataFile: string, opts: { model?: string }) => {
    const rawData = JSON.parse(fs.readFileSync(path.resolve(dataFile), "utf-8"));
    const data = normalizeInputData(rawData);

    let model: EnsembleModel | undefined;
    if (opts.model) {
      model = loadModel(opts.model);
    }

    const report = validateDataQuality(data, model);

    console.log(`\nData Quality Report`);
    console.log(`${"─".repeat(40)}`);
    console.log(`Score: ${report.score}/100 (Grade: ${report.grade})\n`);

    console.log(`Statistics:`);
    console.log(`  Examples: ${report.stats.totalExamples}`);
    console.log(`  Intents: ${report.stats.totalIntents}`);
    console.log(`  Avg examples/intent: ${report.stats.avgExamplesPerIntent}`);
    console.log(`  Vocabulary: ${report.stats.vocabularySize}`);
    console.log(`  Duplicates: ${report.stats.duplicateCount}`);
    console.log(`  Near-duplicates: ${report.stats.nearDuplicateCount}`);
    console.log(`  Suspected mislabels: ${report.stats.suspectedMislabels}`);

    if (report.issues.length > 0) {
      console.log(`\nIssues (${report.issues.length}):`);
      const errors = report.issues.filter((i) => i.severity === "error");
      const warnings = report.issues.filter((i) => i.severity === "warning");
      const infos = report.issues.filter((i) => i.severity === "info");

      if (errors.length > 0) {
        console.log(`\n  ERRORS (${errors.length}):`);
        for (const issue of errors.slice(0, 10)) {
          console.log(`    ${issue.message}`);
          console.log(`      Fix: ${issue.suggestion}`);
        }
      }
      if (warnings.length > 0) {
        console.log(`\n  WARNINGS (${warnings.length}):`);
        for (const issue of warnings.slice(0, 10)) {
          console.log(`    ${issue.message}`);
        }
      }
      if (infos.length > 0) {
        console.log(`\n  INFO (${infos.length}):`);
        for (const issue of infos.slice(0, 5)) {
          console.log(`    ${issue.message}`);
        }
      }
    } else {
      console.log("\nNo issues found — data quality is excellent!");
    }
  });

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════

function loadModel(modelPath: string): EnsembleModel {
  const resolved = path.resolve(modelPath);
  if (!fs.existsSync(resolved)) {
    console.error(`Model file not found: ${resolved}`);
    console.error("Train a model first: nlu-bot-trainer train data.json");
    process.exit(1);
  }
  return deserializeEnsemble(fs.readFileSync(resolved, "utf-8"));
}

/**
 * Normalize various input formats to [{text, intent}]
 * Supports:
 *   - Array of {text, intent}
 *   - {intents: [{name, examples: [{text}]}]} (our native format)
 *   - {nlu: [{intent, examples: ["text"]}]} (Rasa format)
 */
function normalizeInputData(raw: unknown): Array<{ text: string; intent: string }> {
  if (Array.isArray(raw)) {
    return raw.map((item: { text: string; intent: string }) => ({
      text: item.text,
      intent: item.intent,
    }));
  }

  const obj = raw as Record<string, unknown>;

  // Our native format
  if (obj.intents && Array.isArray(obj.intents)) {
    const intents = obj.intents as Array<{ name: string; examples: Array<{ text: string }> }>;
    return intents.flatMap((intent) =>
      intent.examples.map((ex) => ({ text: ex.text, intent: intent.name }))
    );
  }

  // Rasa format
  if (obj.nlu && Array.isArray(obj.nlu)) {
    const nlu = obj.nlu as Array<{ intent: string; examples: string }>;
    return nlu.flatMap((item) => {
      const examples = item.examples
        .split("\n")
        .map((line: string) => line.replace(/^-\s*/, "").trim())
        .filter((line: string) => line.length > 0);
      return examples.map((text: string) => ({ text, intent: item.intent }));
    });
  }

  throw new Error("Unrecognized data format. Expected array of {text, intent} or {intents: [...]}");
}

program.parse();
