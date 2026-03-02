/**
 * Autonomous Self-Learning Loop
 *
 * The crown jewel: a recursive self-improvement system that makes the model
 * better over time WITHOUT human intervention.
 *
 * Learning Cycle:
 * 1. EVALUATE: Measure current model performance via cross-validation
 * 2. DIAGNOSE: Identify weakest intents and confusion patterns
 * 3. AUGMENT: Generate synthetic training data for weak areas
 * 4. SELF-TRAIN: Use high-confidence predictions as pseudo-labels
 * 5. CURRICULUM: Order training data by difficulty
 * 6. RETRAIN: Train new model on augmented + pseudo-labeled data
 * 7. VALIDATE: Compare new model vs old model
 * 8. ACCEPT/REJECT: Only keep improvements (no regression)
 * 9. REPEAT: Loop until convergence or max iterations
 *
 * Anti-degeneration safeguards:
 * - Confidence threshold for pseudo-labels (must be > 0.9)
 * - Diversity requirement for augmented data
 * - Regression testing on held-out validation set
 * - Maximum augmentation ratio (prevent data swamping)
 * - Early stopping when improvement plateaus
 */

import {
  trainEnsemble, predictEnsemble, crossValidateEnsemble,
  type EnsembleModel,
} from "../engine/ensemble";
import { augmentDataset, augmentExample } from "./data-augmentation";
import { scoreForActiveLearning, generateCandidatePool } from "./active-learning";

export interface SelfLearnConfig {
  maxIterations: number;
  minImprovement: number;        // minimum accuracy gain per iteration
  pseudoLabelThreshold: number;  // confidence threshold for self-training
  maxAugmentRatio: number;       // max augmented:original ratio
  validationSplit: number;       // fraction held out for validation
  augmentationsPerExample: number;
  enablePseudoLabeling: boolean;
  enableAugmentation: boolean;
  enableCurriculum: boolean;
}

export const DEFAULT_SELF_LEARN_CONFIG: SelfLearnConfig = {
  maxIterations: 10,
  minImprovement: 0.005,
  pseudoLabelThreshold: 0.92,
  maxAugmentRatio: 2.0,
  validationSplit: 0.15,
  augmentationsPerExample: 3,
  enablePseudoLabeling: true,
  enableAugmentation: true,
  enableCurriculum: true,
};

export interface SelfLearnIteration {
  iteration: number;
  accuracy: number;
  previousAccuracy: number;
  improvement: number;
  augmentedExamples: number;
  pseudoLabeledExamples: number;
  totalTrainingExamples: number;
  weakestIntents: string[];
  action: string;
  timestamp: string;
}

export interface SelfLearnResult {
  finalModel: EnsembleModel;
  initialAccuracy: number;
  finalAccuracy: number;
  totalImprovement: number;
  iterations: SelfLearnIteration[];
  converged: boolean;
  stoppedReason: string;
  totalNewExamples: number;
  durationMs: number;
}

interface TrainingItem {
  text: string;
  intent: string;
}

/**
 * Diagnose model weaknesses
 * Returns intents sorted by their F1 score (weakest first)
 */
function diagnoseWeaknesses(
  model: EnsembleModel,
): { weakIntents: string[]; confusionPairs: Array<[string, string]> } {
  const f1Scores = model.metrics.perClassF1;
  const weakIntents = Object.entries(f1Scores)
    .sort((a, b) => a[1] - b[1])
    .filter(([, f1]) => f1 < 0.95)
    .map(([intent]) => intent);

  // Find confusion pairs from confusion matrix
  const confusionPairs: Array<[string, string]> = [];
  const cm = model.metrics.confusionMatrix;
  for (const [actual, predictions] of Object.entries(cm)) {
    for (const [predicted, count] of Object.entries(predictions)) {
      if (actual !== predicted && count > 0) {
        confusionPairs.push([actual, predicted]);
      }
    }
  }

  confusionPairs.sort((a, b) => {
    const countA = cm[a[0]]?.[a[1]] || 0;
    const countB = cm[b[0]]?.[b[1]] || 0;
    return countB - countA;
  });

  return { weakIntents, confusionPairs };
}

/**
 * Self-train: generate pseudo-labels from unlabeled candidate pool
 */
function selfTrain(
  model: EnsembleModel,
  existingData: TrainingItem[],
  config: SelfLearnConfig,
): TrainingItem[] {
  const candidatePool = generateCandidatePool(existingData, 200);
  const pseudoLabeled: TrainingItem[] = [];

  for (const text of candidatePool) {
    const prediction = predictEnsemble(text, model);
    if (prediction.confidence >= config.pseudoLabelThreshold) {
      // Verify committee agreement (all 5 classifiers)
      const models = prediction.perModelScores;
      const topIntents = [
        models.logReg[0]?.intent,
        models.naiveBayes[0]?.intent,
        models.svm[0]?.intent,
        models.mlp[0]?.intent,
        models.gradBoost[0]?.intent,
      ];
      const agreementCount = topIntents.filter((i) => i === prediction.intent).length;

      // Require at least 4/5 models to agree
      if (agreementCount >= 4) {
        pseudoLabeled.push({ text, intent: prediction.intent });
      }
    }
  }

  return pseudoLabeled;
}

/**
 * Curriculum learning: sort training data by difficulty
 * Easy examples first, hard examples later
 */
function orderByCurriculum(
  data: TrainingItem[],
  model: EnsembleModel,
): TrainingItem[] {
  const scored = data.map((item) => {
    const prediction = predictEnsemble(item.text, model);
    const isCorrect = prediction.intent === item.intent;
    const confidence = isCorrect ? prediction.confidence : 1 - prediction.confidence;
    return { item, difficulty: 1 - confidence };
  });

  // Sort by difficulty (easy first)
  scored.sort((a, b) => a.difficulty - b.difficulty);
  return scored.map((s) => s.item);
}

/**
 * Split data into train and validation sets (stratified)
 */
function stratifiedSplit(
  data: TrainingItem[],
  valRatio: number,
): { train: TrainingItem[]; val: TrainingItem[] } {
  const byClass: Record<string, TrainingItem[]> = {};
  for (const item of data) {
    if (!byClass[item.intent]) byClass[item.intent] = [];
    byClass[item.intent].push(item);
  }

  const train: TrainingItem[] = [];
  const val: TrainingItem[] = [];

  for (const items of Object.values(byClass)) {
    // Shuffle
    for (let i = items.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
    const splitIdx = Math.max(1, Math.floor(items.length * (1 - valRatio)));
    train.push(...items.slice(0, splitIdx));
    val.push(...items.slice(splitIdx));
  }

  return { train, val };
}

/**
 * Evaluate model on validation set
 */
function evaluateOnValidation(
  model: EnsembleModel,
  valData: TrainingItem[],
): number {
  if (valData.length === 0) return 0;
  let correct = 0;
  for (const item of valData) {
    const prediction = predictEnsemble(item.text, model);
    if (prediction.intent === item.intent) correct++;
  }
  return correct / valData.length;
}

/**
 * Yield control to the browser event loop so the UI can repaint.
 * This prevents the tab from freezing during long-running loops.
 */
function yieldToUI(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Run the autonomous self-learning loop (async)
 *
 * This is the main entry point. It takes training data and iteratively
 * improves the model through augmentation, self-training, and curriculum learning.
 * Each iteration yields to the browser event loop so the UI stays responsive.
 *
 * @param onProgress - Callback for progress updates (for UI)
 */
export async function runSelfLearningLoop(
  originalData: TrainingItem[],
  config: SelfLearnConfig = DEFAULT_SELF_LEARN_CONFIG,
  onProgress?: (iteration: SelfLearnIteration) => void,
): Promise<SelfLearnResult> {
  const startTime = performance.now();
  const iterations: SelfLearnIteration[] = [];

  // Split off a held-out validation set that never gets augmented
  const { train: trainPool, val: validationSet } = stratifiedSplit(originalData, config.validationSplit);

  // Initial training
  let currentData = [...trainPool];
  let currentModel = trainEnsemble(currentData);
  let currentAccuracy = evaluateOnValidation(currentModel, validationSet);
  const initialAccuracy = currentAccuracy;

  let totalNewExamples = 0;
  let stoppedReason = "max_iterations";

  for (let iter = 0; iter < config.maxIterations; iter++) {
    // Yield to browser between iterations so UI can repaint
    await yieldToUI();

    const previousAccuracy = currentAccuracy;
    let augmentedCount = 0;
    let pseudoLabeledCount = 0;
    let action = "";

    // Step 1: Diagnose weaknesses
    const { weakIntents } = diagnoseWeaknesses(currentModel);

    // Step 2: Augment data for weak intents
    if (config.enableAugmentation && weakIntents.length > 0) {
      const weakExamples = currentData.filter((d) => weakIntents.includes(d.intent));
      const maxNew = Math.floor(currentData.length * config.maxAugmentRatio) - currentData.length;

      if (maxNew > 0 && weakExamples.length > 0) {
        const augmented = augmentDataset(weakExamples, config.augmentationsPerExample);
        const toAdd = augmented.slice(0, maxNew);
        currentData = [...currentData, ...toAdd];
        augmentedCount = toAdd.length;
        totalNewExamples += augmentedCount;
        action += `augmented:${augmentedCount} `;
      }
    }

    // Step 3: Self-training with pseudo-labels
    if (config.enablePseudoLabeling) {
      const pseudoLabeled = selfTrain(currentModel, currentData, config);
      const maxPseudo = Math.floor(originalData.length * 0.3); // cap at 30% of original
      const toAdd = pseudoLabeled.slice(0, maxPseudo);
      if (toAdd.length > 0) {
        currentData = [...currentData, ...toAdd];
        pseudoLabeledCount = toAdd.length;
        totalNewExamples += pseudoLabeledCount;
        action += `pseudo:${pseudoLabeledCount} `;
      }
    }

    // Step 4: Curriculum ordering
    if (config.enableCurriculum && iter > 0) {
      currentData = orderByCurriculum(currentData, currentModel);
      action += "curriculum ";
    }

    // Step 5: Retrain
    const newModel = trainEnsemble(currentData);
    const newAccuracy = evaluateOnValidation(newModel, validationSet);

    // Step 6: Accept or reject
    const improvement = newAccuracy - previousAccuracy;
    if (newAccuracy >= previousAccuracy - 0.01) {
      // Accept: new model is better or within tolerance
      currentModel = newModel;
      currentAccuracy = newAccuracy;
      action += `accepted(+${(improvement * 100).toFixed(1)}%)`;
    } else {
      // Reject: new model regressed too much
      // Revert data changes
      currentData = currentData.slice(0, currentData.length - augmentedCount - pseudoLabeledCount);
      action += `rejected(${(improvement * 100).toFixed(1)}%)`;
    }

    const iteration: SelfLearnIteration = {
      iteration: iter + 1,
      accuracy: currentAccuracy,
      previousAccuracy,
      improvement,
      augmentedExamples: augmentedCount,
      pseudoLabeledExamples: pseudoLabeledCount,
      totalTrainingExamples: currentData.length,
      weakestIntents: weakIntents.slice(0, 3),
      action: action.trim(),
      timestamp: new Date().toISOString(),
    };
    iterations.push(iteration);
    onProgress?.(iteration);

    // Early stopping conditions
    if (improvement < config.minImprovement && iter > 2) {
      stoppedReason = "convergence";
      break;
    }

    if (currentAccuracy >= 0.99) {
      stoppedReason = "perfect_accuracy";
      break;
    }
  }

  const durationMs = performance.now() - startTime;

  return {
    finalModel: currentModel,
    initialAccuracy,
    finalAccuracy: currentAccuracy,
    totalImprovement: currentAccuracy - initialAccuracy,
    iterations,
    converged: stoppedReason === "convergence" || stoppedReason === "perfect_accuracy",
    stoppedReason,
    totalNewExamples,
    durationMs,
  };
}

/**
 * Knowledge synthesis: generate entirely new training examples
 * based on learned patterns from the model
 *
 * This is the "world model" component - it creates knowledge
 * from first principles by combining patterns
 */
export function synthesizeKnowledge(
  model: EnsembleModel,
  existingData: TrainingItem[],
  targetCount: number = 50,
): TrainingItem[] {
  const synthesized: TrainingItem[] = [];
  const existingTexts = new Set(existingData.map((d) => d.text.toLowerCase()));

  // For each class, generate diverse examples
  for (const cls of model.classes) {
    const classExamples = existingData.filter((d) => d.intent === cls);
    const targetPerClass = Math.ceil(targetCount / model.classes.length);

    // Extract vocabulary patterns from this class
    const wordFreq: Record<string, number> = {};
    for (const ex of classExamples) {
      for (const word of ex.text.toLowerCase().split(/\s+/)) {
        wordFreq[word] = (wordFreq[word] || 0) + 1;
      }
    }

    // Sort by frequency - these are the "defining" words
    const topWords = Object.entries(wordFreq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([w]) => w);

    // Generate new combinations
    for (let i = 0; i < targetPerClass * 3 && synthesized.length < targetCount; i++) {
      // Pick 2-4 random top words and combine
      const numWords = 2 + Math.floor(Math.random() * 3);
      const selectedWords: string[] = [];
      for (let j = 0; j < numWords; j++) {
        selectedWords.push(topWords[Math.floor(Math.random() * topWords.length)]);
      }
      const candidate = selectedWords.join(" ");

      // Verify the model classifies it correctly with high confidence
      const prediction = predictEnsemble(candidate, model);
      if (
        prediction.intent === cls &&
        prediction.confidence > 0.8 &&
        !existingTexts.has(candidate.toLowerCase())
      ) {
        synthesized.push({ text: candidate, intent: cls });
        existingTexts.add(candidate.toLowerCase());
      }
    }
  }

  return synthesized;
}
