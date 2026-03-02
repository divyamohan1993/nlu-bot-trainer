/**
 * NLU Bot Trainer — Public API
 *
 * Browser-native intent classification engine.
 * 5-classifier stacking ensemble (171K params), zero server dependency.
 *
 * @example
 * ```typescript
 * import { trainEnsemble, predictEnsemble } from 'nlu-bot-trainer';
 *
 * const data = [
 *   { text: "hello", intent: "greet" },
 *   { text: "bye", intent: "goodbye" },
 *   // ... more examples
 * ];
 *
 * const model = trainEnsemble(data);
 * const result = predictEnsemble("hi there", model);
 * console.log(result.intent, result.confidence);
 * ```
 *
 * @packageDocumentation
 */

// === Core Engine ===
export {
  trainEnsemble,
  predictEnsemble,
  crossValidateEnsemble,
  serializeEnsemble,
  deserializeEnsemble,
  type EnsembleModel,
} from "./engine/ensemble";

// === Feature Extraction ===
export {
  hashFeaturesTF,
  hashFeaturesDense,
  hashFeaturesSparse,
  hashFeature,
  dotProduct,
  DEFAULT_HASH_DIM,
  quantizeToInt8,
  dequantizeFromInt8,
} from "./engine/feature-hasher";

// === Tokenization ===
export {
  tokenizeV2 as tokenize,
  normalizeText,
  DEFAULT_TOKENIZER_CONFIG,
  type TokenizerConfig,
} from "./engine/tokenizer-v2";

// === Out-of-Scope Detection ===
export {
  detectOOS,
  calibrateOOS,
  evaluateOOS,
  DEFAULT_OOS_CONFIG,
  type OOSConfig,
  type OOSResult,
} from "./engine/oos-detector";

// === Confidence Calibration ===
export {
  learnTemperature,
  applyTemperature,
  calibrateConfidence,
  DEFAULT_CALIBRATION_CONFIG,
  type CalibrationConfig,
  type CalibrationResult,
  type ReliabilityBin,
} from "./engine/confidence-calibration";

// === Prediction Explanation ===
export {
  explainPrediction,
  batchExplain,
  explainMisclassification,
  type ExplanationResult,
  type TokenExplanation,
} from "./engine/explainer";

// === Data Quality ===
export {
  validateDataQuality,
  type DataQualityReport,
  type DataQualityIssue,
  type IssueSeverity,
  type IssueCategory,
} from "./engine/data-quality";

// === Self-Learning ===
export {
  runSelfLearningLoop,
  synthesizeKnowledge,
  DEFAULT_SELF_LEARN_CONFIG,
  type SelfLearnConfig,
  type SelfLearnResult,
  type SelfLearnIteration,
} from "./self-learn/autonomous-loop";

// === Active Learning ===
export {
  scoreForActiveLearning,
  selectDiverseBatch,
  type ActiveLearningCandidate,
} from "./self-learn/active-learning";

// === Data Augmentation ===
export {
  augmentDataset,
  augmentExample,
} from "./self-learn/data-augmentation";

// === Enterprise: Model Registry ===
export {
  registerModel,
  loadRegistry,
  loadModelVersion,
  promoteToChampion,
  rollbackToVersion,
  getChampionModel,
  deleteVersion,
  startABTest,
  recordABResult,
  concludeABTest,
  computeNextVersion,
  type ModelVersion,
  type ModelRegistryState,
  type ABTest,
} from "./enterprise/model-registry";

// === Enterprise: Drift Detection ===
export {
  recordPrediction,
  getDriftReport,
  loadDriftState,
  type DriftState,
} from "./enterprise/drift-detector";

// === Enterprise: Export Formats ===
export {
  exportTrainingData,
  importFromJSON,
  type ExportFormat,
} from "./enterprise/export-formats";
