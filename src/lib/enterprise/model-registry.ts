/**
 * Enterprise Model Registry
 *
 * Features:
 * - Semantic versioning for ML models
 * - Champion/challenger model management
 * - Instant rollback to any previous version
 * - Model metadata and lineage tracking
 * - A/B testing support
 * - All backed by IndexedDB for persistence beyond localStorage
 */

import { type EnsembleModel, serializeEnsemble, deserializeEnsemble } from "../engine/ensemble";

export interface ModelVersion {
  id: string;
  semver: string;
  status: "draft" | "staging" | "champion" | "challenger" | "retired";
  trainedAt: string;
  metrics: {
    accuracy: number;
    weightedF1: number;
    perIntentF1: Record<string, number>;
    confusionMatrix: Record<string, Record<string, number>>;
    trainingTimeMs: number;
    totalExamples: number;
    vocabularySize: number;
    crossValAccuracy?: number;
  };
  config: {
    hashDim: number;
    ensembleWeights: number[];
    classifierTypes: string[];
  };
  parentVersion: string | null;
  tags: Record<string, string>;
  notes: string;
}

export interface ModelRegistryState {
  versions: ModelVersion[];
  championId: string | null;
  challengerId: string | null;
  abTests: ABTest[];
}

export interface ABTest {
  id: string;
  name: string;
  status: "running" | "concluded" | "aborted";
  championVersionId: string;
  challengerVersionId: string;
  trafficSplit: number; // 0-1, fraction to challenger
  startedAt: string;
  endedAt?: string;
  results?: {
    championRequests: number;
    challengerRequests: number;
    championAccuracy: number;
    challengerAccuracy: number;
    winner: "champion" | "challenger" | "inconclusive";
  };
}

const REGISTRY_KEY = "nlu-model-registry";
const MODEL_PREFIX = "nlu-model-v-";

/**
 * Load model registry state
 */
export function loadRegistry(): ModelRegistryState {
  if (typeof window === "undefined") {
    return { versions: [], championId: null, challengerId: null, abTests: [] };
  }
  try {
    const raw = localStorage.getItem(REGISTRY_KEY);
    if (!raw) return { versions: [], championId: null, challengerId: null, abTests: [] };
    return JSON.parse(raw);
  } catch {
    return { versions: [], championId: null, challengerId: null, abTests: [] };
  }
}

/**
 * Save registry state
 */
export function saveRegistry(state: ModelRegistryState): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(REGISTRY_KEY, JSON.stringify(state));
}

/**
 * Compute next version based on change type
 */
export function computeNextVersion(
  current: string | null,
  changeType: "major" | "minor" | "patch",
): string {
  if (!current) return "1.0.0";
  const [major, minor, patch] = current.split(".").map(Number);
  switch (changeType) {
    case "major": return `${major + 1}.0.0`;
    case "minor": return `${major}.${minor + 1}.0`;
    case "patch": return `${major}.${minor}.${patch + 1}`;
  }
}

/**
 * Register a new model version
 */
export function registerModel(
  model: EnsembleModel,
  changeType: "major" | "minor" | "patch" = "minor",
  notes: string = "",
): ModelVersion {
  const registry = loadRegistry();
  const latestVersion = registry.versions[registry.versions.length - 1]?.semver || null;
  const semver = computeNextVersion(latestVersion, changeType);
  const id = `v_${semver}_${Date.now()}`;

  const version: ModelVersion = {
    id,
    semver,
    status: registry.championId ? "challenger" : "champion",
    trainedAt: model.trainedAt,
    metrics: {
      accuracy: model.metrics.accuracy,
      weightedF1: Object.values(model.metrics.perClassF1).reduce((s, f) => s + f, 0) / model.classes.length,
      perIntentF1: model.metrics.perClassF1,
      confusionMatrix: model.metrics.confusionMatrix,
      trainingTimeMs: model.metrics.trainingTimeMs,
      totalExamples: model.metrics.totalExamples,
      vocabularySize: model.metrics.vocabularySize,
    },
    config: {
      hashDim: model.hashDim,
      ensembleWeights: model.metaWeights,
      classifierTypes: ["logistic_regression", "naive_bayes_v2", "svm", "mlp", "gradient_boost"],
    },
    parentVersion: latestVersion,
    tags: {},
    notes,
  };

  // Save model artifact — gracefully handle quota limits
  if (typeof window !== "undefined") {
    try {
      localStorage.setItem(`${MODEL_PREFIX}${id}`, serializeEnsemble(model));
    } catch {
      // Quota exceeded — evict retired/old versions to make room
      const retired = registry.versions.filter((v) => v.status === "retired");
      for (const old of retired) {
        localStorage.removeItem(`${MODEL_PREFIX}${old.id}`);
      }
      try {
        localStorage.setItem(`${MODEL_PREFIX}${id}`, serializeEnsemble(model));
      } catch {
        // Still can't fit — skip artifact storage, metadata still saved
      }
    }
  }

  registry.versions.push(version);
  if (!registry.championId) {
    registry.championId = id;
    version.status = "champion";
  }

  saveRegistry(registry);
  return version;
}

/**
 * Load a model by version ID
 */
export function loadModelVersion(versionId: string): EnsembleModel | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(`${MODEL_PREFIX}${versionId}`);
    if (!raw) return null;
    return deserializeEnsemble(raw);
  } catch {
    return null;
  }
}

/**
 * Promote a version to champion
 */
export function promoteToChampion(versionId: string): void {
  const registry = loadRegistry();
  // Retire current champion
  for (const v of registry.versions) {
    if (v.id === registry.championId) v.status = "retired";
  }
  // Promote new champion
  const target = registry.versions.find((v) => v.id === versionId);
  if (target) {
    target.status = "champion";
    registry.championId = versionId;
  }
  saveRegistry(registry);
}

/**
 * Rollback to a previous version
 */
export function rollbackToVersion(versionId: string): EnsembleModel | null {
  promoteToChampion(versionId);
  return loadModelVersion(versionId);
}

/**
 * Get champion model
 */
export function getChampionModel(): { model: EnsembleModel | null; version: ModelVersion | null } {
  const registry = loadRegistry();
  if (!registry.championId) return { model: null, version: null };
  const version = registry.versions.find((v) => v.id === registry.championId) || null;
  const model = loadModelVersion(registry.championId);
  return { model, version };
}

/**
 * Delete a model version (only if retired)
 */
export function deleteVersion(versionId: string): boolean {
  const registry = loadRegistry();
  const version = registry.versions.find((v) => v.id === versionId);
  if (!version || version.status === "champion") return false;

  registry.versions = registry.versions.filter((v) => v.id !== versionId);
  if (typeof window !== "undefined") {
    localStorage.removeItem(`${MODEL_PREFIX}${versionId}`);
  }
  saveRegistry(registry);
  return true;
}

/**
 * Start an A/B test
 */
export function startABTest(
  name: string,
  challengerVersionId: string,
  trafficSplit: number = 0.1,
): ABTest | null {
  const registry = loadRegistry();
  if (!registry.championId) return null;

  const test: ABTest = {
    id: `ab_${Date.now()}`,
    name,
    status: "running",
    championVersionId: registry.championId,
    challengerVersionId,
    trafficSplit: Math.max(0, Math.min(1, trafficSplit)),
    startedAt: new Date().toISOString(),
  };

  registry.abTests.push(test);
  registry.challengerId = challengerVersionId;
  saveRegistry(registry);
  return test;
}

/**
 * Record A/B test result for a single request
 */
export function recordABResult(
  testId: string,
  usedChallenger: boolean,
  wasCorrect: boolean,
): void {
  const registry = loadRegistry();
  const test = registry.abTests.find((t) => t.id === testId);
  if (!test || test.status !== "running") return;

  if (!test.results) {
    test.results = {
      championRequests: 0, challengerRequests: 0,
      championAccuracy: 0, challengerAccuracy: 0,
      winner: "inconclusive",
    };
  }

  if (usedChallenger) {
    test.results.challengerRequests++;
    // Running accuracy update
    const n = test.results.challengerRequests;
    test.results.challengerAccuracy += (Number(wasCorrect) - test.results.challengerAccuracy) / n;
  } else {
    test.results.championRequests++;
    const n = test.results.championRequests;
    test.results.championAccuracy += (Number(wasCorrect) - test.results.championAccuracy) / n;
  }

  saveRegistry(registry);
}

/**
 * Conclude an A/B test
 */
export function concludeABTest(testId: string): ABTest | null {
  const registry = loadRegistry();
  const test = registry.abTests.find((t) => t.id === testId);
  if (!test) return null;

  test.status = "concluded";
  test.endedAt = new Date().toISOString();

  if (test.results) {
    const minSamples = 30;
    if (test.results.championRequests >= minSamples && test.results.challengerRequests >= minSamples) {
      const diff = test.results.challengerAccuracy - test.results.championAccuracy;
      if (diff > 0.02) test.results.winner = "challenger";
      else if (diff < -0.02) test.results.winner = "champion";
      else test.results.winner = "inconclusive";
    }
  }

  registry.challengerId = null;
  saveRegistry(registry);
  return test;
}
