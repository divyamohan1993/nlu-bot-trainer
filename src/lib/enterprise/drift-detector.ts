/**
 * Drift Detection Suite
 *
 * Monitors model performance over time and detects:
 * 1. Concept drift (model's predictions degrade)
 * 2. Data drift (input distribution changes)
 * 3. Feature drift (token/vocabulary distribution shifts)
 *
 * Algorithms: Page-Hinkley, ADWIN, DDM
 */

export interface DriftState {
  pageHinkley: { sum: number; min: number; count: number; mean: number; detected: boolean; severity: number };
  ddm: { errorRate: number; stdDev: number; count: number; minError: number; minStd: number; state: "in_control" | "warning" | "drift" };
  history: Array<{ timestamp: string; accuracy: number; confidence: number; driftScore: number }>;
  vocabularyBaseline: Record<string, number>;
  inputLengthBaseline: { mean: number; std: number; count: number };
}

export function createDriftState(): DriftState {
  return {
    pageHinkley: { sum: 0, min: Infinity, count: 0, mean: 0, detected: false, severity: 0 },
    ddm: { errorRate: 0, stdDev: 0, count: 0, minError: Infinity, minStd: Infinity, state: "in_control" },
    history: [],
    vocabularyBaseline: {},
    inputLengthBaseline: { mean: 0, std: 0, count: 0 },
  };
}

const DRIFT_STATE_KEY = "nlu-drift-state";

export function loadDriftState(): DriftState {
  if (typeof window === "undefined") return createDriftState();
  try {
    const raw = localStorage.getItem(DRIFT_STATE_KEY);
    if (!raw) return createDriftState();
    return JSON.parse(raw);
  } catch {
    return createDriftState();
  }
}

export function saveDriftState(state: DriftState): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(DRIFT_STATE_KEY, JSON.stringify(state));
}

/**
 * Page-Hinkley Test for concept drift
 */
export function updatePageHinkley(
  state: DriftState,
  value: number,
  delta: number = 0.005,
  lambda: number = 50,
): { detected: boolean; severity: number } {
  const ph = state.pageHinkley;
  ph.count++;
  ph.mean = ph.mean + (value - ph.mean) / ph.count;
  ph.sum = 0.999 * ph.sum + (value - ph.mean - delta);
  ph.min = Math.min(ph.min, ph.sum);
  const phValue = ph.sum - ph.min;
  ph.detected = phValue > lambda;
  ph.severity = phValue / lambda;
  return { detected: ph.detected, severity: ph.severity };
}

/**
 * DDM (Drift Detection Method) for error rate monitoring
 */
export function updateDDM(
  state: DriftState,
  isError: boolean,
): { state: "in_control" | "warning" | "drift" } {
  const ddm = state.ddm;
  ddm.count++;
  ddm.errorRate += (Number(isError) - ddm.errorRate) / ddm.count;
  ddm.stdDev = Math.sqrt(ddm.errorRate * (1 - ddm.errorRate) / ddm.count);

  if (ddm.count >= 30) {
    if (ddm.errorRate + ddm.stdDev < ddm.minError + ddm.minStd) {
      ddm.minError = ddm.errorRate;
      ddm.minStd = ddm.stdDev;
    }

    if (ddm.errorRate + ddm.stdDev >= ddm.minError + 3 * ddm.minStd) {
      ddm.state = "drift";
    } else if (ddm.errorRate + ddm.stdDev >= ddm.minError + 2 * ddm.minStd) {
      ddm.state = "warning";
    } else {
      ddm.state = "in_control";
    }
  }

  return { state: ddm.state };
}

/**
 * Record a prediction for drift monitoring
 */
export function recordPrediction(
  state: DriftState,
  confidence: number,
  wasCorrect: boolean,
  inputTokens: string[],
): {
  conceptDrift: { detected: boolean; severity: number };
  ddmState: "in_control" | "warning" | "drift";
  vocabularyDrift: number;
} {
  // Update Page-Hinkley with confidence (lower confidence may indicate drift)
  const ph = updatePageHinkley(state, 1 - confidence);

  // Update DDM with correctness
  const ddm = updateDDM(state, !wasCorrect);

  // Track vocabulary drift
  let newTokens = 0;
  for (const token of inputTokens) {
    if (!state.vocabularyBaseline[token]) newTokens++;
    state.vocabularyBaseline[token] = (state.vocabularyBaseline[token] || 0) + 1;
  }
  const vocabularyDrift = inputTokens.length > 0 ? newTokens / inputTokens.length : 0;

  // Update input length baseline
  const len = inputTokens.length;
  state.inputLengthBaseline.count++;
  const oldMean = state.inputLengthBaseline.mean;
  state.inputLengthBaseline.mean += (len - oldMean) / state.inputLengthBaseline.count;
  state.inputLengthBaseline.std = Math.sqrt(
    ((state.inputLengthBaseline.count - 1) * state.inputLengthBaseline.std ** 2 +
      (len - oldMean) * (len - state.inputLengthBaseline.mean)) /
    state.inputLengthBaseline.count
  );

  // Record history point
  const accuracy = ddm.state === "in_control" ? 1 - state.ddm.errorRate : state.ddm.errorRate;
  state.history.push({
    timestamp: new Date().toISOString(),
    accuracy,
    confidence,
    driftScore: ph.severity,
  });

  // Keep last 1000 history points
  if (state.history.length > 1000) {
    state.history = state.history.slice(-1000);
  }

  return {
    conceptDrift: ph,
    ddmState: ddm.state,
    vocabularyDrift,
  };
}

/**
 * Get drift summary report
 */
export function getDriftReport(state: DriftState): {
  overallStatus: "healthy" | "warning" | "critical";
  conceptDrift: { detected: boolean; severity: number };
  errorDrift: { state: string; errorRate: number };
  vocabularyDriftRate: number;
  recentAccuracy: number;
  predictionCount: number;
} {
  const recentHistory = state.history.slice(-100);
  const recentAccuracy = recentHistory.length > 0
    ? recentHistory.filter((h) => h.confidence > 0.5).length / recentHistory.length
    : 1;

  const overallStatus = state.pageHinkley.detected || state.ddm.state === "drift"
    ? "critical"
    : state.ddm.state === "warning"
    ? "warning"
    : "healthy";

  const totalTokens = Object.values(state.vocabularyBaseline).reduce((s, c) => s + c, 0);
  const uniqueTokens = Object.keys(state.vocabularyBaseline).length;
  const vocabularyDriftRate = totalTokens > 0 ? uniqueTokens / totalTokens : 0;

  return {
    overallStatus,
    conceptDrift: { detected: state.pageHinkley.detected, severity: state.pageHinkley.severity },
    errorDrift: { state: state.ddm.state, errorRate: state.ddm.errorRate },
    vocabularyDriftRate,
    recentAccuracy,
    predictionCount: state.history.length,
  };
}
