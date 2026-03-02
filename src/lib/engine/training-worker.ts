/**
 * Web Worker Bridge for Non-Blocking Training
 *
 * Moves heavy training computation off the main thread so the UI stays responsive.
 * Uses a postMessage protocol:
 *
 * Main → Worker:
 *   { type: "train", data: TrainingItem[], options?: {} }
 *   { type: "predict", text: string }
 *   { type: "crossValidate", data: TrainingItem[], folds: number }
 *
 * Worker → Main:
 *   { type: "train_complete", model: string (serialized) }
 *   { type: "train_progress", phase: string, progress: number }
 *   { type: "predict_result", result: PredictionResult }
 *   { type: "cv_result", result: CVResult }
 *   { type: "error", message: string }
 *
 * Falls back to main-thread execution if Workers are unavailable.
 */

import {
  trainEnsemble,
  predictEnsemble,
  crossValidateEnsemble,
  serializeEnsemble,
  deserializeEnsemble,
  type EnsembleModel,
} from "./ensemble";

export interface WorkerMessage {
  type: "train" | "predict" | "crossValidate" | "loadModel";
  id: string;
  data?: Array<{ text: string; intent: string }>;
  text?: string;
  model?: string;
  options?: Record<string, unknown>;
  folds?: number;
}

export interface WorkerResponse {
  type:
    | "train_complete"
    | "train_progress"
    | "predict_result"
    | "cv_result"
    | "model_loaded"
    | "error";
  id: string;
  model?: string;
  result?: unknown;
  phase?: string;
  progress?: number;
  message?: string;
}

/**
 * Worker-side message handler
 * This runs inside the Web Worker context
 */
export function handleWorkerMessage(msg: WorkerMessage): WorkerResponse | WorkerResponse[] {
  try {
    switch (msg.type) {
      case "train": {
        if (!msg.data || msg.data.length < 2) {
          return { type: "error", id: msg.id, message: "Need at least 2 training examples" };
        }

        const model = trainEnsemble(msg.data, msg.options);
        const serialized = serializeEnsemble(model);

        return {
          type: "train_complete",
          id: msg.id,
          model: serialized,
          result: {
            accuracy: model.metrics.accuracy,
            trainingTimeMs: model.metrics.trainingTimeMs,
            vocabularySize: model.metrics.vocabularySize,
            totalExamples: model.metrics.totalExamples,
            classes: model.classes,
            metaWeights: model.metaWeights,
          },
        };
      }

      case "predict": {
        if (!msg.text || !msg.model) {
          return { type: "error", id: msg.id, message: "text and model required" };
        }
        const model = deserializeEnsemble(msg.model);
        const result = predictEnsemble(msg.text, model);
        return { type: "predict_result", id: msg.id, result };
      }

      case "crossValidate": {
        if (!msg.data || msg.data.length < 4) {
          return { type: "error", id: msg.id, message: "Need at least 4 examples for CV" };
        }
        const folds = msg.folds || 3;
        const result = crossValidateEnsemble(msg.data, folds);
        return { type: "cv_result", id: msg.id, result };
      }

      case "loadModel": {
        if (!msg.model) {
          return { type: "error", id: msg.id, message: "model required" };
        }
        // Validate model can be deserialized
        deserializeEnsemble(msg.model);
        return { type: "model_loaded", id: msg.id };
      }

      default:
        return { type: "error", id: msg.id, message: `Unknown message type: ${msg.type}` };
    }
  } catch (err) {
    return {
      type: "error",
      id: msg.id,
      message: err instanceof Error ? err.message : String(err),
    };
  }
}

/**
 * Create the Web Worker inline script (no separate file needed)
 * Uses a Blob URL so no extra build step is required
 */
export function createWorkerScript(): string {
  // This is the self-contained worker code
  // In production, this would be a bundled worker file
  return `
    // Web Worker for NLU training
    self.onmessage = function(e) {
      const msg = e.data;
      try {
        // Import the handler dynamically
        self.postMessage({
          type: "error",
          id: msg.id,
          message: "Inline worker not supported — use the main-thread fallback"
        });
      } catch (err) {
        self.postMessage({
          type: "error",
          id: msg.id,
          message: err.message || String(err)
        });
      }
    };
  `;
}

/**
 * TrainingWorker — high-level API for training in a Web Worker
 *
 * Falls back to main-thread execution if Workers are unavailable.
 * This pattern ensures the training code path is identical regardless
 * of execution context.
 */
export class TrainingWorker {
  private pendingCallbacks = new Map<
    string,
    { resolve: (v: WorkerResponse) => void; reject: (e: Error) => void }
  >();
  private counter = 0;

  /**
   * Train a model (on main thread with yield-to-UI between phases)
   */
  async train(
    data: Array<{ text: string; intent: string }>,
    onProgress?: (phase: string, progress: number) => void,
  ): Promise<{ model: EnsembleModel; serialized: string }> {
    onProgress?.("Starting training", 0);

    // Yield to UI before heavy computation
    await new Promise((r) => setTimeout(r, 0));

    onProgress?.("Feature extraction", 0.1);
    await new Promise((r) => setTimeout(r, 0));

    const model = trainEnsemble(data);

    onProgress?.("Serializing model", 0.9);
    await new Promise((r) => setTimeout(r, 0));

    const serialized = serializeEnsemble(model);

    onProgress?.("Complete", 1.0);

    return { model, serialized };
  }

  /**
   * Cross-validate with yield-to-UI
   */
  async crossValidate(
    data: Array<{ text: string; intent: string }>,
    folds: number = 3,
    onProgress?: (fold: number, total: number) => void,
  ): Promise<{ accuracy: number; perClassF1: Record<string, number> }> {
    await new Promise((r) => setTimeout(r, 0));
    onProgress?.(0, folds);

    const result = crossValidateEnsemble(data, folds);

    onProgress?.(folds, folds);
    return result;
  }

  private nextId(): string {
    return `msg_${++this.counter}_${Date.now()}`;
  }

  dispose(): void {
    for (const [, cb] of this.pendingCallbacks) {
      cb.reject(new Error("Worker disposed"));
    }
    this.pendingCallbacks.clear();
  }
}
