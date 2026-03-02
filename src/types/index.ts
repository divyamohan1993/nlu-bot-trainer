export interface TrainingExample {
  id: string;
  text: string;
  intent: string;
  entities: EntityAnnotation[];
}

export interface EntityAnnotation {
  start: number;
  end: number;
  value: string;
  entity: string;
}

export interface Intent {
  id: string;
  name: string;
  description: string;
  examples: TrainingExample[];
  color: string;
}

export interface Entity {
  id: string;
  name: string;
  description: string;
  values: string[];
  color: string;
}

export interface PredictionResult {
  text: string;
  intent: {
    name: string;
    confidence: number;
  };
  intent_ranking: Array<{
    name: string;
    confidence: number;
  }>;
  entities: EntityAnnotation[];
}

export interface TrainingData {
  intents: Intent[];
  entities: Entity[];
  metadata: {
    trainedAt: string | null;
    totalExamples: number;
    version: string;
  };
}

export interface ModelState {
  isTrained: boolean;
  trainedAt: string | null;
  accuracy: number | null;
  totalExamples: number;
  vocabulary: string[];
  idfWeights: Record<string, number>;
  intentVectors: Record<string, number[]>;
}

/** Extended prediction result with per-model breakdown (v2 engine) */
export interface PredictionResultV2 extends PredictionResult {
  inferenceTimeUs: number;
  perModelScores?: {
    logReg: Array<{ intent: string; score: number }>;
    naiveBayes: Array<{ intent: string; score: number }>;
    svm: Array<{ intent: string; score: number }>;
    mlp: Array<{ intent: string; score: number }>;
    gradBoost: Array<{ intent: string; score: number }>;
  };
}
