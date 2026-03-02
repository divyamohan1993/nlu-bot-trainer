import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

// Lazy-loaded model artifacts
let labels: Record<string, string> | null = null;
let vocabulary: Record<string, number> | null = null;
let idf: number[] | null = null;
let modelWeights: {
  stem_weight: number[][];
  stem_bias: number[];
  res_weight: number[][];
  res_bias: number[];
  neck_weight: number[][];
  neck_bias: number[];
  head_weight: number[][];
  head_bias: number[];
} | null = null;

const MODEL_DIR = path.join(process.cwd(), "public", "models", "vuln-classifier");

async function loadArtifacts() {
  if (labels && vocabulary && idf && modelWeights) return true;

  try {
    const [labelsRaw, vocabRaw, weightsRaw] = await Promise.all([
      fs.readFile(path.join(MODEL_DIR, "labels.json"), "utf-8"),
      fs.readFile(path.join(MODEL_DIR, "tfidf_vocab.json"), "utf-8"),
      fs.readFile(path.join(MODEL_DIR, "weights.json"), "utf-8"),
    ]);

    labels = JSON.parse(labelsRaw);
    const vocabData = JSON.parse(vocabRaw);
    vocabulary = vocabData.vocabulary;
    idf = vocabData.idf;
    modelWeights = JSON.parse(weightsRaw);
    return true;
  } catch {
    return false;
  }
}

// TF-IDF vectorization (matching scikit-learn's TfidfVectorizer)
function tfidfVectorize(text: string): number[] {
  if (!vocabulary || !idf) return [];

  const dim = idf.length;
  const vector = new Float64Array(dim);

  // Tokenize: lowercase, split on non-alphanum
  const tokens = text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);

  // Count unigrams
  const counts: Record<string, number> = {};
  for (const token of tokens) {
    counts[token] = (counts[token] || 0) + 1;
  }
  // Count bigrams
  for (let i = 0; i < tokens.length - 1; i++) {
    const bigram = tokens[i] + " " + tokens[i + 1];
    counts[bigram] = (counts[bigram] || 0) + 1;
  }

  // TF-IDF: sublinear_tf = log(1 + count) * idf
  for (const [term, count] of Object.entries(counts)) {
    const idx = vocabulary[term];
    if (idx !== undefined && idx < dim) {
      vector[idx] = Math.log(1 + count) * idf[idx];
    }
  }

  // L2 normalize
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vector[i] * vector[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < dim; i++) vector[i] /= norm;
  }

  return Array.from(vector);
}

// Forward pass through the ResNet-MLP (pure JS, no runtime dependency)
function predict(input: number[]): Array<{ cwe: string; score: number; name: string }> {
  if (!modelWeights || !labels) return [];

  // Stem: Linear(50K, 192) + GELU
  let h = matMulBias(input, modelWeights.stem_weight, modelWeights.stem_bias);
  h = h.map(gelu);

  // ResBlock: Linear(192, 192) + GELU + skip
  const resOut = matMulBias(h, modelWeights.res_weight, modelWeights.res_bias);
  h = h.map((v, i) => v + gelu(resOut[i])); // skip connection

  // Neck: Linear(192, 96) + GELU
  h = matMulBias(h, modelWeights.neck_weight, modelWeights.neck_bias);
  h = h.map(gelu);

  // Head: Linear(96, num_classes)
  const logits = matMulBias(h, modelWeights.head_weight, modelWeights.head_bias);

  // Softmax
  const probs = softmax(logits);

  // Top predictions
  const results = probs
    .map((score, i) => ({
      cwe: labels![String(i)] || `CWE-${i}`,
      score,
      name: labels![String(i)] || "Unknown",
    }))
    .sort((a, b) => b.score - a.score);

  return results;
}

function matMulBias(input: number[], weight: number[][], bias: number[]): number[] {
  const out = new Array(bias.length);
  for (let j = 0; j < bias.length; j++) {
    let sum = bias[j];
    const wj = weight[j];
    for (let i = 0; i < input.length; i++) {
      if (input[i] !== 0) sum += input[i] * wj[i];
    }
    out[j] = sum;
  }
  return out;
}

function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export async function POST(request: NextRequest) {
  const loaded = await loadArtifacts();
  if (!loaded) {
    return NextResponse.json(
      {
        error: "Model not loaded. Place model artifacts in public/models/vuln-classifier/",
        needed: ["labels.json", "tfidf_vocab.json", "weights.json"],
      },
      { status: 503 }
    );
  }

  let body: { text: string; topK?: number };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!body.text || body.text.trim().length < 10) {
    return NextResponse.json(
      { error: "Provide a CVE description (at least 10 characters)" },
      { status: 400 }
    );
  }

  const t0 = performance.now();
  const vector = tfidfVectorize(body.text);
  const results = predict(vector);
  const inferenceMs = performance.now() - t0;

  const topK = Math.min(body.topK || 10, results.length);

  return NextResponse.json({
    predictions: results.slice(0, topK),
    inferenceMs: Math.round(inferenceMs * 100) / 100,
    modelInfo: {
      parameters: "9.7M",
      classes: results.length,
      architecture: "ResNet-MLP (50K→192→192→96→N)",
    },
  });
}
