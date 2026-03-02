/**
 * REST Inference API — POST /api/predict
 *
 * Stateless prediction endpoint for the NLU ensemble model.
 * Loads the model from localStorage-compatible server-side store,
 * or accepts an inline model payload.
 *
 * Request body:
 *   { "text": "track my order", "topK": 5, "explain": false, "oosDetection": false }
 *
 * Response:
 *   {
 *     "intent": "order_status",
 *     "confidence": 0.94,
 *     "ranking": [{ "name": "order_status", "confidence": 0.94 }, ...],
 *     "isOOS": false,
 *     "explanation": { ... },    // only if explain=true
 *     "inferenceTimeUs": 142
 *   }
 */

import { NextRequest, NextResponse } from "next/server";

// Note: these imports work at build time; at runtime the model is loaded from localStorage
// via the client. For the API route, we need the model passed in or pre-loaded.
import { predictEnsemble, deserializeEnsemble, type EnsembleModel } from "@/lib/engine/ensemble";
import { detectOOS, type OOSConfig, DEFAULT_OOS_CONFIG } from "@/lib/engine/oos-detector";
import { explainPrediction } from "@/lib/engine/explainer";
import { calibrateConfidence } from "@/lib/engine/confidence-calibration";

// In-memory model cache for the server process
let cachedModel: EnsembleModel | null = null;
let cachedTemperature: number = 1.0;

export async function POST(request: NextRequest) {
  let body: {
    text?: string;
    texts?: string[];
    topK?: number;
    explain?: boolean;
    oosDetection?: boolean;
    oosConfig?: Partial<OOSConfig>;
    temperature?: number;
    model?: string; // serialized model JSON (optional — for stateless use)
  };

  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  // Validate input
  const texts = body.texts || (body.text ? [body.text] : []);
  if (texts.length === 0) {
    return NextResponse.json(
      { error: "Provide 'text' (string) or 'texts' (string[])" },
      { status: 400 }
    );
  }

  if (texts.some((t) => typeof t !== "string" || t.trim().length === 0)) {
    return NextResponse.json(
      { error: "All texts must be non-empty strings" },
      { status: 400 }
    );
  }

  if (texts.length > 100) {
    return NextResponse.json(
      { error: "Maximum 100 texts per request" },
      { status: 400 }
    );
  }

  // Load or use model
  let model: EnsembleModel | null = cachedModel;
  if (body.model) {
    try {
      model = deserializeEnsemble(body.model);
      cachedModel = model; // cache for subsequent requests
    } catch {
      return NextResponse.json(
        { error: "Invalid model payload" },
        { status: 400 }
      );
    }
  }

  if (body.temperature !== undefined) {
    cachedTemperature = body.temperature;
  }

  if (!model) {
    return NextResponse.json(
      {
        error: "No model loaded. Either POST a 'model' field with serialized model JSON, or train a model via the UI first.",
        hint: "Use POST /api/predict with { model: '<serialized>', text: 'hello' }",
      },
      { status: 404 }
    );
  }

  const topK = Math.min(Math.max(body.topK || 5, 1), model.classes.length);
  const oosConfig: OOSConfig = body.oosConfig
    ? { ...DEFAULT_OOS_CONFIG, ...body.oosConfig }
    : DEFAULT_OOS_CONFIG;

  // Single text prediction
  if (texts.length === 1) {
    const text = texts[0];
    const prediction = predictEnsemble(text, model);

    // Apply calibration
    const ranking = cachedTemperature !== 1.0
      ? calibrateConfidence(prediction.ranking, cachedTemperature)
      : prediction.ranking;

    const result: Record<string, unknown> = {
      intent: ranking[0]?.name || "unknown",
      confidence: ranking[0]?.confidence || 0,
      ranking: ranking.slice(0, topK),
      inferenceTimeUs: prediction.inferenceTimeUs,
    };

    // OOS detection
    if (body.oosDetection) {
      const oos = detectOOS(text, model, oosConfig);
      result.isOOS = oos.isOOS;
      result.oosScore = oos.oosScore;
      result.oosSignals = oos.signals;
      if (oos.isOOS) {
        result.intent = "__oos__";
        result.confidence = 0;
      }
    }

    // Explanation
    if (body.explain) {
      const explanation = explainPrediction(text, model, topK);
      result.explanation = {
        summary: explanation.summary,
        supportingTokens: explanation.supportingTokens.slice(0, 5).map((t) => ({
          token: t.token,
          contribution: Math.round(t.normalizedContribution * 100) / 100,
        })),
        opposingTokens: explanation.opposingTokens.slice(0, 3).map((t) => ({
          token: t.token,
          contribution: Math.round(t.normalizedContribution * 100) / 100,
        })),
        computeTimeMs: explanation.computeTimeMs,
      };
    }

    return NextResponse.json(result);
  }

  // Batch prediction
  const results = texts.map((text) => {
    const prediction = predictEnsemble(text, model!);
    const ranking = cachedTemperature !== 1.0
      ? calibrateConfidence(prediction.ranking, cachedTemperature)
      : prediction.ranking;

    const r: Record<string, unknown> = {
      text,
      intent: ranking[0]?.name || "unknown",
      confidence: ranking[0]?.confidence || 0,
      ranking: ranking.slice(0, topK),
    };

    if (body.oosDetection) {
      const oos = detectOOS(text, model!, oosConfig);
      r.isOOS = oos.isOOS;
      r.oosScore = oos.oosScore;
      if (oos.isOOS) {
        r.intent = "__oos__";
        r.confidence = 0;
      }
    }

    return r;
  });

  return NextResponse.json({ predictions: results });
}

/**
 * GET /api/predict — Health check + model info
 */
export async function GET() {
  return NextResponse.json({
    status: "ok",
    modelLoaded: cachedModel !== null,
    classes: cachedModel?.classes || [],
    classCount: cachedModel?.classes.length || 0,
    hashDim: cachedModel?.hashDim || 0,
    temperature: cachedTemperature,
    endpoints: {
      "POST /api/predict": {
        body: {
          text: "string (single prediction)",
          texts: "string[] (batch prediction, max 100)",
          topK: "number (default 5)",
          explain: "boolean (default false)",
          oosDetection: "boolean (default false)",
          model: "string (serialized model JSON, optional)",
          temperature: "number (calibration temperature, optional)",
        },
      },
    },
  });
}
