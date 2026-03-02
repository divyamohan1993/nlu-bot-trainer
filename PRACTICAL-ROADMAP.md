# Practical Roadmap: From Showcase to Production-Grade

> Research-backed plan to make NLU Bot Trainer a repo that hiring managers, developers, and ML engineers take seriously.

---

## Part 1: Remaining Bugs (Ship-Blockers)

### Bug 1: SVM Softmax Overflow → NaN
**File**: `src/lib/engine/classifiers/svm.ts:113-120`
**Issue**: `Math.exp(score)` without max-subtraction. Large raw SVM margins (e.g., 50+) overflow to `Infinity`, producing `NaN` probabilities.
**Fix**: Subtract max score before exponentiation (standard numerically-stable softmax).

### Bug 2: Active Learning Missing MLP
**File**: `src/lib/self-learn/active-learning.ts:62-75`
**Issue**: `committeeDisagreement()` only checks 4/5 classifiers (omits MLP). Vote entropy is computed over 4 models, not 5.
**Fix**: Add `perModelScores.mlp` to the models array.

### Bug 3: Model Registry Missing MLP
**File**: `src/lib/enterprise/model-registry.ts:138`
**Issue**: `classifierTypes` array lists 4 classifiers, missing `"mlp"`. Stored metadata is incorrect.
**Fix**: Add `"mlp"` to the classifierTypes array.

### Bug 4: Generate API Accepts Negative Count
**File**: `src/app/api/generate/route.ts:54,69`
**Issue**: `!count` passes for negative numbers. `Math.min(-5, 50)` = -5 → prompt says "Generate -5 examples".
**Fix**: Validate `count > 0 && count <= 50`.

### Bug 5: `saveData` No Quota Guard
**File**: `src/lib/store.ts:528-534`
**Issue**: `saveData()` has no `QuotaExceededError` handling. Large training datasets silently fail.
**Fix**: Add try/catch with graceful error (same pattern as `saveEnsembleModel`).

### Bug 6: Tokenizer Contraction Order
**File**: `src/lib/engine/tokenizer-v2.ts:119-121`
**Issue**: Contractions are expanded *before* Unicode apostrophe normalization (line 124). Text with curly apostrophes (`'`) won't match contractions keyed with straight apostrophes.
**Fix**: Move Unicode normalization before contraction expansion.

---

## Part 2: Production Essentials (Week 1-2)

### 2.1 Test Suite
**Priority**: Critical — zero test coverage is the #1 credibility killer.
**Tool**: Vitest (fast, native ESM, zero-config with Next.js)
**Coverage targets**: 80%+ for engine/, 60%+ for self-learn/, smoke tests for all pages.

```
tests/
  unit/
    engine/
      feature-hasher.test.ts       # MurmurHash3 distribution, collision rate
      tokenizer-v2.test.ts         # Normalization, stemming, contraction expansion
      ensemble.test.ts             # Training, prediction, serialization roundtrip
      classifiers/
        logistic-regression.test.ts
        naive-bayes-v2.test.ts
        svm.test.ts
        mlp.test.ts
        gradient-boost.test.ts
    self-learn/
      data-augmentation.test.ts
      active-learning.test.ts
      autonomous-loop.test.ts
    enterprise/
      model-registry.test.ts
      drift-detector.test.ts
  integration/
    train-predict-roundtrip.test.ts # Load data → train → predict → verify
    self-learn-improvement.test.ts  # Run loop → verify accuracy improves
  smoke/
    pages.test.ts                   # Each page renders without crash
```

### 2.2 Publishable npm Package
**Goal**: `npm install nlu-bot-trainer` → works headlessly, no React/Next.js needed.
**Architecture**: Barrel export from `src/lib/index.ts`, build with `tsup` for dual CJS/ESM.

```typescript
// src/lib/index.ts — public API
export { trainEnsemble, predictEnsemble, type EnsembleModel } from "./engine/ensemble";
export { hashFeatures } from "./engine/feature-hasher";
export { tokenize, normalizeText } from "./engine/tokenizer-v2";
export { runSelfLearningLoop, type SelfLearnConfig, type SelfLearnResult } from "./self-learn/autonomous-loop";
export { registerModel, loadRegistry, type ModelVersion } from "./enterprise/model-registry";
export { getDriftReport, recordPrediction } from "./enterprise/drift-detector";
```

**package.json additions**:
```json
{
  "main": "dist/index.cjs",
  "module": "dist/index.mjs",
  "types": "dist/index.d.ts",
  "exports": {
    ".": { "import": "./dist/index.mjs", "require": "./dist/index.cjs", "types": "./dist/index.d.ts" }
  },
  "files": ["dist/"],
  "scripts": {
    "build:lib": "tsup src/lib/index.ts --format cjs,esm --dts --clean"
  }
}
```

### 2.3 REST Inference API
**Current gap**: No way to call the model without the React UI.
**Add**: `POST /api/predict` endpoint.

```typescript
// src/app/api/predict/route.ts
export async function POST(request: NextRequest) {
  const { text, topK } = await request.json();
  const model = loadEnsembleModel();
  if (!model) return NextResponse.json({ error: "No trained model" }, { status: 404 });
  const result = predictEnsemble(text, model);
  return NextResponse.json({
    intent: result.intent,
    confidence: result.confidence,
    ranking: result.ranking.slice(0, topK || 5),
  });
}
```

### 2.4 Input Validation & Error Boundaries
- **Data validation**: Validate intent names (no empty strings, no duplicates, max length), example text (non-empty, reasonable length), entity values.
- **React Error Boundaries**: Wrap each page section so a crash in analytics doesn't kill the whole app.
- **API input sanitization**: All API routes validate request body shape before processing.

---

## Part 3: Credibility Boosters (Week 2-3)

### 3.1 Standard NLU Benchmarks
**Why**: Every serious NLU tool publishes benchmark numbers. Without them, claims are unverifiable.
**Datasets** (all freely available):

| Dataset | Classes | Examples | Domain |
|---------|---------|----------|--------|
| CLINC150 | 150 + OOS | 23,700 | Virtual assistant |
| Banking77 | 77 | 13,083 | Banking support |
| SNIPS | 7 | 14,484 | Voice assistant |
| ATIS | 26 | 5,871 | Airline queries |
| HWU64 | 64 | 25,716 | Home assistant |

**Implementation**: Add `benchmarks/` directory with download scripts, evaluation runner, and results table auto-generated into README.

### 3.2 Out-of-Scope (OOS) Detection
**Why**: Real chatbots receive gibberish, off-topic input, adversarial probes. Currently no way to say "I don't know."
**Approach**: Entropy-based rejection — if prediction entropy exceeds a calibrated threshold, return `{ intent: "__oos__", confidence: 0 }`.
**Calibration**: Use CLINC150's OOS split to find optimal threshold via ROC curve.

### 3.3 Confidence Calibration
**Why**: Raw ensemble confidence ≠ true probability. "92% confident" should mean correct 92% of the time.
**Approach**: Temperature scaling (single learned parameter T, post-hoc on validation set).
**Visualization**: Add reliability diagram to Analytics page showing calibration before/after.

### 3.4 CLI Tool
**Why**: Developers evaluate tools from the terminal, not the browser.

```bash
npx nlu-bot-trainer train ./data/intents.json     # Train from file
npx nlu-bot-trainer predict "track my order"       # Single prediction
npx nlu-bot-trainer evaluate ./data/test.json      # Batch evaluation
npx nlu-bot-trainer benchmark banking77            # Run standard benchmark
npx nlu-bot-trainer serve --port 8080              # Start HTTP inference server
npx nlu-bot-trainer export rasa ./model.json       # Export to Rasa format
```

### 3.5 Data Quality Validator
Catch training data problems before they waste training time:
- Duplicate detection (exact + near-duplicate via Jaccard similarity)
- Class imbalance warnings (< 5 examples per intent)
- Mislabel detection (examples that classify better under a different intent)
- Vocabulary coverage gaps (classes with very few unique tokens)

---

## Part 4: Differentiators (Week 3-4)

### 4.1 Prediction Explanation (Explainable AI)
**Why**: "Why did the model pick this intent?" is the #1 question from bot builders.
**Approach**: LIME-lite — perturb input, observe prediction changes, highlight contributing words.
**UI**: Show highlighted tokens with contribution scores on the Test page.

### 4.2 Conversation Log Import
**Why**: Most teams already have chat logs. They shouldn't have to manually label everything.
**Formats**: CSV, JSON lines, Intercom export, Zendesk export, plain text.
**Flow**: Upload → auto-cluster with k-means → suggest intent names → human reviews → done.

### 4.3 Responsive Layout
**Current issue**: Hard-coded `ml-64` sidebar, unusable on mobile/tablet.
**Fix**: Collapsible sidebar with hamburger menu, responsive grid breakpoints.

### 4.4 Web Worker Training
**Why**: Training blocks the main thread, freezing the UI (especially with 1000+ examples).
**Approach**: Move `trainEnsemble()` to a Web Worker, communicate via `postMessage`.

### 4.5 Model Serving Patterns
- **CDN-hosted models**: Export model as JSON, load from CDN at runtime
- **IndexedDB caching**: Cache trained model in IndexedDB for faster subsequent loads
- **Service Worker**: Offline inference after first load
- **Framework integrations**: React hook (`useNLU`), Express middleware, Fastify plugin

---

## Part 5: Code Cleanup

### Dead Code Removal
- `src/lib/classifier.ts` — Original single-classifier, unused by new UI (keep only if needed for backwards-compat API)
- `src/lib/tokenizer.ts` — Old tokenizer v1, fully replaced by tokenizer-v2

### Missing Type Safety
- `committeeDisagreement()` parameter type doesn't include `mlp` key
- Several `Record<string, any>` that should be properly typed
- `saveData()` should validate the `TrainingData` shape

### Documentation
- JSDoc on all public exports (engine, self-learn, enterprise)
- API route documentation (OpenAPI/Swagger for `/api/predict`, `/api/generate`)
- Architecture diagram in README showing data flow: Input → Tokenizer → Feature Hasher → 5 Classifiers → Meta-Weights → Prediction

---

## Part 6: Sprint Plan

### Week 1: Foundation
| Day | Task | Size |
|-----|------|------|
| 1 | Fix all 6 remaining bugs | S |
| 1 | Set up Vitest, write engine unit tests | M |
| 2 | Write classifier + self-learn tests (80% coverage) | L |
| 3 | Add `/api/predict` REST endpoint + input validation | M |
| 3 | Add React Error Boundaries to all pages | S |
| 4 | Create `src/lib/index.ts` barrel export | S |
| 4 | Set up tsup for npm package build | M |
| 5 | Add data quality validator | M |

### Week 2: Credibility
| Day | Task | Size |
|-----|------|------|
| 1-2 | CLI tool (train, predict, evaluate, benchmark, serve) | L |
| 3 | Banking77 benchmark + results table | M |
| 3 | CLINC150 benchmark + OOS detection | M |
| 4 | Confidence calibration (temperature scaling) | M |
| 5 | Responsive sidebar + mobile layout | M |

### Week 3: Differentiation
| Day | Task | Size |
|-----|------|------|
| 1 | Web Worker for training | M |
| 2 | Prediction explanation (LIME-lite) | L |
| 3 | Conversation log import (CSV/JSON) | M |
| 4 | IndexedDB model caching + Service Worker | M |
| 5 | React hook `useNLU` + Express middleware | M |

### Week 4: Polish
| Day | Task | Size |
|-----|------|------|
| 1 | Dead code removal + type safety pass | S |
| 2 | JSDoc all public APIs + OpenAPI spec | M |
| 3 | Architecture diagram + README overhaul | M |
| 4 | CI/CD: GitHub Actions (lint, test, build, benchmark) | M |
| 5 | npm publish dry run + final review | S |

---

## Part 7: Impact Matrix

| Feature | Effort | Credibility | Practical Use | Differentiation |
|---------|--------|-------------|---------------|-----------------|
| Test suite | M | ***** | *** | ** |
| npm package | M | **** | ***** | **** |
| REST API | S | *** | ***** | ** |
| CLI tool | L | **** | ***** | **** |
| Benchmarks | M | ***** | ** | **** |
| OOS detection | M | **** | ***** | **** |
| Confidence calibration | M | **** | **** | *** |
| Prediction explanation | L | **** | **** | ***** |
| Data quality validator | M | *** | ***** | *** |
| Web Worker | M | ** | **** | ** |
| Responsive layout | M | ** | *** | * |
| Conv. log import | M | ** | **** | *** |

**Top 5 bang-for-buck** (ordered):
1. Test suite — table stakes for any serious repo
2. npm package + barrel exports — makes the engine usable outside the UI
3. CLI tool — how developers actually evaluate tools
4. Benchmarks (Banking77 + CLINC150) — backs up claims with numbers
5. OOS detection — the single most-asked-for feature in production NLU
