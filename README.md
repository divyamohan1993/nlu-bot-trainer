<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/Sentio-4c6ef5?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48cGF0aCBkPSJNMTIgMTZWMTIiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjgiIHI9IjAuNSIgZmlsbD0id2hpdGUiLz48cGF0aCBkPSJNNC45MyA0LjkzTDE5LjA3IDE5LjA3IiBvcGFjaXR5PSIwLjIiLz48L3N2Zz4=&logoColor=white" />
    <img src="https://img.shields.io/badge/Sentio-4c6ef5?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48cGF0aCBkPSJNMTIgMTZWMTIiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjgiIHI9IjAuNSIgZmlsbD0id2hpdGUiLz48cGF0aCBkPSJNNC45MyA0LjkzTDE5LjA3IDE5LjA3IiBvcGFjaXR5PSIwLjIiLz48L3N2Zz4=&logoColor=white" alt="Sentio" />
  </picture>
</p>

<h1 align="center">Sentio</h1>

<p align="center">
  <em>Latin: "I perceive."</em>
</p>

<p align="center">
  <strong>Enterprise-grade NLU bot trainer that runs everywhere — from a Pentium 4 to a GPU cluster.</strong>
</p>

<p align="center">
  A research-grade intent classification engine with a 5-classifier stacking ensemble (~171K parameters),<br/>
  autonomous self-learning, drift detection, and 7-platform export — all client-side in the browser.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#self-learning">Self-Learning</a> &bull;
  <a href="#enterprise">Enterprise</a> &bull;
  <a href="#training-pipeline">Training Pipeline</a> &bull;
  <a href="#api">API</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/TypeScript-5.5-3178c6?logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Next.js-14.2-black?logo=next.js&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/React-18.3-61dafb?logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/Zero_Dependencies-Pure_Math-40c057" alt="Zero ML Dependencies" />
  <img src="https://img.shields.io/badge/License-AGPL_3.0-blue" alt="License" />
</p>

<p align="center">
  <code>nlu</code> &middot; <code>bot-trainer</code> &middot; <code>intent-classification</code> &middot; <code>self-learning</code> &middot; <code>ensemble-ml</code> &middot; <code>browser-ml</code> &middot; <code>zero-dependency</code>
</p>

---

## Why This Exists

Most NLU tools force a choice: use a massive cloud model that costs money per request, or settle for a toy classifier that misclassifies half your inputs.

**Sentio eliminates that tradeoff.** It ships a 5-model, 171K-parameter ensemble — Logistic Regression, Complement Naive Bayes, Linear SVM, a Multi-Layer Perceptron neural network, and Gradient Boosted Stumps — that trains in your browser tab and classifies in microseconds. No Python runtime. No API keys. No GPU. Just math.

The kicker: it improves itself. The autonomous self-learning loop augments weak intents, generates pseudo-labels, applies curriculum learning, and only accepts changes that pass regression testing. You train it once. It gets better on its own.

---

## Quick Start

```bash
git clone https://github.com/divyamohan1993/nlu-bot-trainer.git
cd nlu-bot-trainer
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). That's it. No Docker, no Python, no environment setup.

Sentio ships with 420 pre-loaded training examples across 12 e-commerce customer support intents. Hit **Train** and you'll have a working model in under 3 seconds.

### What you get out of the box

| Intent | Examples | What it catches |
|--------|----------|----------------|
| `greet` | 35 | "hello", "hey there", "good morning" |
| `goodbye` | 35 | "bye", "see you later", "ttyl" |
| `thank_you` | 35 | "thanks", "appreciate it", "you're awesome" |
| `order_status` | 35 | "where is my order", "track my package" |
| `return_product` | 35 | "I want to return this", "return policy" |
| `refund_request` | 35 | "I want my money back", "process refund" |
| `complaint` | 35 | "this product is defective", "terrible quality" |
| `product_inquiry` | 35 | "what sizes available", "is this in stock" |
| `cancel_order` | 35 | "cancel my order", "stop my shipment" |
| `payment_issue` | 35 | "payment failed", "double charged" |
| `account_help` | 35 | "reset password", "can't log in" |
| `speak_to_human` | 35 | "talk to a real person", "get me an agent" |

---

## Architecture

```
                        ┌────────────────────────────────────┐
                        │         User Input Text            │
                        └──────────────┬─────────────────────┘
                                       │
                        ┌──────────────▼─────────────────────┐
                        │     Tokenizer V2 (6 strategies)    │
                        │  word n-grams · char 3/4-grams     │
                        │  syntactic · intent signals         │
                        │  positional · subword               │
                        └──────────────┬─────────────────────┘
                                       │
                        ┌──────────────▼─────────────────────┐
                        │   MurmurHash3 Feature Hasher       │
                        │   1024-dim Float32Array · L2 norm  │
                        └──────────────┬─────────────────────┘
                                       │
         ┌──────────┬──────────┬───────┼───────┬──────────┐
         ▼          ▼          ▼       ▼       ▼          │
    ┌─────────┐┌─────────┐┌────────┐┌─────┐┌────────┐    │
    │Logistic ││Complem. ││ Linear ││ MLP ││Gradient│    │
    │Regress. ││  NB v2  ││  SVM   ││NN   ││ Boost  │    │
    │(SGD+L2) ││(CNB'03) ││Pegasos ││128h ││150 stmp│    │
    │ 12K par ││  7K par ││ 12K par││133K ││ 7K par │    │
    └────┬────┘└────┬────┘└───┬────┘└──┬──┘└───┬────┘    │
         │         │         │        │       │          │
         └─────────┴────┬────┴────────┴───────┘          │
                        │                                 │
           ┌────────────▼──────────────┐                  │
           │  Cross-Validated Weights   │                  │
           │  (log-likelihood grid      │                  │
           │   search, 5-way combos)    │                  │
           └────────────┬──────────────┘                  │
                        │                                 │
                   ┌────────────▼──────────────┐                │
                   │     Prediction Result      │◄───── Drift ──┘
                   │  intent · confidence ·      │     Monitoring
                   │  ranking · per-model scores │
                   └─────────────────────────────┘
```

### Why five classifiers?

Each model family makes different mistakes. Linear models fail on overlapping feature spaces. Naive Bayes struggles with correlated features. SVMs overfit tight margins. Neural networks need lots of data. Boosted stumps miss smooth boundaries.

By combining all five through learned weights, the ensemble's error rate is strictly lower than any individual model. The meta-learner discovers the optimal weighting via 3-fold cross-validated log-likelihood grid search across all valid 5-way weight combinations.

**Typical meta-weights on 420 examples:**

| Model | Params | Weight | Role |
|-------|--------|--------|------|
| Logistic Regression | 12K | ~0% | Smooth probability calibration |
| Complement NB | 7K | ~0% | Strong on class-imbalanced data |
| Linear SVM | 12K | ~40% | Maximum margin separation |
| Neural Net MLP | 133K | ~0% | Non-linear feature interactions (scales with data) |
| Gradient Boost | 7K | ~60% | Non-linear decision boundaries |

> The MLP's 133K parameters dominate the parameter budget. With 420 examples it correctly gets 0% weight (the grid search prevents overfitting). On larger datasets (1K+) the MLP begins to contribute significantly.

---

## Features

### Core NLU Engine

- **MurmurHash3 feature hashing** — O(1) per feature, zero vocabulary mismatch between train and inference. 1024-dimensional signed hashing with collision bias reduction.
- **Multi-strategy tokenization** — Six parallel feature extractors: word n-grams, character 3/4-grams (typo resilience), syntactic features (question/negation/emphasis), 14 domain-specific intent signals, positional features, and subword decomposition.
- **INT8 quantization** — `quantizeToInt8()` compresses Float32 vectors to Int8 with 4x memory reduction. Quantized dot product stays in the int8 domain for cache-friendly inference.
- **4-way unrolled dot product** — SIMD-friendly memory layout with loop unrolling for maximum throughput on any CPU, including decade-old hardware.

### Inference Speed

On a modern machine, inference takes **50-200 microseconds** per query. On constrained hardware (Pentium 4 class), expect **1-5 milliseconds** — still imperceptible.

The entire model (5 classifiers, 171K parameters + meta-weights + tokenizer config) fits in ~2 MB of localStorage. No network calls during inference.

### Training

Training 420 examples across 12 intents takes **1-3 seconds** in the browser. The bottleneck is meta-weight learning (grid search), not classifier training.

| Phase | What happens | Time |
|-------|-------------|------|
| Tokenize | 6-strategy feature extraction on all examples | ~50ms |
| Hash | MurmurHash3 into 1024-dim Float32Arrays | ~30ms |
| Train LR | SGD with L2 regularization, 15 epochs | ~100ms |
| Train CNB | Complement Naive Bayes with Lidstone smoothing | ~50ms |
| Train SVM | Pegasos algorithm, 10 epochs, projection step | ~150ms |
| Train MLP | Neural network (1024→128→12), 50 epochs, backprop | ~2000ms |
| Train GBM | 150 gradient boosted stumps | ~400ms |
| Meta-Learn | 3-fold CV log-likelihood grid search, 5-way combos | ~5000ms |
| Validate | 3-fold stratified cross-validation | ~3000ms |

---

## Self-Learning

Sentio's autonomous self-learning loop is the system's most distinctive capability. It recursively improves the model without human input.

### How it works

```
     ┌─────────────────────────────────────────────────┐
     │                                                 │
     ▼                                                 │
  EVALUATE ──► DIAGNOSE ──► AUGMENT ──► SELF-TRAIN     │
     │                                      │          │
     │         CURRICULUM ◄─────────────────┘          │
     │              │                                  │
     │           RETRAIN                               │
     │              │                                  │
     │          VALIDATE                               │
     │              │                                  │
     │     ┌────────┴────────┐                         │
     │     ▼                 ▼                         │
     │   ACCEPT           REJECT                       │
     │     │             (revert)                      │
     │     └─────────────────┘─────────────────────────┘
```

**Step by step:**

1. **Evaluate** — Measure accuracy on a held-out validation set (15% of data, never augmented).
2. **Diagnose** — Identify intents with F1 < 0.95. Extract confusion pairs from the confusion matrix.
3. **Augment** — Generate synthetic training data for weak intents using 6 techniques: synonym replacement, random insertion, random swap, random deletion, rule-based paraphrasing, and template generation.
4. **Self-Train** — Generate a candidate pool of 200 synthetic utterances. Pseudo-label examples where confidence >= 92% AND 3 of 4 ensemble members agree.
5. **Curriculum** — Reorder training data: easy examples first, hard examples last. Difficulty is measured by the model's own confidence on each example.
6. **Retrain** — Train a fresh ensemble on augmented + pseudo-labeled + curriculum-ordered data.
7. **Validate** — Compare new model against previous model on the held-out validation set.
8. **Accept/Reject** — Accept if accuracy doesn't regress by more than 1%. Reject otherwise and revert all data changes.

### Safeguards against degeneration

Self-learning systems can spiral into garbage if unchecked. We prevent this with:

| Safeguard | Mechanism |
|-----------|-----------|
| Confidence gate | Pseudo-labels require >= 92% confidence |
| Committee consensus | 4 of 5 models must agree on pseudo-label |
| Held-out validation | 15% of data is never augmented or pseudo-labeled |
| Augmentation cap | Augmented data capped at 2x original dataset size |
| Pseudo-label cap | Pseudo-labels capped at 30% of original dataset |
| Regression testing | New model must not lose > 1% accuracy vs. previous |
| Early stopping | Stops when improvement < 0.5% for 3 consecutive iterations |
| Maximum iterations | Hard limit of 10 iterations (configurable to 50) |

### Configuration

All parameters are exposed in the Self-Learn UI:

```typescript
{
  maxIterations: 10,              // Up to 50
  minImprovement: 0.005,          // 0.5% minimum per iteration
  pseudoLabelThreshold: 0.92,     // Confidence gate
  maxAugmentRatio: 2.0,           // Max 2x original data
  validationSplit: 0.15,          // 15% held out
  augmentationsPerExample: 3,     // 3 variants per weak example
  enablePseudoLabeling: true,     // Toggle pseudo-labels
  enableAugmentation: true,       // Toggle augmentation
  enableCurriculum: true,         // Toggle curriculum ordering
}
```

---

## Enterprise

### Model Registry

Every trained model is automatically registered with semantic versioning:

- **Major** version — Architecture changes (new classifier types, different hash dimensions)
- **Minor** version — Standard retraining (new data, hyperparameter changes)
- **Patch** version — Self-learning improvements (augmentation, pseudo-labels)

Each version stores:
- Full metrics (accuracy, per-intent F1, confusion matrix, training time)
- Configuration (hash dim, ensemble weights, classifier types)
- Lineage (parent version, training notes, tags)
- The serialized model artifact (can be loaded for rollback or A/B testing)

**Status lifecycle:** `draft` → `staging` → `champion` / `challenger` → `retired`

### A/B Testing

Run champion vs. challenger experiments with configurable traffic splitting:

1. Select a challenger version from the registry
2. Set traffic split (e.g., 10% to challenger)
3. Run the test — each prediction is routed and accuracy is tracked
4. Conclude when you have enough data (minimum 30 samples per arm)
5. Winner is declared with a 2% improvement threshold. Ties are "inconclusive."

### Drift Detection

Three algorithms monitor production health in real-time:

| Algorithm | What it detects | How it works |
|-----------|----------------|-------------|
| **Page-Hinkley** | Concept drift | Cumulative sum test on prediction confidence. Triggers when PH statistic exceeds threshold (lambda=50). |
| **DDM** | Error rate drift | Monitors error rate + standard deviation. Warning at 2-sigma, drift at 3-sigma above historical minimum. |
| **Vocabulary drift** | Distribution shift | Tracks new token ratio in incoming requests vs. training vocabulary baseline. |

The dashboard shows real-time status: **Healthy** / **Warning** / **Critical**.

### 7-Platform Export

Export training data in any major NLU format:

| Format | Platform | Use case |
|--------|----------|----------|
| **Rasa YAML v3.1** | Rasa Open Source | Self-hosted NLU |
| **Dialogflow ES JSON** | Google Cloud | Google Assistant, Contact Center AI |
| **Amazon Lex V2 JSON** | AWS | Alexa, Connect |
| **Microsoft LUIS 7.0.0** | Azure | Bot Framework, Power Virtual Agents |
| **Wit.ai JSON** | Meta | Messenger, WhatsApp bots |
| **CSV** | Any | Data analysis, spreadsheets, custom pipelines |
| **JSON** | Universal | Sentio native format |

Import is supported for JSON (Sentio native format) and Rasa format.

---

## Training Pipeline

For datasets beyond what the browser can handle (50K+ examples), or when you want GPU-accelerated embeddings and hyperparameter optimization, use the Python training pipeline.

### Local training

```bash
cd training
pip install -r requirements.txt

# Quick mode (~30 seconds)
python train_nlu.py --input training_data.json --output results/ --quick

# Full optimization (~15 minutes)
python train_nlu.py --input training_data.json --output results/ --optimize --n-trials 100

# With sentence embeddings + knowledge distillation
python train_nlu.py --input training_data.json --output results/ --optimize --embeddings --distill
```

The Python pipeline trains 7 classifiers (SVM, Logistic Regression, Random Forest, Gradient Boosting, SGD, Voting Ensemble, + embedding-based variants), runs Optuna hyperparameter optimization, and exports a `model_ts_compatible.json` that drops directly into the TypeScript app.

### GCloud Spot VM training

For heavy optimization runs, spin up a spot VM that auto-deletes after 2 hours:

```bash
# Create spot VM (~$0.07 for 2 hours on e2-standard-4)
./gcloud-spot-vm.sh --project=YOUR_PROJECT_ID

# Upload data, train, download results in one command
./sync_results.sh full INSTANCE_NAME us-central1-a
```

| Machine Type | vCPU | RAM | Spot $/hr | Best For |
|---|---|---|---|---|
| `e2-standard-4` | 4 | 16GB | ~$0.034 | Default (scikit-learn) |
| `e2-standard-8` | 8 | 32GB | ~$0.067 | Large datasets (50K+) |
| `e2-highmem-4` | 4 | 32GB | ~$0.045 | Sentence-transformers |
| `n1-standard-4 + T4` | 4 | 15GB | ~$0.12 | GPU embeddings |

### Synthetic data generation

Generate diverse training examples using Google Gemini:

```bash
export GOOGLE_API_KEY=your-key-here

# Generate 50 examples per intent (~$0.02 total)
python generate_synthetic.py \
  --input training_data.json \
  --output augmented_data.json \
  --provider gemini \
  --model gemini-2.0-flash \
  --count 50
```

### External datasets

Download and merge public NLU datasets:

```bash
# Best for customer support
python download_datasets.py --dataset customer_support --output datasets/

# Multi-domain benchmark
python download_datasets.py --dataset clinc150,banking77 --output datasets/ --merge
```

| Dataset | Intents | Examples | Domain |
|---------|---------|----------|--------|
| **CLINC150** | 150 | 23,700 | Multi-domain + OOS |
| **Banking77** | 77 | 13,083 | Banking |
| **Customer Support** | 27 | 3,000+ | Customer service |
| **HWU64** | 64 | ~25,000 | 21 domains |
| **SNIPS** | 7 | ~14,000 | Voice assistant |
| **ATIS** | 26 | ~5,800 | Travel |

---

## API

### Core Engine

```typescript
import { trainEnsemble, predictEnsemble } from "@/lib/engine/ensemble";

// Train
const model = trainEnsemble([
  { text: "hello", intent: "greet" },
  { text: "goodbye", intent: "farewell" },
  // ... more examples
]);

// Predict (returns in microseconds)
const result = predictEnsemble("hey there", model);

console.log(result.intent);           // "greet"
console.log(result.confidence);       // 0.94
console.log(result.inferenceTimeUs);  // 127
console.log(result.ranking);          // [{ name: "greet", confidence: 0.94 }, ...]
console.log(result.perModelScores);   // { logReg: [...], naiveBayes: [...], svm: [...], mlp: [...], gradBoost: [...] }
```

### Self-Learning

```typescript
import { runSelfLearningLoop } from "@/lib/self-learn/autonomous-loop";

const result = runSelfLearningLoop(trainingData, {
  maxIterations: 10,
  pseudoLabelThreshold: 0.92,
  enableAugmentation: true,
  enablePseudoLabeling: true,
  enableCurriculum: true,
}, (iteration) => {
  console.log(`Iteration ${iteration.iteration}: ${(iteration.accuracy * 100).toFixed(1)}%`);
});

console.log(`Improved from ${result.initialAccuracy} to ${result.finalAccuracy}`);
console.log(`Added ${result.totalNewExamples} synthetic examples`);
```

### Model Registry

```typescript
import { registerModel, promoteToChampion, rollbackToVersion } from "@/lib/enterprise/model-registry";

const version = registerModel(model, "minor", "Retrained with augmented data");
promoteToChampion(version.id);
// Later:
const oldModel = rollbackToVersion("v_1.0.0_...");
```

### Export

```typescript
import { exportTrainingData } from "@/lib/enterprise/export-formats";

const { content, filename, mimeType } = exportTrainingData(data, "rasa");
// content: YAML string for Rasa NLU v3.1
// filename: "nlu-training-data.yml"
// mimeType: "text/yaml"
```

---

## Project Structure

```
sentio/
├── src/
│   ├── app/                          # Next.js pages
│   │   ├── page.tsx                  # Dashboard
│   │   ├── analytics/page.tsx        # Confusion matrix, F1, drift monitoring
│   │   ├── entities/page.tsx         # Entity management
│   │   ├── intents/page.tsx          # Intent management
│   │   ├── models/page.tsx           # Model registry, A/B testing
│   │   ├── self-learn/page.tsx       # Autonomous improvement
│   │   ├── test/page.tsx             # Chat-style test playground
│   │   └── train/page.tsx            # Ensemble training
│   │
│   ├── components/                   # React components
│   │   ├── Sidebar.tsx               # 8-page navigation (Alt+1-8)
│   │   ├── StatsCard.tsx             # Metric display cards
│   │   └── ConfidenceBar.tsx         # Animated confidence bars
│   │
│   ├── lib/
│   │   ├── engine/                   # Core NLU engine
│   │   │   ├── feature-hasher.ts     # MurmurHash3, quantization
│   │   │   ├── tokenizer-v2.ts       # 6-strategy tokenizer
│   │   │   ├── ensemble.ts           # Stacking ensemble
│   │   │   └── classifiers/
│   │   │       ├── logistic-regression.ts
│   │   │       ├── naive-bayes-v2.ts
│   │   │       ├── svm.ts
│   │   │       ├── mlp.ts
│   │   │       └── gradient-boost.ts
│   │   │
│   │   ├── self-learn/               # Self-learning pipeline
│   │   │   ├── autonomous-loop.ts    # Recursive improvement
│   │   │   ├── data-augmentation.ts  # EDA, templates, paraphrasing
│   │   │   └── active-learning.ts    # 5 query strategies
│   │   │
│   │   ├── enterprise/               # Enterprise features
│   │   │   ├── model-registry.ts     # Versioning, A/B testing
│   │   │   ├── drift-detector.ts     # Page-Hinkley, DDM
│   │   │   └── export-formats.ts     # 7-platform export
│   │   │
│   │   ├── store.ts                  # State persistence
│   │   ├── classifier.ts             # Legacy classifier (v1)
│   │   ├── tokenizer.ts              # Legacy tokenizer (v1)
│   │   └── entity-extractor.ts       # Dictionary-based NER
│   │
│   └── types/
│       └── index.ts                  # TypeScript interfaces
│
├── training/                         # Python training pipeline
│   ├── train_nlu.py                  # Multi-classifier + Optuna HPO
│   ├── generate_synthetic.py         # Gemini/Vertex AI synthetic data
│   ├── download_datasets.py          # Kaggle/HuggingFace datasets
│   ├── gcloud-spot-vm.sh             # GCP spot VM setup
│   ├── sync_results.sh               # VM file sync
│   ├── export_from_app.js            # Export training data from app
│   ├── requirements.txt              # Python dependencies
│   └── training_data.json            # Default 420-example dataset
│
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.js
```

---

## Performance Characteristics

### Browser (TypeScript engine)

| Metric | Value | Notes |
|--------|-------|-------|
| Parameters | 171,772 | 5 classifiers (LR + NB + SVM + MLP + GBM) |
| Inference latency | 1-6 ms | Modern hardware, 12 classes, 420 examples |
| Training time | 30-60 seconds | Full 5-model ensemble + 5-way meta-weight learning |
| Model size | ~2 MB | 5 classifiers in localStorage |
| Memory footprint | ~20 MB | During training peak |
| Cross-validation | 20-40 seconds | 3-fold stratified |
| Self-learning loop | 60-180 seconds | 10 iterations with augmentation |

### Python pipeline

| Metric | Value | Notes |
|--------|-------|-------|
| Quick training | ~30 seconds | TF-IDF + 5 classifiers |
| Full optimization | 5-15 minutes | Optuna, 100 trials per model |
| With embeddings | 15-30 minutes | sentence-transformers + distillation |
| Spot VM cost | $0.07-0.24 | 2-hour session, auto-delete |
| Synthetic data | $0.01-0.05 | 600 examples via Gemini Flash |

---

## Algorithms Reference

<details>
<summary><strong>MurmurHash3 Feature Hashing</strong></summary>

Maps arbitrary string features to a fixed-size vector without maintaining a vocabulary dictionary. Uses MurmurHash3 (32-bit) with signed hashing: each feature is hashed to an index and a sign (+1/-1), which reduces collision bias through random sign cancellation.

**Advantages over vocabulary-based approaches:**
- O(1) memory per feature (no dictionary growth)
- Zero vocabulary mismatch between train and inference
- Supports infinite feature spaces (char n-grams, subword)
- 1024 dimensions provides a good accuracy-size tradeoff for NLU-scale vocabularies

**Reference:** Weinberger et al., "Feature Hashing for Large Scale Multitask Learning", ICML 2009.

</details>

<details>
<summary><strong>Complement Naive Bayes</strong></summary>

Standard Multinomial NB is biased toward majority classes because it estimates P(word|class) from only that class's documents. Complement NB instead estimates P(word|NOT class) and classifies by minimizing the complement class probability. This makes it significantly more robust to class imbalance.

Weight normalization (dividing by the sum of absolute log-weights) makes predictions invariant to document length.

**Reference:** Rennie et al., "Tackling the Poor Assumptions of Naive Bayes Text Classifiers", ICML 2003.

</details>

<details>
<summary><strong>Pegasos SVM</strong></summary>

Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) solves the SVM optimization problem via stochastic gradient descent on the primal objective. Learning rate is eta = 1/(lambda * t), giving O(1/t) convergence — optimal for SVMs.

The projection step ensures ||w|| <= 1/sqrt(lambda), which acts as implicit regularization.

Probability estimates use Platt scaling (softmax approximation over raw SVM scores).

**Reference:** Shalev-Shwartz et al., "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM", Mathematical Programming 2011.

</details>

<details>
<summary><strong>Gradient Boosted Stumps</strong></summary>

Decision stumps (depth-1 trees) are the weakest learner, but 150 of them with gradient boosting become surprisingly powerful. Each stump finds the single best feature + threshold split that minimizes the residual loss.

Feature and sample subsampling (50% features, 80% samples per round) prevents overfitting and adds diversity. Shrinkage factor of 0.1 ensures each stump contributes only a small correction. Softmax normalization on raw log-odds scores produces calibrated probabilities.

</details>

<details>
<summary><strong>Multi-Layer Perceptron (MLP)</strong></summary>

A single hidden-layer neural network (1024→128→12) trained with SGD and backpropagation. Xavier weight initialization prevents vanishing/exploding gradients. ReLU activation on the hidden layer enables non-linear feature interactions. Softmax output with cross-entropy loss for proper probability calibration.

At 132,748 parameters, the MLP is the largest model in the ensemble. It requires more training data than the linear models to shine — with 420 examples it tends to overfit during cross-validation, but the meta-learner correctly assigns it 0% weight. On datasets of 1K+ examples, it becomes a major contributor.

**Training:** 50 epochs, learning rate 0.01 with linear decay, L2 regularization (lambda=0.0001).

</details>

<details>
<summary><strong>Page-Hinkley Drift Detection</strong></summary>

A sequential hypothesis test for detecting abrupt changes in a data stream. Maintains a cumulative sum of deviations from the running mean. When the difference between the cumulative sum and its minimum exceeds threshold lambda, drift is declared.

Parameters: delta = 0.005 (minimum magnitude of change to detect), lambda = 50 (detection threshold).

**Reference:** Page, "Continuous Inspection Schemes", Biometrika 1954; Hinkley, "Inference About the Change-Point from Cumulative Sum Tests", Biometrika 1971.

</details>

<details>
<summary><strong>DDM (Drift Detection Method)</strong></summary>

Monitors the online error rate as a Bernoulli process. Under stable conditions, the error rate and its standard deviation have a known minimum. When the current error rate deviates significantly (2-sigma for warning, 3-sigma for drift), the model is flagged.

Requires a minimum of 30 samples before activating to avoid false positives.

**Reference:** Gama et al., "Learning with Drift Detection", SBIA 2004.

</details>

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Framework | Next.js 14.2 | App Router, SSG, zero-config |
| UI | React 18 + Tailwind CSS 3.4 | Dark theme, glass morphism, WCAG 2.2 |
| Language | TypeScript 5.5 (strict mode) | Type safety across 50+ source files |
| ML Engine | Pure TypeScript | No Python dependency at runtime |
| State | localStorage + IndexedDB | Persistent across sessions, no backend |
| Training | Python 3.12 + scikit-learn + Optuna | Heavy optimization when needed |
| Infra | GCloud Spot VMs | $0.07 for 2-hour training sessions |
| Synthetic Data | Google Gemini API | $0.02 for 600 examples |

### Design System

Dark theme with a carefully crafted surface hierarchy:

- `surface-0` (#0a0a0f) — Page background
- `surface-1` (#12121a) — Sidebar
- `surface-2` (#1a1a25) — Input fields
- `surface-3` (#222233) — Cards, sections
- `surface-4` (#2a2a3d) — Hover states

Glass morphism cards with `backdrop-filter: blur(12px)` and 6% white borders.

Brand palette: blue (#4c6ef5) to purple (#be4bdb) gradient. Each intent gets a unique color for instant visual identification.

---

## Accessibility

- **WCAG 2.2 AA compliant** — All interactive elements have visible focus indicators, proper contrast ratios, and semantic HTML.
- **Keyboard navigation** — Alt+1 through Alt+8 for instant page switching. Arrow keys for sidebar navigation. Full tab order.
- **Screen reader support** — ARIA labels, roles, live regions for dynamic content, proper heading hierarchy.
- **Reduced motion** — All animations respect `prefers-reduced-motion: reduce`.
- **Skip navigation** — "Skip to main content" link for keyboard users.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and pull request guidelines.

## Security

See [SECURITY.md](SECURITY.md) for our security policy and how to report vulnerabilities.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) — see the LICENSE file for details.

---

<p align="center">
  <strong>Sentio</strong> — <em>I perceive.</em>
</p>
<p align="center">
  Built for <a href="https://dmj.one">dmj.one</a> &bull; Aatmanirbhar Bharat
</p>
