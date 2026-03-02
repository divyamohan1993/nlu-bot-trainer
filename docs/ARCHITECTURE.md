# Architecture Deep Dive

This document covers the technical details of Sentio's ML engine. For a high-level overview, see the [README](../README.md).

---

## NLU Ensemble Architecture

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
    │Regress. ││  NB v2  ││  SVM   ││     ││ Boost  │    │
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
                   ┌────▼─────────────────────┐           │
                   │   Prediction Result      │◄── Drift ─┘
                   │  intent · confidence ·    │  Monitoring
                   │  ranking · per-model      │
                   └──────────────────────────┘
```

### Why Five Classifiers?

Each model family makes different mistakes. Linear models fail on overlapping feature spaces. Naive Bayes struggles with correlated features. SVMs overfit tight margins. Neural networks need lots of data. Boosted stumps miss smooth boundaries.

By combining all five through learned weights, the ensemble's error rate is strictly lower than any individual. The meta-learner discovers optimal weighting via 3-fold cross-validated log-likelihood grid search.

---

## Algorithm Reference

### MurmurHash3 Feature Hashing

Maps arbitrary string features to a fixed 1024-dimensional vector using MurmurHash3 (32-bit) with signed hashing. Each feature hashes to an index and a sign (+1/-1), reducing collision bias.

**Key properties:**
- O(1) memory per feature (no dictionary growth)
- Zero vocabulary mismatch between train and inference
- Supports infinite feature spaces (char n-grams, subword)

**Reference:** Weinberger et al., "Feature Hashing for Large Scale Multitask Learning", ICML 2009.

### Complement Naive Bayes

Standard Multinomial NB is biased toward majority classes. Complement NB instead estimates P(word|NOT class) and classifies by minimizing complement probability. Weight normalization makes predictions invariant to document length.

**Reference:** Rennie et al., "Tackling the Poor Assumptions of Naive Bayes Text Classifiers", ICML 2003.

### Pegasos SVM

Solves the SVM optimization via stochastic gradient descent on the primal objective. Learning rate eta = 1/(lambda * t) gives O(1/t) convergence. Probability estimates via softmax approximation over raw SVM scores.

**Reference:** Shalev-Shwartz et al., "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM", Mathematical Programming 2011.

### Gradient Boosted Stumps

150 decision stumps (depth-1 trees) with gradient boosting. Feature/sample subsampling (50%/80%) prevents overfitting. Shrinkage factor 0.1 ensures each stump contributes a small correction.

### Multi-Layer Perceptron

Single hidden-layer neural network (1024→128→N). Xavier initialization, ReLU activation, SGD with backpropagation. At 132,748 parameters, the MLP dominates the parameter budget but requires 1K+ examples to outperform linear models.

### Drift Detection

| Algorithm | Detects | Mechanism |
|-----------|---------|-----------|
| Page-Hinkley | Concept drift | Cumulative sum test on confidence (lambda=50) |
| DDM | Error rate drift | Error rate monitoring at 2/3-sigma thresholds |
| Vocabulary | Distribution shift | New token ratio vs. training baseline |

---

## Vulnerability Classifier Architecture

```
TF-IDF (50K features, unigrams + bigrams, sublinear TF)
  │
  ├─ Stem: Linear(50K, 192) + GELU  [fused BatchNorm]
  │
  ├─ ResBlock: Linear(192, 192) + GELU + skip connection  [fused BatchNorm]
  │
  ├─ Neck: Linear(192, 96) + GELU  [fused BatchNorm]
  │
  └─ Head: Linear(96, 349)  →  Softmax
```

**Training:** 50 epochs, focal loss (gamma=2), cosine annealing, mixup augmentation, SparseDataset for memory efficiency.

**Inference:** BatchNorm is fused into Linear layers at export time, reducing the forward pass to 4 matrix-vector multiplies + 3 GELU activations + 1 skip addition + softmax. Pure JavaScript, no ML runtime.

**Parameters:** 9,690,589 (9.7M)

---

## Vulnerability Triage Architecture

The `/vulnerability` page provides 4 independent triage modes:

```
                    ┌──────────────────────────────────────────┐
                    │           Vulnerability Page              │
                    │  ┌──────┬──────┬──────────┬───────────┐  │
                    │  │ Desc │ CVE  │  Code    │   Dep     │  │
                    │  │      │ ID   │  Scanner │   Scan    │  │
                    │  └──┬───┴──┬───┴────┬─────┴─────┬─────┘  │
                    └─────┼──────┼────────┼───────────┼────────┘
                          │      │        │           │
                          ▼      │        ▼           │
                   /api/classify │   Client-side      │
                     -vuln       │   regex engine      │
                          ▲      ▼        │           ▼
                          │  /api/nvd     │      /api/osv
                          │  -lookup      │      -scan
                          │      │        │           │
                          │      ▼        │           ▼
                          │  NVD API v2   │      OSV API
                          │      │        │           │
                          │      └──►─────┘           │
                          └───────┘                   │
                                                      ▼
                                              osv.dev/v1/query
```

### Code Scanner Engine

Client-side vulnerability detection via regex pattern matching:

- **~30 patterns** across 9 categories (SQLi, XSS, CMDi, Path Traversal, Secrets, Crypto, Deserialization, SSRF, Buffer Overflow)
- **Language detection** — heuristic-based (JS/TS, Python, Java, C/C++, Go, PHP)
- **CWE enrichment** — each finding maps to a CWE and gets severity, description, remediation from `cwe-database.ts`
- **Performance** — <5ms for typical code snippets, zero network calls

### NVD Proxy (`/api/nvd-lookup`)

CORS proxy for the NVD REST API v2:
- Validates CVE ID format (`CVE-YYYY-NNNNN+`)
- Extracts CVSS v3.1 (score, vector, severity), falling back to v2
- Parses CPE 2.3 URIs into human-readable product names
- 10-second timeout with AbortController
- Optional `NVD_API_KEY` for 50 req/30s (vs 5 without)

### OSV Proxy (`/api/osv-scan`)

Batched dependency vulnerability scanner via OSV.dev:
- Parses client-side: `package.json`, `requirements.txt`, `pom.xml`
- Queries OSV in batches of 10 concurrent requests
- 100-dependency cap, valid ecosystem validation
- Extracts CVSS from `severity[]`, `database_specific.cvss`, or summary text

---

## Performance Characteristics

### Browser (TypeScript engine)

| Metric | Value | Notes |
|--------|-------|-------|
| Parameters | 171,772 | 5 classifiers |
| Inference latency | 1-6 ms | Modern hardware, 12 classes |
| Training time | 30-60 seconds | Full ensemble + meta-weights |
| Model size | ~2 MB | localStorage |
| Memory footprint | ~20 MB | Training peak |

### Vulnerability Classifier

| Metric | Value | Notes |
|--------|-------|-------|
| Parameters | 9.7M | ResNet-MLP |
| Top-1 accuracy | 71% | 349 CWE classes |
| Top-5 accuracy | 85% | 349 CWE classes |
| Inference latency | 1-30 ms | Server-side, varies by hardware |
| Model size | 87 MB | weights.json |

### Python Training Pipeline

| Metric | Value |
|--------|-------|
| Quick training | ~30 seconds |
| Full optimization | 5-15 minutes |
| With embeddings | 15-30 minutes |
| Spot VM cost | $0.07-0.24 per session |

---

## Design System

Dark theme with a surface hierarchy:

| Token | Color | Usage |
|-------|-------|-------|
| `surface-0` | #0a0a0f | Page background |
| `surface-1` | #12121a | Sidebar |
| `surface-2` | #1a1a25 | Input fields |
| `surface-3` | #222233 | Cards, sections |
| `surface-4` | #2a2a3d | Hover states |

Brand: blue (#4c6ef5) to purple (#be4bdb) gradient. Glass morphism cards with `backdrop-filter: blur(12px)`.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Framework | Next.js 14.2 | App Router, SSG, standalone output |
| UI | React 18 + Tailwind CSS 3.4 | Dark theme, WCAG 2.2 |
| Language | TypeScript 5.5 (strict) | Type safety |
| ML Engine | Pure TypeScript | Zero runtime dependencies |
| State | localStorage | Persistent, no backend |
| Training | Python 3.12 + PyTorch + scikit-learn | Heavy optimization |
| Infra | GCloud Spot VMs | $0.07/session |
