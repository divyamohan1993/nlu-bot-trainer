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
  <strong>Two ML engines. Zero cloud dependencies. One Next.js app.</strong>
</p>

<p align="center">
  Train intent classifiers in your browser. Triage CVE vulnerabilities with severity, OWASP mapping, and remediation guidance.<br/>
  Both run on pure TypeScript math — no Python runtime, no API keys, no GPU required.
</p>

<p align="center">
  <a href="#what-sentio-does">What It Does</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#nlu-bot-trainer">NLU Engine</a> &bull;
  <a href="#vulnerability-triage">Vuln Triage</a> &bull;
  <a href="#deploy-your-own">Deploy</a> &bull;
  <a href="#train-your-own-models">Train</a> &bull;
  <a href="#scalability">Scalability</a> &bull;
  <a href="#api-reference">API</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/TypeScript-5.5-3178c6?logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Next.js-14.2-black?logo=next.js&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/React-18.3-61dafb?logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/Zero_Dependencies-Pure_Math-40c057" alt="Zero ML Dependencies" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/License-AGPL_3.0-blue" alt="License" />
</p>

---

## What Sentio Does

| Capability | NLU Bot Trainer | Vulnerability Triage |
|------------|----------------|---------------------|
| **What** | Train intent classifiers for chatbots | 4-mode vulnerability triage: CVE text, CVE ID lookup, code scanning, dependency audit |
| **How** | 5-classifier stacking ensemble (171K params) | ResNet-MLP (9.7M params) + NVD API + regex engine + OSV API |
| **Output** | Intent + confidence + per-model scores | CWE + severity + OWASP + remediation + CVSS + affected products + fix versions |
| **Runs where** | 100% in-browser, zero server | ML: API route · Code scanner: browser · NVD/OSV: proxy routes |
| **Training** | In-browser (30s) or Python pipeline | PyTorch on GCP VM (~2 hours) |
| **Data** | 420 pre-loaded examples, 12 intents | 224K+ CVEs + NVD live data + OSV vulnerability database |

---

## Quick Start

```bash
git clone https://github.com/divyamohan1993/nlu-bot-trainer.git
cd nlu-bot-trainer
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Both engines are ready.

**Or with Docker:**

```bash
docker compose up --build
```

### NLU — Try it now

1. Open the app → 420 pre-loaded e-commerce support examples across 12 intents
2. Click **Train** → Working model in under 3 seconds
3. Go to /test → Type "where is my package?" → See `order_status` at 95%+ confidence

### Vulnerability Triage — Try it now

1. Go to /vulnerability → 4 tabs: Description, CVE Lookup, Code Scanner, Dependency Scan
2. **CVE Lookup:** Type `CVE-2021-44228` → CVSS 10.0 CRITICAL, 342 affected products, ML classification
3. **Code Scanner:** Click "Python Vulns" sample → 5 findings in <5ms, zero network calls
4. **Dependency Scan:** Click "npm package.json" sample → OSV vulnerability report with fix versions

---

## NLU Bot Trainer

A research-grade intent classification engine with autonomous self-learning. Five classifiers vote through learned meta-weights to produce predictions that no single model can match alone.

### Architecture

```
  User Input → Tokenizer V2 (6 strategies) → MurmurHash3 (1024-dim)
                                                    │
         ┌──────────┬──────────┬──────────┬─────────┼─────────┐
         ▼          ▼          ▼          ▼         ▼         │
    Logistic   Complement   Linear     MLP     Gradient      │
    Regress.    NB v2       SVM       128h     Boost         │
     12K par    7K par      12K par   133K     7K par        │
         └──────────┴────┬─────┴──────────┴─────────┘         │
                         ▼                                     │
              Cross-Validated Meta-Weights              Drift ─┘
                         ▼                            Monitoring
                 Prediction Result
```

**Why five classifiers?** Each fails differently. Linear models miss overlapping features. Naive Bayes struggles with correlations. SVMs overfit tight margins. Neural nets need lots of data. Boosted stumps miss smooth boundaries. The ensemble's error rate is strictly lower than any individual.

### Key Capabilities

- **Self-learning loop** — Evaluates → diagnoses weak intents → augments data → pseudo-labels high-confidence predictions → curriculum-orders → retrains → validates. Accepts only if accuracy doesn't regress. Fully autonomous.
- **Drift detection** — Page-Hinkley (concept drift), DDM (error rate drift), vocabulary distribution monitoring. Real-time dashboard.
- **Model registry** — Semantic versioning, champion/challenger lifecycle, A/B testing with configurable traffic splits.
- **7-platform export** — Rasa, Dialogflow, Lex, LUIS, Wit.ai, CSV, JSON.
- **Zero dependencies** — Every algorithm (MurmurHash3, Pegasos SVM, CNB, backprop MLP, gradient boosted stumps) implemented from scratch in TypeScript.

### Performance

| Metric | Value |
|--------|-------|
| Inference | 1–6 ms (modern), 50–200μs (optimized path) |
| Training | 30–60 seconds (full ensemble + meta-weights) |
| Model size | ~2 MB (localStorage) |
| Parameters | 171,772 |

For deep algorithm references and the math behind each classifier, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Vulnerability Triage

Four ways to find vulnerabilities — pick the one that matches what you have.

| Tab | You have... | You get... |
|-----|------------|-----------|
| **CVE Description** | Prose text describing a vulnerability | CWE classification + severity + OWASP + remediation |
| **CVE ID Lookup** | A CVE ID (e.g., CVE-2021-44228) | NVD metadata (CVSS, products, dates) + ML classification + CWE agreement check |
| **Code Scanner** | Source code (any language) | Line-by-line vulnerability findings with CWE mapping and fix guidance |
| **Dependency Scan** | package.json / requirements.txt / pom.xml | Known vulnerable dependencies with severity, fix versions, and CVE links |

### CVE ID Lookup

Type a CVE ID, get everything: CVSS score and vector, severity rating, affected products, NVD references, plus ML-powered CWE classification cross-referenced against NVD's ground truth.

### Code Scanner

Paste code, get instant results. ~30 regex patterns detect SQL injection, XSS, command injection, path traversal, hardcoded secrets, insecure crypto, deserialization, SSRF, and buffer overflows. Runs entirely client-side in <5ms — zero network calls, zero data leaves your browser.

> Pattern-based scanning catches common vulnerability patterns but cannot analyze data flow. Use alongside SAST tools for comprehensive coverage.

### Dependency Scan

Paste a manifest file. The scanner parses your dependencies, queries the [OSV vulnerability database](https://osv.dev), and returns known CVEs with severity, fix versions, and CWE enrichment. Supports npm (package.json), PyPI (requirements.txt), and Maven (pom.xml).

### ML Classification

For each CVE description (direct input or fetched via CVE ID), the classifier returns:

- **CWE ID + human-readable name** — Not just "CWE-89", but "SQL Injection"
- **Severity** — Critical / High / Medium / Low, mapped from CWE category and exploit impact
- **OWASP Top 10 2021 category** — Where this weakness fits in the security landscape
- **Remediation guidance** — 3–4 actionable steps specific to the weakness category
- **Top 5 predictions** — Ranked alternatives with confidence scores

### Model Details

- **Architecture:** ResNet-MLP (TF-IDF → 192 → 192+skip → 96 → 349 classes)
- **Parameters:** 9.7M
- **Training data:** 224K+ CVEs (NVD, 1999–2025)
- **Accuracy:** 71% top-1, 85% top-5 across 349 CWE categories
- **Inference:** 1–30ms server-side

> 71% across 349 categories is a triage starting point, not a final determination. The model helps security teams prioritize — it doesn't replace expert analysis.

### CWE Enrichment Database

All 349 supported CWE classes include enrichment data. The top ~50 most critical CWEs (SQL injection, XSS, buffer overflow, RCE, etc.) have hand-curated descriptions and remediation. The remaining ~300 use category-based enrichment with MITRE CWE names.

---

## Deploy Your Own

### Docker (recommended)

```bash
docker compose up --build
# App available at http://localhost:3000
```

### Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/divyamohan1993/nlu-bot-trainer)

### GCP / Any VM

```bash
git clone https://github.com/divyamohan1993/nlu-bot-trainer.git
cd nlu-bot-trainer
npm ci && npm run build
node .next/standalone/server.js
# Reverse proxy port 3000 with Nginx/Caddy
```

For detailed deployment guides (Nginx config, systemd service, model weights management), see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## Train Your Own Models

### NLU — Browser Training

Just use the app. Add intents, add examples, click Train. The 5-classifier ensemble trains in 30–60 seconds entirely in your browser. No server, no setup.

For large datasets (50K+), use the Python pipeline:

```bash
cd training
pip install -r requirements.txt
python train_nlu.py --input data.json --output results/ --optimize
```

### Vulnerability Classifier — PyTorch

The vulnerability model was trained on a GCP VM with PyTorch:

```bash
# On a VM with 16GB+ RAM
cd training/vuln-classifier
pip install -r requirements.txt
python train.py --epochs 50 --batch-size 256
python export_weights.py  # → weights.json + tfidf_vocab.json + labels.json
```

Exported weights drop directly into `public/models/vuln-classifier/` for the Next.js API route.

For full training guides (data preparation, hyperparameters, checkpoints, custom datasets), see [docs/TRAINING.md](docs/TRAINING.md).

---

## Self-Hosting — The Honest Answer

**"Can I run this with Ollama?"** — No. Sentio's models are classifiers, not LLMs. Ollama serves large language models. These are different things.

**What actually runs where:**

| Engine | Where It Runs | Infrastructure Needed |
|--------|--------------|----------------------|
| NLU Bot Trainer | In the browser | None. Zero. The user's browser IS the compute. |
| Vulnerability Classifier | Next.js API route | A Node.js server (Docker container, Vercel, any VM) |

The NLU engine needs no hosting at all — it trains and infers in the browser tab. The vulnerability classifier is a stateless API route that loads model weights on startup.

For ONNX-based serving (Triton, ONNX Runtime), the training pipeline exports ONNX format. See [docs/SELF-HOSTING.md](docs/SELF-HOSTING.md).

---

## Scalability

| Tier | Users | NLU | Vuln Classifier | Infra |
|------|-------|-----|-----------------|-------|
| 0 | 0–10K | Client-side (free) | Single container | 1 VM / Vercel |
| 1 | 10K–100K | Client-side (free) | 2–4 containers behind LB | Horizontal scale |
| 2 | 100K–1M | Client-side (free) | Auto-scaling group | Cloud Run / ECS |
| 3 | 1M+ | Client-side (free) | Multi-region deployment | CDN + edge |

**NLU scales infinitely for free.** Every user brings their own compute — the browser. 10 users or 10 million users, the server load is identical (serving static files).

**Vulnerability classifier scales horizontally.** It's a stateless API route. No sessions, no database, no shared state. Add containers behind a load balancer.

---

## API Reference

### NLU Engine

```typescript
import { trainEnsemble, predictEnsemble } from "@/lib/engine/ensemble";

const model = trainEnsemble([
  { text: "hello", intent: "greet" },
  { text: "track my order", intent: "order_status" },
]);

const result = predictEnsemble("hey there", model);
// result.intent → "greet"
// result.confidence → 0.94
// result.ranking → [{ name: "greet", confidence: 0.94 }, ...]
```

### Self-Learning

```typescript
import { runSelfLearningLoop } from "@/lib/self-learn/autonomous-loop";

const result = runSelfLearningLoop(trainingData, {
  maxIterations: 10,
  pseudoLabelThreshold: 0.92,
  enableAugmentation: true,
});
// result.finalAccuracy, result.totalNewExamples
```

### Vulnerability Classification

```
POST /api/classify-vuln
Content-Type: application/json

{ "text": "SQL injection in login endpoint...", "topK": 5 }
```

Response:
```json
{
  "predictions": [{
    "cwe": "CWE-89",
    "score": 0.71,
    "name": "SQL Injection",
    "severity": "High",
    "owasp": "A03:2021 Injection",
    "remediation": ["Use parameterized queries...", "..."],
    "category": "Injection"
  }],
  "inferenceMs": 1.2,
  "modelInfo": { "parameters": "9.7M", "classes": 349, "architecture": "ResNet-MLP" }
}
```

### CVE ID Lookup

```
GET /api/nvd-lookup?cveId=CVE-2021-44228
```

Returns NVD metadata: description, CVSS v3.1 score/vector/severity, ground-truth CWEs, affected products (CPE), references, and timestamps. Optional `NVD_API_KEY` env var for higher rate limits.

### Dependency Vulnerability Scan

```
POST /api/osv-scan
Content-Type: application/json

{ "dependencies": [{ "name": "lodash", "version": "4.17.20", "ecosystem": "npm" }] }
```

Returns known vulnerabilities per dependency from [OSV.dev](https://osv.dev): vuln ID, summary, severity, fix version, published date, and CWE IDs. Max 100 dependencies per request.

### Export

```typescript
import { exportTrainingData } from "@/lib/enterprise/export-formats";

const { content, filename } = exportTrainingData(data, "rasa");
// Rasa YAML v3.1, Dialogflow ES, Lex V2, LUIS, Wit.ai, CSV, JSON
```

---

## Documentation

| Guide | What's Covered |
|-------|---------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Algorithm deep dive, ensemble math, classifier internals, design system |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker, Vercel, GCP, Nginx config, model weights |
| [TRAINING.md](docs/TRAINING.md) | Browser training, Python pipeline, vulnerability model training, custom data |
| [SELF-HOSTING.md](docs/SELF-HOSTING.md) | What runs where, ONNX export, the honest Ollama answer |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, PR process, code standards |
| [SECURITY.md](SECURITY.md) | Supported versions, reporting vulnerabilities |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Accessibility

WCAG 2.2 AA compliant. Full keyboard navigation (Alt+1–8 page switching). ARIA labels, screen reader support, reduced motion, skip navigation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## License

[GNU Affero General Public License v3.0](LICENSE)

---

<p align="center">
  <strong>Sentio</strong> — <em>I perceive.</em>
</p>
<p align="center">
  Built for <a href="https://dmj.one">dmj.one</a> &bull; Aatmanirbhar Bharat
</p>
