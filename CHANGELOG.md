# Changelog

All notable changes to Sentio are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.2.0] — 2026-03-02

### Added
- **4-tab vulnerability triage** — CVE Description (existing), CVE ID Lookup, Code Scanner, Dependency Scan
  - **CVE ID Lookup** — Type a CVE ID (e.g., CVE-2021-44228), fetches NVD metadata (CVSS score, severity, affected products, references), then runs ML classification with CWE agreement check
  - **Code Scanner** — Paste code, get instant client-side vulnerability detection (~30 regex patterns across 9 categories: SQLi, XSS, Command Injection, Path Traversal, Hardcoded Secrets, Insecure Crypto, Deserialization, SSRF, Buffer Overflow). Zero network calls, runs in <5ms
  - **Dependency Scan** — Paste package.json, requirements.txt, or pom.xml. Queries OSV API for known vulnerabilities with CVSS severity, fix versions, and CWE enrichment
- **NVD API proxy** (`/api/nvd-lookup`) — CORS proxy for NVD REST API v2 with optional API key support (5 req/30s free, 50 req/30s with key)
- **OSV API proxy** (`/api/osv-scan`) — Batched dependency vulnerability scanning via OSV.dev (10 concurrent, 100 dep cap, no auth needed)
- **Shared vulnerability types** (`src/lib/vuln/types.ts`) — Centralized type definitions for all 4 vulnerability triage modes
- **Dependency file parser** (`src/lib/vuln/dependency-parser.ts`) — Client-side parser for npm, PyPI, and Maven manifests

### Changed
- **Vulnerability page** — Complete rewrite from single-mode to 4-tab interface with independent state per tab, ARIA tablist keyboard navigation (ArrowLeft/ArrowRight)
- **.env.example** — Added `NVD_API_KEY` documentation

---

## [2.1.0] — 2026-03-02

### Added
- **CWE enrichment database** — All 349 supported CWE classes now return human-readable names, descriptions, severity ratings, OWASP Top 10 2021 mapping, and remediation guidance
  - Top ~50 critical CWEs hand-curated with specific remediation steps
  - Remaining ~300 CWEs enriched via category-based fallback with MITRE CWE names
- **Vulnerability UI redesign** — Top prediction card with severity badge, "What This Means" description, "How to Fix" remediation bullets, ranked alternatives with confidence bars
- **Docker support** — Multi-stage Dockerfile (Node 20 Alpine), docker-compose.yml, `output: "standalone"` in Next.js config
- **Documentation directory** (`docs/`)
  - `ARCHITECTURE.md` — Algorithm deep dive, ensemble math, classifier internals, design system
  - `DEPLOYMENT.md` — Docker, Vercel, GCP, Nginx config, model weights management
  - `TRAINING.md` — Browser training, Python NLU pipeline, vulnerability model training, custom datasets
  - `SELF-HOSTING.md` — What runs where, ONNX export, honest Ollama/deployment answer

### Changed
- **README.md** — Complete restructure covering both NLU Bot Trainer and Vulnerability Triage capabilities, deployment story, training guides, scalability tiers, and honest value proposition
- **Vulnerability API** (`/api/classify-vuln`) — Response now includes `name`, `description`, `severity`, `owasp`, `remediation[]`, and `category` for each prediction
- Updated SECURITY.md, CONTRIBUTING.md with vulnerability classifier scope and Docker workflow

---

## [2.0.0] — 2026-03-02

### Added
- **Multi-Layer Perceptron (MLP)** neural network classifier (1024→128→12, 132,748 parameters)
  - Xavier weight initialization, ReLU activation, SGD with backpropagation
  - Full serialize/deserialize support for localStorage persistence
- **5-way ensemble meta-weight learning** via log-likelihood grid search over 3-fold CV
- Total parameter count: **~171,772** across 5 classifiers

### Changed
- Gradient Boosted Stumps increased from 50 to 150 rounds (7,200 parameters, up from 2,400)
- Meta-weight objective changed from accuracy-primary to **log-likelihood** (proper scoring rule)
- Architecture grid in train UI expanded to 5 columns showing per-model parameter counts

### Fixed
- **Double softmax bug** — removed redundant temperature-scaled softmax in `predictEnsemble` that was flattening all confidence to ~9%
- **GBM sigmoid+normalize** — changed from sigmoid-per-class normalization to proper softmax on raw log-odds scores
- **Meta-weight grid search bias** — accuracy-primary objective favored near-uniform LR predictions; log-likelihood correctly rewards confident correct predictions

---

## [1.0.0] — 2026-03-02

The foundational release. A complete NLU training and inference system running entirely in the browser.

### Core Engine
- **4-classifier stacking ensemble**: Logistic Regression (SGD + L2), Complement Naive Bayes (Rennie et al. 2003), Linear SVM (Pegasos), Gradient Boosted Stumps
- **MurmurHash3 feature hashing** into 1024-dimensional Float32Array vectors — zero vocabulary serialization
- **Multi-strategy tokenizer**: word n-grams (1-3), character n-grams (3-5), syntactic features, intent signals, positional encoding, BPE-inspired subword segmentation
- **Cross-validated meta-weight learning** via grid search over 3-fold CV
- **Sub-millisecond inference** — single classification in ~1-6ms on modern hardware

### Self-Learning
- **Autonomous improvement loop**: Evaluate → Diagnose → Augment → Self-Train → Curriculum → Retrain → Validate → Accept/Reject
- **Data augmentation strategies**: synonym substitution, random insertion/deletion/swap, back-paraphrase templates
- **Pseudo-labeling** with configurable confidence threshold (default ≥0.85)
- **Curriculum learning**: easy-to-hard ordering based on prediction confidence
- **Anti-degeneration safeguards**: confidence gate (≥92%), 3/4 committee consensus, held-out validation set, augmentation caps

### Enterprise Features
- **Model registry** with semantic versioning (major/minor/patch) and champion/challenger lifecycle
- **A/B testing** framework with configurable traffic splits and statistical comparison
- **Drift detection**: Page-Hinkley test (concept drift), DDM (error drift), vocabulary distribution monitoring
- **7-platform export**: Rasa YAML v3.1, Dialogflow ES, Amazon Lex V2, LUIS, Wit.ai, CSV, raw JSON

### Training Pipeline
- **GCloud spot VM** orchestration scripts for heavy training ($0.07/2hr session on e2-standard-4)
- **Python training bridge**: scikit-learn models, Optuna hyperparameter optimization, sentence-transformers
- **Synthetic data generation** via Gemini/Vertex AI
- **Public dataset downloader**: CLINC150, BANKING77, HWU64, ATIS, SNIPS

### UI & Pages
- **Dashboard** — model status, version count, drift indicator, quick actions
- **Data Management** — intent CRUD, example management, color-coded intents, import/export
- **Train** — ensemble training with live progress, architecture visualization, multi-format export
- **Test** — real-time inference with per-model breakdown, confidence visualization
- **Analytics** — per-intent F1 bars, confusion matrix heatmap, ensemble weight circles, drift monitoring
- **Self-Learn** — configurable autonomous loop with iteration history and results summary
- **Models** — version list, promote/delete, A/B test management, version detail panel
- **Settings** — configuration management

### Design
- Dark theme with glass morphism design system
- WCAG 2.2 compliant — full keyboard navigation (Alt+1-8), ARIA labels, screen reader support
- Responsive layout with collapsible sidebar

### Tech Stack
- Next.js 14.2, React 18.3, TypeScript 5.5 (strict), Tailwind CSS 3.4
- Zero ML runtime dependencies — all algorithms implemented from scratch in TypeScript
