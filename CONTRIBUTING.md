# Contributing to Sentio

Thank you for considering contributing to Sentio! This document outlines the process and standards.

## Development Setup

```bash
git clone https://github.com/divyamohan1993/nlu-bot-trainer.git
cd nlu-bot-trainer
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

**With Docker:**

```bash
docker compose up --build
```

## Pull Request Process

1. **Fork and branch** from `main`. Use descriptive branch names: `feature/mlp-classifier`, `fix/softmax-normalization`.
2. **Write code** following existing patterns. TypeScript strict mode, no `any` unless absolutely necessary.
3. **Test manually** — Train the model, test predictions on the /test page, verify per-model scores. For vulnerability changes, test on /vulnerability with sample CVEs.
4. **Run the build** — `npm run build` must pass with zero errors.
5. **Open a PR** — Use the PR template. Describe what and why, not how.

## Code Standards

- **TypeScript strict mode** — No implicit `any`, no unused variables.
- **Classifier pattern** — New classifiers follow the same interface: `train*()`, `predict*()`, `serialize*()`, `deserialize*()`. See `src/lib/engine/classifiers/mlp.ts` for the template.
- **Pure math** — No external ML dependencies. All algorithms implemented from scratch in TypeScript.
- **Performance** — Inference must stay under 10ms. Measure with `performance.now()`.
- **Accessibility** — WCAG 2.2 AA. Semantic HTML, ARIA labels, keyboard navigation.

## Architecture Rules

1. All NLU ML runs client-side in the browser. No server-side computation for the NLU engine.
2. NLU models serialize to JSON and persist in localStorage (5–10MB budget).
3. The meta-weight grid search uses log-likelihood as the objective function.
4. New classifiers must integrate into the ensemble via `combineScores()` in `ensemble.ts`.
5. The vulnerability classifier runs server-side as a Next.js API route.

## Vulnerability Triage System

The vulnerability page has 4 independent modes. Each has different code paths:

| Mode | Key files | Runs where |
|------|----------|-----------|
| CVE Description | `/api/classify-vuln`, `cwe-database.ts` | Server-side |
| CVE ID Lookup | `/api/nvd-lookup`, `/api/classify-vuln` | Server-side (NVD proxy → ML) |
| Code Scanner | `src/lib/vuln/code-scanner.ts` | Client-side only |
| Dependency Scan | `src/lib/vuln/dependency-parser.ts`, `/api/osv-scan` | Client parse → server proxy |

### CWE Database Maintenance

The CWE enrichment database lives in `src/lib/vuln/cwe-database.ts`. When adding support for new CWEs:

1. Add the CWE name to the `CWE_NAMES` dictionary
2. For critical CWEs, add a full entry to `CURATED_CWES` with description, severity, OWASP mapping, and remediation
3. For less common CWEs, the category-based fallback handles enrichment automatically
4. Verify with `npx tsc --noEmit` — the file must compile clean

### Code Scanner Patterns

Vulnerability patterns live in `src/lib/vuln/code-scanner.ts`. When adding new patterns:

1. Add to the `PATTERNS` array with `id`, `name`, `cwe`, `severity`, `regex`, `languages`, `message`, and `fix`
2. Test against real-world code samples — minimize false positives
3. Add a sample to `CODE_SAMPLES` if the pattern covers a new category
4. Remember: this is regex-based pattern matching, not data-flow analysis. Don't overclaim

### Dependency Parser

The dependency file parser lives in `src/lib/vuln/dependency-parser.ts`. Supports:
- **npm**: `package.json` (JSON.parse, reads `dependencies` + `devDependencies`)
- **PyPI**: `requirements.txt` (line regex, handles `==`, `>=`, comments)
- **Maven**: `pom.xml` (XML regex for `<dependency>` blocks)

### Model Weights

Vulnerability classifier weights live in `public/models/vuln-classifier/`. These are large files (~87MB) tracked in the repo. When retraining:

1. Train with `training/vuln-classifier/train.py`
2. Export with `training/vuln-classifier/export_weights.py`
3. Copy `weights.json`, `tfidf_vocab.json`, `labels.json` to `public/models/vuln-classifier/`

### Docker Workflow

```bash
# Build and run
docker compose up --build

# Build image only
docker build -t sentio .

# Run with custom port
docker run -p 8080:3000 sentio
```

## Reporting Issues

- **Bugs** — Use the bug report template. Include browser, OS, and steps to reproduce.
- **Features** — Use the feature request template. Explain the use case.
- **Security** — See [SECURITY.md](SECURITY.md). Do NOT open public issues for vulnerabilities.

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 License.
