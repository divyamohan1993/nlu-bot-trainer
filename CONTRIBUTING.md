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

## Pull Request Process

1. **Fork and branch** from `main`. Use descriptive branch names: `feature/mlp-classifier`, `fix/softmax-normalization`.
2. **Write code** following existing patterns. TypeScript strict mode, no `any` unless absolutely necessary.
3. **Test manually** — Train the model, test predictions on the /test page, verify per-model scores.
4. **Run the build** — `npm run build` must pass with zero errors.
5. **Open a PR** — Use the PR template. Describe what and why, not how.

## Code Standards

- **TypeScript strict mode** — No implicit `any`, no unused variables.
- **Classifier pattern** — New classifiers follow the same interface: `train*()`, `predict*()`, `serialize*()`, `deserialize*()`. See `src/lib/engine/classifiers/mlp.ts` for the template.
- **Pure math** — No external ML dependencies. All algorithms implemented from scratch in TypeScript.
- **Performance** — Inference must stay under 10ms. Measure with `performance.now()`.
- **Accessibility** — WCAG 2.2 AA. Semantic HTML, ARIA labels, keyboard navigation.

## Architecture Rules

1. All ML runs client-side in the browser. No server-side computation for the core engine.
2. Models serialize to JSON and persist in localStorage (5-10MB budget).
3. The meta-weight grid search uses log-likelihood as the objective function.
4. New classifiers must integrate into the ensemble via `combineScores()` in `ensemble.ts`.

## Reporting Issues

- **Bugs** — Use the bug report template. Include browser, OS, and steps to reproduce.
- **Features** — Use the feature request template. Explain the use case.
- **Security** — See [SECURITY.md](SECURITY.md). Do NOT open public issues for vulnerabilities.

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 License.
