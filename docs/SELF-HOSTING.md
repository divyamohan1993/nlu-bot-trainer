# Self-Hosting and Model Deployment

## The Ollama Question

> "Can I deploy Sentio's models in Ollama?"

**No — and here's why that's actually good news.**

Ollama serves **large language models** (LLMs) — generative models with billions of parameters that produce text. Sentio's models are **classifiers** — they map input to categories, not generate text. This is a fundamentally different architecture.

The good news: Sentio's models are **far simpler to deploy** than LLMs.

---

## How Sentio's Models Actually Work

### NLU Ensemble (171K parameters)

**Deployment: Zero infrastructure needed.**

The NLU engine runs entirely in the browser. There is no model to "deploy" — it ships as TypeScript code that executes client-side. Any web browser is the inference server.

To embed in your own application:

```typescript
import { trainEnsemble, predictEnsemble } from "@/lib/engine/ensemble";

// Train once (1-3 seconds)
const model = trainEnsemble(yourTrainingData);

// Predict (microseconds)
const result = predictEnsemble("hello there", model);
console.log(result.intent);      // "greet"
console.log(result.confidence);  // 0.94
```

The model serializes to ~2 MB of JSON and persists in localStorage. No GPU, no Python, no server.

### Vulnerability Classifier (9.7M parameters)

**Deployment: Next.js API route or any Node.js server.**

The vulnerability classifier uses TF-IDF + a fused ResNet-MLP. Inference is pure JavaScript — matrix multiplies and activation functions. No ML runtime dependency.

**Option 1: Run the full app**
```bash
docker build -t sentio .
docker run -p 3000:3000 -v ./public/models:/app/public/models sentio
```

**Option 2: Use the API endpoints directly**
```bash
# ML classification
curl -X POST http://localhost:3000/api/classify-vuln \
  -H "Content-Type: application/json" \
  -d '{"text": "SQL injection in login form via username parameter", "topK": 5}'

# CVE ID lookup (fetches NVD metadata + ML classification)
curl http://localhost:3000/api/nvd-lookup?cveId=CVE-2021-44228

# Dependency vulnerability scan
curl -X POST http://localhost:3000/api/osv-scan \
  -H "Content-Type: application/json" \
  -d '{"dependencies": [{"name": "lodash", "version": "4.17.20", "ecosystem": "npm"}]}'
```

**Option 3: Extract the inference code**

The entire inference pipeline is in one file: `src/app/api/classify-vuln/route.ts`. It loads three JSON files and does math. You can port this to any language or framework.

---

## ONNX Export (Advanced)

The training pipeline can export to ONNX format for use with production serving frameworks:

```bash
cd training/vuln-classifier
pip install onnx onnxscript
python export_weights.py --checkpoint output/best_model.pt --format onnx
```

ONNX models can be served with:
- **ONNX Runtime** — Lightweight inference in Python, C++, C#, Java
- **Triton Inference Server** — NVIDIA's production serving framework
- **TensorFlow Serving** — After ONNX→TF conversion
- **AWS SageMaker** — Managed ONNX endpoint

---

## Edge Deployment

The NLU ensemble already runs on the edge (browser). For the vulnerability classifier:

- **Cloudflare Workers**: The inference code is pure JS. With model weights stored in R2 or KV, it can run on Cloudflare's edge network. The 87 MB weights exceed the Worker bundle limit, but can be streamed from R2.
- **Vercel Edge Functions**: Similar constraints — use external storage for weights.
- **Deno Deploy / Bun**: The inference code has no Node.js-specific dependencies.

---

## Performance Characteristics

| Model | Parameters | Inference | Memory | Storage |
|-------|-----------|-----------|--------|---------|
| NLU Ensemble | 171K | 0.3-1ms | ~20 MB | ~2 MB |
| Vuln Classifier | 9.7M | 1-30ms | ~350 MB | ~87 MB |

The NLU ensemble is negligible. The vulnerability classifier's memory footprint is dominated by the 50K-dimensional TF-IDF vocabulary loaded into memory.

---

## When You Actually Need an LLM

If your use case requires:
- **Generating** remediation text (not just looking it up)
- **Deep source code analysis** — data-flow analysis, taint tracking, inter-procedural vulnerabilities (Sentio's code scanner catches common patterns via regex, but can't do semantic analysis)
- **Conversational** security triage
- **Explaining** vulnerabilities in natural language

Then you need an LLM, and tools like Ollama, vLLM, or cloud APIs (Claude, GPT) are appropriate. Sentio's tools can work **alongside** an LLM — use the classifier for fast CWE categorization, the code scanner for quick pattern detection, and the dependency scanner for known CVEs, then send results to an LLM for deeper analysis.
