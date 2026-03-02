# Deployment Guide

## Local Development

```bash
git clone https://github.com/divyamohan1993/nlu-bot-trainer.git
cd nlu-bot-trainer
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The NLU engine works immediately — no setup needed.

### Vulnerability classifier

The vulnerability classifier requires model weights (~87 MB). Download them:

```bash
bash scripts/download-model.sh
```

This places `weights.json` in `public/models/vuln-classifier/`. Without it, the NLU features still work — only the `/vulnerability` page requires the weights.

---

## Docker

```bash
# Build
docker build -t sentio .

# Run
docker run -p 3000:3000 sentio
```

Or with Docker Compose:

```bash
docker compose up
```

The Docker image includes everything except the 87 MB vulnerability classifier weights. Mount them:

```bash
docker run -p 3000:3000 -v ./public/models:/app/public/models sentio
```

### Image size

The multi-stage build produces a ~200 MB image (Node.js Alpine + Next.js standalone output). The vulnerability model weights add ~87 MB if baked in.

---

## Vercel

One-click deploy:

1. Push to GitHub
2. Import the repository on [vercel.com/new](https://vercel.com/new)
3. Vercel auto-detects Next.js — no configuration needed
4. Deploy

**Note:** The vulnerability classifier weights (87 MB) exceed Vercel's serverless function size limit. For Vercel deployments, the NLU features work perfectly; the vulnerability page will show a "model not loaded" message unless you use an external storage solution for the weights.

---

## GCP Instance

For a full deployment including the vulnerability classifier:

```bash
# Create a small instance
gcloud compute instances create sentio \
  --machine-type=e2-small \
  --zone=asia-south1-a \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB

# SSH in
gcloud compute ssh sentio --zone=asia-south1-a

# On the VM:
sudo apt update && sudo apt install -y nodejs npm nginx
git clone https://github.com/divyamohan1993/nlu-bot-trainer.git
cd nlu-bot-trainer
npm install
bash scripts/download-model.sh
npm run build
npm start  # Runs on port 3000
```

Set up Nginx as a reverse proxy and Cloudflare DNS proxy in front.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 3000 | Server port |
| `NVD_API_KEY` | — | Optional: NVD API key for CVE ID Lookup (5 req/30s without, 50 with). Free at [nvd.nist.gov](https://nvd.nist.gov/developers/request-an-api-key) |
| `GOOGLE_API_KEY` | — | Optional: for synthetic data generation via Gemini |
| `NEXT_PUBLIC_DEFAULT_CONFIDENCE_THRESHOLD` | 0.65 | Prediction confidence threshold |

---

## External API Dependencies

The vulnerability triage features call external APIs through server-side proxy routes:

| Feature | External API | Auth | Rate Limit |
|---------|-------------|------|-----------|
| CVE ID Lookup | `services.nvd.nist.gov/rest/json/cves/2.0` | Optional API key | 5 req/30s (free), 50 req/30s (key) |
| Dependency Scan | `api.osv.dev/v1/query` | None | No published limit |
| Code Scanner | None — client-side only | — | — |
| ML Classification | Local inference — no external call | — | — |

If deploying behind a corporate firewall, allowlist `services.nvd.nist.gov` and `api.osv.dev` for outbound HTTPS.

---

## Model Weights

The vulnerability classifier weights are not included in the Git repository (87 MB). Options:

1. **Download script:** `bash scripts/download-model.sh` (tries GitHub release, falls back to GCP instructions)
2. **Train your own:** See [TRAINING.md](TRAINING.md)
3. **GitHub Release:** Download from the [v2.0.0 release](https://github.com/divyamohan1993/nlu-bot-trainer/releases/tag/v2.0.0)

The NLU ensemble trains in-browser and needs no pre-downloaded weights.
