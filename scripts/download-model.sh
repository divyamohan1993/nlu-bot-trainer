#!/bin/bash
# Download vuln-classifier model weights from the GCP VM or release
# Usage: bash scripts/download-model.sh

set -euo pipefail

MODEL_DIR="public/models/vuln-classifier"
WEIGHTS_FILE="$MODEL_DIR/weights.json"

if [ -f "$WEIGHTS_FILE" ]; then
    echo "Model weights already exist at $WEIGHTS_FILE"
    echo "Delete the file first if you want to re-download."
    exit 0
fi

mkdir -p "$MODEL_DIR"

echo "Downloading vuln-classifier model weights (87 MB)..."

# Try downloading from GitHub release first
RELEASE_URL="https://github.com/divyamohan1993/nlu-bot-trainer/releases/download/v2.0.0/weights.json"
if curl -sL --fail -o "$WEIGHTS_FILE" "$RELEASE_URL" 2>/dev/null; then
    echo "Downloaded from GitHub release"
else
    echo "GitHub release not available."
    echo ""
    echo "To get the weights, either:"
    echo "  1. Copy from GCP VM:"
    echo "     gcloud compute scp vuln-trainer:/opt/vuln-trainer/output/weights.json $WEIGHTS_FILE --zone=asia-south2-a"
    echo ""
    echo "  2. Or re-run training:"
    echo "     cd training/vuln-classifier && bash setup_vm.sh"
    exit 1
fi

echo ""
echo "Model artifacts:"
ls -lh "$MODEL_DIR/"
echo ""
echo "Ready! Start the dev server with: npm run dev"
