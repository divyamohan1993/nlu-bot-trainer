#!/usr/bin/env bash
# =============================================================================
# GCloud Spot VM Setup for NLU Training
# =============================================================================
# Usage:
#   chmod +x gcloud-spot-vm.sh
#   ./gcloud-spot-vm.sh [--gpu]
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Billing enabled on project
#   - Compute Engine API enabled
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration -- EDIT THESE
# ---------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-shanti-thakur}"   # Confirm your project name
ZONE="us-central1-a"                             # Cheapest zone for spot VMs
INSTANCE_NAME="nlu-trainer-$(date +%s)"
MAX_RUN_HOURS=2                                  # Auto-delete after this
USE_GPU=false

# Parse args
for arg in "$@"; do
  case $arg in
    --gpu) USE_GPU=true ;;
    --project=*) PROJECT_ID="${arg#*=}" ;;
    --zone=*) ZONE="${arg#*=}" ;;
    --hours=*) MAX_RUN_HOURS="${arg#*=}" ;;
  esac
done

# ---------------------------------------------------------------------------
# Machine type selection
# ---------------------------------------------------------------------------
# For NLU training (not deep learning), CPU is usually sufficient:
#   - e2-standard-4   : 4 vCPU, 16GB RAM, ~$0.034/hr spot  -- good default
#   - e2-standard-8   : 8 vCPU, 32GB RAM, ~$0.067/hr spot  -- larger datasets
#   - e2-highmem-4    : 4 vCPU, 32GB RAM, ~$0.045/hr spot  -- memory-heavy
#   - c2-standard-8   : 8 vCPU, 32GB RAM, ~$0.08/hr spot   -- compute-heavy
#   - n1-standard-4 + T4: ~$0.12/hr spot (GPU)              -- if embedding gen
#
# For 2-hour NLU training with scikit-learn + sentence-transformers:
#   e2-standard-4 is plenty. Total cost: ~$0.07

if [ "$USE_GPU" = true ]; then
  MACHINE_TYPE="n1-standard-4"
  ACCELERATOR="--accelerator=count=1,type=nvidia-tesla-t4"
  # T4 GPU zones: us-central1-a, us-east1-c, europe-west4-b
  echo "[INFO] GPU mode: n1-standard-4 + T4 GPU in $ZONE"
  echo "[INFO] Estimated spot cost: ~\$0.12/hr (\$0.24 for 2 hours)"
else
  MACHINE_TYPE="e2-standard-4"
  ACCELERATOR=""
  echo "[INFO] CPU mode: e2-standard-4 (4 vCPU, 16GB RAM)"
  echo "[INFO] Estimated spot cost: ~\$0.034/hr (\$0.07 for 2 hours)"
fi

DISK_SIZE="15"  # GB -- minimal for NLU training

echo "[INFO] Project: $PROJECT_ID"
echo "[INFO] Zone: $ZONE"
echo "[INFO] Instance: $INSTANCE_NAME"
echo "[INFO] Max runtime: ${MAX_RUN_HOURS}h"
echo ""

# ---------------------------------------------------------------------------
# Startup script (embedded)
# ---------------------------------------------------------------------------
STARTUP_SCRIPT=$(cat <<'STARTUP_EOF'
#!/bin/bash
set -euo pipefail

LOG_FILE="/var/log/nlu-training.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "NLU Training VM Startup - $(date -u)"
echo "=========================================="

# ---- Self-destruct timer ----
MAX_HOURS=__MAX_RUN_HOURS__
echo "[TIMER] Auto-shutdown in ${MAX_HOURS} hours"
(sleep $((MAX_HOURS * 3600)) && echo "[TIMER] Max runtime reached. Shutting down..." && shutdown -h now) &
TIMER_PID=$!

# ---- System packages ----
echo "[SETUP] Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3-pip git wget curl jq unzip

# If python3.12 not available in default repos, use deadsnakes
if ! command -v python3.12 &>/dev/null; then
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -qq
  apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
fi

# Ensure python3.12 is the default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 2>/dev/null || true

# ---- GPU drivers (if GPU attached) ----
if lspci | grep -i nvidia &>/dev/null; then
  echo "[SETUP] NVIDIA GPU detected. Installing drivers..."
  apt-get install -y -qq linux-headers-$(uname -r)
  # Install CUDA toolkit (minimal)
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -qq
  apt-get install -y -qq cuda-toolkit-12-4
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
  nvidia-smi || echo "[WARN] nvidia-smi failed"
fi

# ---- Python virtual environment ----
echo "[SETUP] Creating Python venv..."
WORK_DIR="/opt/nlu-training"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
python3.12 -m venv venv
source venv/bin/activate

echo "[SETUP] Installing Python packages..."
pip install --upgrade pip wheel setuptools

# Core ML packages
pip install \
  scikit-learn>=1.5 \
  sentence-transformers>=3.0 \
  optuna>=4.0 \
  numpy>=1.26 \
  pandas>=2.2 \
  joblib>=1.4 \
  onnxruntime>=1.19 \
  kagglehub \
  google-cloud-aiplatform>=1.70 \
  google-cloud-storage>=2.18

# Optional: torch CPU-only (smaller) if no GPU
if ! lspci | grep -i nvidia &>/dev/null; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch
fi

echo "[SETUP] Python environment ready."
python3 --version
pip list | grep -E "(scikit|sentence|optuna|torch|onnx)"

# ---- Download training script from GCS (or use embedded) ----
# If you uploaded train_nlu.py to a GCS bucket:
# gsutil cp gs://${GCS_BUCKET}/training/train_nlu.py "$WORK_DIR/"
# gsutil cp gs://${GCS_BUCKET}/training/training_data.json "$WORK_DIR/"

# Otherwise, the script should be SCP'd to the VM after creation.
echo "[SETUP] Waiting for training data and script..."
echo "[SETUP] SCP your files with:"
echo "  gcloud compute scp train_nlu.py $(hostname):$WORK_DIR/ --zone=__ZONE__"
echo "  gcloud compute scp training_data.json $(hostname):$WORK_DIR/ --zone=__ZONE__"

# ---- If training data exists, run automatically ----
if [ -f "$WORK_DIR/train_nlu.py" ] && [ -f "$WORK_DIR/training_data.json" ]; then
  echo "[TRAIN] Starting training..."
  cd "$WORK_DIR"
  source venv/bin/activate
  python3 train_nlu.py \
    --input training_data.json \
    --output results/ \
    --optimize \
    2>&1 | tee training_output.log

  echo "[TRAIN] Training complete!"

  # Upload results to GCS if bucket is set
  if [ -n "${GCS_BUCKET:-}" ]; then
    gsutil -m cp -r results/ "gs://${GCS_BUCKET}/training-results/$(date +%Y%m%d-%H%M%S)/"
    echo "[UPLOAD] Results uploaded to GCS"
  fi
fi

echo "[READY] VM is ready at $(date -u)"
echo "[READY] Connect with: gcloud compute ssh $(hostname) --zone=__ZONE__"
STARTUP_EOF
)

# Replace placeholders
STARTUP_SCRIPT="${STARTUP_SCRIPT//__MAX_RUN_HOURS__/$MAX_RUN_HOURS}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__ZONE__/$ZONE}"

# ---------------------------------------------------------------------------
# Create the Spot VM
# ---------------------------------------------------------------------------
echo "[CREATE] Creating Spot VM..."

# Build the gcloud command
CMD="gcloud compute instances create $INSTANCE_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --boot-disk-size=${DISK_SIZE}GB \
  --boot-disk-type=pd-standard \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script=<(echo \"$STARTUP_SCRIPT\") \
  --labels=purpose=nlu-training,auto-delete=true \
  --tags=nlu-training"

# Add GPU accelerator if requested
if [ -n "$ACCELERATOR" ]; then
  CMD="$CMD $ACCELERATOR --maintenance-policy=TERMINATE"
fi

echo "[CMD] $CMD"
echo ""

# Execute
eval "$CMD"

echo ""
echo "=========================================="
echo "Spot VM Created Successfully!"
echo "=========================================="
echo ""
echo "Instance: $INSTANCE_NAME"
echo "Zone:     $ZONE"
echo "Project:  $PROJECT_ID"
echo ""
echo "Next steps:"
echo "  1. Wait ~2 min for startup script to finish"
echo "  2. Upload training files:"
echo "     gcloud compute scp training/train_nlu.py $INSTANCE_NAME:/opt/nlu-training/ --zone=$ZONE"
echo "     gcloud compute scp training/training_data.json $INSTANCE_NAME:/opt/nlu-training/ --zone=$ZONE"
echo "  3. SSH in and run training:"
echo "     gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "     cd /opt/nlu-training && source venv/bin/activate"
echo "     python3 train_nlu.py --input training_data.json --output results/ --optimize"
echo "  4. Download results:"
echo "     gcloud compute scp --recurse $INSTANCE_NAME:/opt/nlu-training/results/ ./results/ --zone=$ZONE"
echo "  5. VM auto-deletes after ${MAX_RUN_HOURS} hours, or delete manually:"
echo "     gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet"
echo ""

# ---------------------------------------------------------------------------
# Optional: Schedule auto-delete with Cloud Scheduler as backup
# ---------------------------------------------------------------------------
# gcloud scheduler jobs create http "delete-$INSTANCE_NAME" \
#   --schedule="0 */2 * * *" \
#   --uri="https://compute.googleapis.com/compute/v1/projects/$PROJECT_ID/zones/$ZONE/instances/$INSTANCE_NAME" \
#   --http-method=DELETE \
#   --oauth-service-account-email="$(gcloud iam service-accounts list --filter='compute' --format='value(email)' | head -1)"
