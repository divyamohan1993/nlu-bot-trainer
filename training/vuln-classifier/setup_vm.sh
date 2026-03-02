#!/bin/bash
# =============================================================================
# CVE-CWE Vulnerability Classifier — VM Setup & Training Script
# =============================================================================
# Idempotent. Run on a fresh GCP Ubuntu 22.04 c2-standard-4 spot instance.
# Downloads dataset, installs deps, trains model, uploads artifacts to GCS.
#
# Usage:
#   # From local machine — create VM and run training:
#   gcloud compute instances create vuln-trainer \
#     --zone=asia-south2-a \
#     --machine-type=c2-standard-4 \
#     --provisioning-model=SPOT \
#     --instance-termination-action=STOP \
#     --image-family=ubuntu-2204-lts \
#     --image-project=ubuntu-os-cloud \
#     --boot-disk-size=50GB \
#     --metadata-from-file=startup-script=setup_vm.sh
#
#   # Or SSH in and run manually:
#   gcloud compute ssh vuln-trainer --zone=asia-south2-a
#   sudo bash /path/to/setup_vm.sh
#
# The script will:
#   1. Install Python 3.12, PyTorch (CPU), scikit-learn, HuggingFace datasets
#   2. Clone the repo / copy training script
#   3. Download CVE-CWE dataset from HuggingFace
#   4. Train 10M-param ResNet-MLP (focal loss, mixup, cosine LR)
#   5. Export to ONNX + INT8 quantized
#   6. Upload artifacts to GCS bucket
# =============================================================================

set -euo pipefail

LOG_FILE="/var/log/vuln-trainer.log"
WORK_DIR="/opt/vuln-trainer"
GCS_BUCKET="${GCS_BUCKET:-}"  # Set via env or metadata

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=============================================="
log "CVE-CWE Vulnerability Classifier — VM Setup"
log "=============================================="

# 1. System packages
log "Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git wget curl > /dev/null 2>&1
log "System packages installed"

# 2. Working directory
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# 3. Python virtual environment
if [ ! -d "venv" ]; then
    log "Creating Python virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 4. Install Python dependencies
log "Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet \
    torch --index-url https://download.pytorch.org/whl/cpu \
    scikit-learn \
    datasets \
    pandas \
    onnx \
    onnxruntime \
    tqdm \
    pyarrow

log "Python packages installed:"
python3 -c "import torch; print('  PyTorch:', torch.__version__)"
python3 -c "import sklearn; print('  scikit-learn:', sklearn.__version__)"
python3 -c "import onnxruntime; print('  ONNX Runtime:', onnxruntime.__version__)"

# 5. Copy or download training script
TRAIN_SCRIPT="$WORK_DIR/train.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    log "Downloading training script from repo..."
    # Try to get from the repo first
    REPO_URL="https://raw.githubusercontent.com/divyamohan1993/nlu-bot-trainer/main/training/vuln-classifier/train.py"
    wget -q -O "$TRAIN_SCRIPT" "$REPO_URL" 2>/dev/null || true

    if [ ! -s "$TRAIN_SCRIPT" ]; then
        log "Could not download from repo. Please copy train.py to $WORK_DIR"
        log "Use: gcloud compute scp train.py vuln-trainer:$WORK_DIR/ --zone=asia-south2-a"
        exit 1
    fi
fi
log "Training script ready: $TRAIN_SCRIPT"

# 6. Run training
log "Starting training..."
cd "$WORK_DIR"
python3 train.py \
    --epochs 50 \
    --batch-size 512 \
    --lr 0.003 \
    2>&1 | tee -a "$LOG_FILE"

# 7. Upload to GCS (if bucket configured)
if [ -n "$GCS_BUCKET" ]; then
    log "Uploading artifacts to gs://$GCS_BUCKET/vuln-classifier/..."
    gsutil -m cp -r "$WORK_DIR/output/"* "gs://$GCS_BUCKET/vuln-classifier/"
    gsutil -m cp "$LOG_FILE" "gs://$GCS_BUCKET/vuln-classifier/training.log"
    log "Upload complete"
else
    log "No GCS_BUCKET set. Artifacts are in $WORK_DIR/output/"
    log "Download with: gcloud compute scp --recurse vuln-trainer:$WORK_DIR/output/ ./ --zone=asia-south2-a"
fi

log "=============================================="
log "DONE. Model artifacts in $WORK_DIR/output/"
log "=============================================="
