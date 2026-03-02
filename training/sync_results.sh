#!/usr/bin/env bash
# =============================================================================
# Sync training results between GCP VM and local machine
# =============================================================================
# Usage:
#   # Download results from VM to local
#   ./sync_results.sh download [INSTANCE_NAME] [ZONE]
#
#   # Upload training data to VM
#   ./sync_results.sh upload [INSTANCE_NAME] [ZONE]
#
#   # Full workflow: upload data, run training, download results
#   ./sync_results.sh full [INSTANCE_NAME] [ZONE]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE_DIR="/opt/nlu-training"

# Defaults
INSTANCE="${2:-nlu-trainer}"
ZONE="${3:-us-central1-a}"
PROJECT="${GCP_PROJECT_ID:-shanti-thakur}"

ACTION="${1:-help}"

case "$ACTION" in

  # ---- Upload training data and scripts to VM ----
  upload)
    echo "[SYNC] Uploading training files to $INSTANCE..."

    # Export training data from the NLU trainer app (if not already exported)
    if [ ! -f "$SCRIPT_DIR/training_data.json" ]; then
      echo "[SYNC] No training_data.json found."
      echo "[SYNC] Export from the NLU Trainer UI (Train page -> Export)"
      echo "[SYNC] Or create it manually with the expected format."
      exit 1
    fi

    # Upload files
    gcloud compute scp \
      "$SCRIPT_DIR/train_nlu.py" \
      "$SCRIPT_DIR/training_data.json" \
      "$SCRIPT_DIR/requirements.txt" \
      "$INSTANCE:$REMOTE_DIR/" \
      --zone="$ZONE" \
      --project="$PROJECT"

    echo "[SYNC] Upload complete!"
    echo "[SYNC] SSH in: gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT"
    ;;

  # ---- Download results from VM ----
  download)
    echo "[SYNC] Downloading results from $INSTANCE..."

    LOCAL_RESULTS="$SCRIPT_DIR/results"
    mkdir -p "$LOCAL_RESULTS"

    gcloud compute scp --recurse \
      "$INSTANCE:$REMOTE_DIR/results/" \
      "$LOCAL_RESULTS/" \
      --zone="$ZONE" \
      --project="$PROJECT"

    echo "[SYNC] Results downloaded to: $LOCAL_RESULTS"
    echo ""

    # Show what we got
    if [ -d "$LOCAL_RESULTS" ]; then
      echo "Files:"
      ls -la "$LOCAL_RESULTS/"
      echo ""

      # Quick summary if training_results.json exists
      if [ -f "$LOCAL_RESULTS/training_results.json" ]; then
        echo "Training Summary:"
        python3 -c "
import json
with open('$LOCAL_RESULTS/training_results.json') as f:
    r = json.load(f)
for m in r.get('results', [])[:5]:
    print(f\"  {m['name']:30s}  CV F1={m['cv_mean']:.4f}  Acc={m['accuracy']:.4f}\")
print(f\"  Total time: {r.get('timing', {}).get('total_seconds', 0):.1f}s\")
" 2>/dev/null || true
      fi

      # Check if TS-compatible model exists
      if [ -f "$LOCAL_RESULTS/model_ts_compatible.json" ]; then
        echo ""
        echo "TypeScript-compatible model ready!"
        echo "Size: $(du -h "$LOCAL_RESULTS/model_ts_compatible.json" | cut -f1)"
        echo ""
        echo "To use in the NLU Trainer app:"
        echo "  1. Open the app in browser"
        echo "  2. Go to Train page"
        echo "  3. Import model_ts_compatible.json"
      fi
    fi
    ;;

  # ---- Full workflow ----
  full)
    echo "[SYNC] Full training workflow"
    echo "=================================="

    # Step 1: Upload
    echo ""
    echo "[STEP 1/4] Uploading training data..."
    "$0" upload "$INSTANCE" "$ZONE"

    # Step 2: Run training on VM
    echo ""
    echo "[STEP 2/4] Running training on VM..."
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --command="
      cd $REMOTE_DIR
      source venv/bin/activate
      pip install -r requirements.txt 2>/dev/null
      python3 train_nlu.py \\
        --input training_data.json \\
        --output results/ \\
        --optimize \\
        --n-trials 50
    "

    # Step 3: Download results
    echo ""
    echo "[STEP 3/4] Downloading results..."
    "$0" download "$INSTANCE" "$ZONE"

    # Step 4: Cleanup option
    echo ""
    echo "[STEP 4/4] Training complete!"
    echo ""
    read -p "Delete the VM instance? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --quiet
      echo "[SYNC] VM deleted."
    else
      echo "[SYNC] VM kept running. Remember to delete it when done!"
      echo "  gcloud compute instances delete $INSTANCE --zone=$ZONE --project=$PROJECT"
    fi
    ;;

  # ---- Help ----
  help|*)
    echo "NLU Training Sync Tool"
    echo "======================"
    echo ""
    echo "Usage:"
    echo "  $0 upload   [INSTANCE] [ZONE]  - Upload training data to VM"
    echo "  $0 download [INSTANCE] [ZONE]  - Download results from VM"
    echo "  $0 full     [INSTANCE] [ZONE]  - Full workflow (upload, train, download)"
    echo ""
    echo "Defaults:"
    echo "  INSTANCE: nlu-trainer"
    echo "  ZONE:     us-central1-a"
    echo "  PROJECT:  $PROJECT (set GCP_PROJECT_ID to override)"
    echo ""
    echo "Prerequisites:"
    echo "  1. Create VM:  ./gcloud-spot-vm.sh"
    echo "  2. Export training data from the NLU Trainer UI as training_data.json"
    echo "  3. Place training_data.json in this directory ($SCRIPT_DIR)"
    ;;

esac
