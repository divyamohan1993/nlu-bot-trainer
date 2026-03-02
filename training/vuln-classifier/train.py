#!/usr/bin/env python3
"""
CVE-CWE Vulnerability Classifier - 10M Parameter Training Pipeline
====================================================================
Trains a ResNet-style MLP on CVE descriptions to predict CWE weakness types.

Architecture: TF-IDF (50K sparse) -> 192 -> 192 (skip) -> 96 -> N classes
Parameters:  ~9.7M (varies with class count)
Techniques:  Focal loss, label smoothing, Mixup, cosine annealing, BatchNorm, GELU

Usage:
  python train.py                          # Full pipeline
  python train.py --epochs 100             # Override epochs
  python train.py --checkpoint resume      # Resume from checkpoint

Requirements:
  pip install torch scikit-learn datasets onnx onnxruntime pandas tqdm

March 2026 research-grade techniques:
  - Focal loss for extreme class imbalance (296 CWE types)
  - Label smoothing (0.05) for calibration
  - Mixup data augmentation for regularization
  - Cosine annealing with warm restarts
  - Residual connections in MLP
  - GELU activation (smoother than ReLU)
  - Gradient clipping for stability
  - INT8 dynamic quantization for deployment
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, top_k_accuracy_score
from scipy.sparse import csr_matrix
from tqdm import tqdm

# =============================================================================
# Config
# =============================================================================

CHECKPOINT_DIR = Path("./checkpoints")
OUTPUT_DIR = Path("./output")
TFIDF_FEATURES = 50_000
HIDDEN_1 = 192
HIDDEN_2 = 192
HIDDEN_3 = 96
MIN_CLASS_SAMPLES = 10       # Drop CWE types with fewer examples
FOCAL_GAMMA = 2.0            # Focal loss gamma
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
GRAD_CLIP = 1.0
CHECKPOINT_EVERY = 5         # epochs


# =============================================================================
# Model: ResNet-style MLP with skip connections
# =============================================================================

class ResBlock(nn.Module):
    """Residual block: Linear -> BatchNorm -> GELU -> Dropout + skip."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VulnClassifier(nn.Module):
    """
    ResNet-style MLP for vulnerability classification.

    Architecture:
      Input (50K sparse) -> Linear(50K, 192) -> BN -> GELU -> Dropout(0.3)
        -> ResBlock(192, dropout=0.2)   [skip connection]
        -> Linear(192, 96) -> BN -> GELU -> Dropout(0.1)
        -> Linear(96, num_classes)

    Total params: 50K*192 + 192 + 192*192 + 192 + 192*96 + 96 + 96*C
                = 9,600,000 + 37,056 + 18,528 + 96*C
                ~ 9.65M + 96*C
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.BatchNorm1d(HIDDEN_1),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.res_block = ResBlock(HIDDEN_2, dropout=0.2)
        self.neck = nn.Sequential(
            nn.Linear(HIDDEN_2, HIDDEN_3),
            nn.BatchNorm1d(HIDDEN_3),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(HIDDEN_3, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_block(x)
        x = self.neck(x)
        return self.head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Focal Loss (handles extreme class imbalance)
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal loss: focuses learning on hard, misclassified examples.
    Lin et al., "Focal Loss for Dense Object Detection" (2017).
    With label smoothing for calibration.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0,
                 weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# =============================================================================
# Mixup data augmentation
# =============================================================================

class SparseDataset(Dataset):
    """Dataset that lazily converts sparse rows to dense tensors per batch."""

    def __init__(self, X_sparse, y):
        self.X = X_sparse
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(
            self.X[idx].toarray().squeeze(0), dtype=torch.float32
        )
        return x, self.y[idx]


def mixup_batch(x, y, alpha=0.2):
    """Mixup: linearly interpolate pairs of examples + labels."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: weighted sum of losses against both targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Data loading
# =============================================================================

def load_cve_dataset():
    """Load CVE-CWE dataset from HuggingFace."""
    log("Loading CVE-CWE dataset from HuggingFace...")

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "stasvinokur/cve-and-cwe-dataset-1999-2025",
            split="train",
        )
        df = ds.to_pandas()
        log("Loaded %d CVE records from HuggingFace" % len(df))
    except Exception as e:
        log("HuggingFace load failed: %s" % str(e))
        log("Trying local cache...")
        cache_path = Path("./cve_cwe_data.parquet")
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            log("Loaded %d records from local cache" % len(df))
        else:
            raise RuntimeError(
                "Cannot load dataset. Ensure 'datasets' is installed "
                "and you have internet access, or provide a local cache."
            )

    return df


def preprocess_data(df, min_samples=MIN_CLASS_SAMPLES):
    """Clean text, filter classes, encode labels."""
    log("Preprocessing data...")

    # Identify text and label columns
    text_col = None
    label_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "description" in col_lower or "desc" in col_lower or "text" in col_lower:
            if text_col is None:
                text_col = col
        if "cwe" in col_lower and ("id" in col_lower or "name" in col_lower or "type" in col_lower):
            if label_col is None:
                label_col = col

    # Fallback: try common column names
    if text_col is None:
        for candidate in ["Description", "description", "cve_description", "text"]:
            if candidate in df.columns:
                text_col = candidate
                break
    if label_col is None:
        for candidate in ["CWE_ID", "cwe_id", "CWE-ID", "cwe", "CWE", "cwe_name"]:
            if candidate in df.columns:
                label_col = candidate
                break

    if text_col is None or label_col is None:
        log("Available columns: %s" % str(list(df.columns)))
        raise ValueError(
            "Cannot identify text column (found: %s) "
            "or label column (found: %s)" % (text_col, label_col)
        )

    log("Text column: %s, Label column: %s" % (text_col, label_col))

    # Drop rows with missing values
    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    # Clean text
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 20]

    # Clean labels
    df["label"] = df["label"].astype(str).str.strip()
    df = df[~df["label"].isin(["", "NVD-CWE-noinfo", "NVD-CWE-Other", "nan"])]

    # Filter rare classes
    class_counts = df["label"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df = df[df["label"].isin(valid_classes)]

    log("After filtering: %d examples, %d classes" % (len(df), df["label"].nunique()))
    log("Class distribution: min=%d, max=%d, median=%.0f" % (
        class_counts[valid_classes].min(),
        class_counts[valid_classes].max(),
        class_counts[valid_classes].median(),
    ))

    return df


# =============================================================================
# Training loop
# =============================================================================

def train_model(
    X_train_sparse,
    y_train,
    X_val_sparse,
    y_val,
    num_classes,
    class_weights,
    epochs=50,
    batch_size=512,
    lr=0.003,
    resume_path=None,
):
    """Train the ResNet-MLP with all the bells and whistles."""

    input_dim = X_train_sparse.shape[1]
    model = VulnClassifier(input_dim, num_classes)
    log("Model parameters: {:,}".format(model.param_count()))

    # Use lazy sparse->dense conversion per batch (memory efficient)
    log("Creating sparse-aware data loaders (lazy dense conversion)...")
    train_dataset = SparseDataset(X_train_sparse, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False,
    )
    val_dataset = SparseDataset(X_val_sparse, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, num_workers=2)

    # Loss with class weights and focal loss
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    criterion = FocalLoss(
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        weight=weight_tensor,
    )

    # Optimizer: AdamW (decoupled weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # LR scheduler: Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    start_epoch = 0
    best_val_acc = 0.0
    best_model_state = None

    # Resume from checkpoint
    if resume_path and Path(resume_path).exists():
        log("Resuming from checkpoint: %s" % resume_path)
        ckpt = torch.load(resume_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        log("Resumed at epoch %d, best_val_acc=%.4f" % (start_epoch, best_val_acc))

    # Store references for preemption handler
    globals()["model"] = model
    globals()["optimizer"] = optimizer
    globals()["scheduler"] = scheduler

    log("")
    log("Starting training: %d epochs, batch_size=%d, lr=%s" % (epochs, batch_size, lr))
    log("Techniques: Focal loss (gamma=%.1f), label smoothing=%.2f, "
        "Mixup (alpha=%.1f), cosine annealing, gradient clipping=%.1f" % (
            FOCAL_GAMMA, LABEL_SMOOTHING, MIXUP_ALPHA, GRAD_CLIP))

    for epoch in range(start_epoch, epochs):
        globals()["current_epoch"] = epoch
        globals()["best_val_acc"] = best_val_acc

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Epoch %d/%d" % (epoch + 1, epochs), leave=False)
        for batch_x, batch_y in pbar:
            # Mixup augmentation
            mixed_x, y_a, y_b, lam = mixup_batch(batch_x, batch_y, MIXUP_ALPHA)

            optimizer.zero_grad()
            logits = model(mixed_x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            pbar.set_postfix(loss="%.4f" % loss.item(), acc="%.3f" % (correct / total))

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        all_val_logits = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
                all_val_logits.append(logits)

        val_acc = val_correct / val_total
        train_acc = correct / total
        avg_train_loss = total_loss / total
        avg_val_loss = val_loss / val_total

        # Top-5 accuracy
        all_logits = torch.cat(all_val_logits)
        all_probs = F.softmax(all_logits, dim=-1).numpy()
        k = min(5, num_classes)
        top5_acc = top_k_accuracy_score(y_val, all_probs, k=k)

        log("Epoch %d/%d: train_loss=%.4f, train_acc=%.3f, "
            "val_loss=%.4f, val_acc=%.3f, top5_acc=%.3f, lr=%.6f" % (
                epoch + 1, epochs, avg_train_loss, train_acc,
                avg_val_loss, val_acc, top5_acc, scheduler.get_last_lr()[0]))

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            log("  -> New best val_acc: %.4f" % best_val_acc)

        # Checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc)

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        log("\nLoaded best model (val_acc=%.4f)" % best_val_acc)

    return model


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc):
    """Save training checkpoint to disk and optionally to GCS."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / ("checkpoint_epoch_%d.pt" % (epoch + 1))
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
    }, path)
    log("  Checkpoint saved: %s" % path)

    # Try GCS upload for durability
    bucket = os.environ.get("GCS_BUCKET", "")
    if bucket:
        os.system("gsutil -q cp %s gs://%s/checkpoints/" % (path, bucket))


# Graceful shutdown on spot preemption
def handle_preemption(signum, frame):
    log("\nPREEMPTION SIGNAL - saving emergency checkpoint...")
    if "model" in globals() and "optimizer" in globals():
        save_checkpoint(
            globals()["model"], globals()["optimizer"],
            globals()["scheduler"], globals().get("current_epoch", 0),
            globals().get("best_val_acc", 0.0),
        )
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_preemption)


# =============================================================================
# Export
# =============================================================================

def export_onnx(model, input_dim, output_path):
    """Export to ONNX format."""
    model.eval()
    dummy = torch.randn(1, input_dim)
    torch.onnx.export(
        model, dummy, output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    log("ONNX model exported: %s (%.1f MB)" % (output_path, size_mb))


def export_quantized_onnx(onnx_path, output_path):
    """INT8 dynamic quantization for 4x size reduction."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            onnx_path, output_path,
            weight_type=QuantType.QInt8,
        )
        size_mb = Path(output_path).stat().st_size / 1024 / 1024
        log("Quantized ONNX model: %s (%.1f MB)" % (output_path, size_mb))
    except ImportError:
        log("onnxruntime.quantization not available, skipping INT8 export")


def export_metadata(label_encoder, tfidf, metrics, output_dir):
    """Export label mapping, TF-IDF vocabulary, and metrics."""
    # Label mapping
    labels = {int(i): str(name) for i, name in enumerate(label_encoder.classes_)}
    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    log("Labels exported: %d classes" % len(labels))

    # TF-IDF vocabulary (needed for inference)
    vocab = tfidf.vocabulary_
    idf = tfidf.idf_.tolist()
    tfidf_meta = {
        "vocabulary": {str(k): int(v) for k, v in vocab.items()},
        "idf": idf,
        "max_features": TFIDF_FEATURES,
        "ngram_range": list(tfidf.ngram_range),
        "sublinear_tf": tfidf.sublinear_tf,
    }
    with open(output_dir / "tfidf_vocab.json", "w") as f:
        json.dump(tfidf_meta, f)
    vocab_size_mb = Path(output_dir / "tfidf_vocab.json").stat().st_size / 1024 / 1024
    log("TF-IDF vocabulary exported: %d terms (%.1f MB)" % (len(vocab), vocab_size_mb))

    # Metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log("Metrics exported")


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, X_test_sparse, y_test, label_encoder):
    """Detailed evaluation on test set."""
    model.eval()

    # Evaluate in batches to avoid OOM
    test_dataset = SparseDataset(X_test_sparse, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=2)

    all_logits = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            all_logits.append(logits)

    all_logits = torch.cat(all_logits)
    probs = F.softmax(all_logits, dim=-1).numpy()
    preds = all_logits.argmax(dim=-1).numpy()

    acc = (preds == y_test).mean()
    k = min(5, len(label_encoder.classes_))
    top5 = top_k_accuracy_score(y_test, probs, k=k)

    log("\nTest Results:")
    log("  Top-1 Accuracy: %.4f (%.1f%%)" % (acc, acc * 100))
    log("  Top-5 Accuracy: %.4f (%.1f%%)" % (top5, top5 * 100))

    # Per-class report
    report = classification_report(
        y_test, preds,
        target_names=label_encoder.classes_,
        output_dict=True,
    )

    # Inference speed benchmark
    single_input = torch.tensor(
        X_test_sparse[:1].toarray(), dtype=torch.float32
    )
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(single_input)
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = np.mean(times[10:])
    p99_ms = np.percentile(times[10:], 99)
    log("  Inference latency: avg=%.2fms, p99=%.2fms" % (avg_ms, p99_ms))

    return {
        "top1_accuracy": float(acc),
        "top5_accuracy": float(top5),
        "num_classes": len(label_encoder.classes_),
        "num_test_examples": len(y_test),
        "inference_avg_ms": round(avg_ms, 2),
        "inference_p99_ms": round(p99_ms, 2),
        "per_class_f1": {
            k: round(v["f1-score"], 4)
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train CVE-CWE Vulnerability Classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--min-samples", type=int, default=MIN_CLASS_SAMPLES,
                        help="Minimum examples per CWE class")
    args = parser.parse_args()

    log("=" * 70)
    log("CVE-CWE Vulnerability Classifier - 10M Parameter Training Pipeline")
    log("=" * 70)
    log("PyTorch %s, CUDA available: %s" % (torch.__version__, torch.cuda.is_available()))
    log("CPU threads: %d" % torch.get_num_threads())
    t_start = time.time()

    # 1. Load data
    df = load_cve_dataset()

    # 2. Preprocess
    df = preprocess_data(df, min_samples=args.min_samples)

    # 3. Train/val/test split (80/10/10)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
    log("Split: train=%d, val=%d, test=%d" % (len(train_df), len(val_df), len(test_df)))

    # 4. TF-IDF feature extraction
    log("Extracting TF-IDF features (max_features=%d)..." % TFIDF_FEATURES)
    tfidf = TfidfVectorizer(
        max_features=TFIDF_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        dtype=np.float32,
    )
    X_train = tfidf.fit_transform(train_df["text"])
    X_val = tfidf.transform(val_df["text"])
    X_test = tfidf.transform(test_df["text"])
    log("TF-IDF shape: %s" % str(X_train.shape))

    # 5. Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_val = le.transform(val_df["label"])
    y_test = le.transform(test_df["label"])
    num_classes = len(le.classes_)
    log("Classes: %d" % num_classes)

    # 6. Compute class weights for focal loss (inverse frequency)
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = len(y_train) / (num_classes * class_counts + 1e-8)
    class_weights = np.clip(class_weights, 0.1, 10.0)

    # 7. Train
    model = train_model(
        X_train, y_train, X_val, y_val,
        num_classes=num_classes,
        class_weights=class_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_path=args.checkpoint,
    )

    # 8. Evaluate
    metrics = evaluate_model(model, X_test, y_test, le)

    # 9. Export
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    onnx_path = str(OUTPUT_DIR / "vuln_classifier.onnx")
    export_onnx(model, X_train.shape[1], onnx_path)

    quantized_path = str(OUTPUT_DIR / "vuln_classifier_int8.onnx")
    export_quantized_onnx(onnx_path, quantized_path)

    export_metadata(le, tfidf, metrics, OUTPUT_DIR)

    # 10. Summary
    elapsed = time.time() - t_start
    log("\n" + "=" * 70)
    log("TRAINING COMPLETE")
    log("=" * 70)
    log("Total time: %.1f minutes" % (elapsed / 60))
    log("Parameters: {:,}".format(model.param_count()))
    log("Test accuracy: %.1f%% (top-1), %.1f%% (top-5)" % (
        metrics["top1_accuracy"] * 100, metrics["top5_accuracy"] * 100))
    log("Macro F1: %.4f" % metrics["macro_f1"])
    log("Inference: %.2fms avg" % metrics["inference_avg_ms"])
    log("\nArtifacts in %s/:" % OUTPUT_DIR)
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        unit = "KB" if size < 1024 * 1024 else "MB"
        size_val = size / 1024 if unit == "KB" else size / (1024 * 1024)
        log("  %s: %.1f %s" % (f.name, size_val, unit))

    # Upload to GCS if available
    bucket = os.environ.get("GCS_BUCKET", "")
    if bucket:
        log("\nUploading to gs://%s/vuln-classifier/..." % bucket)
        os.system("gsutil -m cp -r %s/* gs://%s/vuln-classifier/" % (OUTPUT_DIR, bucket))
        log("Upload complete.")


def log(msg):
    print("[TRAIN] %s" % msg, flush=True)


if __name__ == "__main__":
    main()
