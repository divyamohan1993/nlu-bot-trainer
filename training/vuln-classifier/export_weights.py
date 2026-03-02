#!/usr/bin/env python3
"""
Export PyTorch model weights to JSON for pure-JS browser inference.
Fuses BatchNorm into preceding Linear layers (standard inference optimization).

Usage:
  python export_weights.py                    # From output/ directory
  python export_weights.py --checkpoint output/best_model.pt --output output/weights.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# Must match train.py architecture constants
HIDDEN_1 = 192
HIDDEN_2 = 192
HIDDEN_3 = 96


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class VulnClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
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


def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d):
    """
    Fuse BatchNorm into preceding Linear layer.

    BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Linear+BN: y = gamma * (W*x + b - mean) / sqrt(var + eps) + beta

    Fused: W' = gamma / sqrt(var+eps) * W
           b' = gamma * (b - mean) / sqrt(var+eps) + beta
    """
    gamma = bn.weight.data           # [out_features]
    beta = bn.bias.data              # [out_features]
    mean = bn.running_mean.data      # [out_features]
    var = bn.running_var.data        # [out_features]
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)  # [out_features]

    # Fused weight: scale each row of W
    W_fused = linear.weight.data * scale.unsqueeze(1)  # [out, in] * [out, 1]
    b_fused = scale * (linear.bias.data - mean) + beta

    return W_fused.numpy(), b_fused.numpy()


def gelu_np(x):
    """GELU activation (numpy)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def export_weights(checkpoint_path: str, output_path: str):
    """Load checkpoint, fuse BN, export to JSON."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get model dimensions from checkpoint
    state = ckpt if not isinstance(ckpt, dict) else ckpt.get("model_state_dict", ckpt)

    input_dim = state["stem.0.weight"].shape[1]
    num_classes = state["head.weight"].shape[0]
    print(f"Model: input_dim={input_dim}, num_classes={num_classes}")
    print(f"Parameters: {sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor)):,}")

    # Reconstruct model and load weights
    model = VulnClassifier(input_dim, num_classes)
    model.load_state_dict(state)
    model.eval()

    # Fuse BatchNorm into Linear layers
    print("Fusing BatchNorm layers...")

    # 1. Stem: Linear(50K, 192) + BN(192)
    stem_W, stem_b = fuse_linear_bn(model.stem[0], model.stem[1])

    # 2. ResBlock: Linear(192, 192) + BN(192)
    res_W, res_b = fuse_linear_bn(model.res_block.net[0], model.res_block.net[1])

    # 3. Neck: Linear(192, 96) + BN(96)
    neck_W, neck_b = fuse_linear_bn(model.neck[0], model.neck[1])

    # 4. Head: Linear(96, C) -- no BN, use directly
    head_W = model.head.weight.data.numpy()
    head_b = model.head.bias.data.numpy()

    print(f"  stem:  W={stem_W.shape}, b={stem_b.shape}")
    print(f"  res:   W={res_W.shape}, b={res_b.shape}")
    print(f"  neck:  W={neck_W.shape}, b={neck_b.shape}")
    print(f"  head:  W={head_W.shape}, b={head_b.shape}")

    # Verify fused model matches original
    print("Verifying fused output matches original...")
    with torch.no_grad():
        dummy = torch.randn(1, input_dim)
        original_out = model(dummy).numpy()

        # Manual fused forward pass
        h = dummy.numpy() @ stem_W.T + stem_b
        h = gelu_np(h)
        res = h @ res_W.T + res_b
        h = h + gelu_np(res)
        h = h @ neck_W.T + neck_b
        h = gelu_np(h)
        fused_out = h @ head_W.T + head_b

        max_diff = np.max(np.abs(original_out - fused_out))
        print(f"  Max difference: {max_diff:.8f}")
        if max_diff > 0.01:
            print("  WARNING: Large difference! Check fusion logic.")
        else:
            print("  OK -- fused model matches original.")

    # Export as JSON
    print("Exporting to JSON...")

    def to_list(arr, precision=6):
        """Convert numpy array to nested list with controlled precision."""
        if arr.ndim == 1:
            return [round(float(x), precision) for x in arr]
        return [[round(float(x), precision) for x in row] for row in arr]

    weights = {
        "stem_weight": to_list(stem_W),
        "stem_bias": to_list(stem_b),
        "res_weight": to_list(res_W),
        "res_bias": to_list(res_b),
        "neck_weight": to_list(neck_W),
        "neck_bias": to_list(neck_b),
        "head_weight": to_list(head_W),
        "head_bias": to_list(head_b),
        "architecture": {
            "input_dim": int(input_dim),
            "hidden1": HIDDEN_1,
            "hidden2": HIDDEN_2,
            "hidden3": HIDDEN_3,
            "num_classes": int(num_classes),
            "activation": "gelu",
            "bn_fused": True,
        },
    }

    with open(output_path, "w") as f:
        json.dump(weights, f, separators=(",", ":"))

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Weights exported: {output_path} ({size_mb:.1f} MB)")

    # Also export a smaller version with 4-digit precision
    small_path = output_path.replace(".json", "_small.json")
    weights_small = {
        "stem_weight": to_list(stem_W, 4),
        "stem_bias": to_list(stem_b, 4),
        "res_weight": to_list(res_W, 4),
        "res_bias": to_list(res_b, 4),
        "neck_weight": to_list(neck_W, 4),
        "neck_bias": to_list(neck_b, 4),
        "head_weight": to_list(head_W, 4),
        "head_bias": to_list(head_b, 4),
        "architecture": weights["architecture"],
    }
    with open(small_path, "w") as f:
        json.dump(weights_small, f, separators=(",", ":"))
    small_size = Path(small_path).stat().st_size / 1024 / 1024
    print(f"Small weights:    {small_path} ({small_size:.1f} MB)")

    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model weights to JSON")
    parser.add_argument("--checkpoint", default="output/best_model.pt",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--output", default="output/weights.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    export_weights(args.checkpoint, args.output)
