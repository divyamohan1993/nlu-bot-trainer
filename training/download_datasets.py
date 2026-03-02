#!/usr/bin/env python3
"""
Kaggle & HuggingFace NLU Dataset Downloader & Preprocessor
===========================================================
Downloads, preprocesses, and converts popular NLU/intent classification
datasets into the format expected by train_nlu.py.

Usage:
  python download_datasets.py --dataset all --output datasets/
  python download_datasets.py --dataset clinc150 --output datasets/
  python download_datasets.py --list

Requirements:
  pip install kagglehub datasets pandas

For Kaggle datasets, you need either:
  - KAGGLE_USERNAME and KAGGLE_KEY env vars, OR
  - ~/.kaggle/kaggle.json with your API credentials
  Get yours at: https://www.kaggle.com/settings -> API -> Create New Token
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Dataset registry
# =============================================================================

DATASETS = {
    # -------------------------------------------------------------------------
    # 1. CLINC150 (Out-of-Scope Intent Detection)
    # -------------------------------------------------------------------------
    # 150 intents, 23,700 in-scope + 1,200 out-of-scope queries
    # Best for: Multi-domain intent classification, OOS detection
    # Paper: "An Evaluation Dataset for Intent Classification and OOS Detection"
    "clinc150": {
        "name": "CLINC150 / OOS Intent Dataset",
        "source": "kaggle",
        "kaggle_slug": "stefanlarson/outofscope-intent-classification-dataset",
        "huggingface_id": "clinc_oos",
        "description": "150 intents across 10 domains + out-of-scope. 23,700 examples.",
        "domains": ["banking", "credit_cards", "kitchen", "home", "auto", "travel",
                     "utility", "work", "small_talk", "meta"],
        "size": "~2MB",
        "license": "CC BY-SA 3.0",
    },

    # -------------------------------------------------------------------------
    # 2. SNIPS NLU Benchmark
    # -------------------------------------------------------------------------
    # 7 intents, ~14,000 examples (2,000 per intent)
    # Best for: Voice assistant / smart speaker intents
    "snips": {
        "name": "SNIPS NLU Benchmark",
        "source": "huggingface",
        "huggingface_id": "snips_built_in_intents",
        "description": "7 intents for smart speaker: weather, music, restaurant, etc.",
        "domains": ["smart_speaker", "voice_assistant"],
        "size": "~1MB",
        "license": "CC0",
    },

    # -------------------------------------------------------------------------
    # 3. ATIS (Airline Travel Information System)
    # -------------------------------------------------------------------------
    # 26 intents, ~5,800 examples
    # Classic NLU benchmark for travel domain
    "atis": {
        "name": "ATIS (Airline Travel Information)",
        "source": "huggingface",
        "huggingface_id": "tuetschek/atis",
        "description": "26 intents for airline travel queries. Classic NLU benchmark.",
        "domains": ["travel", "airlines"],
        "size": "~800KB",
        "license": "Research",
    },

    # -------------------------------------------------------------------------
    # 4. Customer Support Intent Dataset (Kaggle)
    # -------------------------------------------------------------------------
    # 27 intents, ~3,000+ examples from real customer support
    "customer_support": {
        "name": "Customer Support Intent Dataset",
        "source": "kaggle",
        "kaggle_slug": "scodepy/customer-support-intent-dataset",
        "description": "27 customer support intents: refund, cancel, track order, etc.",
        "domains": ["customer_service", "ecommerce"],
        "size": "~500KB",
        "license": "CC0",
    },

    # -------------------------------------------------------------------------
    # 5. Intent Recognition Dataset (Kaggle)
    # -------------------------------------------------------------------------
    # General-purpose intent recognition
    "intent_recognition": {
        "name": "Intent Recognition Dataset",
        "source": "kaggle",
        "kaggle_slug": "himanshunayal/intent-recognition-dataset",
        "description": "General-purpose intent recognition with multiple domains.",
        "domains": ["general", "multi-domain"],
        "size": "~300KB",
        "license": "CC0",
    },

    # -------------------------------------------------------------------------
    # 6. Banking77 (PolyAI)
    # -------------------------------------------------------------------------
    # 77 intents, 13,083 examples from banking customer service
    "banking77": {
        "name": "Banking77",
        "source": "huggingface",
        "huggingface_id": "PolyAI/banking77",
        "description": "77 fine-grained banking intents. 13,083 customer queries.",
        "domains": ["banking", "finance"],
        "size": "~1.5MB",
        "license": "CC BY 4.0",
    },

    # -------------------------------------------------------------------------
    # 7. HWU64 (Heriot-Watt University)
    # -------------------------------------------------------------------------
    # 64 intents across 21 domains, ~25,000 examples
    "hwu64": {
        "name": "HWU64 Multi-Domain",
        "source": "huggingface",
        "huggingface_id": "xingkunliuxtracta/hwu64",
        "description": "64 intents across 21 domains. Great for multi-domain NLU.",
        "domains": ["alarm", "audio", "calendar", "cooking", "datetime", "email",
                     "general", "iot", "lists", "music", "news", "play", "qa",
                     "recommendation", "social", "takeaway", "transport", "weather"],
        "size": "~2MB",
        "license": "Research",
    },
}


# =============================================================================
# Downloaders
# =============================================================================

def download_from_kaggle(slug: str, output_dir: str) -> str:
    """Download a Kaggle dataset. Returns the path to downloaded files."""
    try:
        import kagglehub
        log(f"Downloading from Kaggle: {slug}")
        path = kagglehub.dataset_download(slug)
        log(f"Downloaded to: {path}")
        return path
    except ImportError:
        log("kagglehub not installed. Trying kaggle CLI...")
        os.makedirs(output_dir, exist_ok=True)
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", output_dir, "--unzip"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to download {slug}. Install kagglehub or kaggle CLI.\n"
                f"stderr: {result.stderr}"
            )
        return output_dir


def download_from_huggingface(dataset_id: str, output_dir: str) -> dict:
    """Download a HuggingFace dataset. Returns the dataset dict."""
    try:
        from datasets import load_dataset
        log(f"Downloading from HuggingFace: {dataset_id}")
        ds = load_dataset(dataset_id)
        log(f"Downloaded: {list(ds.keys())} splits")
        return ds
    except ImportError:
        raise RuntimeError("Install 'datasets' package: pip install datasets")


# =============================================================================
# Preprocessors -- convert each dataset to unified format
# =============================================================================

def preprocess_clinc150(data_path: str, output_dir: str) -> str:
    """Preprocess CLINC150 dataset."""
    import pandas as pd

    output_file = os.path.join(output_dir, "clinc150_processed.json")

    # Try HuggingFace first (more reliable)
    try:
        from datasets import load_dataset
        ds = load_dataset("clinc_oos", "plus")

        examples = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                for item in ds[split]:
                    text = item.get("text", "")
                    intent = item.get("intent", 42)  # 42 = oos
                    # Map integer label to name if needed
                    if isinstance(intent, int):
                        intent_name = ds[split].features["intent"].int2str(intent)
                    else:
                        intent_name = str(intent)

                    if text and intent_name != "oos":
                        examples.append({"text": text, "intent": intent_name})

        save_processed(examples, output_file)
        return output_file

    except Exception as e:
        log(f"HuggingFace download failed: {e}. Trying Kaggle...")

    # Fall back to Kaggle
    path = download_from_kaggle("stefanlarson/outofscope-intent-classification-dataset", data_path)
    csv_files = list(Path(path).rglob("*.csv")) + list(Path(path).rglob("*.json"))

    examples = []
    for f in csv_files:
        if f.suffix == ".csv":
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                text = str(row.get("text", row.iloc[0]))
                intent = str(row.get("intent", row.get("label", row.iloc[-1])))
                if intent != "oos" and text:
                    examples.append({"text": text, "intent": intent})
        elif f.suffix == ".json":
            with open(f) as jf:
                data = json.load(jf)
                if isinstance(data, dict):
                    for split_name, split_data in data.items():
                        if isinstance(split_data, list):
                            for item in split_data:
                                if isinstance(item, list) and len(item) >= 2:
                                    text, intent = item[0], item[1]
                                    if intent != "oos":
                                        examples.append({"text": text, "intent": intent})

    save_processed(examples, output_file)
    return output_file


def preprocess_banking77(data_path: str, output_dir: str) -> str:
    """Preprocess Banking77 dataset."""
    output_file = os.path.join(output_dir, "banking77_processed.json")

    from datasets import load_dataset
    ds = load_dataset("PolyAI/banking77")

    examples = []
    for split in ["train", "test"]:
        if split in ds:
            for item in ds[split]:
                text = item["text"]
                label = item["label"]
                intent_name = ds[split].features["label"].int2str(label)
                examples.append({"text": text, "intent": intent_name})

    save_processed(examples, output_file)
    return output_file


def preprocess_snips(data_path: str, output_dir: str) -> str:
    """Preprocess SNIPS dataset."""
    output_file = os.path.join(output_dir, "snips_processed.json")

    from datasets import load_dataset
    ds = load_dataset("snips_built_in_intents")

    examples = []
    for split in ds.keys():
        for item in ds[split]:
            text = item["text"]
            label = item["label"]
            intent_name = ds[split].features["label"].int2str(label)
            examples.append({"text": text, "intent": intent_name})

    save_processed(examples, output_file)
    return output_file


def preprocess_atis(data_path: str, output_dir: str) -> str:
    """Preprocess ATIS dataset."""
    output_file = os.path.join(output_dir, "atis_processed.json")

    from datasets import load_dataset
    ds = load_dataset("tuetschek/atis")

    examples = []
    for split in ds.keys():
        for item in ds[split]:
            text = item.get("text", "")
            intent = item.get("intent", "")
            if text and intent:
                examples.append({"text": text, "intent": intent})

    save_processed(examples, output_file)
    return output_file


def preprocess_customer_support(data_path: str, output_dir: str) -> str:
    """Preprocess Customer Support Intent dataset from Kaggle."""
    import pandas as pd

    output_file = os.path.join(output_dir, "customer_support_processed.json")

    path = download_from_kaggle("scodepy/customer-support-intent-dataset", data_path)
    csv_files = list(Path(path).rglob("*.csv"))

    examples = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Common column names
        text_col = next((c for c in df.columns if c.lower() in ["text", "query", "utterance", "sentence"]), df.columns[0])
        intent_col = next((c for c in df.columns if c.lower() in ["intent", "label", "category", "class"]), df.columns[-1])

        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            intent = str(row[intent_col]).strip()
            if text and intent and text != "nan" and intent != "nan":
                examples.append({"text": text, "intent": intent})

    save_processed(examples, output_file)
    return output_file


def preprocess_intent_recognition(data_path: str, output_dir: str) -> str:
    """Preprocess general Intent Recognition dataset from Kaggle."""
    import pandas as pd

    output_file = os.path.join(output_dir, "intent_recognition_processed.json")

    path = download_from_kaggle("himanshunayal/intent-recognition-dataset", data_path)
    csv_files = list(Path(path).rglob("*.csv"))

    examples = []
    for f in csv_files:
        df = pd.read_csv(f)
        text_col = next((c for c in df.columns if c.lower() in ["text", "query", "utterance", "sentence"]), df.columns[0])
        intent_col = next((c for c in df.columns if c.lower() in ["intent", "label", "category", "class"]), df.columns[-1])

        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            intent = str(row[intent_col]).strip()
            if text and intent and text != "nan" and intent != "nan":
                examples.append({"text": text, "intent": intent})

    save_processed(examples, output_file)
    return output_file


def preprocess_hwu64(data_path: str, output_dir: str) -> str:
    """Preprocess HWU64 dataset."""
    output_file = os.path.join(output_dir, "hwu64_processed.json")

    from datasets import load_dataset
    ds = load_dataset("xingkunliuxtracta/hwu64")

    examples = []
    for split in ds.keys():
        for item in ds[split]:
            text = item.get("text", "")
            label = item.get("label", -1)
            if isinstance(label, int):
                intent_name = ds[split].features["label"].int2str(label)
            else:
                intent_name = str(label)
            if text and intent_name:
                examples.append({"text": text, "intent": intent_name})

    save_processed(examples, output_file)
    return output_file


# =============================================================================
# Utility functions
# =============================================================================

PREPROCESSORS = {
    "clinc150": preprocess_clinc150,
    "snips": preprocess_snips,
    "atis": preprocess_atis,
    "customer_support": preprocess_customer_support,
    "intent_recognition": preprocess_intent_recognition,
    "banking77": preprocess_banking77,
    "hwu64": preprocess_hwu64,
}


def save_processed(examples: list[dict], output_file: str):
    """Save processed examples in the unified format."""
    from collections import Counter

    # Group by intent for the NLU trainer format
    intent_groups: dict[str, list] = {}
    for ex in examples:
        intent = ex["intent"]
        if intent not in intent_groups:
            intent_groups[intent] = []
        intent_groups[intent].append({"text": ex["text"], "entities": []})

    output = {
        "intents": [
            {
                "name": intent_name,
                "description": f"Auto-imported intent: {intent_name}",
                "examples": exs,
            }
            for intent_name, exs in sorted(intent_groups.items())
        ],
        "entities": [],
        "metadata": {
            "totalExamples": len(examples),
            "nIntents": len(intent_groups),
            "source": "auto-downloaded",
        },
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    counts = Counter(ex["intent"] for ex in examples)
    log(f"Saved {len(examples)} examples, {len(intent_groups)} intents to {output_file}")
    log(f"Top 5 intents: {counts.most_common(5)}")


def merge_datasets(files: list[str], output_file: str, prefix_intents: bool = True):
    """Merge multiple processed dataset files into one."""
    all_examples = []

    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)

        source = Path(fpath).stem.replace("_processed", "")

        for intent_obj in data.get("intents", []):
            intent_name = intent_obj["name"]
            if prefix_intents:
                intent_name = f"{source}/{intent_name}"
            for ex in intent_obj.get("examples", []):
                all_examples.append({"text": ex["text"], "intent": intent_name})

    save_processed(all_examples, output_file)
    log(f"Merged {len(files)} datasets -> {output_file}")


def log(msg: str):
    print(f"[DATASETS] {msg}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess NLU datasets")
    parser.add_argument("--dataset", "-d", default="all",
                        help="Dataset to download (or 'all'). Use --list to see options.")
    parser.add_argument("--output", "-o", default="datasets/", help="Output directory")
    parser.add_argument("--list", "-l", action="store_true", help="List available datasets")
    parser.add_argument("--merge", "-m", action="store_true",
                        help="Merge all downloaded datasets into one file")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable NLU Datasets:")
        print("=" * 70)
        for key, info in DATASETS.items():
            print(f"\n  {key}")
            print(f"    Name:    {info['name']}")
            print(f"    Source:  {info['source']}")
            print(f"    Desc:    {info['description']}")
            print(f"    Domains: {', '.join(info['domains'])}")
            print(f"    Size:    {info['size']}")
            print(f"    License: {info['license']}")
        print("\nUsage: python download_datasets.py --dataset <name> --output datasets/")
        print("       python download_datasets.py --dataset all --output datasets/")
        return

    os.makedirs(args.output, exist_ok=True)
    raw_dir = os.path.join(args.output, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if args.dataset == "all":
        datasets_to_download = list(DATASETS.keys())
    else:
        datasets_to_download = [d.strip() for d in args.dataset.split(",")]

    processed_files = []

    for ds_name in datasets_to_download:
        if ds_name not in PREPROCESSORS:
            log(f"Unknown dataset: {ds_name}. Skipping.")
            continue

        log(f"\n{'='*60}")
        log(f"Processing: {ds_name}")
        log(f"{'='*60}")

        try:
            output_file = PREPROCESSORS[ds_name](raw_dir, args.output)
            processed_files.append(output_file)
            log(f"Success: {output_file}")
        except Exception as e:
            log(f"Failed to process {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # Merge if requested
    if args.merge and len(processed_files) > 1:
        merge_file = os.path.join(args.output, "merged_all.json")
        merge_datasets(processed_files, merge_file)

    log(f"\nDone! Processed {len(processed_files)} datasets.")
    log(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
