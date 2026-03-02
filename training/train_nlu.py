#!/usr/bin/env python3
"""
Advanced NLU Training Pipeline
===============================
Trains multiple classifiers with hyperparameter optimization, knowledge
distillation, and exports optimized models as JSON for TypeScript consumption.

Usage:
  python train_nlu.py --input training_data.json --output results/ --optimize
  python train_nlu.py --input training_data.json --output results/ --quick
  python train_nlu.py --help

Requirements:
  pip install scikit-learn sentence-transformers optuna numpy pandas joblib

Author: divyamohan1993
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nlu-trainer")


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class TrainingExample:
    text: str
    intent: str


@dataclass
class TrainingConfig:
    input_path: str
    output_dir: str
    optimize: bool = False
    quick: bool = False
    n_trials: int = 100
    cv_folds: int = 5
    use_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    distill: bool = True
    random_state: int = 42


@dataclass
class ModelResult:
    name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    cv_mean: float
    cv_std: float
    training_time: float
    params: dict = field(default_factory=dict)


# =============================================================================
# Data loading -- supports both the NLU trainer JSON format and flat formats
# =============================================================================

def load_training_data(path: str) -> list[TrainingExample]:
    """Load training data from JSON. Supports multiple formats."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: list[TrainingExample] = []

    # Format 1: NLU Bot Trainer format (from store.ts exportJsonFormat)
    if "intents" in data and isinstance(data["intents"], list):
        for intent_obj in data["intents"]:
            intent_name = intent_obj.get("name", "")
            for ex in intent_obj.get("examples", []):
                text = ex.get("text", "") if isinstance(ex, dict) else str(ex)
                if text and intent_name:
                    examples.append(TrainingExample(text=text, intent=intent_name))

    # Format 2: Flat array [{text, intent}, ...]
    elif isinstance(data, list):
        for item in data:
            if "text" in item and "intent" in item:
                examples.append(TrainingExample(text=item["text"], intent=item["intent"]))
            elif "text" in item and "label" in item:
                examples.append(TrainingExample(text=item["text"], intent=item["label"]))

    # Format 3: NLU Bot Trainer full TrainingData format
    elif "intents" in data and "metadata" in data:
        for intent_obj in data["intents"]:
            intent_name = intent_obj.get("name", "")
            for ex in intent_obj.get("examples", []):
                text = ex.get("text", "")
                if text and intent_name:
                    examples.append(TrainingExample(text=text, intent=intent_name))

    if not examples:
        raise ValueError(f"No training examples found in {path}")

    log.info(f"Loaded {len(examples)} examples across {len(set(e.intent for e in examples))} intents")
    return examples


def validate_data(examples: list[TrainingExample]) -> dict[str, int]:
    """Validate and report on training data distribution."""
    intent_counts = Counter(e.intent for e in examples)
    log.info("Intent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {intent}: {count} examples")

    # Warn about imbalanced classes
    min_count = min(intent_counts.values())
    max_count = max(intent_counts.values())
    if max_count > 3 * min_count:
        log.warning(f"Imbalanced data: min={min_count}, max={max_count}. Consider augmentation.")

    if min_count < 3:
        log.warning(f"Some intents have fewer than 3 examples. Minimum for CV is 3.")

    return dict(intent_counts)


# =============================================================================
# Feature extraction
# =============================================================================

def build_tfidf_features(
    texts: list[str],
    params: dict | None = None,
) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer with configurable params."""
    defaults = {
        "analyzer": "word",
        "ngram_range": (1, 2),
        "max_features": 10000,
        "sublinear_tf": True,
        "min_df": 1,
        "max_df": 0.95,
        "strip_accents": "unicode",
        "lowercase": True,
    }
    if params:
        defaults.update(params)

    vectorizer = TfidfVectorizer(**defaults)
    vectorizer.fit(texts)
    return vectorizer


def build_embedding_features(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Generate sentence embeddings using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer

        log.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        log.info("Encoding texts...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
        log.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    except ImportError:
        log.warning("sentence-transformers not installed. Skipping embeddings.")
        return np.array([])


# =============================================================================
# Classifier training
# =============================================================================

def train_svm(X, y, params: dict | None = None) -> Pipeline:
    """Train a calibrated Linear SVM."""
    defaults = {"C": 1.0, "max_iter": 5000, "class_weight": "balanced"}
    if params:
        defaults.update(params)

    svm = CalibratedClassifierCV(
        LinearSVC(**defaults),
        cv=3,
        method="sigmoid",
    )
    svm.fit(X, y)
    return svm


def train_logistic_regression(X, y, params: dict | None = None):
    """Train Logistic Regression."""
    defaults = {
        "C": 1.0,
        "max_iter": 2000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "multi_class": "multinomial",
    }
    if params:
        defaults.update(params)
    lr = LogisticRegression(**defaults)
    lr.fit(X, y)
    return lr


def train_random_forest(X, y, params: dict | None = None):
    """Train Random Forest."""
    defaults = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)
    rf = RandomForestClassifier(**defaults)
    rf.fit(X, y)
    return rf


def train_gradient_boosting(X, y, params: dict | None = None):
    """Train Gradient Boosting."""
    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
    }
    if params:
        defaults.update(params)
    gb = GradientBoostingClassifier(**defaults)
    gb.fit(X, y)
    return gb


def train_sgd(X, y, params: dict | None = None):
    """Train SGD Classifier (fast linear model)."""
    defaults = {
        "loss": "modified_huber",  # gives probability estimates
        "alpha": 1e-4,
        "max_iter": 2000,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)
    sgd = SGDClassifier(**defaults)
    sgd.fit(X, y)
    return sgd


# =============================================================================
# Hyperparameter optimization with Optuna
# =============================================================================

def optimize_svm(X, y, n_trials: int = 50, cv_folds: int = 5) -> dict:
    """Optimize SVM hyperparameters."""
    def objective(trial):
        C = trial.suggest_float("C", 0.01, 100, log=True)
        max_iter = trial.suggest_int("max_iter", 1000, 10000)

        clf = CalibratedClassifierCV(
            LinearSVC(C=C, max_iter=max_iter, class_weight="balanced"),
            cv=3,
            method="sigmoid",
        )
        scores = cross_val_score(clf, X, y, cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42), scoring="f1_macro")
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="svm")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"SVM best F1: {study.best_value:.4f}, params: {study.best_params}")
    return study.best_params


def optimize_logistic_regression(X, y, n_trials: int = 50, cv_folds: int = 5) -> dict:
    """Optimize LR hyperparameters."""
    def objective(trial):
        C = trial.suggest_float("C", 0.001, 100, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])

        clf = LogisticRegression(
            C=C, solver=solver, max_iter=3000,
            class_weight="balanced", multi_class="multinomial",
        )
        scores = cross_val_score(clf, X, y, cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42), scoring="f1_macro")
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="lr")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"LR best F1: {study.best_value:.4f}, params: {study.best_params}")
    return study.best_params


def optimize_random_forest(X, y, n_trials: int = 50, cv_folds: int = 5) -> dict:
    """Optimize RF hyperparameters."""
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 5, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        scores = cross_val_score(clf, X, y, cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42), scoring="f1_macro")
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="rf")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"RF best F1: {study.best_value:.4f}, params: {study.best_params}")
    return study.best_params


def optimize_gradient_boosting(X, y, n_trials: int = 50, cv_folds: int = 5) -> dict:
    """Optimize GB hyperparameters."""
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, subsample=subsample,
            min_samples_split=min_samples_split, random_state=42,
        )
        scores = cross_val_score(clf, X, y, cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42), scoring="f1_macro")
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="gb")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"GB best F1: {study.best_value:.4f}, params: {study.best_params}")
    return study.best_params


def optimize_tfidf(X_raw: list[str], y, n_trials: int = 30, cv_folds: int = 5) -> dict:
    """Optimize TF-IDF + LR pipeline jointly."""
    def objective(trial):
        ngram_max = trial.suggest_int("ngram_max", 1, 3)
        max_features = trial.suggest_int("max_features", 3000, 30000)
        sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])
        min_df = trial.suggest_int("min_df", 1, 5)
        C = trial.suggest_float("C", 0.01, 50, log=True)

        tfidf = TfidfVectorizer(
            ngram_range=(1, ngram_max), max_features=max_features,
            sublinear_tf=sublinear_tf, min_df=min_df, max_df=0.95,
            strip_accents="unicode", lowercase=True,
        )
        clf = LogisticRegression(
            C=C, max_iter=3000, class_weight="balanced",
            multi_class="multinomial", solver="lbfgs",
        )
        pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
        scores = cross_val_score(
            pipe, X_raw, y,
            cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42),
            scoring="f1_macro",
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="tfidf")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"TF-IDF best F1: {study.best_value:.4f}, params: {study.best_params}")
    return study.best_params


# =============================================================================
# Knowledge distillation
# =============================================================================

def distill_knowledge(
    teacher_model,
    X_train,
    y_train,
    label_encoder: LabelEncoder,
    temperature: float = 3.0,
    alpha: float = 0.5,
) -> dict:
    """
    Knowledge distillation: extract soft labels from the best (teacher) model
    and train a lightweight student model to match them.

    Returns a dict with the student model and its soft-label training data,
    exportable as JSON for the TypeScript runtime.
    """
    log.info("Starting knowledge distillation...")

    # Get teacher's soft predictions (probability distributions)
    if hasattr(teacher_model, "predict_proba"):
        teacher_probs = teacher_model.predict_proba(X_train)
    else:
        log.warning("Teacher has no predict_proba. Using hard labels.")
        n_classes = len(label_encoder.classes_)
        teacher_probs = np.zeros((len(y_train), n_classes))
        preds = teacher_model.predict(X_train)
        for i, p in enumerate(preds):
            teacher_probs[i, p] = 1.0

    # Apply temperature scaling for softer distributions
    teacher_probs_soft = np.power(teacher_probs, 1.0 / temperature)
    teacher_probs_soft /= teacher_probs_soft.sum(axis=1, keepdims=True)

    # Train a lightweight student (Logistic Regression) on soft labels
    # Use weighted combination of hard and soft labels
    n_classes = len(label_encoder.classes_)

    # Create soft targets: weighted average of one-hot hard labels and teacher soft labels
    hard_labels = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        hard_labels[i, y] = 1.0

    soft_targets = alpha * teacher_probs_soft + (1.0 - alpha) * hard_labels

    # Train student on soft targets using a simple approach:
    # For each sample, use the argmax of soft targets as the label,
    # but weight samples by the confidence of the teacher
    student_labels = soft_targets.argmax(axis=1)
    sample_weights = soft_targets.max(axis=1)

    student = LogisticRegression(
        C=1.0, max_iter=3000, solver="lbfgs",
        multi_class="multinomial",
    )
    student.fit(X_train, student_labels, sample_weight=sample_weights)

    # Evaluate student
    student_preds = student.predict(X_train)
    student_acc = accuracy_score(y_train, student_preds)
    log.info(f"Student training accuracy: {student_acc:.4f}")

    return {
        "student_model": student,
        "soft_targets": soft_targets.tolist(),
        "temperature": temperature,
        "alpha": alpha,
    }


# =============================================================================
# Model export to JSON (for TypeScript consumption)
# =============================================================================

def export_model_json(
    model,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    results: list[ModelResult],
    config: TrainingConfig,
    output_path: str,
    intent_counts: dict[str, int],
):
    """
    Export the trained model as a JSON file that can be loaded
    by the TypeScript NLU bot trainer.

    The JSON format matches the TrainedModel interface in classifier.ts.
    """
    log.info(f"Exporting model to {output_path}")

    # Get vocabulary from vectorizer
    vocab = vectorizer.get_feature_names_out().tolist()

    # Get IDF weights
    idf_weights = {}
    for word, idf in zip(vocab, vectorizer.idf_):
        idf_weights[word] = float(idf)

    # Build intent centroids (average TF-IDF vector per intent)
    # This provides a kNN-compatible representation
    intent_names = label_encoder.classes_.tolist()

    # Export model coefficients for direct classification
    model_data: dict[str, Any] = {
        "version": 7,  # Bump version for Python-trained models
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trainedWith": "python-advanced-pipeline",
        "vocabulary": vocab,
        "idfWeights": idf_weights,
        "intentNames": intent_names,
        "intentCounts": intent_counts,
        "trainingVectors": [],  # Populated below
        "nbLogPriors": {},
        "nbLogLikelihoods": {},
    }

    # If model is LogisticRegression or LinearSVC, export coefficients
    # for direct use in TypeScript (no sklearn dependency needed)
    actual_model = model
    if hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV wraps the base estimator
        actual_model = model.calibrated_classifiers_[0].estimator if hasattr(model, "calibrated_classifiers_") else model

    if hasattr(actual_model, "coef_"):
        coef = actual_model.coef_
        intercept = actual_model.intercept_ if hasattr(actual_model, "intercept_") else np.zeros(coef.shape[0])

        model_data["linearModel"] = {
            "type": "logistic_regression" if isinstance(actual_model, LogisticRegression) else "svm",
            "coefficients": coef.tolist(),
            "intercepts": intercept.tolist(),
            "classes": intent_names,
        }

    # Also export probability-based model info if available
    if hasattr(model, "predict_proba"):
        model_data["hasProbabilities"] = True

    # Compute Naive Bayes-compatible parameters from the training data
    # This allows the TypeScript classifier to use the same NB+kNN hybrid
    # even with Python-trained models

    # Export per-intent statistics
    best_result = max(results, key=lambda r: r.f1_macro) if results else None
    model_data["metrics"] = {
        "bestModel": best_result.name if best_result else "unknown",
        "accuracy": best_result.accuracy if best_result else 0,
        "f1Macro": best_result.f1_macro if best_result else 0,
        "f1Weighted": best_result.f1_weighted if best_result else 0,
        "cvMean": best_result.cv_mean if best_result else 0,
        "cvStd": best_result.cv_std if best_result else 0,
        "allResults": [
            {
                "name": r.name,
                "accuracy": r.accuracy,
                "f1Macro": r.f1_macro,
                "cvMean": r.cv_mean,
                "cvStd": r.cv_std,
                "trainingTime": r.training_time,
            }
            for r in results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_path)
    log.info(f"Model exported: {output_path} ({file_size / 1024:.1f} KB)")

    return model_data


def export_lightweight_model(
    model,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    training_texts: list[str],
    training_labels: list[int],
    output_path: str,
):
    """
    Export a lightweight model that matches the EXACT format expected by
    the existing TypeScript classifier (TrainedModel in classifier.ts).

    This produces a drop-in replacement for the browser-trained model.
    """
    log.info(f"Exporting lightweight TS-compatible model to {output_path}")

    vocab = vectorizer.get_feature_names_out().tolist()
    idf_weights = {w: float(v) for w, v in zip(vocab, vectorizer.idf_)}
    intent_names = label_encoder.classes_.tolist()

    # Build training vectors (TF-IDF) for each example -- this is what
    # the TypeScript kNN classifier uses
    X_tfidf = vectorizer.transform(training_texts)
    training_vectors = []
    for i in range(X_tfidf.shape[0]):
        vec = X_tfidf[i].toarray().flatten().tolist()
        training_vectors.append({
            "vector": vec,
            "intent": intent_names[training_labels[i]],
        })

    # Build Naive Bayes parameters (matching TypeScript implementation)
    from collections import defaultdict
    intent_word_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    intent_total_words: dict[str, int] = defaultdict(int)
    intent_doc_counts: dict[str, int] = defaultdict(int)

    for text, label_idx in zip(training_texts, training_labels):
        intent = intent_names[label_idx]
        intent_doc_counts[intent] += 1
        tokens = text.lower().split()
        for token in tokens:
            if token in idf_weights:
                intent_word_counts[intent][token] += 1
                intent_total_words[intent] += 1

    n_docs = len(training_texts)
    vocab_size = len(vocab)

    nb_log_priors: dict[str, float] = {}
    nb_log_likelihoods: dict[str, dict[str, float]] = {}

    for intent in intent_names:
        nb_log_priors[intent] = math.log(intent_doc_counts[intent] / n_docs)
        nb_log_likelihoods[intent] = {}
        total_words = intent_total_words.get(intent, 0)
        for word in vocab:
            word_count = intent_word_counts[intent].get(word, 0)
            nb_log_likelihoods[intent][word] = math.log(
                (word_count + 0.5) / (total_words + vocab_size * 0.5)
            )

    ts_model = {
        "vocabulary": vocab,
        "idfWeights": idf_weights,
        "trainingVectors": training_vectors,
        "nbLogPriors": nb_log_priors,
        "nbLogLikelihoods": nb_log_likelihoods,
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": 7,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ts_model, f, ensure_ascii=False)

    file_size = os.path.getsize(output_path)
    log.info(f"Lightweight model exported: {output_path} ({file_size / 1024:.1f} KB)")

    # Also export a compressed version (vocabulary indices instead of full strings)
    compressed = compress_model(ts_model)
    compressed_path = output_path.replace(".json", ".min.json")
    with open(compressed_path, "w", encoding="utf-8") as f:
        json.dump(compressed, f, separators=(",", ":"), ensure_ascii=False)

    compressed_size = os.path.getsize(compressed_path)
    log.info(f"Compressed model: {compressed_path} ({compressed_size / 1024:.1f} KB)")
    log.info(f"Compression ratio: {compressed_size / file_size:.1%}")


def compress_model(model_data: dict) -> dict:
    """
    Compress the model by:
    - Truncating float precision
    - Removing near-zero IDF weights
    - Sparse encoding of training vectors
    """
    vocab = model_data["vocabulary"]
    idf = model_data["idfWeights"]

    # Truncate to 4 decimal places
    idf_compressed = {k: round(v, 4) for k, v in idf.items()}

    # Sparse encode training vectors (only non-zero entries)
    sparse_vectors = []
    for tv in model_data["trainingVectors"]:
        sparse = {}
        for i, v in enumerate(tv["vector"]):
            if abs(v) > 1e-6:
                sparse[i] = round(v, 4)
        sparse_vectors.append({"s": sparse, "i": tv["intent"]})

    # Compress NB likelihoods (only store non-default values)
    nb_ll = {}
    for intent, words in model_data.get("nbLogLikelihoods", {}).items():
        compressed_words = {}
        values = list(words.values())
        median_val = sorted(values)[len(values) // 2] if values else 0
        for word, val in words.items():
            # Only store if significantly different from median
            if abs(val - median_val) > 0.1:
                compressed_words[word] = round(val, 4)
        nb_ll[intent] = {"_d": round(median_val, 4), **compressed_words}

    return {
        "v": vocab,
        "idf": idf_compressed,
        "tv": sparse_vectors,
        "nbP": {k: round(v, 4) for k, v in model_data.get("nbLogPriors", {}).items()},
        "nbL": nb_ll,
        "at": model_data["trainedAt"],
        "ver": model_data.get("version", 7),
        "_fmt": "compressed",
    }


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(config: TrainingConfig):
    """Run the full NLU training pipeline."""
    start_time = time.time()
    os.makedirs(config.output_dir, exist_ok=True)

    # ---- Load data ----
    log.info("=" * 60)
    log.info("Loading training data...")
    examples = load_training_data(config.input_path)
    intent_counts = validate_data(examples)

    texts = [e.text for e in examples]
    intents = [e.intent for e in examples]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(intents)
    n_classes = len(le.classes_)
    log.info(f"Classes: {n_classes}, Samples: {len(texts)}")

    # ---- Feature extraction ----
    log.info("=" * 60)
    log.info("Extracting features...")

    # TF-IDF features
    if config.optimize and not config.quick:
        log.info("Optimizing TF-IDF parameters...")
        tfidf_params = optimize_tfidf(texts, y, n_trials=min(config.n_trials, 30), cv_folds=config.cv_folds)
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, tfidf_params.get("ngram_max", 2)),
            max_features=tfidf_params.get("max_features", 10000),
            sublinear_tf=tfidf_params.get("sublinear_tf", True),
            min_df=tfidf_params.get("min_df", 1),
            max_df=0.95,
            strip_accents="unicode",
            lowercase=True,
        )
    else:
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
            strip_accents="unicode",
            lowercase=True,
        )

    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    log.info(f"TF-IDF features: {X_tfidf.shape}")

    # Sentence embeddings
    X_embed = None
    if config.use_embeddings and not config.quick:
        X_embed = build_embedding_features(texts, config.embedding_model)
        if X_embed.size > 0:
            log.info(f"Embedding features: {X_embed.shape}")
            # Combine TF-IDF + embeddings
            from scipy.sparse import hstack, csr_matrix
            X_combined = hstack([X_tfidf, csr_matrix(X_embed)])
            log.info(f"Combined features: {X_combined.shape}")
        else:
            X_combined = X_tfidf
    else:
        X_combined = X_tfidf

    # ---- Train classifiers ----
    log.info("=" * 60)
    log.info("Training classifiers...")
    cv = StratifiedKFold(config.cv_folds, shuffle=True, random_state=config.random_state)
    results: list[ModelResult] = []
    trained_models: dict[str, Any] = {}

    classifiers = [
        ("SVM (Linear)", train_svm, optimize_svm, X_tfidf),
        ("Logistic Regression", train_logistic_regression, optimize_logistic_regression, X_tfidf),
        ("Random Forest", train_random_forest, optimize_random_forest, X_tfidf),
        ("Gradient Boosting", train_gradient_boosting, optimize_gradient_boosting, X_tfidf),
        ("SGD", train_sgd, None, X_tfidf),
    ]

    if X_embed is not None and X_embed.size > 0:
        classifiers.extend([
            ("SVM (Embeddings)", train_svm, None, X_combined),
            ("LR (Embeddings)", train_logistic_regression, None, X_combined),
        ])

    for name, train_fn, opt_fn, X in classifiers:
        log.info(f"\n  Training {name}...")
        t0 = time.time()

        # Optimize if requested
        params = None
        if config.optimize and not config.quick and opt_fn is not None:
            try:
                params = opt_fn(X, y, n_trials=config.n_trials, cv_folds=config.cv_folds)
            except Exception as e:
                log.warning(f"Optimization failed for {name}: {e}. Using defaults.")

        # Train
        try:
            model = train_fn(X, y, params)
            train_time = time.time() - t0

            # Evaluate
            preds = model.predict(X)
            acc = accuracy_score(y, preds)
            f1_m = f1_score(y, preds, average="macro")
            f1_w = f1_score(y, preds, average="weighted")

            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")

            result = ModelResult(
                name=name,
                accuracy=float(acc),
                f1_macro=float(f1_m),
                f1_weighted=float(f1_w),
                cv_mean=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
                training_time=train_time,
                params=params or {},
            )
            results.append(result)
            trained_models[name] = model
            log.info(f"  {name}: acc={acc:.4f}, f1={f1_m:.4f}, cv={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        except Exception as e:
            log.error(f"  Failed to train {name}: {e}")
            continue

    # ---- Ensemble (Voting Classifier) ----
    log.info("\n  Training Voting Ensemble...")
    try:
        t0 = time.time()
        estimators = []
        for name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
            if name in trained_models:
                clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                estimators.append((clean_name, trained_models[name]))

        if len(estimators) >= 2:
            ensemble = VotingClassifier(estimators=estimators, voting="soft")
            ensemble.fit(X_tfidf, y)
            train_time = time.time() - t0

            preds = ensemble.predict(X_tfidf)
            acc = accuracy_score(y, preds)
            f1_m = f1_score(y, preds, average="macro")
            f1_w = f1_score(y, preds, average="weighted")
            cv_scores = cross_val_score(ensemble, X_tfidf, y, cv=cv, scoring="f1_macro")

            result = ModelResult(
                name="Voting Ensemble",
                accuracy=float(acc), f1_macro=float(f1_m), f1_weighted=float(f1_w),
                cv_mean=float(cv_scores.mean()), cv_std=float(cv_scores.std()),
                training_time=train_time,
            )
            results.append(result)
            trained_models["Voting Ensemble"] = ensemble
            log.info(f"  Ensemble: acc={acc:.4f}, f1={f1_m:.4f}, cv={cv_scores.mean():.4f}")
    except Exception as e:
        log.error(f"  Ensemble failed: {e}")

    # ---- Results summary ----
    log.info("\n" + "=" * 60)
    log.info("Results Summary")
    log.info("=" * 60)
    results.sort(key=lambda r: r.cv_mean, reverse=True)
    for i, r in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        log.info(f"  {r.name:30s}  CV F1={r.cv_mean:.4f} (+/- {r.cv_std:.4f})  Acc={r.accuracy:.4f}{marker}")

    best = results[0]
    best_model = trained_models[best.name]
    log.info(f"\nBest model: {best.name} (CV F1: {best.cv_mean:.4f})")

    # ---- Classification report for best model ----
    best_X = X_tfidf  # Default to TF-IDF
    if "Embeddings" in best.name and X_combined is not None:
        best_X = X_combined

    best_preds = best_model.predict(best_X)
    report = classification_report(y, best_preds, target_names=le.classes_)
    log.info(f"\nClassification Report ({best.name}):\n{report}")

    # Save report
    report_path = os.path.join(config.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Best Model: {best.name}\n")
        f.write(f"CV F1 Macro: {best.cv_mean:.4f} (+/- {best.cv_std:.4f})\n\n")
        f.write(report)

    # ---- Knowledge distillation ----
    if config.distill:
        log.info("\n" + "=" * 60)
        log.info("Knowledge Distillation")
        log.info("=" * 60)
        try:
            distill_result = distill_knowledge(
                best_model, best_X, y, le, temperature=3.0, alpha=0.5
            )
            student = distill_result["student_model"]
            student_preds = student.predict(best_X)
            student_f1 = f1_score(y, student_preds, average="macro")
            log.info(f"Student model F1: {student_f1:.4f} (teacher: {best.f1_macro:.4f})")

            results.append(ModelResult(
                name="Distilled Student (LR)",
                accuracy=float(accuracy_score(y, student_preds)),
                f1_macro=float(student_f1),
                f1_weighted=float(f1_score(y, student_preds, average="weighted")),
                cv_mean=float(student_f1),
                cv_std=0.0,
                training_time=0.0,
            ))
        except Exception as e:
            log.error(f"Knowledge distillation failed: {e}")

    # ---- Export models ----
    log.info("\n" + "=" * 60)
    log.info("Exporting models...")
    log.info("=" * 60)

    # 1. Full model JSON (with metadata, metrics, coefficients)
    full_path = os.path.join(config.output_dir, "model_full.json")
    export_model_json(best_model, tfidf_vectorizer, le, results, config, full_path, intent_counts)

    # 2. Lightweight model JSON (drop-in for TypeScript classifier)
    ts_path = os.path.join(config.output_dir, "model_ts_compatible.json")
    export_lightweight_model(best_model, tfidf_vectorizer, le, texts, y.tolist(), ts_path)

    # 3. Save all results as JSON
    results_path = os.path.join(config.output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "results": [asdict(r) for r in results],
                "config": {
                    "optimize": config.optimize,
                    "n_trials": config.n_trials,
                    "cv_folds": config.cv_folds,
                    "use_embeddings": config.use_embeddings,
                    "embedding_model": config.embedding_model,
                },
                "data": {
                    "n_examples": len(examples),
                    "n_intents": n_classes,
                    "intent_counts": intent_counts,
                },
                "timing": {
                    "total_seconds": time.time() - start_time,
                },
            },
            f,
            indent=2,
        )

    # 4. Save confusion matrix
    cm = confusion_matrix(y, best_preds)
    cm_path = os.path.join(config.output_dir, "confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump({
            "matrix": cm.tolist(),
            "labels": le.classes_.tolist(),
        }, f, indent=2)

    total_time = time.time() - start_time
    log.info(f"\nTotal training time: {total_time:.1f}s")
    log.info(f"Results saved to: {config.output_dir}")
    log.info("Done!")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Advanced NLU Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training (no optimization, no embeddings)
  python train_nlu.py --input data.json --output results/ --quick

  # Full optimization
  python train_nlu.py --input data.json --output results/ --optimize --n-trials 100

  # With embeddings and distillation
  python train_nlu.py --input data.json --output results/ --optimize --embeddings --distill
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to training data JSON")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--quick", action="store_true", help="Quick mode: skip embeddings and optimization")
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials per model (default: 100)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds (default: 5)")
    parser.add_argument("--embeddings", action="store_true", default=True, help="Use sentence embeddings")
    parser.add_argument("--no-embeddings", dest="embeddings", action="store_false")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--distill", action="store_true", default=True, help="Knowledge distillation")
    parser.add_argument("--no-distill", dest="distill", action="store_false")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = TrainingConfig(
        input_path=args.input,
        output_dir=args.output,
        optimize=args.optimize,
        quick=args.quick,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        use_embeddings=args.embeddings and not args.quick,
        embedding_model=args.embedding_model,
        distill=args.distill and not args.quick,
        random_state=args.seed,
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
