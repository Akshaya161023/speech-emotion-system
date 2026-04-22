"""
train_ml_models.py
==================
Train scikit-learn emotion classifiers:
  1. Random Forest Classifier
  2. Support Vector Machine (SVM)

Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
         https://zenodo.org/record/1188976

Run this script AFTER downloading the RAVDESS dataset to data/RAVDESS/
Usage:
    python training/train_ml_models.py --data_dir data/RAVDESS
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_extraction import extract_features
from src.emotion_model import RAVDESS_EMOTIONS, EMOTION_LABELS

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─── 1. DATA LOADING ──────────────────────────────────────────────────────────

def load_ravdess_dataset(data_dir: str, target_emotions: list = None) -> tuple:
    """
    Walk through RAVDESS directory structure, extract features & labels.

    RAVDESS folder structure:
        data/RAVDESS/
            Actor_01/
                03-01-05-01-01-01-01.wav   ← emotion code = 05 = angry
                ...
            Actor_02/
                ...

    Args:
        data_dir        : Root directory containing Actor_XX folders
        target_emotions : List of emotion labels to keep (None = use all)

    Returns:
        (X_features array, y_labels list)
    """
    data_path  = Path(data_dir)
    audio_files = list(data_path.rglob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(
            f"No .wav files found in '{data_dir}'.\n"
            "Please download RAVDESS from: https://zenodo.org/record/1188976\n"
            "Extract to data/RAVDESS/ maintaining the Actor_XX folder structure."
        )

    print(f"\n📁 Found {len(audio_files)} audio files in {data_dir}")
    print("─" * 60)

    X, y = [], []
    skipped = 0

    for audio_file in tqdm(audio_files, desc="Extracting features"):
        # Parse filename to get emotion
        parts = audio_file.stem.split("-")
        if len(parts) != 7:
            skipped += 1
            continue

        emotion_code  = parts[2]
        emotion_label = RAVDESS_EMOTIONS.get(emotion_code)
        if emotion_label is None:
            skipped += 1
            continue

        # Filter emotions if target list specified
        if target_emotions and emotion_label not in target_emotions:
            continue

        try:
            features = extract_features(str(audio_file))
            X.append(features)
            y.append(emotion_label)
        except Exception as e:
            print(f"  ⚠️  Skipped {audio_file.name}: {e}")
            skipped += 1

    print(f"\n✅ Loaded {len(X)} samples | Skipped {skipped} files")
    if not X:
        raise RuntimeError("No features extracted. Check data directory and file naming.")

    return np.array(X), np.array(y)


def load_tess_dataset(data_dir: str) -> tuple:
    """
    Walk through TESS directory structure, extract features & labels.

    TESS folder structure:
        data/TESS/
            OAF_angry/  ← folder name encodes emotion
                OAF_back_angry.wav
                ...
            OAF_happy/
                ...

    Args:
        data_dir : Root directory containing TESS emotion folders

    Returns:
        (X_features array, y_labels list)
    """
    data_path   = Path(data_dir)
    audio_files = list(data_path.rglob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(
            f"No .wav files found in '{data_dir}'.\n"
            "Please download TESS from: https://tspace.library.utoronto.ca/handle/1807/24487\n"
            "Extract to data/TESS/ maintaining the emotion-labelled folder structure."
        )

    print(f"\n📁 Found {len(audio_files)} TESS audio files in {data_dir}")

    TESS_EMOTION_MAP = {
        "angry"   : "angry",
        "disgust" : "disgust",
        "fear"    : "fearful",
        "happy"   : "happy",
        "neutral" : "neutral",
        "sad"     : "sad",
        "ps"      : "surprised",   # pleasant surprise
    }

    X, y = [], []
    for audio_file in tqdm(audio_files, desc="Extracting TESS features"):
        # Emotion is encoded in the parent folder name, e.g. OAF_angry
        folder_name   = audio_file.parent.name.lower()
        emotion_label = None
        for key, val in TESS_EMOTION_MAP.items():
            if key in folder_name:
                emotion_label = val
                break

        if emotion_label is None:
            continue

        try:
            features = extract_features(str(audio_file))
            X.append(features)
            y.append(emotion_label)
        except Exception as e:
            pass

    print(f"✅ Loaded {len(X)} TESS samples")
    return np.array(X), np.array(y)


# ─── 2. PREPROCESSING ─────────────────────────────────────────────────────────

def preprocess_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Scale features and encode labels.

    Returns:
        (X_scaled, y_encoded, scaler, label_encoder)
    """
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\n🏷️  Emotion classes: {list(le.classes_)}")

    # Feature scaling (StandardScaler → mean=0, std=1)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, scaler, le


# ─── 3. MODEL TRAINING ────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, n_estimators: int = 200, verbose: bool = True) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train      : Training features
        y_train      : Encoded training labels
        n_estimators : Number of trees
        verbose      : Print training progress

    Returns:
        Trained RandomForestClassifier
    """
    if verbose:
        print("\n🌲 Training Random Forest Classifier...")

    rf = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth    = None,          # Grow full trees
        min_samples_split = 4,
        min_samples_leaf  = 2,
        max_features = "sqrt",        # Recommended for classification
        class_weight = "balanced",    # Handle class imbalance
        random_state = 42,
        n_jobs       = -1,            # Use all CPU cores
        verbose      = 0,
    )
    rf.fit(X_train, y_train)

    if verbose:
        print("  ✅ Random Forest trained!")

    return rf


def train_svm(X_train, y_train, verbose: bool = True) -> SVC:
    """
    Train a Support Vector Machine (SVM) classifier with RBF kernel.

    Args:
        X_train : Training features
        y_train : Encoded training labels
        verbose : Print progress

    Returns:
        Trained SVC
    """
    if verbose:
        print("\n🔷 Training SVM Classifier...")

    svm = SVC(
        kernel       = "rbf",
        C            = 10,
        gamma        = "scale",
        probability  = True,          # Enable predict_proba
        class_weight = "balanced",
        random_state = 42,
        max_iter     = 5000,
    )
    svm.fit(X_train, y_train)

    if verbose:
        print("  ✅ SVM trained!")

    return svm


# ─── 4. EVALUATION ────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, label_encoder, model_name: str) -> dict:
    """
    Evaluate a trained classifier and print/return metrics.

    Returns:
        Dict with accuracy, precision, recall, f1
    """
    y_pred   = model.predict(X_test)
    class_names = label_encoder.classes_

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'═'*55}")
    print(f"  📊 {model_name} Evaluation")
    print(f"{'═'*55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'─'*55}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, model_name)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(cm: np.ndarray, class_names: list, model_name: str):
    """Save a confusion matrix heatmap as PNG."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, cbar=True,
    )
    plt.title(f"Confusion Matrix – {model_name}", fontsize=14, pad=15)
    plt.ylabel("True Label",      fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    save_path = MODELS_DIR / f"confusion_matrix_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Confusion matrix saved: {save_path}")


# ─── 5. CROSS-VALIDATION ──────────────────────────────────────────────────────

def cross_validate_model(model, X, y, cv: int = 5, model_name: str = "Model") -> float:
    """
    Run k-fold cross-validation and report mean accuracy.

    Returns:
        Mean CV accuracy
    """
    print(f"\n🔁 {cv}-Fold Cross-Validation for {model_name}...")
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"   Folds: {[f'{s:.3f}' for s in scores]}")
    print(f"   Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


# ─── 6. SAVE ARTIFACTS ────────────────────────────────────────────────────────

def save_model(model, path: str):
    """Serialize a scikit-learn model with joblib."""
    joblib.dump(model, path, compress=3)
    print(f"  💾 Model saved: {path}")


def save_preprocessing(scaler, label_encoder):
    """Save the feature scaler and label encoder."""
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl", compress=3)
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl", compress=3)
    print(f"  💾 Scaler & encoder saved to {MODELS_DIR}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train emotion detection ML models")
    parser.add_argument("--data_dir",   default="data/RAVDESS", help="Path to RAVDESS dataset root")
    parser.add_argument("--tess_dir",   default=None,            help="Optional: Path to TESS dataset root")
    parser.add_argument("--emotions",   nargs="+",
                        default=["angry", "happy", "neutral", "sad"],
                        help="Emotion classes to train on")
    parser.add_argument("--test_size",  default=0.2, type=float, help="Test split ratio")
    parser.add_argument("--cv",         default=5,   type=int,   help="Cross-validation folds")
    parser.add_argument("--skip_cv",    action="store_true",     help="Skip cross-validation (faster)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║    Speech Emotion Detection — ML Model Training      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n  Target emotions : {args.emotions}")
    print(f"  Data directory  : {args.data_dir}")
    print(f"  Test split      : {args.test_size:.0%}")

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y = load_ravdess_dataset(args.data_dir, target_emotions=args.emotions)

    # Optionally merge TESS data
    if args.tess_dir and Path(args.tess_dir).exists():
        X_tess, y_tess = load_tess_dataset(args.tess_dir)
        X = np.vstack([X, X_tess])
        y = np.concatenate([y, y_tess])
        print(f"\n📊 Combined dataset: {len(X)} samples")

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_scaled, y_encoded, scaler, le = preprocess_data(X, y)
    save_preprocessing(scaler, le)

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size    = args.test_size,
        random_state = 42,
        stratify     = y_encoded,
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)} samples")

    results = {}

    # ── Train Random Forest ───────────────────────────────────────────────────
    rf = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, le, "Random Forest")
    results["Random Forest"] = rf_metrics
    save_model(rf, MODELS_DIR / "rf_model.pkl")

    if not args.skip_cv:
        cross_validate_model(rf, X_scaled, y_encoded, cv=args.cv, model_name="Random Forest")

    # ── Train SVM ─────────────────────────────────────────────────────────────
    svm = train_svm(X_train, y_train)
    svm_metrics = evaluate_model(svm, X_test, y_test, le, "SVM")
    results["SVM"] = svm_metrics
    save_model(svm, MODELS_DIR / "svm_model.pkl")

    if not args.skip_cv:
        cross_validate_model(svm, X_scaled, y_encoded, cv=args.cv, model_name="SVM")

    # ── Comparison Table ──────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║            MODEL COMPARISON SUMMARY                  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
    for name, m in results.items():
        print(f"  {name:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
    print("╚══════════════════════════════════════════════════════╝")

    winner = max(results, key=lambda k: results[k]["f1"])
    print(f"\n🏆 Best model: {winner}  (F1 = {results[winner]['f1']:.4f})")
    print("\n✅ Training complete! Models saved to ./models/")


if __name__ == "__main__":
    main()
