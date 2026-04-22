"""
train_demo_model.py
===================
Quick demo trainer — creates synthetic emotion data and trains a Random Forest
so the Streamlit app works immediately WITHOUT needing the RAVDESS dataset.

Run this first to get the app working:
    python training/train_demo_model.py

Then replace with real models by running:
    python training/train_ml_models.py --data_dir data/RAVDESS
"""

import sys
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]
N_FEATURES     = 95       # Must match feature_extraction.py output
N_SAMPLES      = 1200     # Samples per class
RANDOM_STATE   = 42


def generate_synthetic_data(n_samples_per_class: int = N_SAMPLES) -> tuple:
    """
    Generate synthetic emotion feature vectors with class-specific distributions.
    Each emotion has a different mean vector to make them separable.

    Feature intuition:
      - MFCC 1-40  : spectral envelope
      - Delta 1-40 : temporal dynamics
      - Chroma 1-12: harmonic content
      - ZCR, RMS, Centroid, Rolloff: energy/brightness scalars
    """
    np.random.seed(RANDOM_STATE)

    # ── Emotion-specific feature means (hand-crafted for realism) ─────────────
    # 40 MFCC + 40 MFCC-delta + 12 Chroma + 3 scalars (ZCR, RMS, Centroid) = 95
    emotion_profiles = {
        "angry"  : np.concatenate([
            np.linspace(-5, 10, 40),   # High-energy MFCCs
            np.linspace( 2,  8, 40),   # Strong deltas (fast changes)
            np.full(12, 0.6),           # Mid chroma
            [0.15, 0.08, 4500],         # ZCR, RMS, Centroid
        ]),
        "happy"  : np.concatenate([
            np.linspace( 2,  8, 40),   # Positive MFCCs
            np.linspace( 1,  5, 40),   # Moderate deltas
            np.full(12, 0.7),           # High chroma (melodic)
            [0.10, 0.06, 4000],
        ]),
        "neutral": np.concatenate([
            np.linspace(-2,  2, 40),   # Flat MFCCs
            np.linspace( 0,  1, 40),   # Low deltas
            np.full(12, 0.4),           # Low chroma
            [0.06, 0.03, 2500],         # Low ZCR, low RMS, dull centroid
        ]),
        "sad"    : np.concatenate([
            np.linspace(-8, -2, 40),   # Negative MFCCs (low energy)
            np.linspace(-3,  1, 40),   # Negative deltas (slow, falling)
            np.full(12, 0.3),           # Low chroma
            [0.04, 0.02, 1800],         # Very low energy/brightness
        ]),
    }

    X_all, y_all = [], []
    for emotion, mean_vec in emotion_profiles.items():
        feat_len  = len(mean_vec)   # should be N_FEATURES (95)
        noise_std = 1.5 + (0.5 * np.random.rand(feat_len))
        samples   = np.random.normal(
            loc   = mean_vec,
            scale = noise_std,
            size  = (n_samples_per_class, feat_len),
        )
        X_all.append(samples)
        y_all.extend([emotion] * n_samples_per_class)

    X = np.vstack(X_all)
    y = np.array(y_all)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def main():
    print("=" * 56)
    print("   Speech Emotion Detection -- Demo Model Trainer")
    print("   (Uses synthetic data -- replace with RAVDESS later)")
    print("=" * 56)

    # ── Generate synthetic data ───────────────────────────────────────────────
    print(f"\nGenerating {N_SAMPLES * len(EMOTION_LABELS):,} synthetic samples...")
    X, y = generate_synthetic_data(N_SAMPLES)
    print(f"   X shape: {X.shape} | Classes: {EMOTION_LABELS}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(EMOTION_LABELS)
    y_enc = le.transform(y)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # ── Train / test split ────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_enc,
        test_size    = 0.2,
        random_state = RANDOM_STATE,
        stratify     = y_enc,
    )
    print(f"   Train: {len(X_tr):,} | Test: {len(X_te):,} samples")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n[RF] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators = 300,
        max_depth    = None,
        class_weight = "balanced",
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(X_te))
    print(f"   RF Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print(classification_report(y_te, rf.predict(X_te), target_names=le.classes_, zero_division=0))

    # ── SVM ───────────────────────────────────────────────────────────────────
    print("[SVM] Training SVM...")
    svm = SVC(
        kernel       = "rbf",
        C            = 10,
        gamma        = "scale",
        probability  = True,
        class_weight = "balanced",
        random_state = RANDOM_STATE,
    )
    svm.fit(X_tr, y_tr)
    svm_acc = accuracy_score(y_te, svm.predict(X_te))
    print(f"   SVM Accuracy: {svm_acc:.4f} ({svm_acc*100:.2f}%)")

    # ── Save everything ───────────────────────────────────────────────────────
    print("\nSaving models...")
    joblib.dump(rf,     MODELS_DIR / "rf_model.pkl",       compress=3)
    joblib.dump(svm,    MODELS_DIR / "svm_model.pkl",      compress=3)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl",         compress=3)
    joblib.dump(le,     MODELS_DIR / "label_encoder.pkl",  compress=3)

    print("  [OK] rf_model.pkl       saved")
    print("  [OK] svm_model.pkl      saved")
    print("  [OK] scaler.pkl         saved")
    print("  [OK] label_encoder.pkl  saved")

    print("\n" + "=" * 56)
    print("  Demo models ready! Launch the app with:")
    print("     streamlit run app.py")
    print("=" * 56)


if __name__ == "__main__":
    main()
