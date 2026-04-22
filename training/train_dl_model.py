"""
train_dl_model.py
=================
Train deep learning emotion classifiers:
  1. 1-D Convolutional Neural Network (CNN)
  2. LSTM (Long Short-Term Memory) Network

Uses TensorFlow/Keras (free, open-source).

Usage:
    python training/train_dl_model.py --data_dir data/RAVDESS --model cnn
    python training/train_dl_model.py --data_dir data/RAVDESS --model lstm
"""

import os
import sys
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_extraction import extract_features_dl
from src.emotion_model import RAVDESS_EMOTIONS
from training.train_ml_models import load_ravdess_dataset, save_preprocessing

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)


# ─── 1. DATA LOADING FOR DL ───────────────────────────────────────────────────

def load_dl_dataset(data_dir: str, target_emotions: list = None) -> tuple:
    """
    Similar to RAVDESS loader, but uses the extended DL feature set (223-d).

    Returns:
        (X array [n_samples, 223], y list)
    """
    data_path   = Path(data_dir)
    audio_files = list(data_path.rglob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in '{data_dir}'.")

    print(f"\n📁 Found {len(audio_files)} audio files")

    X, y = [], []
    for audio_file in tqdm(audio_files, desc="Extracting DL features"):
        parts = audio_file.stem.split("-")
        if len(parts) != 7:
            continue
        emotion_code  = parts[2]
        emotion_label = RAVDESS_EMOTIONS.get(emotion_code)
        if emotion_label is None:
            continue
        if target_emotions and emotion_label not in target_emotions:
            continue
        try:
            features = extract_features_dl(str(audio_file))
            X.append(features)
            y.append(emotion_label)
        except Exception:
            pass

    return np.array(X), np.array(y)


# ─── 2. MODEL ARCHITECTURES ───────────────────────────────────────────────────

def build_cnn_model(input_dim: int, n_classes: int) -> Model:
    """
    Build a 1-D CNN model for emotion classification.

    Architecture:
      Input (223,1) → [Conv1D → BN → ReLU → Pool] x3 → Flatten → Dense → Dropout → Output (softmax)

    Args:
        input_dim : Feature vector length (223 for DL features)
        n_classes : Number of emotion classes

    Returns:
        Compiled Keras Model
    """
    inputs = keras.Input(shape=(input_dim, 1), name="audio_features")

    # ── Block 1 ───────────────────────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # ── Block 2 ───────────────────────────────────────────────────────────────
    x = layers.Conv1D(128, kernel_size=5, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # ── Block 3 ───────────────────────────────────────────────────────────────
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    # ── Dense Head ────────────────────────────────────────────────────────────
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="CNN_Emotion_Classifier")
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


def build_lstm_model(input_dim: int, n_classes: int) -> Model:
    """
    Build a Bidirectional LSTM model for emotion classification.

    Architecture:
      Input (223,1) → BiLSTM → BiLSTM → Dense → Dropout → Output (softmax)

    Args:
        input_dim : Feature vector length
        n_classes : Number of emotion classes

    Returns:
        Compiled Keras Model
    """
    inputs = keras.Input(shape=(input_dim, 1), name="audio_features")

    # ── Bidirectional LSTM layers ─────────────────────────────────────────────
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)

    # ── Dense head ────────────────────────────────────────────────────────────
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Emotion_Classifier")
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=5e-4),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


# ─── 3. TRAINING ──────────────────────────────────────────────────────────────

def train_model(
    model, X_train, y_train, X_val, y_val,
    epochs: int = 100, batch_size: int = 32, model_name: str = "model"
) -> keras.callbacks.History:
    """
    Train a Keras model with callbacks for early stopping and LR scheduling.

    Args:
        model      : Compiled Keras model
        X_train    : Training features [n, features, 1]
        y_train    : Training labels
        X_val      : Validation features
        y_val      : Validation labels
        epochs     : Max training epochs
        batch_size : Mini-batch size
        model_name : Used for checkpoint filename

    Returns:
        Keras History object
    """
    checkpoint_path = MODELS_DIR / f"{model_name}_best.h5"

    callbacks = [
        EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 15,
            restore_best_weights = True,
            verbose              = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 7,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        ModelCheckpoint(
            filepath   = str(checkpoint_path),
            monitor    = "val_accuracy",
            save_best_only = True,
            verbose    = 0,
        ),
    ]

    print(f"\n🧠 Training {model.name}  (max {epochs} epochs, batch={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = epochs,
        batch_size      = batch_size,
        callbacks       = callbacks,
        verbose         = 1,
    )

    print(f"\n  ✅ Training complete!")
    print(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    return history


# ─── 4. EVALUATION & PLOTTING ─────────────────────────────────────────────────

def evaluate_dl_model(model, X_test, y_test, label_encoder, model_name: str):
    """Evaluate a Keras model and print metrics."""
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    class_names  = label_encoder.classes_

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'═'*55}")
    print(f"  📊 {model_name} Evaluation")
    print(f"{'═'*55}")
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))


def plot_training_history(history, model_name: str):
    """Save loss & accuracy curves as PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} Training History", fontsize=15)

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = MODELS_DIR / f"training_history_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Training history saved: {save_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DL emotion detection models")
    parser.add_argument("--data_dir",   default="data/RAVDESS", help="Path to RAVDESS dataset")
    parser.add_argument("--model",      default="cnn",           choices=["cnn", "lstm", "both"])
    parser.add_argument("--emotions",   nargs="+",
                        default=["angry", "happy", "neutral", "sad"])
    parser.add_argument("--epochs",     default=100,  type=int)
    parser.add_argument("--batch_size", default=32,   type=int)
    parser.add_argument("--test_size",  default=0.2,  type=float)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Speech Emotion Detection — Deep Learning Training   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n  Model type      : {args.model}")
    print(f"  Target emotions : {args.emotions}")
    print(f"  Epochs          : {args.epochs}")

    # ── Load & preprocess ─────────────────────────────────────────────────────
    X, y = load_dl_dataset(args.data_dir, target_emotions=args.emotions)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"\n  Classes ({n_classes}): {list(le.classes_)}")

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Save preprocessing if ML models not already saved
    if not (MODELS_DIR / "scaler.pkl").exists():
        save_preprocessing(scaler, le)

    # Reshape for CNN/LSTM: (samples, features, 1)
    X_3d = X_sc.reshape(X_sc.shape[0], X_sc.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_3d, y_enc,
        test_size    = args.test_size,
        random_state = 42,
        stratify     = y_enc,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size    = 0.1,
        random_state = 42,
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    input_dim = X_sc.shape[1]

    # ── Train CNN ─────────────────────────────────────────────────────────────
    if args.model in ("cnn", "both"):
        cnn_model   = build_cnn_model(input_dim, n_classes)
        cnn_model.summary()
        cnn_history = train_model(
            cnn_model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, batch_size=args.batch_size, model_name="cnn"
        )
        evaluate_dl_model(cnn_model, X_test, y_test, le, "CNN")
        plot_training_history(cnn_history, "CNN")
        cnn_model.save(MODELS_DIR / "cnn_model.h5")
        print(f"  💾 CNN model saved: {MODELS_DIR / 'cnn_model.h5'}")

    # ── Train LSTM ────────────────────────────────────────────────────────────
    if args.model in ("lstm", "both"):
        lstm_model   = build_lstm_model(input_dim, n_classes)
        lstm_model.summary()
        lstm_history = train_model(
            lstm_model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, batch_size=args.batch_size, model_name="lstm"
        )
        evaluate_dl_model(lstm_model, X_test, y_test, le, "BiLSTM")
        plot_training_history(lstm_history, "BiLSTM")
        # Save with a separate name to avoid overwriting CNN model
        lstm_model.save(MODELS_DIR / "lstm_model.h5")
        print(f"  💾 LSTM model saved: {MODELS_DIR / 'lstm_model.h5'}")
        # Also copy as cnn_model.h5 if CNN not trained
        if args.model == "lstm":
            import shutil
            shutil.copy(MODELS_DIR / "lstm_model.h5", MODELS_DIR / "cnn_model.h5")

    print("\n✅ Deep Learning training complete!")


if __name__ == "__main__":
    main()
