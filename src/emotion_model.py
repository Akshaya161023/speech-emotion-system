"""
emotion_model.py
================
Emotion detection module supporting:
  - Scikit-learn models  : RandomForest, SVM
  - Deep Learning models : CNN (1-D), LSTM

Handles loading pre-trained models and running inference.
"""

import os
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path


# ─── Emotion Label Mapping ────────────────────────────────────────────────────
# RAVDESS / TESS emotion codes → human-readable labels
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# 4-class subset used for this project (simplified)
EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]

EMOTION_EMOJIS = {
    "angry"    : "😡",
    "happy"    : "😊",
    "neutral"  : "😐",
    "sad"      : "😢",
    "calm"     : "😌",
    "fearful"  : "😨",
    "disgust"  : "🤢",
    "surprised": "😲",
}

EMOTION_COLORS = {
    "angry"    : "#EF4444",
    "happy"    : "#F59E0B",
    "neutral"  : "#6B7280",
    "sad"      : "#3B82F6",
    "calm"     : "#10B981",
    "fearful"  : "#8B5CF6",
    "disgust"  : "#84CC16",
    "surprised": "#EC4899",
}

# ─── Model Paths ─────────────────────────────────────────────────────────────
MODELS_DIR   = Path(__file__).parent.parent / "models"
RF_MODEL_PATH  = MODELS_DIR / "rf_model.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.h5"
SCALER_PATH    = MODELS_DIR / "scaler.pkl"
ENCODER_PATH   = MODELS_DIR / "label_encoder.pkl"


# ─── Model Loader ─────────────────────────────────────────────────────────────

class EmotionPredictor:
    """
    Unified interface for emotion prediction using any trained model.

    Usage:
        predictor = EmotionPredictor(model_type='rf')
        emotion, confidence, probabilities = predictor.predict(features)
    """

    def __init__(self, model_type: str = "rf"):
        """
        Initialize the predictor with the chosen model type.

        Args:
            model_type : One of 'rf' (Random Forest), 'svm', 'cnn'
        """
        self.model_type = model_type
        self.model      = None
        self.scaler     = None
        self.encoder    = None
        self._load_preprocessing()
        self._load_model()

    # ── Preprocessing helpers ─────────────────────────────────────────────────

    def _load_preprocessing(self):
        """Load the saved feature scaler and label encoder."""
        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)
        if ENCODER_PATH.exists():
            self.encoder = joblib.load(ENCODER_PATH)

    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using the saved scaler (if available)."""
        features = features.reshape(1, -1)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        return features

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Load the appropriate model from disk."""
        try:
            if self.model_type == "rf":
                if RF_MODEL_PATH.exists():
                    self.model = joblib.load(RF_MODEL_PATH)
                else:
                    raise FileNotFoundError(f"Random Forest model not found at {RF_MODEL_PATH}")

            elif self.model_type == "svm":
                if SVM_MODEL_PATH.exists():
                    self.model = joblib.load(SVM_MODEL_PATH)
                else:
                    raise FileNotFoundError(f"SVM model not found at {SVM_MODEL_PATH}")

            elif self.model_type == "cnn":
                import tensorflow as tf
                if CNN_MODEL_PATH.exists():
                    self.model = tf.keras.models.load_model(CNN_MODEL_PATH)
                else:
                    raise FileNotFoundError(f"CNN model not found at {CNN_MODEL_PATH}")

            else:
                raise ValueError(f"Unknown model type: {self.model_type}. Choose 'rf', 'svm', or 'cnn'.")

        except FileNotFoundError as e:
            self.model = None
            print(f"⚠️ Warning: {e}")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> tuple:
        """
        Run emotion prediction on a feature vector.

        Args:
            features : 1-D numpy feature array (95 or 223 features)

        Returns:
            (emotion_label, confidence_float, probabilities_dict)
        """
        if self.model is None:
            return "Model not loaded", 0.0, {}

        X = self._preprocess_features(features)

        if self.model_type in ("rf", "svm"):
            # ── Scikit-learn inference ────────────────────────────────────────
            pred_idx = self.model.predict(X)[0]

            if hasattr(self.model, "predict_proba"):
                proba     = self.model.predict_proba(X)[0]
                classes   = self.model.classes_
                prob_dict = {str(c): float(p) for c, p in zip(classes, proba)}
                confidence = float(max(proba))
            else:
                # SVM with linear kernel may lack predict_proba
                prob_dict  = {}
                confidence = 1.0

            emotion = str(pred_idx)
            # Decode if encoder is available
            if self.encoder is not None:
                emotion = self.encoder.inverse_transform([pred_idx])[0]

        elif self.model_type == "cnn":
            # ── TensorFlow/Keras inference ────────────────────────────────────
            if self.encoder is not None:
                classes = list(self.encoder.classes_)
            else:
                classes = EMOTION_LABELS

            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            proba      = self.model.predict(X_reshaped, verbose=0)[0]
            pred_idx   = int(np.argmax(proba))
            emotion    = classes[pred_idx]
            confidence = float(proba[pred_idx])
            prob_dict  = {cls: float(p) for cls, p in zip(classes, proba)}

        return emotion, confidence, prob_dict

    def is_loaded(self) -> bool:
        """Return True if a model is successfully loaded."""
        return self.model is not None


# ─── Label Utilities ──────────────────────────────────────────────────────────

def get_emotion_emoji(emotion: str) -> str:
    """Return the emoji for a given emotion label."""
    return EMOTION_EMOJIS.get(emotion.lower(), "🎭")

def get_emotion_color(emotion: str) -> str:
    """Return the hex color for a given emotion label."""
    return EMOTION_COLORS.get(emotion.lower(), "#7C3AED")

def parse_ravdess_filename(filename: str) -> dict:
    """
    Parse a RAVDESS filename and extract metadata.

    RAVDESS filename format:
        Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Example: 03-01-05-01-01-01-12.wav
              ↑  ↑  ↑  ↑  ↑  ↑  ↑
              M  VC Em In St Re Ac

    Args:
        filename : RAVDESS filename (without directory)

    Returns:
        Dictionary with parsed metadata
    """
    parts = Path(filename).stem.split("-")
    if len(parts) != 7:
        return {}

    emotion_code = parts[2]
    return {
        "modality"      : parts[0],
        "vocal_channel" : parts[1],
        "emotion"       : RAVDESS_EMOTIONS.get(emotion_code, "unknown"),
        "emotion_code"  : emotion_code,
        "intensity"     : "normal" if parts[3] == "01" else "strong",
        "statement"     : "kids are talking" if parts[4] == "01" else "dogs are sitting",
        "repetition"    : parts[5],
        "actor_id"      : int(parts[6]),
        "actor_gender"  : "female" if int(parts[6]) % 2 == 0 else "male",
    }
