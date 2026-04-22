"""
feature_extraction.py
=====================
Extracts audio features (MFCC, Chroma, Mel-Spectrogram, ZCR, RMS)
from preprocessed audio signals for emotion classification.
"""

import numpy as np
import librosa
from src.audio_utils import load_and_preprocess, SAMPLE_RATE


# ─── Feature Configuration ───────────────────────────────────────────────────
N_MFCC     = 40    # Number of MFCC coefficients
N_CHROMA   = 12    # Chromagram bins
N_MEL      = 128   # Mel filterbanks
HOP_LENGTH = 512   # Hop length for STFT
N_FFT      = 2048  # FFT window size


# ─── Individual Feature Extractors ───────────────────────────────────────────

def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract MFCC (Mel-Frequency Cepstral Coefficients).
    MFCCs capture the timbral texture of sound — excellent for speech/emotion.

    Returns: Mean of each MFCC across frames → shape (n_mfcc,)
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return np.mean(mfccs, axis=1)


def extract_mfcc_delta(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract MFCC delta (first-order derivative) — captures temporal dynamics.

    Returns: Mean of delta-MFCCs → shape (n_mfcc,)
    """
    mfccs       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mfcc_delta  = librosa.feature.delta(mfccs)
    return np.mean(mfcc_delta, axis=1)


def extract_chroma(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract Chroma features (pitch class profiles).
    Useful for capturing harmonic content and musical emotion cues.

    Returns: Mean of chroma → shape (12,)
    """
    stft   = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_chroma=N_CHROMA)
    return np.mean(chroma, axis=1)


def extract_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract Mel-Spectrogram features.
    Represents audio energy across frequency bands on the mel scale.

    Returns: Mean of mel bands → shape (N_MEL,)
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.mean(mel_db, axis=1)


def extract_zcr(y: np.ndarray) -> float:
    """
    Extract Zero Crossing Rate — how often the signal crosses zero.
    Higher ZCR often indicates noise or high-frequency content.

    Returns: Mean ZCR (scalar)
    """
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    return float(np.mean(zcr))


def extract_rms(y: np.ndarray) -> float:
    """
    Extract Root Mean Square Energy — represents loudness/intensity.

    Returns: Mean RMS energy (scalar)
    """
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    return float(np.mean(rms))


def extract_spectral_centroid(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Extract Spectral Centroid — the 'center of mass' of the spectrum.
    Brighter sounds have a higher spectral centroid.

    Returns: Mean spectral centroid (scalar)
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return float(np.mean(centroid))


def extract_spectral_rolloff(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Extract Spectral Rolloff — frequency below which 85% of energy is contained.

    Returns: Mean spectral rolloff (scalar)
    """
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return float(np.mean(rolloff))


# ─── Combined Feature Vector ──────────────────────────────────────────────────

def extract_features(file_path: str) -> np.ndarray:
    """
    Main feature extraction pipeline.
    Extracts and concatenates all features into a single 1-D feature vector.

    Feature vector composition:
      - MFCC mean          : 40 features
      - MFCC delta mean    : 40 features
      - Chroma mean        : 12 features
      - Mel-Spectrogram    : 128 features  ← omitted for ML models (too large)
      - ZCR                : 1  feature
      - RMS energy         : 1  feature
      - Spectral centroid  : 1  feature
      - Spectral rolloff   : 1  feature
                             ────────────
    Total (without Mel)   : 95 features
    Total (with Mel)      : 223 features

    Args:
        file_path : Path to audio file

    Returns:
        1-D numpy feature array (95 features for ML, 223 for DL)
    """
    y = load_and_preprocess(file_path)

    mfcc       = extract_mfcc(y)              # 40
    mfcc_delta = extract_mfcc_delta(y)        # 40
    chroma     = extract_chroma(y)            # 12
    zcr        = np.array([extract_zcr(y)])   # 1
    rms        = np.array([extract_rms(y)])   # 1
    centroid   = np.array([extract_spectral_centroid(y)])  # 1

    features = np.concatenate([mfcc, mfcc_delta, chroma, zcr, rms, centroid])
    return features  # shape: (95,)


def extract_features_dl(file_path: str) -> np.ndarray:
    """
    Extended feature extraction for deep learning models (CNN/LSTM).
    Includes Mel-Spectrogram features for richer representation.

    Returns:
        1-D numpy feature array (223 features)
    """
    y = load_and_preprocess(file_path)

    mfcc       = extract_mfcc(y)
    mfcc_delta = extract_mfcc_delta(y)
    chroma     = extract_chroma(y)
    mel        = extract_mel_spectrogram(y)
    zcr        = np.array([extract_zcr(y)])
    rms        = np.array([extract_rms(y)])
    centroid   = np.array([extract_spectral_centroid(y)])

    features = np.concatenate([mfcc, mfcc_delta, chroma, mel, zcr, rms, centroid])
    return features  # shape: (223,)


def extract_features_from_array(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract features directly from a numpy audio array (e.g., from mic recording).
    No file I/O required.

    Args:
        y  : Preprocessed audio signal
        sr : Sample rate

    Returns:
        1-D numpy feature array (95 features)
    """
    mfcc       = extract_mfcc(y, sr)
    mfcc_delta = extract_mfcc_delta(y, sr)
    chroma     = extract_chroma(y, sr)
    zcr        = np.array([extract_zcr(y)])
    rms        = np.array([extract_rms(y)])
    centroid   = np.array([extract_spectral_centroid(y, sr)])

    return np.concatenate([mfcc, mfcc_delta, chroma, zcr, rms, centroid])
