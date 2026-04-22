"""
audio_utils.py
==============
Utility functions for loading, processing, and saving audio files.
Handles WAV, MP3, and other formats supported by librosa.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
from pathlib import Path


# ─── Constants ───────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050       # Standard sample rate for librosa
DURATION    = 3.0         # Fixed clip duration in seconds (for model input)
N_SAMPLES   = int(SAMPLE_RATE * DURATION)


# ─── Core Audio Functions ─────────────────────────────────────────────────────

def load_audio(file_path: str, sr: int = SAMPLE_RATE, duration: float = None) -> tuple:
    """
    Load an audio file and return the signal and sample rate.

    Args:
        file_path : Path to the audio file (WAV, MP3, FLAC, etc.)
        sr        : Target sample rate (resample if needed)
        duration  : Max duration in seconds to load (None = full file)

    Returns:
        (y, sr) – numpy array signal and sample rate integer
    """
    try:
        y, sr_loaded = librosa.load(file_path, sr=sr, duration=duration, mono=True)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio '{file_path}': {e}")


def pad_or_truncate(y: np.ndarray, target_len: int = N_SAMPLES) -> np.ndarray:
    """
    Pad (with zeros) or truncate a signal to a fixed length.

    Args:
        y          : 1-D numpy audio array
        target_len : Desired length in samples

    Returns:
        Audio array of exactly target_len samples
    """
    if len(y) < target_len:
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')
    else:
        y = y[:target_len]
    return y


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to [-1, 1] range.

    Args:
        y : 1-D numpy audio array

    Returns:
        Normalized audio array
    """
    max_val = np.max(np.abs(y))
    if max_val == 0:
        return y
    return y / max_val


def save_temp_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """
    Save raw audio bytes to a temporary file and return the file path.
    Useful for passing microphone recordings to librosa/Whisper.

    Args:
        audio_bytes : Raw audio bytes from Streamlit recorder
        suffix      : File extension

    Returns:
        Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        return tmp.name


def get_waveform(y: np.ndarray, sr: int = SAMPLE_RATE) -> tuple:
    """
    Generate time axis and signal for waveform plotting.

    Args:
        y  : Audio signal
        sr : Sample rate

    Returns:
        (time_array, signal_array)
    """
    time = np.linspace(0, len(y) / sr, num=len(y))
    return time, y


def get_audio_duration(file_path: str) -> float:
    """
    Return audio file duration in seconds without loading the full signal.

    Args:
        file_path : Path to audio file

    Returns:
        Duration in seconds (float)
    """
    return librosa.get_duration(path=file_path)


def load_and_preprocess(file_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline: load → normalize → pad/truncate.
    Returns a clean, fixed-length signal ready for feature extraction.

    Args:
        file_path : Path to audio file

    Returns:
        Preprocessed numpy audio array of length N_SAMPLES
    """
    y, _ = load_audio(file_path)
    y    = normalize_audio(y)
    y    = pad_or_truncate(y)
    return y
