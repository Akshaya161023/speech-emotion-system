"""
speech_to_text.py
=================
Local (offline) Speech-to-Text using OpenAI Whisper.
No API keys, no cloud — 100% free and runs on CPU.

Available Whisper models (smallest → largest, faster → more accurate):
  tiny   → fastest, ~39M params
  base   → good balance, ~74M params
  small  → better accuracy, ~244M params
  medium → high accuracy, ~769M params
  large  → best accuracy, ~1.5B params
"""

import os
import time
import tempfile
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ─── Whisper Loader ───────────────────────────────────────────────────────────

_whisper_model = None          # cached model (avoid re-loading every call)

def load_whisper(model_size: str = "base"):
    """
    Load and cache the Whisper ASR model.
    Downloads the model on first use (~74 MB for 'base').

    Args:
        model_size : Whisper model size ('tiny', 'base', 'small', 'medium', 'large')

    Returns:
        Loaded Whisper model object
    """
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            print(f"[INFO] Loading Whisper '{model_size}' model...", flush=True)
            _whisper_model = whisper.load_model(model_size)
            print("[OK] Whisper model loaded.", flush=True)
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed. Run: pip install openai-whisper"
            )
    return _whisper_model


# ─── Transcription Functions ──────────────────────────────────────────────────

def transcribe_audio_file(
    file_path: str,
    model_size: str = "base",
    language: str   = "en",
) -> dict:
    """
    Transcribe an audio file to text using local Whisper.

    Args:
        file_path  : Path to audio file (WAV, MP3, FLAC, etc.)
        model_size : Whisper model size (default: 'base')
        language   : Language code (default: 'en' for English)

    Returns:
        Dictionary with keys:
          - 'text'     : Full transcription string
          - 'segments' : List of segment dicts (with timestamps)
          - 'language' : Detected language
          - 'duration' : Processing time in seconds
    """
    model      = load_whisper(model_size)
    start_time = time.time()

    try:
        result = model.transcribe(
            file_path,
            language   = language,
            fp16       = False,        # Use FP32 for CPU compatibility
            verbose    = False,
        )
    except Exception as e:
        if "cannot reshape tensor of 0 elements" in str(e) or "unspecified dimension size -1" in str(e):
            # This happens with Numpy 2.0+ and Whisper when NO speech is present
            result = {"text": "[No voice detected in audio - instrumental/silent track]", "segments": []}
        else:
            raise e

    elapsed = time.time() - start_time

    return {
        "text"     : result.get("text", "").strip(),
        "segments" : result.get("segments", []),
        "language" : result.get("language", language),
        "duration" : round(elapsed, 2),
    }


def transcribe_audio_array(
    y: np.ndarray,
    sr: int        = 16000,
    model_size: str = "base",
    language: str   = "en",
) -> dict:
    """
    Transcribe audio from a numpy array (e.g., from mic recording).
    Saves to a temp file internally, then runs Whisper.

    Args:
        y          : Audio signal (numpy array, mono)
        sr         : Sample rate (Whisper expects 16000 Hz)
        model_size : Whisper model size
        language   : Language code

    Returns:
        Same dict as transcribe_audio_file()
    """
    import soundfile as sf

    # Write numpy array to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, y, sr)

    try:
        result = transcribe_audio_file(tmp_path, model_size=model_size, language=language)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result


def format_transcript_with_timestamps(segments: list) -> list:
    """
    Format Whisper segments into readable timestamp-text pairs.

    Args:
        segments : List of Whisper segment dicts

    Returns:
        List of dicts: [{'start': '0:00', 'end': '0:02', 'text': '...'}]
    """
    formatted = []
    for seg in segments:
        start = _seconds_to_mmss(seg.get("start", 0))
        end   = _seconds_to_mmss(seg.get("end",   0))
        text  = seg.get("text", "").strip()
        if text:
            formatted.append({"start": start, "end": end, "text": text})
    return formatted


def _seconds_to_mmss(seconds: float) -> str:
    """Convert float seconds → 'M:SS' string."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


# ─── Model Info ───────────────────────────────────────────────────────────────

WHISPER_MODEL_INFO = {
    "tiny"  : {"params": "39 M",    "speed": "~32x real-time"},
    "base"  : {"params": "74 M",    "speed": "~16x real-time"},
    "small" : {"params": "244 M",   "speed": "~6x real-time"},
    "medium": {"params": "769 M",   "speed": "~2x real-time"},
    "large" : {"params": "1,550 M", "speed": "~1x real-time"},
}

def get_model_info(model_size: str) -> str:
    """Return a human-readable string describing the selected Whisper model."""
    info = WHISPER_MODEL_INFO.get(model_size, {})
    return f"Whisper '{model_size}' | {info.get('params','?')} params | Speed: {info.get('speed','?')}"
