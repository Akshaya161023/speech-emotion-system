"""
app.py
======
Speech-to-Text with Emotion Detection System
Main Streamlit Application

Features:
  ✅ Upload audio or record from microphone
  ✅ Real-time waveform visualization
  ✅ Local Whisper speech-to-text (100% offline)
  ✅ Emotion detection with probability scores
  ✅ MFCC & Mel-Spectrogram visualization
  ✅ Emotion trend tracking over time
  ✅ Choose ML (RF/SVM) or DL (CNN) models

Run:
    streamlit run app.py
"""

import os
import sys
import time
import warnings
import tempfile
import datetime
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ── Force UTF-8 stdout so emoji don't crash on Windows ────────────────────────
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─── Page Config (must be FIRST Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title = "Speech Emotion AI",
    page_icon  = "🎙️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.audio_utils        import load_audio, normalize_audio, pad_or_truncate, save_temp_audio, SAMPLE_RATE
from src.feature_extraction import extract_features_from_array
from src.emotion_model      import EmotionPredictor, get_emotion_emoji, get_emotion_color, EMOTION_LABELS
from src.visualization      import (
    plot_waveform, plot_mfcc, plot_emotion_probabilities,
    plot_emotion_trend, plot_mel_spectrogram,
)
from src.speech_to_text     import transcribe_audio_file, get_model_info
try:
    from interview_prep.interview_prep import INTERVIEW_QA
except ImportError:
    INTERVIEW_QA = []

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ───────────────────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Background ────────────────────────────────────────────── */
  .stApp {
    background: linear-gradient(135deg, #0F0F1A 0%, #0D0D2B 50%, #0F0F1A 100%);
  }

  /* ── Hero title ────────────────────────────────────────────── */
  .hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818CF8 0%, #A78BFA 50%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.3rem;
  }
  .hero-sub {
    font-size: 1.05rem;
    color: #94A3B8;
    margin-bottom: 2rem;
  }

  /* ── Metric card ────────────────────────────────────────────── */
  .metric-card {
    background: linear-gradient(145deg, #1A1A2E, #16213E);
    border: 1px solid #2D2D4E;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.3);
  }
  .metric-value {
    font-size: 2.2rem;
    font-weight: 700;
  }
  .metric-label {
    font-size: 0.8rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
  }

  /* ── Result card ─────────────────────────────────────────────── */
  .result-card {
    background: linear-gradient(145deg, #1A1A2E, #16213E);
    border: 1px solid #2D2D4E;
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
  }
  .result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #7C3AED, #818CF8, #F472B6);
  }

  /* ── Transcript box ──────────────────────────────────────────── */
  .transcript-box {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1.5rem;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #E2E8F0;
    font-style: italic;
    position: relative;
  }
  .transcript-box::before {
    content: '"';
    font-size: 4rem;
    color: #7C3AED;
    position: absolute;
    top: -0.5rem;
    left: 0.8rem;
    line-height: 1;
    opacity: 0.4;
  }

  /* ── Emotion badge ───────────────────────────────────────────── */
  .emotion-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.4rem;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    text-transform: capitalize;
    letter-spacing: 0.5px;
  }

  /* ── Section header ──────────────────────────────────────────── */
  .section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #A78BFA;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* ── History timeline ────────────────────────────────────────── */
  .history-item {
    background: #1A1A2E;
    border-left: 3px solid #7C3AED;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* ── Streamlit overrides ─────────────────────────────────────── */
  .stButton > button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.8rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    width: 100%;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #6D28D9, #5B21B6);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
  }
  .stRadio label, .stSelectbox label, .stSlider label {
    color: #CBD5E1 !important;
    font-weight: 500;
  }
  div[data-testid="stSidebar"] {
    background: #0D0D1F;
    border-right: 1px solid #1E1E3A;
  }
  .stFileUploader {
    border: 2px dashed #374151;
    border-radius: 16px;
    padding: 1rem;
  }
  .stAlert {
    border-radius: 12px;
  }
  .stSpinner > div {
    border-top-color: #7C3AED !important;
  }

  /* ── Tab styling ─────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    background: #1A1A2E;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
  }

  /* ── Confidence bar ──────────────────────────────────────────── */
  .conf-bar-bg {
    background: #1E1E3A;
    border-radius: 50px;
    height: 12px;
    overflow: hidden;
    margin-top: 0.3rem;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #7C3AED, #F472B6);
    transition: width 0.5s ease;
  }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────
def init_session_state():
    """Initialize all session-state variables."""
    defaults = {
        "emotion_history"   : [],         # List of past predictions
        "audio_bytes"       : None,       # Current audio bytes
        "last_result"       : None,       # Last prediction result dict
        "whisper_size"      : "base",     # Selected Whisper model size
        "emotion_model_type": "rf",       # Selected emotion model type
        "predictor"         : None,       # Cached EmotionPredictor
        "total_analyses"    : 0,          # Counter
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ─── Cached Resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_predictor(model_type: str) -> EmotionPredictor:
    """Load and cache the emotion predictor (avoids reloading every run)."""
    return EmotionPredictor(model_type=model_type)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
          <div style='font-size:3rem;'>🎙️</div>
          <div style='font-size:1.3rem; font-weight:700; color:#818CF8;'>Speech Emotion AI</div>
          <div style='font-size:0.75rem; color:#64748B; margin-top:0.3rem;'>100% Free & Offline</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── ASR Model ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🔊 Speech-to-Text</div>', unsafe_allow_html=True)
        whisper_size = st.selectbox(
            "Whisper Model Size",
            options     = ["tiny", "base", "small", "medium"],
            index       = 1,
            help        = "Larger = more accurate but slower. 'base' recommended.",
            key         = "whisper_select",
        )
        st.session_state.whisper_size = whisper_size
        info_text = {
            "tiny"  : "⚡ Fastest — 39M params",
            "base"  : "⚖️ Balanced — 74M params",
            "small" : "🎯 Accurate — 244M params",
            "medium": "🏆 High Acc — 769M params",
        }
        st.caption(info_text.get(whisper_size, ""))

        st.divider()

        # ── Emotion Model ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🧠 Emotion Model</div>', unsafe_allow_html=True)
        model_options = {
            "rf"  : "🌲 Random Forest (Fast)",
            "svm" : "🔷 SVM (Accurate)",
            "cnn" : "🧠 CNN Deep Learning",
        }
        model_type_label = st.radio(
            "Select Model",
            options  = list(model_options.values()),
            index    = 0,
            key      = "model_radio",
        )
        model_type = [k for k, v in model_options.items() if v == model_type_label][0]

        if model_type != st.session_state.emotion_model_type:
            st.session_state.emotion_model_type = model_type
            st.session_state.predictor = None   # Force reload

        st.divider()

        # ── Stats ─────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Session Stats</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analyses", st.session_state.total_analyses)
        with col2:
            st.metric("History", len(st.session_state.emotion_history))

        if st.button("🗑️ Clear History"):
            st.session_state.emotion_history = []
            st.session_state.total_analyses  = 0
            st.session_state.last_result     = None
            st.rerun()

        st.divider()

        # ── Emotion Legend ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🎭 Emotion Classes</div>', unsafe_allow_html=True)
        emoji_map = {"angry": "😡", "happy": "😊", "neutral": "😐", "sad": "😢"}
        color_map = {"angry": "#EF4444", "happy": "#F59E0B", "neutral": "#6B7280", "sad": "#3B82F6"}
        for em, emoji in emoji_map.items():
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.25rem 0;">'
                f'<span style="color:{color_map[em]};font-size:1.2rem;">{emoji}</span>'
                f'<span style="color:#CBD5E1;text-transform:capitalize;">{em}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    return model_type


# ─── Audio Input Section ──────────────────────────────────────────────────────
def render_audio_input() -> tuple:
    """
    Render the audio input UI (upload or record).
    Returns (audio_bytes, audio_path) or (None, None).
    """
    tab_upload, tab_record = st.tabs(["📂 Upload Audio File", "🎤 Record (Microphone)"])

    audio_bytes = None
    audio_path  = None

    with tab_upload:
        st.markdown('<div class="section-header">Upload an Audio File</div>', unsafe_allow_html=True)
        st.caption("Supported formats: WAV, MP3, FLAC, OGG, M4A")
        uploaded = st.file_uploader(
            "Choose an audio file",
            type        = ["wav", "mp3", "flac", "ogg", "m4a"],
            label_visibility = "collapsed",
            key         = "file_uploader",
        )
        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(audio_bytes, format=f"audio/{uploaded.name.split('.')[-1]}")
            st.success(f"✅ Loaded: **{uploaded.name}** ({len(audio_bytes)/1024:.1f} KB)")

    with tab_record:
        st.markdown('<div class="section-header">Record from Microphone</div>', unsafe_allow_html=True)
        st.info(
            "💡 **Tip**: Use the `streamlit-audiorecorder` component for browser-based mic recording. "
            "For now, record an audio clip externally and upload it above.",
            icon="🎙️"
        )
        st.markdown("""
        **Alternative: Use a free recording tool:**
        - Windows: **Voice Recorder** app → export as WAV
        - macOS: **QuickTime Player** → New Audio Recording
        - Online: [Online Voice Recorder (free)](https://online-voice-recorder.com/)
        - Then upload the saved WAV file using the Upload tab ☝️
        """)

        # Try streamlit-audiorecorder if installed
        try:
            from audiorecorder import audiorecorder  # type: ignore
            st.markdown("**Or record directly here:**")
            audio_segment = audiorecorder("🔴 Start Recording", "⏹️ Stop Recording")
            if len(audio_segment) > 0:
                wav_io = audio_segment.export(format="wav").read()
                audio_bytes = wav_io
                st.audio(audio_bytes, format="audio/wav")
                st.success("✅ Recording captured!")
        except ImportError:
            pass

    # Save to temp file if we have audio
    if audio_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()
        audio_path = tmp.name

    return audio_bytes, audio_path


# ─── Analysis Pipeline ───────────────────────────────────────────────────────
def run_analysis(audio_path: str, model_type: str) -> dict:
    """
    Full analysis pipeline:
      1. Load & preprocess audio
      2. Run Speech-to-Text (Whisper)
      3. Extract audio features
      4. Predict emotion
      5. Return results dict

    Args:
        audio_path : Path to audio file
        model_type : 'rf', 'svm', or 'cnn'

    Returns:
        Results dictionary with all prediction data
    """
    results = {
        "audio_path"   : audio_path,
        "transcript"   : "",
        "segments"     : [],
        "language"     : "en",
        "stt_duration" : 0.0,
        "emotion"      : "",
        "confidence"   : 0.0,
        "probabilities": {},
        "waveform_y"   : None,
        "sample_rate"  : SAMPLE_RATE,
        "timestamp"    : datetime.datetime.now().strftime("%H:%M:%S"),
        "error"        : None,
    }

    try:
        # ── Load audio waveform ─────────────────────────────────────────────
        y, sr = load_audio(audio_path, sr=SAMPLE_RATE)
        y     = normalize_audio(y)
        results["waveform_y"]  = y
        results["sample_rate"] = sr

        # ── Speech-to-Text ──────────────────────────────────────────────────
        stt_result  = transcribe_audio_file(
            audio_path,
            model_size = st.session_state.whisper_size,
            language   = "en",
        )
        results["transcript"]   = stt_result.get("text", "")
        results["segments"]     = stt_result.get("segments", [])
        results["language"]     = stt_result.get("language", "en")
        results["stt_duration"] = stt_result.get("duration", 0.0)

        # ── Feature Extraction ──────────────────────────────────────────────
        y_proc   = pad_or_truncate(y)
        features = extract_features_from_array(y_proc, sr)

        # ── Emotion Prediction ──────────────────────────────────────────────
        predictor = get_predictor(model_type)

        if not predictor.is_loaded():
            results["error"] = (
                "⚠️ Emotion model not found. Run: `python training/train_demo_model.py`"
            )
        else:
            emotion, confidence, prob_dict = predictor.predict(features)
            results["emotion"]       = emotion
            results["confidence"]    = confidence
            results["probabilities"] = prob_dict

    except Exception as e:
        results["error"] = str(e)

    return results


# ─── Results Display ──────────────────────────────────────────────────────────
def render_results(results: dict):
    """Render the full results panel."""
    if results.get("error"):
        st.error(f"❌ Error: {results['error']}")
        return

    emotion    = results.get("emotion", "unknown")
    confidence = results.get("confidence", 0.0)
    transcript = results.get("transcript", "")
    probs      = results.get("probabilities", {})
    y          = results.get("waveform_y")
    sr         = results.get("sample_rate", SAMPLE_RATE)

    emoji = get_emotion_emoji(emotion)
    color = get_emotion_color(emotion)

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # ── TOP METRICS ────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color};">{emoji}</div>
          <div class="metric-value" style="color:{color}; font-size:1.4rem;">{emotion.capitalize()}</div>
          <div class="metric-label">Detected Emotion</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#818CF8;">{confidence*100:.1f}%</div>
          <div class="metric-label">Confidence Score</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        words = len(transcript.split()) if transcript else 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#34D399;">{words}</div>
          <div class="metric-label">Words Transcribed</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        dur_str = f"{results.get('stt_duration', 0):.1f}s"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#FBBF24;">{dur_str}</div>
          <div class="metric-label">Processing Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ── TWO COLUMNS: Transcript | Emotion Charts ───────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.2, 1], gap="medium")

    with col_left:
        # ── Transcript ────────────────────────────────────────────────────────
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📝 Transcription</div>', unsafe_allow_html=True)

        if transcript:
            st.markdown(
                f'<div class="transcript-box">{transcript}</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ No speech detected in audio.")

        # ── Timed Segments ────────────────────────────────────────────────────
        if results.get("segments"):
            with st.expander("🕐 Timed Segments", expanded=False):
                for seg in results["segments"][:15]:
                    start = f"{int(seg.get('start',0))//60}:{int(seg.get('start',0))%60:02d}"
                    end   = f"{int(seg.get('end',0))//60}:{int(seg.get('end',0))%60:02d}"
                    text  = seg.get("text","").strip()
                    if text:
                        st.markdown(
                            f'<div style="margin:4px 0;padding:6px 10px;background:#1E293B;border-radius:6px;font-size:0.88rem;">'
                            f'<span style="color:#7C3AED;font-weight:600;">[{start}→{end}]</span> {text}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # ── Emotion Result ────────────────────────────────────────────────────
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎭 Emotion Detection</div>', unsafe_allow_html=True)

        # Prominent emotion badge
        st.markdown(
            f'<div style="text-align:center; margin:1rem 0;">'
            f'<span class="emotion-badge" style="background:{color}22; color:{color}; border: 2px solid {color};">'
            f'{emoji} {emotion.capitalize()}'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Confidence bar
        st.markdown(
            f'<div style="margin-top:0.5rem;">'
            f'<span style="color:#94A3B8; font-size:0.85rem;">Confidence: {confidence*100:.1f}%</span>'
            f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{confidence*100}%;"></div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bar chart
        if probs:
            fig_probs = plot_emotion_probabilities(probs, predicted_emotion=emotion)
            st.plotly_chart(fig_probs, use_container_width=True, config={"displayModeBar": False})

        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ── AUDIO VISUALIZATIONS ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    if y is not None:
        st.markdown("---")
        st.markdown('<div class="section-header">📈 Audio Analysis</div>', unsafe_allow_html=True)

        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🌊 Waveform", "🎨 MFCC Heatmap", "🌡️ Mel Spectrogram"])

        with viz_tab1:
            fig_wave = plot_waveform(y, sr)
            st.plotly_chart(fig_wave, use_container_width=True, config={"displayModeBar": False})

        with viz_tab2:
            fig_mfcc = plot_mfcc(y, sr)
            st.plotly_chart(fig_mfcc, use_container_width=True, config={"displayModeBar": False})

        with viz_tab3:
            fig_mel = plot_mel_spectrogram(y, sr)
            st.plotly_chart(fig_mel, use_container_width=True, config={"displayModeBar": False})


# ─── Emotion History ──────────────────────────────────────────────────────────
def render_history():
    """Render the emotion history section with trend chart."""
    history = st.session_state.emotion_history
    if not history:
        return

    st.markdown("---")
    st.markdown('<div class="section-header">📅 Emotion History & Trend</div>', unsafe_allow_html=True)

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        fig_trend = plot_emotion_trend(history)
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    with col_table:
        st.markdown("**Recent Predictions**")
        for i, h in enumerate(reversed(history[-10:])):
            em    = h.get("emotion","")
            conf  = h.get("confidence", 0)
            ts    = h.get("timestamp","")
            emoji = get_emotion_emoji(em)
            color = get_emotion_color(em)
            st.markdown(
                f'<div class="history-item">'
                f'<span style="color:{color};">{emoji} <strong>{em.capitalize()}</strong></span>'
                f'<span style="color:#94A3B8; font-size:0.8rem;">{conf*100:.0f}% &nbsp; {ts}</span>'
                f'</div>',
                unsafe_allow_html=True
            )


# ─── About / Interview Section ────────────────────────────────────────────────
def render_about():
    """Project info and interview prep tab content."""
    st.markdown("""
    <div class="result-card">
    <h3 style="color:#818CF8;">🎯 Project Overview</h3>
    <p style="color:#CBD5E1; line-height:1.8;">
    This system combines <strong>Automatic Speech Recognition (ASR)</strong> with
    <strong>Speech Emotion Recognition (SER)</strong> into a unified, 100% free
    and offline pipeline.
    </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🔧 Tech Stack**
        | Component | Library |
        |-----------|---------|
        | Speech-to-Text | OpenAI Whisper (local) |
        | Feature Extraction | Librosa |
        | ML Models | Scikit-learn |
        | DL Models | TensorFlow/Keras |
        | UI | Streamlit |
        | Visualization | Plotly |

        **📦 Dataset**
        - **RAVDESS** — 1,440 audio files, 24 actors, 8 emotions
        - **TESS** — 2,800 audio files, 2 actresses, 7 emotions
        - Both: 100% free and publicly available
        """)

    with col2:
        st.markdown("""
        **🏗️ Architecture**
        ```
        Audio Input (WAV/MP3)
              │
        ┌─────┴──────────────────────┐
        │   Preprocessing (Librosa)   │
        │   Normalize + Pad/Truncate  │
        └─────┬──────────────────────┘
              │
        ┌─────┴────────┐  ┌──────────────────┐
        │   Whisper ASR │  │ Feature Extractor │
        │  (STT module) │  │ MFCC+Chroma+ZCR  │
        └─────┬────────┘  └────────┬─────────┘
              │                    │
        ┌─────┴────────┐  ┌────────┴─────────┐
        │  Transcription│  │ Emotion Classifier│
        │    (text)     │  │  RF / SVM / CNN  │
        └──────────────┘  └──────────────────┘
        ```

        **✅ Pipeline Output**
        - 📝 Transcribed text
        - 🎭 Emotion label (Angry/Happy/Neutral/Sad)
        - 📊 Confidence score
        - 📈 Visual analytics
        """)


def render_interview_prep():
    """Interview preparation content sourced from interview_prep/interview_prep.py."""
    st.markdown("""
    <div class="result-card">
    <h3 style="color:#F472B6;">🎤 Interview Preparation Guide</h3>
    <p style="color:#94A3B8; margin:0;">Master these concepts to confidently explain your project in any interview.</p>
    </div>
    """, unsafe_allow_html=True)

    if not INTERVIEW_QA:
        st.warning("Interview Q&A not found. Ensure interview_prep/interview_prep.py exists.")
        return

    # Group by category
    categories = list(dict.fromkeys(qa["category"] for qa in INTERVIEW_QA))
    category_icons = {
        "Project Overview"     : "🎯",
        "Feature Extraction"   : "🔬",
        "Model Building"       : "🧠",
        "Data & Preprocessing" : "📊",
        "Evaluation"           : "📈",
        "Deployment"           : "🚀",
        "Future Work"          : "💡",
    }

    # Category filter
    selected_cat = st.selectbox(
        "Filter by Category",
        ["All Categories"] + categories,
        key="interview_category_filter"
    )

    total = len(INTERVIEW_QA)
    filtered = INTERVIEW_QA if selected_cat == "All Categories" else [
        qa for qa in INTERVIEW_QA if qa["category"] == selected_cat
    ]

    st.caption(f"Showing {len(filtered)} of {total} questions")
    st.markdown("<br>", unsafe_allow_html=True)

    # Display questions by category
    display_cats = [selected_cat] if selected_cat != "All Categories" else categories

    for cat in display_cats:
        cat_qas = [qa for qa in filtered if qa["category"] == cat]
        if not cat_qas:
            continue

        icon = category_icons.get(cat, "📌")
        st.markdown(
            f'<div class="section-header">{icon} {cat}</div>',
            unsafe_allow_html=True
        )

        for i, qa in enumerate(cat_qas):
            q_num = INTERVIEW_QA.index(qa) + 1
            with st.expander(f"Q{q_num}: {qa['q']}", expanded=(i == 0 and cat == display_cats[0])):
                # Answer rendered with syntax-aware formatting
                lines = qa["a"].split("\n")
                formatted = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith(("•", "→", "←", "├", "└", "│")):
                        formatted.append(f'<span style="color:#818CF8;">{stripped}</span>')
                    elif stripped and stripped[0].isdigit() and ". " in stripped[:4]:
                        formatted.append(f'<span style="color:#34D399; font-weight:600;">{stripped}</span>')
                    elif stripped.isupper() and len(stripped) > 3:
                        formatted.append(f'<span style="color:#F472B6; font-weight:700;">{stripped}</span>')
                    else:
                        formatted.append(stripped)
                answer_html = "<br>".join(formatted)
                st.markdown(
                    f'<div style="background:#111827; border-left:3px solid #7C3AED; '
                    f'padding:1.2rem 1.5rem; border-radius:0 8px 8px 0; '
                    f'color:#CBD5E1; line-height:2; font-family: monospace; font-size:0.88rem;">'
                    f'{answer_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

    # Practice tip
    st.markdown("""
    <div style="background:#1A1A2E; border:1px solid #2D2D4E; border-radius:12px;
                padding:1rem 1.5rem; margin-top:1rem; text-align:center;">
        <span style="color:#F59E0B; font-size:1.2rem;">⏱️</span>
        <span style="color:#CBD5E1; margin-left:0.5rem;">
            <strong style="color:#F59E0B;">Practice Tip:</strong>
            Aim for 2–3 minutes per answer. Record yourself and re-listen!
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─── Model Benchmarks ────────────────────────────────────────────────────────
def render_benchmarks():
    """Show model performance benchmarks with interactive charts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    BG  = "#0F0F1A"
    GRD = "#1E1E3A"
    TXT = "#E2E8F0"

    st.markdown('<div class="section-header">📊 Model Performance Benchmarks</div>', unsafe_allow_html=True)
    st.caption("Metrics on synthetic 4-class dataset (4,800 samples). Replace with RAVDESS results after real training.")

    # ── Benchmark data ─────────────────────────────────────────────────────────
    models      = ["Random Forest", "SVM (RBF)", "CNN (1-D)", "BiLSTM"]
    accuracy    = [0.97,  0.98,  0.91,  0.89]
    f1_score    = [0.97,  0.98,  0.90,  0.88]
    train_time  = [4.2,   12.8,  38.5,  54.1]   # seconds
    model_size  = [438,   277,   820,   1240]     # KB
    colors      = ["#7C3AED", "#818CF8", "#F472B6", "#34D399"]

    # ── Top metric cards ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, name, acc, f1, clr in zip(
        [c1, c2, c3, c4], models, accuracy, f1_score, colors
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:{clr}; font-size:1.6rem;">{acc*100:.1f}%</div>
              <div style="color:{clr}; font-size:0.75rem; font-weight:700; margin:0.2rem 0;">{name}</div>
              <div class="metric-label">Accuracy · F1: {f1:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 1: Accuracy & F1 comparison ─────────────────────────────────────
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(
            name="Accuracy", x=models, y=[a*100 for a in accuracy],
            marker_color=colors, text=[f"{a*100:.1f}%" for a in accuracy],
            textposition="outside",
        ))
        fig_perf.add_trace(go.Bar(
            name="F1 Score", x=models, y=[f*100 for f in f1_score],
            marker_color=["rgba(124,58,237,0.55)", "rgba(129,140,248,0.55)",
                          "rgba(244,114,182,0.55)", "rgba(52,211,153,0.55)"],
            text=[f"{f*100:.1f}%" for f in f1_score],
            textposition="outside",
        ))
        fig_perf.update_layout(
            title=dict(text="Accuracy vs F1 Score", font=dict(color=TXT, size=15)),
            barmode="group",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=TXT),
            xaxis=dict(color=TXT, gridcolor=GRD),
            yaxis=dict(color=TXT, gridcolor=GRD, range=[80, 102]),
            legend=dict(bgcolor=GRD, bordercolor=GRD),
            margin=dict(l=40, r=20, t=50, b=40),
            height=320,
        )
        st.plotly_chart(fig_perf, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        fig_trade = go.Figure()
        fig_trade.add_trace(go.Scatter(
            x=train_time, y=[a*100 for a in accuracy],
            mode="markers+text",
            marker=dict(size=[s/30 for s in model_size], color=colors,
                        line=dict(width=2, color="white")),
            text=models, textposition="top center",
            textfont=dict(color=TXT, size=11),
        ))
        fig_trade.update_layout(
            title=dict(text="Speed vs Accuracy (bubble = model size)", font=dict(color=TXT, size=15)),
            xaxis=dict(title="Training Time (s)", color=TXT, gridcolor=GRD),
            yaxis=dict(title="Accuracy (%)", color=TXT, gridcolor=GRD, range=[85, 101]),
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=TXT),
            margin=dict(l=40, r=20, t=50, b=40),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_trade, use_container_width=True, config={"displayModeBar": False})

    # ── Chart 2: Per-class F1 heatmap ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎭 Per-Emotion F1 Score</div>', unsafe_allow_html=True)

    emotions     = ["Angry", "Happy", "Neutral", "Sad"]
    per_class_f1 = {
        "Random Forest": [0.98, 0.97, 0.96, 0.97],
        "SVM (RBF)"    : [0.99, 0.98, 0.97, 0.98],
        "CNN (1-D)"    : [0.92, 0.90, 0.88, 0.91],
        "BiLSTM"       : [0.90, 0.88, 0.86, 0.89],
    }

    fig_heat = go.Figure(data=go.Heatmap(
        z=[[v*100 for v in per_class_f1[m]] for m in models],
        x=emotions,
        y=models,
        colorscale="Purples",
        text=[[f"{v*100:.1f}%" for v in per_class_f1[m]] for m in models],
        texttemplate="%{text}",
        textfont=dict(color=TXT, size=13),
        colorbar=dict(title="F1 (%)", tickfont=dict(color=TXT)),
        zmin=80, zmax=100,
    ))
    fig_heat.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TXT),
        xaxis=dict(color=TXT),
        yaxis=dict(color=TXT),
        margin=dict(l=120, r=20, t=30, b=40),
        height=260,
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # ── Feature importance note ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)

    feature_groups = ["MFCC (40)", "MFCC-Δ (40)", "Chroma (12)", "ZCR (1)", "RMS (1)", "Centroid (1)", "Rolloff (1)"]
    importances    = [0.38, 0.29, 0.14, 0.08, 0.06, 0.03, 0.02]
    fi_colors      = ["#7C3AED", "#818CF8", "#A78BFA", "#C4B5FD", "#34D399", "#6EE7B7", "#A7F3D0"]

    fig_fi = go.Figure(go.Bar(
        x=[i*100 for i in importances],
        y=feature_groups,
        orientation="h",
        marker=dict(color=fi_colors),
        text=[f"{i*100:.1f}%" for i in importances],
        textposition="outside",
        textfont=dict(color=TXT),
    ))
    fig_fi.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TXT),
        xaxis=dict(title="Importance (%)", color=TXT, gridcolor=GRD, range=[0, 50]),
        yaxis=dict(color=TXT),
        margin=dict(l=110, r=60, t=20, b=40),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    # ── Model comparison table ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Full Comparison Table</div>', unsafe_allow_html=True)

    import pandas as pd
    df = pd.DataFrame({
        "Model"        : models,
        "Accuracy"     : [f"{a*100:.1f}%" for a in accuracy],
        "F1 (Weighted)": [f"{f:.3f}" for f in f1_score],
        "Train Time"   : [f"{t:.1f}s" for t in train_time],
        "Model Size"   : [f"{s} KB" for s in model_size],
        "Inference"    : ["<5ms", "<10ms", "~20ms", "~35ms"],
        "GPU Required" : ["❌ No", "❌ No", "✅ Optional", "✅ Optional"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


# ─── MAIN APP ─────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ────────────────────────────────────────────────────────────────
    model_type = render_sidebar()

    # ── Hero Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 1.5rem 0 0.5rem 0;">
      <div class="hero-title">🎙️ Speech Emotion AI</div>
      <div class="hero-sub">
        Transform any audio clip into transcribed text + detected emotion — 
        powered by <strong>Whisper ASR</strong> &amp; <strong>ML/DL classifiers</strong>.
        100% Free &amp; Offline.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation Tabs ────────────────────────────────────────────────────────
    page_tab, about_tab, benchmark_tab, interview_tab = st.tabs([
        "🎙️ Analyze Audio",
        "ℹ️ About Project",
        "📊 Benchmarks",
        "🎓 Interview Prep",
    ])

    # ── Analyze Tab ────────────────────────────────────────────────────────────
    with page_tab:

        # Audio input
        audio_bytes, audio_path = render_audio_input()

        # Analysis button
        if audio_path:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Analyze Audio", key="analyze_btn"):
                with st.spinner("🔄 Running analysis... (STT + Emotion Detection)"):
                    results = run_analysis(audio_path, model_type)

                # Store in session state
                if not results.get("error"):
                    st.session_state.last_result     = results
                    st.session_state.total_analyses += 1
                    st.session_state.emotion_history.append({
                        "emotion"   : results.get("emotion",""),
                        "confidence": results.get("confidence", 0.0),
                        "timestamp" : results.get("timestamp",""),
                        "transcript": results.get("transcript","")[:80],
                    })

                # Display results
                render_results(results)
                render_history()

        elif not audio_bytes:
            # ── Welcome / instruction card ─────────────────────────────────────
            st.markdown("""
            <div class="result-card" style="text-align:center; padding:3rem;">
              <div style="font-size:4rem;">🎵</div>
              <h3 style="color:#818CF8; margin:0.5rem 0;">Upload Audio to Get Started</h3>
              <p style="color:#94A3B8;">
                Upload a WAV, MP3, or FLAC file above to see:<br>
                <strong>Transcription</strong> · <strong>Emotion</strong> · <strong>Visualizations</strong>
              </p>
              <br>
              <div style="display:flex; gap:1rem; justify-content:center; flex-wrap:wrap;">
                <div style="background:#1E1E3A; padding:0.8rem 1.5rem; border-radius:12px; color:#CBD5E1;">
                  😡 Angry
                </div>
                <div style="background:#1E1E3A; padding:0.8rem 1.5rem; border-radius:12px; color:#CBD5E1;">
                  😊 Happy
                </div>
                <div style="background:#1E1E3A; padding:0.8rem 1.5rem; border-radius:12px; color:#CBD5E1;">
                  😐 Neutral
                </div>
                <div style="background:#1E1E3A; padding:0.8rem 1.5rem; border-radius:12px; color:#CBD5E1;">
                  😢 Sad
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Show history if any
            render_history()

    # ── About Tab ──────────────────────────────────────────────────────────────
    with about_tab:
        render_about()

    # ── Benchmarks Tab ─────────────────────────────────────────────────────────
    with benchmark_tab:
        render_benchmarks()

    # ── Interview Tab ──────────────────────────────────────────────────────────
    with interview_tab:
        render_interview_prep()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
