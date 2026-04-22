"""
visualization.py
================
Rich visualization utilities for the Streamlit UI:
  - Interactive waveform plot
  - MFCC heatmap
  - Emotion probability bar chart
  - Emotion trend chart over time
"""

import numpy as np
import librosa
import librosa.display
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit

from src.emotion_model import EMOTION_COLORS, EMOTION_EMOJIS


# ─── Color Palette ────────────────────────────────────────────────────────────
BG_COLOR          = "#0F0F1A"
GRID_COLOR        = "#1E1E3A"
TEXT_COLOR        = "#E2E8F0"
PRIMARY_COLOR     = "#7C3AED"
WAVE_COLOR        = "#818CF8"
WAVE_FILL_COLOR   = "rgba(129, 140, 248, 0.15)"


# ─── Waveform Visualization ───────────────────────────────────────────────────

def plot_waveform(y: np.ndarray, sr: int = 22050, title: str = "Audio Waveform") -> go.Figure:
    """
    Create an interactive waveform plot using Plotly.

    Args:
        y     : Audio signal
        sr    : Sample rate
        title : Chart title

    Returns:
        Plotly Figure
    """
    # Downsample for faster rendering (max 5000 points)
    step = max(1, len(y) // 5000)
    y_ds = y[::step]
    t    = np.linspace(0, len(y) / sr, num=len(y_ds))

    fig = go.Figure()

    # Fill area under curve
    fig.add_trace(go.Scatter(
        x=t, y=y_ds,
        mode="lines",
        line=dict(color=WAVE_COLOR, width=1.5),
        fill="tozeroy",
        fillcolor=WAVE_FILL_COLOR,
        name="Signal",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=16)),
        xaxis=dict(
            title="Time (s)",
            color=TEXT_COLOR,
            gridcolor=GRID_COLOR,
            showgrid=True,
        ),
        yaxis=dict(
            title="Amplitude",
            color=TEXT_COLOR,
            gridcolor=GRID_COLOR,
            showgrid=True,
            zeroline=True,
            zerolinecolor=GRID_COLOR,
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=40, r=20, t=50, b=40),
        height=200,
        showlegend=False,
    )
    return fig


# ─── MFCC Heatmap ─────────────────────────────────────────────────────────────

def plot_mfcc(y: np.ndarray, sr: int = 22050, n_mfcc: int = 40) -> go.Figure:
    """
    Create an interactive MFCC heatmap.

    Args:
        y      : Audio signal
        sr     : Sample rate
        n_mfcc : Number of MFCC coefficients

    Returns:
        Plotly Figure
    """
    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    frame_dur = 512 / sr  # hop length / sr
    time_axis = np.arange(mfccs.shape[1]) * frame_dur

    fig = go.Figure(data=go.Heatmap(
        z=mfccs,
        x=time_axis,
        y=[f"MFCC {i+1}" for i in range(n_mfcc)],
        colorscale="Viridis",
        colorbar=dict(title="Energy", tickfont=dict(color=TEXT_COLOR)),
    ))

    fig.update_layout(
        title=dict(text="MFCC Features", font=dict(color=TEXT_COLOR, size=16)),
        xaxis=dict(title="Time (s)", color=TEXT_COLOR, gridcolor=GRID_COLOR),
        yaxis=dict(title="MFCC Coefficient", color=TEXT_COLOR, gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=80, r=20, t=50, b=40),
        height=300,
    )
    return fig


# ─── Emotion Probability Bar Chart ───────────────────────────────────────────

def plot_emotion_probabilities(
    prob_dict: dict,
    predicted_emotion: str = None,
    title: str = "Emotion Probabilities"
) -> go.Figure:
    """
    Create a horizontal bar chart of emotion prediction probabilities.

    Args:
        prob_dict          : Dict of {emotion: probability}
        predicted_emotion  : The top predicted emotion (highlighted)
        title              : Chart title

    Returns:
        Plotly Figure
    """
    if not prob_dict:
        return go.Figure()

    emotions = list(prob_dict.keys())
    probs    = [prob_dict[e] * 100 for e in emotions]
    emojis   = [EMOTION_EMOJIS.get(e.lower(), "🎭") for e in emotions]
    colors   = [
        EMOTION_COLORS.get(e.lower(), PRIMARY_COLOR)
        if e.lower() == (predicted_emotion or "").lower()
        else "#2D2D4E"
        for e in emotions
    ]
    labels   = [f"{emoji} {emo.capitalize()}" for emoji, emo in zip(emojis, emotions)]

    fig = go.Figure(go.Bar(
        x=probs,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="#7C3AED", width=1),
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=16)),
        xaxis=dict(
            title="Probability (%)",
            color=TEXT_COLOR,
            gridcolor=GRID_COLOR,
            range=[0, 110],
        ),
        yaxis=dict(color=TEXT_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=120, r=60, t=50, b=40),
        height=280,
        showlegend=False,
    )
    return fig


# ─── Emotion Trend Over Time ──────────────────────────────────────────────────

def plot_emotion_trend(history: list, title: str = "Emotion Trend Over Time") -> go.Figure:
    """
    Plot the emotion trend across multiple audio predictions.

    Args:
        history : List of dicts [{'timestamp': '...', 'emotion': '...', 'confidence': 0.9}]
        title   : Chart title

    Returns:
        Plotly Figure
    """
    if not history:
        return go.Figure()

    timestamps  = [h.get("timestamp", f"#{i+1}") for i, h in enumerate(history)]
    emotions    = [h.get("emotion", "").capitalize() for h in history]
    confidences = [h.get("confidence", 0) * 100 for h in history]
    colors      = [EMOTION_COLORS.get(h.get("emotion", "").lower(), PRIMARY_COLOR) for h in history]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Emotion Label Over Time", "Confidence Score (%)"),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    # ── Row 1: Emotion timeline ───────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=emotions,
            mode="lines+markers+text",
            marker=dict(size=12, color=colors, symbol="circle"),
            line=dict(color=PRIMARY_COLOR, width=2, dash="dot"),
            text=emotions,
            textposition="top center",
            textfont=dict(color=TEXT_COLOR, size=11),
            name="Emotion",
        ),
        row=1, col=1
    )

    # ── Row 2: Confidence line ────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=confidences,
            mode="lines+markers",
            marker=dict(size=8, color=WAVE_COLOR),
            line=dict(color=WAVE_COLOR, width=2),
            fill="tozeroy",
            fillcolor=WAVE_FILL_COLOR,
            name="Confidence",
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=16)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=60, r=20, t=80, b=40),
        height=380,
        showlegend=False,
    )

    for row in [1, 2]:
        fig.update_xaxes(color=TEXT_COLOR, gridcolor=GRID_COLOR, row=row, col=1)
        fig.update_yaxes(color=TEXT_COLOR, gridcolor=GRID_COLOR, row=row, col=1)

    return fig


# ─── Mel Spectrogram ─────────────────────────────────────────────────────────

def plot_mel_spectrogram(y: np.ndarray, sr: int = 22050) -> go.Figure:
    """
    Plot a Mel-Spectrogram heatmap.

    Args:
        y  : Audio signal
        sr : Sample rate

    Returns:
        Plotly Figure
    """
    mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db  = librosa.power_to_db(mel, ref=np.max)
    t       = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr, hop_length=512)
    freqs   = librosa.mel_frequencies(n_mels=64, fmin=0, fmax=sr // 2)

    fig = go.Figure(data=go.Heatmap(
        z=mel_db,
        x=t,
        y=[f"{int(f)} Hz" for f in freqs],
        colorscale="Plasma",
        colorbar=dict(title="dB", tickfont=dict(color=TEXT_COLOR)),
    ))

    fig.update_layout(
        title=dict(text="Mel Spectrogram", font=dict(color=TEXT_COLOR, size=16)),
        xaxis=dict(title="Time (s)", color=TEXT_COLOR, gridcolor=GRID_COLOR),
        yaxis=dict(title="Frequency", color=TEXT_COLOR, gridcolor=GRID_COLOR, tickvals=[]),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=60, r=20, t=50, b=40),
        height=280,
    )
    return fig
