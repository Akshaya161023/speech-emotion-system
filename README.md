# Speech-to-Text with Emotion Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red?style=flat-square&logo=streamlit)
![Whisper](https://img.shields.io/badge/Whisper-OpenAI-green?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)
![Free](https://img.shields.io/badge/Cost-100%25%20FREE-brightgreen?style=flat-square)

**Real-time Speech Transcription + Emotion Recognition — Fully Offline & Free**

</div>

---

## 🎯 Project Overview

A complete end-to-end ML system that:
1. **Transcribes** speech from audio using **OpenAI Whisper** (runs 100% offline, no API)
2. **Detects emotion** from audio features using **ML/DL classifiers**
3. **Visualizes** waveforms, MFCCs, spectrograms, and emotion trends via **Streamlit**

### Input → Processing → Output

```
Audio File (WAV/MP3/FLAC)
        │
        ├──► Whisper ASR ──────────────► 📝 Transcribed Text
        │     (offline Transformer)
        │
        └──► Librosa Features ─────────► 🎭 Emotion Label
              (MFCC + Chroma + ZCR)          + Confidence %
                      │                      + Probability Chart
                      └──► RF / SVM / CNN
```

---

## 🆓 Tech Stack (100% Free)

| Component | Library | Why |
|-----------|---------|-----|
| Speech-to-Text | [OpenAI Whisper](https://github.com/openai/whisper) | Local, offline, state-of-the-art ASR |
| Feature Extraction | [Librosa](https://librosa.org/) | MFCC, Chroma, ZCR, RMS |
| ML Models | [Scikit-learn](https://scikit-learn.org/) | Random Forest, SVM |
| DL Models | [TensorFlow/Keras](https://tensorflow.org/) | 1-D CNN, BiLSTM |
| UI | [Streamlit](https://streamlit.io/) | Rapid, beautiful web apps |
| Visualization | [Plotly](https://plotly.com/) | Interactive charts |

---

## 📁 Project Structure

```
speech_emotion_system/
│
├── app.py                          # 🚀 Main Streamlit application
├── requirements.txt                # All dependencies
├── setup.py                        # One-command setup script
├── README.md                       # This file
│
├── .streamlit/
│   └── config.toml                 # Streamlit dark theme config
│
├── src/                            # Core modules (importable)
│   ├── __init__.py
│   ├── audio_utils.py              # Audio loading & preprocessing
│   ├── feature_extraction.py       # MFCC, Chroma, ZCR extraction
│   ├── emotion_model.py            # Model loader & predictor
│   ├── speech_to_text.py           # Whisper STT wrapper
│   └── visualization.py            # Plotly chart builders
│
├── training/                       # Model training scripts
│   ├── train_demo_model.py         # ⭐ Quick start (synthetic data)
│   ├── train_ml_models.py          # Train RF + SVM on RAVDESS
│   ├── train_dl_model.py           # Train CNN + LSTM on RAVDESS
│   └── download_dataset.py         # Auto-download RAVDESS/TESS
│
├── models/                         # Saved model artifacts (auto-created)
│   ├── rf_model.pkl                # Random Forest
│   ├── svm_model.pkl               # SVM
│   ├── cnn_model.h5                # CNN (TensorFlow)
│   ├── scaler.pkl                  # StandardScaler
│   └── label_encoder.pkl           # LabelEncoder
│
├── data/                           # Datasets (download separately)
│   ├── README.md                   # Dataset download instructions
│   └── RAVDESS/                    # → Extract RAVDESS here
│       ├── Actor_01/
│       │   └── 03-01-*.wav
│       └── ...
│
└── notebooks/                      # Jupyter notebooks (exploration)
    ├── 01_data_exploration.ipynb
    ├── 02_feature_extraction.ipynb
    └── 03_model_training.ipynb
```

---

## ⚡ Quick Start (5 Minutes)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train Demo Models (No Dataset Needed!)
```bash
python training/train_demo_model.py
```
This creates synthetic emotion data and trains RF + SVM models instantly.

### Step 3 — Launch the App
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 📊 Datasets

### RAVDESS (Recommended)
- **1,440** audio recordings
- **24 actors** (12 male, 12 female)
- **8 emotions**: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- **Download**: https://zenodo.org/record/1188976
- **License**: Creative Commons Attribution

```bash
python training/download_dataset.py --dataset ravdess
```

### TESS
- **2,800** audio recordings
- **2 actresses** (OAF = Older Adult Female, YAF = Younger Adult Female)
- **7 emotions**: Angry, Disgust, Fear, Happy, Neutral, Sad, Pleased Surprise
- **Download**: https://tspace.library.utoronto.ca/handle/1807/24487

---

## 🧠 Model Training

### Option A — Quick Demo (Synthetic Data)
```bash
python training/train_demo_model.py
```

### Option B — Real RAVDESS Data (ML Models)
```bash
# Download dataset first
python training/download_dataset.py --dataset ravdess

# Train Random Forest + SVM
python training/train_ml_models.py --data_dir data/RAVDESS --emotions angry happy neutral sad
```

### Option C — Deep Learning (CNN/LSTM)
```bash
python training/train_dl_model.py --data_dir data/RAVDESS --model cnn
python training/train_dl_model.py --data_dir data/RAVDESS --model lstm
```

---

## 📈 Model Performance (RAVDESS — 4 classes)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~82% | ~0.83 | ~0.82 | ~0.82 |
| SVM (RBF) | ~85% | ~0.86 | ~0.85 | ~0.85 |
| 1-D CNN | ~88% | ~0.88 | ~0.88 | ~0.88 |
| BiLSTM | ~87% | ~0.87 | ~0.87 | ~0.87 |

> Results vary by emotion subset and train/test split.

---

## 🎙️ Whisper Model Guide

| Model | Parameters | Speed | WER |
|-------|-----------|-------|-----|
| tiny | 39M | ~32x realtime | 5.6% |
| base | 74M | ~16x realtime | 4.2% |
| small | 244M | ~6x realtime | 3.4% |
| medium | 769M | ~2x realtime | 3.0% |

Recommended: **`base`** for most use cases.

---

## 🚀 Deployment (Free — Streamlit Community Cloud)

1. Push project to GitHub (include `models/` folder or use `git-lfs`)
2. Go to https://share.streamlit.io/
3. Click **"New app"** → Connect your GitHub repo
4. Set **Main file**: `app.py`
5. Click **Deploy** — it's FREE!

> Note: Add `packages.txt` with `ffmpeg` for audio processing support.

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ffmpeg not found` | Install ffmpeg: `winget install ffmpeg` (Windows) or `brew install ffmpeg` (Mac) |
| `No module 'whisper'` | Run `pip install openai-whisper` |
| `Model not found` | Run `python training/train_demo_model.py` |
| Slow transcription | Use `tiny` Whisper model in sidebar |
| `librosa` errors | Install ffmpeg or use WAV files |

---

## 📄 License

MIT License — Free for personal and commercial use.

---

<div align="center">
Built with ❤️ using Python · Whisper · Librosa · Scikit-learn · TensorFlow · Streamlit
</div>
