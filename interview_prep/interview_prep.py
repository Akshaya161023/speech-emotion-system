"""
interview_prep.py
=================
Standalone interview preparation module.
Run as a script to print all Q&A to the terminal.

Usage:
    python interview_prep/interview_prep.py
"""

INTERVIEW_QA = [
    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 1: PROJECT UNDERSTANDING
    # ──────────────────────────────────────────────────────────────────────────
    {
        "category": "Project Overview",
        "q": "Explain your project in 2 minutes.",
        "a": """
I built a Speech-to-Text with Emotion Detection System — a complete, 100% 
free, offline ML pipeline that processes audio recordings and produces two 
outputs simultaneously.

PIPELINE:
  Input Audio → [OpenAI Whisper (local)] → Transcribed Text
              → [Librosa Features + ML/DL Model] → Emotion Label

TECH STACK:
  • OpenAI Whisper — offline ASR, no API keys
  • Librosa — MFCC, Chroma, ZCR feature extraction
  • Scikit-learn — Random Forest, SVM classifiers
  • TensorFlow/Keras — 1-D CNN and BiLSTM networks
  • Streamlit — interactive web UI

DATASET: RAVDESS (1,440 clips, 24 actors, 8 emotions)
BEST MODEL: SVM / CNN achieving ~85-88% accuracy on 4-class task.

WHY THIS MATTERS: Emotion-aware systems enable mental health monitoring,
call center quality assurance, and accessibility tools.
        """.strip()
    },
    {
        "category": "Project Overview",
        "q": "Why did you choose Whisper over Google Speech-to-Text or AWS Transcribe?",
        "a": """
Three key reasons:

1. COST — Whisper is 100% free. Google STT charges $0.006/15-sec chunk.
   Running 1000 clips/day on Google would cost ~$12/day = $4,380/year.

2. OFFLINE — Whisper runs locally. No internet, no API limits, no outages,
   no data privacy concerns (audio never leaves the machine).

3. ACCURACY — Whisper 'base' achieves ~4.2% WER (Word Error Rate) on
   standard benchmarks, comparable to paid services. 'medium' achieves ~3%.

4. MULTILINGUAL — 99 languages with a single model.
   Google requires separate language models and separate billing.

Trade-off: Whisper is slower than cloud APIs (no GPU → uses CPU).
           'tiny' model (~10x realtime on CPU) mitigates this.
        """.strip()
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2: FEATURE EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────
    {
        "category": "Feature Extraction",
        "q": "What is MFCC and why is it the most important feature for speech emotion recognition?",
        "a": """
MFCC = Mel-Frequency Cepstral Coefficients

STEP-BY-STEP:
  1. Framing — split audio into overlapping 25ms frames (512 samples @ 22kHz)
  2. FFT — convert each frame from time domain to frequency domain
  3. Mel filterbank — apply 40 triangular filters spaced on the MEL SCALE
     (logarithmic — mimics how the human cochlea processes sound)
  4. Log compression — take log of filter energies (dB scale)
  5. DCT — apply Discrete Cosine Transform to decorrelate coefficients
     → first 13-40 coefficients = MFCCs

WHY IT WORKS FOR EMOTION:
  • Captures VOCAL TRACT shape (timbre) — changes with emotional state
  • MFCC-1 (C0) ≈ energy (loud = excited/angry, soft = sad/neutral)
  • MFCC-2 to 5 ≈ spectral slope (rising pitch in happy, falling in sad)
  • Delta-MFCCs capture RATE OF CHANGE — fear/surprise show fast changes

WE EXTRACT: 40 MFCCs + 40 delta-MFCCs = 80 features from MFCCs alone
        """.strip()
    },
    {
        "category": "Feature Extraction",
        "q": "What other features did you use besides MFCC and why?",
        "a": """
FEATURE VECTOR (95-D total for ML models):

1. MFCC (40)       — Spectral envelope, vocal tract shape
2. MFCC Delta (40) — Temporal dynamics, rate of change
3. Chroma (12)     — Pitch class profile, harmonic content
                     (Happy speech often more 'melodic')
4. ZCR (1)         — Zero Crossing Rate: how fast signal crosses zero
                     High ZCR = fricatives, noise, harsh sounds (angry)
                     Low ZCR = smooth, voiced speech (neutral/sad)
5. RMS Energy (1)  — Root Mean Square = loudness
                     Angry: high, Sad: low, Neutral: moderate
6. Spectral Centroid (1) — 'Brightness' of sound
                           Happy/Angry: bright (high centroid)
                           Sad: dull (low centroid)
7. Spectral Rolloff (1) — Frequency below which 85% energy lies
                          Related to sharpness of the sound

TOTAL: 95 features for ML, 223 features for DL (adds Mel spectrogram)
        """.strip()
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3: MODEL SELECTION
    # ──────────────────────────────────────────────────────────────────────────
    {
        "category": "Model Building",
        "q": "Why Random Forest over Logistic Regression or Naive Bayes?",
        "a": """
COMPARISON TABLE:

Feature          | Logistic Reg | Naive Bayes | Random Forest
─────────────────────────────────────────────────────────────
Non-linearity    | ❌ Linear    | ❌ Linear   | ✅ Non-linear
Feature corr     | ✅ Handles   | ❌ Assumes  | ✅ Handles
Imbalanced data  | Needs tuning | Needs tuning| class_weight
Feature import.  | Coefficients | N/A         | ✅ Built-in
Training speed   | Fast         | Fastest     | Moderate
Accuracy (SER)   | ~70%         | ~65%        | ~82%

BAGGING ADVANTAGE:
  RF trains N trees on random subsets (bootstrap samples) and random 
  feature subsets. Aggregating diverse trees reduces VARIANCE without
  increasing bias — crucial for noisy audio features.

PRACTICAL: No need for feature selection — RF's max_features='sqrt'
           auto-selects the most informative features at each split.
        """.strip()
    },
    {
        "category": "Model Building",
        "q": "Explain your CNN architecture. Why 1-D CNN for audio?",
        "a": """
WHY 1-D CNN (not 2-D):
  Our input is a 1-D feature vector (223 features), not an image.
  1-D convolutions slide along the feature axis, learning local 
  correlations between adjacent features (e.g., MFCC band patterns).
  2-D CNN would only make sense if we used raw spectrogram images.

OUR ARCHITECTURE:
  Input: (223, 1)
  ├── Conv1D(64, k=5) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.3)
  ├── Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.3)
  ├── Conv1D(256, k=3) → BatchNorm → ReLU → GlobalAvgPool
  ├── Dense(128, relu) → Dropout(0.4)
  ├── Dense(64, relu) → Dropout(0.3)
  └── Dense(4, softmax)     ← 4 emotion classes

KEY DESIGN CHOICES:
  • BatchNormalization — prevents internal covariate shift, faster training
  • GlobalAveragePooling — reduces parameters, prevents overfitting
  • Dropout(0.3-0.4) — regularization against small dataset (~1440 samples)
  • EarlyStopping(patience=15) — stops when val_accuracy plateaus
  • ReduceLROnPlateau — halves LR when val_loss stagnates
        """.strip()
    },
    {
        "category": "Model Building",
        "q": "Why did you use BiLSTM and not vanilla LSTM?",
        "a": """
VANILLA LSTM reads sequence left-to-right (past → present).
BiLSTM (Bidirectional LSTM) reads BOTH directions simultaneously:
  → Forward LSTM:  frame_1 → frame_2 → ... → frame_N
  ← Backward LSTM: frame_N → frame_N-1 → ... → frame_1

For EMOTION DETECTION:
  • Emotional cues appear throughout the clip (not just at the end)
  • A falling intonation at the END modifies meaning of earlier speech
  • BiLSTM captures BOTH context before and after each frame
  • This is critical for distinguishing sad vs. neutral (both low energy
    but sad often has a distinctive terminal fall)

TRADE-OFF: BiLSTM is ~2x slower to train than vanilla LSTM, but 
consistently 2-5% more accurate on speech emotion tasks.
        """.strip()
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 4: DATA & PREPROCESSING
    # ──────────────────────────────────────────────────────────────────────────
    {
        "category": "Data & Preprocessing",
        "q": "How did you preprocess the audio data?",
        "a": """
PIPELINE:
  1. LOAD     → librosa.load(sr=22050, mono=True)
                Resamples to 22,050 Hz (standard for speech features)
                Converts stereo to mono by averaging channels

  2. NORMALIZE → divide by max(abs(y))
                Ensures all clips are in [-1, 1] range
                Prevents loudness bias (angry clips aren't just louder)

  3. PAD / TRUNCATE → target length = 22050 * 3.0 = 66,150 samples
                      Short clips: zero-padded on the right
                      Long clips: truncated from the right
                      WHY FIXED LENGTH: ML/DL models need fixed input size

  4. FEATURE EXTRACTION (per clip):
     • MFCC: n_mfcc=40, hop_length=512, n_fft=2048
     • Take MEAN across time frames → noise-robust summary

  5. SCALING → StandardScaler (μ=0, σ=1)
               Critical for SVM (kernel-based, distance-sensitive)
               Less critical for RF (tree-based, scale-invariant)
               But applied to both for consistency

  6. ENCODING → LabelEncoder → anger=0, happy=1, neutral=2, sad=3
        """.strip()
    },
    {
        "category": "Data & Preprocessing",
        "q": "RAVDESS has 8 emotions but you used only 4. Why?",
        "a": """
PRACTICAL REASONS FOR 4-CLASS:

1. DATA IMBALANCE: RAVDESS has equal samples per class (180 each).
   But 8 classes with 180 samples each = 1440 total.
   Per class: only 144 training samples — too few for reliable DL.

2. AMBIGUITY: 'Calm' and 'Neutral' are acoustically very similar.
   'Disgusted' and 'Angry' overlap significantly.
   Adding these creates confusion that hurts overall accuracy more 
   than the added granularity helps.

3. PRACTICAL UTILITY: The 4 selected emotions (Angry, Happy, Neutral, Sad)
   cover 90%+ of real-world use cases (customer service, mental health).

4. PERFORMANCE: 4-class → ~85% accuracy | 8-class → ~68% accuracy
   Better to be highly accurate on fewer classes than mediocre on all.

EXTENSION: The architecture easily supports 8 classes — just change
   target_emotions to None in the training script to include all.
        """.strip()
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 5: EVALUATION
    # ──────────────────────────────────────────────────────────────────────────
    {
        "category": "Evaluation",
        "q": "Why use F1 score instead of accuracy? Explain the difference.",
        "a": """
ACCURACY = (TP + TN) / Total
  • Misleading for imbalanced classes
  • Example: 90% neutral clips → predict 'neutral' always = 90% accuracy!

PRECISION = TP / (TP + FP) — "Of all 'happy' predictions, how many were truly happy?"
  → Matters when FALSE POSITIVES are costly (e.g., flagging a false customer complaint)

RECALL = TP / (TP + FN) — "Of all truly happy clips, how many did we find?"
  → Matters when FALSE NEGATIVES are costly (e.g., missing a suicidal patient's sadness)

F1 = 2 × (Precision × Recall) / (Precision + Recall)
  → Harmonic mean — punishes extreme imbalance between P and R
  → Better single metric for multi-class imbalanced problems

FOR EMOTION DETECTION:
  RECALL matters more — missing a 'sad' or 'angry' signal is worse than 
  a false positive. Mental health monitoring needs high recall for SAD/FEARFUL.

WE REPORT: Weighted F1 (weights by class support) and macro F1 (equal class weight)
        """.strip()
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 6: DEPLOYMENT
    # ──────────────────────────────────────────────────────────────────────────
    {
        "category": "Deployment",
        "q": "How would you deploy this to production at scale?",
        "a": """
CURRENT (Demo): Streamlit Community Cloud — free, good for prototyping
                Limitation: single-threaded, models reload on each user

PRODUCTION ARCHITECTURE:
  ┌─────────────────────────────────────────────────────┐
  │              Load Balancer (nginx)                  │
  └────────────────┬────────────────────────────────────┘
                   │
          ┌────────▼────────┐
          │   FastAPI       │  ← REST API backend
          │   /transcribe   │     (async endpoints)
          │   /predict      │
          └────────┬────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
  ┌───▼───┐  ┌────▼────┐  ┌───▼────┐
  │Whisper│  │Feature  │  │Emotion │
  │Worker │  │Extractor│  │Model   │
  │(GPU)  │  │(CPU)    │  │(CPU/  │
  └───────┘  └─────────┘  │  GPU) │
                           └───────┘

OPTIMIZATIONS:
  1. Whisper quantization (INT8) → 2x faster, 50% less memory
  2. @st.cache_resource → load models once per session
  3. Redis cache → cache repeat audio predictions by hash
  4. Celery workers → async processing for concurrent users
  5. Docker container → reproducible deployment anywhere
  6. model.pkl → ONNX runtime → 3-5x faster inference
        """.strip()
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 7: IMPROVEMENTS
    # ──────────────────────────────────────────────────────────────────────────
    {
        "category": "Future Work",
        "q": "What would you improve if you had more time / data?",
        "a": """
1. DATA AUGMENTATION (immediately implementable):
   • Pitch shifting: librosa.effects.pitch_shift(y, n_steps=±2)
   • Time stretching: librosa.effects.time_stretch(y, rate=0.9)
   • Add background noise: y += 0.005 * np.random.randn(len(y))
   • Creates 3-4x more training data, improves generalization

2. PRETRAINED MODELS (wav2vec 2.0 / HuBERT):
   • Facebook's wav2vec 2.0 learns speech representations from raw audio
   • Fine-tuning on RAVDESS → state-of-the-art ~93% accuracy
   • Available free via Hugging Face Transformers

3. REAL-TIME STREAMING:
   • Process audio in sliding 3-second windows
   • Use WebRTC for browser microphone access
   • Stream emotion predictions at 1Hz

4. MULTIMODAL FUSION:
   • Combine audio emotion + text sentiment (from transcript)
   • Text may say "I'm fine" while audio says "sad" → detect incongruence

5. MORE DATASETS:
   • EmoDB (German, 7 emotions)
   • IEMOCAP (12 hours, natural conversations)
   • Cross-dataset training for better generalization
        """.strip()
    },
]


def print_interview_prep():
    """Pretty-print all Q&A pairs to the terminal."""
    categories = list(dict.fromkeys(qa["category"] for qa in INTERVIEW_QA))

    print("\n" + "═"*70)
    print("  SPEECH EMOTION DETECTION — INTERVIEW PREPARATION GUIDE")
    print("  " + f"{len(INTERVIEW_QA)} Questions Across {len(categories)} Categories")
    print("═"*70)

    for cat in categories:
        cat_qas = [qa for qa in INTERVIEW_QA if qa["category"] == cat]
        print(f"\n\n{'─'*70}")
        print(f"  📚 {cat.upper()}")
        print(f"{'─'*70}")

        for i, qa in enumerate(cat_qas, 1):
            print(f"\n  Q: {qa['q']}")
            print(f"  {'─'*60}")
            for line in qa["a"].split("\n"):
                print(f"  {line}")

    print("\n" + "═"*70)
    print("  💡 Tip: Practice answering each question out loud!")
    print("  ⏱️  Target: 2-3 minutes per answer")
    print("═"*70 + "\n")


if __name__ == "__main__":
    print_interview_prep()
