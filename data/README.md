# Dataset Download Instructions

## RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song

**Download URL**: https://zenodo.org/record/1188976

### Emotions (8 classes)
| Code | Emotion |
|------|---------|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | **Angry** |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

### Filename Structure
```
03-01-05-01-01-01-12.wav
│  │  │  │  │  │  └── Actor ID (01-24)
│  │  │  │  │  └───── Repetition (01=1st, 02=2nd)
│  │  │  │  └──────── Statement (01="Kids are talking", 02="Dogs are sitting")
│  │  │  └─────────── Intensity (01=normal, 02=strong)
│  │  └────────────── Emotion (01-08, see table above)
│  └───────────────── Vocal Channel (01=speech, 02=song)
└──────────────────── Modality (01=AV, 02=video, 03=audio only)
```

### Quick Download (Auto)
```bash
python training/download_dataset.py --dataset ravdess
```

### Manual Download Steps
1. Go to: https://zenodo.org/record/1188976
2. Download: **Audio_Speech_Actors_01-24.zip** (~24 MB)
3. Extract into this `data/RAVDESS/` folder
4. Verify structure:
   ```
   data/RAVDESS/
     Actor_01/
       03-01-01-01-01-01-01.wav
       03-01-02-01-01-01-01.wav
       ...
     Actor_02/
     ...
     Actor_24/
   ```

---

## TESS — Toronto Emotional Speech Set

**Download URL**: https://tspace.library.utoronto.ca/handle/1807/24487

### Emotions (7 classes)
- Angry, Disgust, Fear, Happy, Neutral, Sad, Pleasant Surprise (ps)

### Folder Structure
```
data/TESS/
  OAF_angry/       ← Older Adult Female, angry
    OAF_back_angry.wav
    OAF_bad_angry.wav
    ...
  OAF_disgust/
  OAF_Fear/
  OAF_happy/
  OAF_neutral/
  OAF_Pleasant_surprise/
  OAF_Sad/
  YAF_angry/       ← Younger Adult Female, angry
  YAF_disgust/
  ...
```

### Kaggle Alternative (Easier)
```bash
pip install kaggle
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
unzip toronto-emotional-speech-set-tess.zip -d data/TESS/
```

---

## Dataset Statistics

| Dataset | Files | Actors/Speakers | Emotions | License |
|---------|-------|-----------------|----------|---------|
| RAVDESS | 1,440 | 24 (12M + 12F) | 8 | CC BY |
| TESS | 2,800 | 2F | 7 | Free Academic |
| Combined | 4,240 | 26 | Up to 8 | Mixed |

---

> ⚠️ **Important**: After placing datasets here, run the training script:
> ```bash
> python training/train_ml_models.py --data_dir data/RAVDESS
> ```
