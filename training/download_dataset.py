"""
download_dataset.py
===================
Helper script to download RAVDESS and TESS datasets automatically.

RAVDESS: https://zenodo.org/record/1188976
TESS   : https://tspace.library.utoronto.ca/handle/1807/24487

Usage:
    python training/download_dataset.py --dataset ravdess
    python training/download_dataset.py --dataset tess
    python training/download_dataset.py --dataset both
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# RAVDESS — 24 actors, 8 emotions
# Full dataset (~24 GB, audio + video). We download speech-only files.
RAVDESS_SPEECH_URLS = [
    f"https://zenodo.org/record/1188976/files/Audio_Speech_Actors_{str(i).zfill(2)}-{str(i+1).zfill(2)}.zip"
    for i in range(1, 25, 2)
]
RAVDESS_SINGLE_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"

# TESS — 2 actresses, 7 emotions
TESS_URL = "https://tspace.library.utoronto.ca/bitstream/1807/24487/2/OAF-006.zip"


class ProgressBar(tqdm):
    """TQDM progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: str):
    """Download a file with a progress bar."""
    print(f"\n  📥 Downloading: {url.split('/')[-1]}")
    with ProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest_path) as t:
        urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
    print(f"  ✅ Downloaded: {dest_path}")


def extract_zip(zip_path: str, extract_to: str):
    """Extract a ZIP archive."""
    print(f"  📦 Extracting {Path(zip_path).name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)
    print(f"  ✅ Extracted to: {extract_to}")


def download_ravdess():
    """
    Download RAVDESS Speech dataset from Zenodo.

    Dataset structure after extraction:
        data/RAVDESS/
            Actor_01/
                03-01-01-01-01-01-01.wav
                ...
            Actor_02/
            ...
    """
    ravdess_dir = DATA_DIR / "RAVDESS"
    ravdess_dir.mkdir(exist_ok=True)

    if list(ravdess_dir.rglob("*.wav")):
        print(f"\n⚡ RAVDESS already found in {ravdess_dir} — skipping download.")
        return

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          Downloading RAVDESS Dataset                  ║")
    print("║  Source: https://zenodo.org/record/1188976            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("""
  ⚠️  The RAVDESS Audio_Speech dataset is ~24 MB.
  It contains 1,440 audio files across 24 actors with 8 emotions.
  Actors 01-24 are included (12 male, 12 female).
    """)

    zip_url  = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = str(DATA_DIR / "ravdess_speech.zip")

    try:
        download_file(zip_url, zip_path)
        extract_zip(zip_path, str(ravdess_dir))
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("""
  Manual Download Instructions:
  1. Visit: https://zenodo.org/record/1188976
  2. Download: Audio_Speech_Actors_01-24.zip  (~24 MB)
  3. Extract to: data/RAVDESS/
  4. Structure should be:
       data/RAVDESS/Actor_01/03-01-*.wav
       data/RAVDESS/Actor_02/...
        """)
        return

    wav_files = list(ravdess_dir.rglob("*.wav"))
    print(f"\n  ✅ RAVDESS ready! {len(wav_files)} audio files in {ravdess_dir}")


def download_tess():
    """
    Download TESS dataset.

    TESS Structure:
        data/TESS/
            OAF_angry/   ← OAF = Older Adult Female
                OAF_back_angry.wav
                ...
            OAF_disgust/
            YAF_happy/   ← YAF = Younger Adult Female
            ...
    """
    tess_dir = DATA_DIR / "TESS"
    tess_dir.mkdir(exist_ok=True)

    if list(tess_dir.rglob("*.wav")):
        print(f"\n⚡ TESS already found in {tess_dir} — skipping download.")
        return

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          Downloading TESS Dataset                     ║")
    print("║  Source: U of Toronto Dataverse                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("""
  ⚠️  TESS (Toronto Emotional Speech Set) contains 2,800 audio files.
  Two actresses, 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
    """)

    print("""
  TESS requires manual download (institutional auth required):
  1. Visit: https://tspace.library.utoronto.ca/handle/1807/24487
  2. Click "Download all files" or download individual zip files
  3. Extract to: data/TESS/
  4. Structure should be:
       data/TESS/OAF_angry/*.wav
       data/TESS/OAF_disgust/*.wav
       data/TESS/YAF_happy/*.wav
       ...

  Alternative (Kaggle mirror):
  1. Install Kaggle CLI: pip install kaggle
  2. Run: kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
  3. Extract to: data/TESS/
    """)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download emotion detection datasets")
    parser.add_argument("--dataset", default="ravdess", choices=["ravdess", "tess", "both"])
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║     Speech Emotion Detection — Dataset Downloader    ║")
    print("╚══════════════════════════════════════════════════════╝")

    if args.dataset in ("ravdess", "both"):
        download_ravdess()

    if args.dataset in ("tess", "both"):
        download_tess()

    print("\n\n  Next steps:")
    print("  1. Train ML models:  python training/train_ml_models.py --data_dir data/RAVDESS")
    print("  2. Train DL models:  python training/train_dl_model.py  --data_dir data/RAVDESS")
    print("  3. Or run demo:      python training/train_demo_model.py")
    print("  4. Launch app:       streamlit run app.py")


if __name__ == "__main__":
    main()
