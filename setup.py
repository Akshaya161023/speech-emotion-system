"""
setup.py
========
One-command setup script for the Speech Emotion Detection project.

Usage:
    python setup.py

This will:
  1. Check Python version
  2. Install all requirements
  3. Check for ffmpeg
  4. Create necessary directories
  5. Train demo models with synthetic data
  6. Launch the Streamlit app
"""

import sys
import os
import subprocess
import platform
from pathlib import Path


def run_cmd(cmd: list, check: bool = True) -> bool:
    """Run a shell command and return success/failure."""
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(f"      {e.stderr.strip()[:200]}")
        return False


def check_python_version():
    """Ensure Python 3.8+."""
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
    if version < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        sys.exit(1)
    print("  ✅ Python version OK")


def check_ffmpeg():
    """Check if ffmpeg is installed (needed by librosa and whisper)."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ ffmpeg found")
            return True
    except FileNotFoundError:
        pass

    print("  ⚠️  ffmpeg NOT found — some audio formats may not work.")

    os_name = platform.system()
    if os_name == "Windows":
        print("""
  Install ffmpeg on Windows:
    Option 1: winget install Gyan.FFmpeg
    Option 2: choco install ffmpeg
    Option 3: Download from https://ffmpeg.org/download.html
              and add to PATH
        """)
    elif os_name == "Darwin":
        print("  Install: brew install ffmpeg")
    else:
        print("  Install: sudo apt install ffmpeg  (Ubuntu/Debian)")

    return False


def create_directories():
    """Create required project directories."""
    dirs = ["models", "data/RAVDESS", "data/TESS", "notebooks"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("  ✅ Directories created")


def install_requirements():
    """Install Python dependencies from requirements.txt."""
    print("  📦 Installing Python packages... (this may take a few minutes)")
    success = run_cmd([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
        "--upgrade", "--quiet"
    ])
    if success:
        print("  ✅ Packages installed")
    else:
        print("  ⚠️  Some packages failed. Try manually: pip install -r requirements.txt")
    return success


def train_demo_models():
    """Train quick demo models with synthetic data."""
    print("  🤖 Training demo emotion models (synthetic data)...")
    success = run_cmd([sys.executable, "training/train_demo_model.py"])
    if success:
        print("  ✅ Demo models trained")
    else:
        print("  ⚠️  Model training failed. Run manually: python training/train_demo_model.py")
    return success


def launch_app():
    """Launch the Streamlit app."""
    print("\n  🚀 Launching Streamlit app...")
    print("  → Open http://localhost:8501 in your browser\n")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n  👋 App stopped.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║    Speech Emotion Detection — Project Setup          ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Change to project root
    os.chdir(Path(__file__).parent)

    steps = [
        ("Checking Python version",   check_python_version),
        ("Creating directories",      create_directories),
        ("Installing requirements",   install_requirements),
        ("Checking ffmpeg",           check_ffmpeg),
        ("Training demo models",      train_demo_models),
    ]

    for i, (label, fn) in enumerate(steps, 1):
        print(f"\nStep {i}/{len(steps)}: {label}")
        print("─" * 50)
        fn()

    print("\n" + "═" * 54)
    print("  ✅ Setup complete!")
    print("═" * 54)
    print("""
  Next steps:
  ─────────────────────────────────────────────────────
  1. Launch the app:
       streamlit run app.py

  2. Train on real RAVDESS data (optional, more accurate):
       python training/download_dataset.py --dataset ravdess
       python training/train_ml_models.py --data_dir data/RAVDESS

  3. Train deep learning model (optional):
       python training/train_dl_model.py --data_dir data/RAVDESS
  ─────────────────────────────────────────────────────
  """)

    launch = input("  Launch app now? [Y/n]: ").strip().lower()
    if launch in ("", "y", "yes"):
        launch_app()


if __name__ == "__main__":
    main()
