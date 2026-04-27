"""
Central configuration for the Mental Health Sentiment Detection App.
All paths, constants, and settings live here.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
# Safer: dynamically locate project root based on known folder name
ROOT_DIR = Path(__file__).resolve()
while ROOT_DIR.name != "mental-health-sentiment-app":
    ROOT_DIR = ROOT_DIR.parent

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
YOUTUBE_DATA_DIR = DATA_DIR / "youtube"
MODELS_DIR = ROOT_DIR / "models"

# Ensure directories exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, YOUTUBE_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model artefacts ───────────────────────────────────────────────────────────
MODEL_PATH = MODELS_DIR / "logistic_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# ── YouTube API ───────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
MAX_COMMENTS = 100

if not YOUTUBE_API_KEY:
    print("[WARNING] YOUTUBE_API_KEY is not set.")

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP: dict[int, str] = {
    0: "normal",
    1: "sad",
    2: "anxiety",
    3: "depressive indicators",
}

# ── TF-IDF / model hyperparameters ────────────────────────────────────────────
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 20000

LR_MAX_ITER = 1000
LR_C = 1.0
RANDOM_STATE = 42

# ── Interpretation templates ─────────────────────────────────────────────────
INTERPRETATIONS: dict[str, str] = {
    "normal": "No strong negative emotional signals detected.",
    "sad": "This comment shows signs of sadness.",
    "anxiety": "This comment reflects possible stress or worry.",
    "depressive indicators": "This comment shows strong negative emotional signals.",
}