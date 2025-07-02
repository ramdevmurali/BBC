from pathlib import Path

# --- Project Root ---
PROJECT_ROOT = Path(__file__).parent.parent

# --- Data Paths ---
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bbc"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "outputs"

PROCESSED_DATA_FILE = PROCESSED_DATA_PATH / "bbc_data.csv"

# --- Model Configurations ---
ZERO_SHOT_CLASSIFIER_MODEL = "valhalla/distilbart-mnli-12-3"
NER_MODEL = "dslim/bert-base-NER"
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

# --- Task-Specific Configurations ---
SUB_CATEGORY_LABELS = {
    "business": [
        "stock market",
        "company news",
        "mergers and acquisitions",
        "economy",
        "corporate earnings"
    ],
    "entertainment": [
        "movie review",
        "music news",
        "celebrity gossip",
        "theatre",
        "literature",
        "awards ceremony"
    ],
    "sport": [
        "football",
        "cricket",
        "rugby",
        "tennis",
        "athletics",
        "olympics"
    ],
    "politics": [
        "uk politics",
        "world politics",
        "elections",
        "government policy"
    ],
    "tech": [
        "mobile technology",
        "software and services",
        "gadgets",
        "internet culture",
        "cybersecurity"
    ]
}

TARGET_MONTH = "April"