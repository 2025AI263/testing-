"""Central configuration for the YouTube Comment Toxicity Analyzer."""
import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file (which must never be committed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- YouTube Data API ---
# Put the key in .env as  YOUTUBE_API_KEY=...  (see .env.example). Never hard-code it here.
API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# Video IDs and/or channel handles (@name) / channel IDs (UC...) to collect from.
VIDEO_IDS = ["dQw4w9WgXcQ"]
CHANNELS = ["@mkbhd", "@MrBeast"]   # leave empty to skip channel discovery

MAX_COMMENTS_PER_VIDEO = 500        # safety cap per video
MAX_VIDEOS_PER_CHANNEL = 25         # most recent N videos per channel

# --- Storage ---
DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_PATH   = os.path.join(DATA_DIR, "raw_comments.jsonl")      # extract -> transform
CLEAN_CSV  = os.path.join(DATA_DIR, "comments_clean.csv")      # load output (used by the dashboard)
DB_PATH    = os.path.join(DATA_DIR, "youtube_toxicity.db")     # load output
REPORT_DIR = os.path.join(DATA_DIR, "reports")

# --- Privacy ---
# Commenter handles are personal data. They are dropped during transform unless you opt in.
KEEP_AUTHORS = os.getenv("KEEP_AUTHORS", "0") == "1"

# --- Toxicity model ---
# "lexicon" (offline keyword scorer, zero deps) or "transformer" (detoxify; needs torch).
TOXICITY_BACKEND = os.getenv("TOXICITY_BACKEND", "lexicon")
TOXIC_THRESHOLD  = 0.5
