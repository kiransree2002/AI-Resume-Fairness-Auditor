"""
Project Configuration (v2 — Post Interviewer Feedback)
======================================================
Central location for all file paths, constants, and settings.
Change values here once — every module reads from this file.
"""

from pathlib import Path

# -----------------------------------------------------------
# DIRECTORY PATHS (relative to project root)
# -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SAVED_JDS_DIR = PROJECT_ROOT / "saved_jds"  # NEW: HR JD Library

# -----------------------------------------------------------
# DATASET FILENAMES
# -----------------------------------------------------------
RESUME_CSV = DATA_DIR / "Resume.csv"
POSTINGS_CSV = DATA_DIR / "postings.csv"

# -----------------------------------------------------------
# JD INPUT MODES (How HR provides a Job Description)
# -----------------------------------------------------------
JD_MODE_URL = "url"          # HR pastes a web link
JD_MODE_TEXT = "text"        # HR types/pastes JD text manually
JD_MODE_SAVED = "saved"      # HR selects from saved JD library

# -----------------------------------------------------------
# DEFAULT GATE CONFIGURATION (Interviewer's Preferred Order)
# -----------------------------------------------------------
# NEW: Skills first, then Education, then Experience
# HR can override this per job role via the editable review screen
DEFAULT_GATE_ORDER = ["skills", "education", "experience"]

DEFAULT_GATE_WEIGHTS = {
    "skills": 0.50,       # Skills matter most for tech roles
    "education": 0.25,    # Degree is secondary filter
    "experience": 0.25    # Years of experience is secondary filter
}

# -----------------------------------------------------------
# SCORING SETTINGS
# -----------------------------------------------------------
RANDOM_SEED = 42
MAX_RESUMES_FOR_TESTING = 50  # Use None to process all 2,484
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# -----------------------------------------------------------
# COMBINED SCORE WEIGHTS
# -----------------------------------------------------------
GATE_WEIGHT = 0.70
TFIDF_WEIGHT = 0.30

# -----------------------------------------------------------
# SHORTLIST THRESHOLDS
# -----------------------------------------------------------
SHORTLIST_THRESHOLD = 70
MAYBE_THRESHOLD = 50

# -----------------------------------------------------------
# ENSURE OUTPUT DIRECTORIES EXIST
# -----------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVED_JDS_DIR.mkdir(parents=True, exist_ok=True)  # NEW