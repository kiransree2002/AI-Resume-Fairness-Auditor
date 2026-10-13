"""
Data Cleaning Pipeline
======================
Loads the raw Kaggle Resume dataset (Resume.csv), cleans all text fields,
and saves a production-ready cleaned CSV for downstream use.

This script processes ALL 2,484 real resumes — no sampling, no synthetic data.

Team: DS/DA + AI/ML (Joint Week 1 Deliverable)
"""

import logging
import re
import html
import unicodedata
from pathlib import Path

import pandas as pd

from config import RESUME_CSV, OUTPUT_DIR

# -----------------------------------------------------------
# LOGGING SETUP (Standard practice: use logging, not print)
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------
# TEXT CLEANING FUNCTIONS
# -----------------------------------------------------------

def clean_resume_text(text: str) -> str:
    """
    Cleans a single raw resume text string.

    Processing steps:
        1. Unescape HTML entities (&amp; → &, &nbsp; → space)
        2. Normalize unicode characters (smart quotes, bullets)
        3. Remove URLs, emails, phone numbers (PII removal)
        4. Remove non-ASCII artifacts (\\xa0, \\u2022, etc.)
        5. Preserve tech tokens: C++, C#, .NET
        6. Collapse excessive whitespace into single spaces

    Args:
        text: Raw resume text from the Resume_str column.

    Returns:
        Cleaned, standardized text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: HTML unescape
    text = html.unescape(text)

    # Step 2: Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Step 3: Remove PII (Personally Identifiable Information)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    text = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' ', text)

    # Step 4: Remove non-ASCII artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Step 5: Remove special chars BUT preserve C++, C#, .NET
    text = re.sub(r'[^\w\s\+\#\.\,\-\/\:]', ' ', text)

    # Step 6: Collapse whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_category_label(category: str) -> str:
    """
    Standardizes category labels from the Kaggle dataset.

    Converts labels like 'INFORMATION-TECHNOLOGY' to 'Information Technology'
    for cleaner display in charts and reports.

    Args:
        category: Raw category string from the Category column.

    Returns:
        Title-cased, human-readable category name.
    """
    if not isinstance(category, str):
        return "Unknown"
    return category.replace("-", " ").title()


# -----------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------

def load_and_clean_resumes() -> pd.DataFrame:
    """
    Loads the full Kaggle Resume dataset and applies all cleaning steps.

    Returns:
        Cleaned DataFrame with columns:
            - id: Original resume ID
            - raw_text: Original uncleaned text (preserved for reference)
            - cleaned_text: Fully cleaned and standardized text
            - category: Original category label
            - category_clean: Human-readable category name
            - text_length: Character count of cleaned text
    """
    # Validate file exists
    if not RESUME_CSV.exists():
        logger.error(f"Dataset not found at: {RESUME_CSV}")
        raise FileNotFoundError(f"Resume dataset not found at {RESUME_CSV}")

    # Load raw data
    logger.info(f"Loading dataset from {RESUME_CSV}...")
    df = pd.read_csv(RESUME_CSV)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values detected:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values found in any column.")

    # Apply cleaning
    logger.info("Cleaning resume text (this may take 30-60 seconds)...")
    df["cleaned_text"] = df["Resume_str"].apply(clean_resume_text)
    df["category_clean"] = df["Category"].apply(clean_category_label)
    df["text_length"] = df["cleaned_text"].str.len()

    # Select and rename columns for downstream use
    df_clean = df[["ID", "Resume_str", "cleaned_text", "Category", "category_clean", "text_length"]].copy()
    df_clean.rename(columns={"ID": "id", "Resume_str": "raw_text", "Category": "category"}, inplace=True)

    # Log summary statistics
    logger.info(f"Cleaning complete. Text length stats:")
    logger.info(f"  Mean:   {df_clean['text_length'].mean():.0f} characters")
    logger.info(f"  Median: {df_clean['text_length'].median():.0f} characters")
    logger.info(f"  Min:    {df_clean['text_length'].min()} characters")
    logger.info(f"  Max:    {df_clean['text_length'].max()} characters")

    return df_clean


def save_cleaned_data(df: pd.DataFrame) -> Path:
    """
    Saves the cleaned DataFrame to the output directory.

    Args:
        df: Cleaned DataFrame from load_and_clean_resumes().

    Returns:
        Path to the saved CSV file.
    """
    output_path = OUTPUT_DIR / "resumes_cleaned.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved cleaned dataset to {output_path} ({len(df)} rows)")
    return output_path


# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STARTING DATA CLEANING PIPELINE")
    logger.info("=" * 60)

    df_cleaned = load_and_clean_resumes()
    saved_path = save_cleaned_data(df_cleaned)

    # Quick preview of first 3 cleaned resumes
    logger.info("\nPreview of first 3 cleaned resumes:")
    for i in range(3):
        row = df_cleaned.iloc[i]
        logger.info(f"\n  [{row['category_clean']}] (ID: {row['id']})")
        logger.info(f"  {row['cleaned_text'][:200]}...")

    logger.info("\n" + "=" * 60)
    logger.info("DATA CLEANING COMPLETE")
    logger.info("=" * 60)