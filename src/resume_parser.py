"""
Resume Parsing Pipeline (Production)
====================================
Processes real resume text into standardized structured JSON profiles.
Extracts real entities only (Skills, Experience, Education, Category).
Zero synthetic data.

Team: AI/ML (Week 1 Core Deliverable)
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from config import OUTPUT_DIR
from text_cleaner import clean_text
from skill_extractor import extract_skills, extract_education

# -----------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------
# REALISTIC EXPERIENCE PARSER (WITH VALIDATION)
# -----------------------------------------------------------

def extract_validated_experience(text: str) -> float:
    """
    Extracts stated experience years while ignoring calendar years (e.g., 2018, 1999).
    Caps realistic experience to between 0 and 45 years.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    patterns = [
        r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience',
        r'experience\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)',
        r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+in\b'
    ]

    found_years = []
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for m in matches:
            try:
                val = float(m)
                # Ignore calendar years like 1990-2026 and unrealistic numbers
                if 0.5 <= val <= 45.0:
                    found_years.append(val)
            except ValueError:
                continue

    return max(found_years) if found_years else 0.0


# -----------------------------------------------------------
# SINGLE RESUME PARSER
# -----------------------------------------------------------

def parse_resume(
    resume_id: Any,
    raw_text: str,
    category: str = "Unspecified"
) -> Dict[str, Any]:
    """
    Parses a single real resume into a structured profile.

    Args:
        resume_id: Unique identifier for the candidate.
        raw_text: Raw or uncleaned resume string.
        category: Job category/domain label.

    Returns:
        Structured dictionary profile containing only real extracted data.
    """
    cleaned = clean_text(raw_text)

    skills = extract_skills(cleaned)
    education = extract_education(cleaned)
    experience_years = extract_validated_experience(cleaned)

    # Soft skills list to separate from hard technical skills
    soft_skills_set = {
        "Communication", "Leadership", "Team Leadership", "Problem Solving",
        "Critical Thinking", "Time Management", "Collaboration",
        "Presentation Skills", "Project Management", "Stakeholder Management",
        "Agile", "Scrum"
    }

    technical_skills = [s for s in skills if s not in soft_skills_set]
    soft_skills = [s for s in skills if s in soft_skills_set]

    structured_profile = {
        "candidate_id": f"RES_{resume_id}",
        "category": category,
        "education": {
            "degrees": education,
            "has_higher_education": len(education) > 0
        },
        "experience": {
            "years": experience_years
        },
        "skills": {
            "technical": technical_skills,
            "soft": soft_skills,
            "total_count": len(skills)
        },
        "raw_character_count": len(raw_text) if isinstance(raw_text, str) else 0
    }

    return structured_profile


# -----------------------------------------------------------
# BATCH RESUME PARSER
# -----------------------------------------------------------

def parse_all_resumes(
    cleaned_csv_path: Optional[Path] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Parses all resumes from the cleaned CSV dataset.

    Args:
        cleaned_csv_path: Path to resumes_cleaned.csv.
        limit: Optional maximum number of resumes to parse.

    Returns:
        List of structured JSON profiles.
    """
    if cleaned_csv_path is None:
        cleaned_csv_path = OUTPUT_DIR / "resumes_cleaned.csv"

    if not cleaned_csv_path.exists():
        logger.error(f"File not found: {cleaned_csv_path}. Run data_cleaner.py first.")
        return []

    logger.info(f"Loading cleaned data from: {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path)

    if limit:
        df = df.head(limit)
        logger.info(f"Limiting parse to first {limit} resumes for testing.")

    profiles = []
    total = len(df)
    logger.info(f"Parsing {total} real resumes into structured JSON...")

    for idx, row in df.iterrows():
        p = parse_resume(
            resume_id=row["id"],
            raw_text=row["cleaned_text"],
            category=row["category_clean"]
        )
        profiles.append(p)

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            logger.info(f"  Parsed {idx + 1}/{total} resumes...")

    return profiles


# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING PRODUCTION RESUME PARSER")
    logger.info("=" * 60)

    # Test on the first 5 real resumes
    sample_profiles = parse_all_resumes(limit=5)

    logger.info(f"\nSample Extracted Profile #1 (Real Data):")
    logger.info(json.dumps(sample_profiles[0], indent=2))

    logger.info(f"\nSample Extracted Profile #2 (Real Data):")
    logger.info(json.dumps(sample_profiles[1], indent=2))

    # Save test batch to output
    sample_out = OUTPUT_DIR / "sample_parsed_resumes.json"
    with open(sample_out, "w", encoding="utf-8") as f:
        json.dump(sample_profiles, f, indent=2)

    logger.info(f"\nSaved test profiles to: {sample_out}")
    logger.info("=" * 60)