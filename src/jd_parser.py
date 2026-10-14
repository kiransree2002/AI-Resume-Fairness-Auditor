"""
Job Description Parser & Library Manager (Production v2)
========================================================
Handles:
1. Live web scraping of Job Descriptions from URLs
2. Parsing and structuring requirements into editable JSON format
3. Dual-path importance scoring (Explicit clue detection + Domain logic fallback)
4. Persistent JD Library management (Save, Load, List saved JDs)

Team: AI/ML (Week 1 Rebuild Deliverable)
"""

import json
import logging
import re
import urllib3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from config import (
    SAVED_JDS_DIR,
    DEFAULT_GATE_ORDER,
    DEFAULT_GATE_WEIGHTS
)
from text_cleaner import clean_text
from skill_extractor import extract_skills, extract_experience_years, extract_education

# Disable SSL warnings for local Windows environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
# 1. IMPORTANCE CLUE WEIGHTS & DOMAIN LOGIC
# -----------------------------------------------------------
CLUE_WEIGHTS = {
    # High Importance (Must Haves: 0.85 - 1.0)
    "must": 0.95, "required": 0.95, "mandatory": 0.95, "essential": 0.90,
    "proficient": 0.90, "expert": 0.90, "strong experience": 0.85, "proven track record": 0.85,

    # Medium Importance (Preferred: 0.55 - 0.75)
    "preferred": 0.70, "should have": 0.65, "good to have": 0.65,
    "experience with": 0.60, "familiar with": 0.55, "advantage": 0.60,

    # Low Importance (Nice to Have: 0.20 - 0.40)
    "nice to have": 0.35, "bonus": 0.30, "plus": 0.30,
    "familiarity": 0.30, "basic knowledge": 0.25, "exposure to": 0.25
}

ROLE_SKILL_PRIORITIES = {
    "data science": {
        "core": ["Python", "Machine Learning", "Sql", "Statistics", "Scikit-Learn", "Pandas"],
        "preferred": ["Deep Learning", "Pytorch", "Tensorflow", "Aws", "Tableau", "Git"],
        "nice_to_have": ["Docker", "Kubernetes", "Flask", "Fastapi", "Spark"]
    },
    "data analyst": {
        "core": ["Sql", "Excel", "Tableau", "Power Bi", "Python", "Data Analysis"],
        "preferred": ["R", "Statistics", "Pandas", "Statistical Analysis"],
        "nice_to_have": ["Aws", "Snowflake", "Bigquery", "Machine Learning"]
    },
    "software engineer": {
        "core": ["Python", "Java", "C++", "Git", "Rest Api", "Sql"],
        "preferred": ["Docker", "Linux", "Ci/Cd", "Aws", "Fastapi", "Django"],
        "nice_to_have": ["Kubernetes", "Redis", "Mongodb"]
    }
}


# -----------------------------------------------------------
# 2. WEB SCRAPER (ROBUST)
# -----------------------------------------------------------

def scrape_job_url(url: str) -> str:
    """
    Fetches raw text content from a public job posting URL.
    Strips scripts, styles, navigation, headers, and footers.
    Includes robust error handling for Windows SSL/network glitches.
    """
    if not url or not url.startswith("http"):
        logger.warning(f"Invalid URL provided: {url}")
        return ""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive"
    }

    try:
        logger.info(f"Connecting to web link: {url}")
        response = requests.get(url, headers=headers, timeout=15, verify=False)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
                tag.extract()
            text = soup.get_text(separator=" ")
            logger.info(f"Successfully scraped {len(text)} raw characters from URL.")
            return text
        else:
            logger.warning(f"Webpage returned HTTP status code: {response.status_code}")
            return ""
    except requests.exceptions.Timeout:
        logger.error("Request timed out after 15 seconds.")
        return ""
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Network connection error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Scraping failed with error: {type(e).__name__}: {e}")
        return ""


# -----------------------------------------------------------
# 3. IMPORTANCE INFERENCE ENGINE
# -----------------------------------------------------------

def infer_skill_importance(skill: str, full_raw_text: str, role_title: str) -> Dict[str, Any]:
    """
    Infers the numerical importance weight (0.0 to 1.0) and reason for a skill.
    Path A: Check surrounding sentence for explicit clue words.
    Path B: Fall back to domain priority logic based on job title.
    """
    sentences = re.split(r'[.\n•\-\;]', full_raw_text)
    skill_lower = skill.lower()

    # Path A: Explicit Clue Words
    for sent in sentences:
        sent_lower = sent.lower()
        if skill_lower in sent_lower:
            for clue, weight in CLUE_WEIGHTS.items():
                if clue in sent_lower:
                    category = "Must-Have" if weight >= 0.8 else ("Preferred" if weight >= 0.5 else "Nice-to-Have")
                    return {
                        "skill": skill,
                        "importance": weight,
                        "category": category,
                        "source": f"Explicit clue ('{clue}')"
                    }

    # Path B: Domain Logic Fallback
    matched_domain = "data science"
    for r in ROLE_SKILL_PRIORITIES.keys():
        if r in role_title.lower() or r in full_raw_text[:300].lower():
            matched_domain = r
            break

    priorities = ROLE_SKILL_PRIORITIES[matched_domain]

    if any(skill.lower() == s.lower() for s in priorities["core"]):
        return {
            "skill": skill,
            "importance": 0.90,
            "category": "Must-Have",
            "source": f"Domain Logic (Core for {matched_domain.title()})"
        }
    elif any(skill.lower() == s.lower() for s in priorities["preferred"]):
        return {
            "skill": skill,
            "importance": 0.65,
            "category": "Preferred",
            "source": f"Domain Logic (Preferred for {matched_domain.title()})"
        }
    else:
        return {
            "skill": skill,
            "importance": 0.35,
            "category": "Nice-to-Have",
            "source": f"Domain Logic (Supplementary for {matched_domain.title()})"
        }


# -----------------------------------------------------------
# 4. STRUCTURED JD GENERATOR (EDITABLE FORMAT)
# -----------------------------------------------------------

def parse_job_description(
    raw_or_scraped_text: str,
    role_title: str = "Unspecified Role",
    company_name: str = "Company",
    source_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Converts raw JD text into an HR-editable structured JSON schema.
    """
    cleaned = clean_text(raw_or_scraped_text)

    skills = extract_skills(cleaned)
    exp_years = extract_experience_years(cleaned)
    education = extract_education(cleaned)

    weighted_skills = [
        infer_skill_importance(skill, raw_or_scraped_text, role_title)
        for skill in skills
    ]

    must_skills = [s for s in weighted_skills if s["category"] == "Must-Have"]
    preferred_skills = [s for s in weighted_skills if s["category"] in ["Preferred", "Nice-to-Have"]]

    structured_jd = {
        "metadata": {
            "job_title": role_title,
            "company_name": company_name,
            "source_url": source_url if source_url else "Direct Text / Dataset",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "gate_configuration": {
            "gate_order": list(DEFAULT_GATE_ORDER),
            "gate_weights": dict(DEFAULT_GATE_WEIGHTS)
        },
        "requirements": {
            "skills": {
                "must_have": must_skills,
                "preferred": preferred_skills,
                "total_skills_detected": len(skills)
            },
            "education": {
                "required_degrees": education if education else ["Bachelor's"],
                "importance": 0.85
            },
            "experience": {
                "min_years": exp_years if exp_years > 0 else 2.0,
                "importance": 0.80
            }
        },
        "raw_character_count": len(raw_or_scraped_text)
    }

    return structured_jd


# -----------------------------------------------------------
# 5. JD LIBRARY MANAGEMENT (SAVE, LOAD, LIST)
# -----------------------------------------------------------

def save_jd(jd_data: Dict[str, Any], custom_name: Optional[str] = None) -> Path:
    """
    Saves a structured JD to the saved_jds/ folder for future reuse.
    """
    title = custom_name if custom_name else jd_data.get("metadata", {}).get("job_title", "custom_job")
    safe_filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', title.lower().strip()) + ".json"
    file_path = SAVED_JDS_DIR / safe_filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(jd_data, f, indent=2)

    logger.info(f"Saved Job Description to library: {file_path}")
    return file_path


def load_jd(filename_or_title: str) -> Optional[Dict[str, Any]]:
    """
    Loads a saved Job Description JSON from the saved_jds/ folder.
    """
    if not filename_or_title.endswith(".json"):
        safe_filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename_or_title.lower().strip()) + ".json"
    else:
        safe_filename = filename_or_title

    file_path = SAVED_JDS_DIR / safe_filename

    if not file_path.exists():
        logger.error(f"Saved JD not found at: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        jd_data = json.load(f)

    logger.info(f"Loaded Job Description from library: {file_path}")
    return jd_data


def list_saved_jds() -> List[Dict[str, Any]]:
    """
    Lists all saved Job Descriptions stored in the saved_jds/ folder.
    """
    saved_files = list(SAVED_JDS_DIR.glob("*.json"))
    summary = []

    for fpath in saved_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                meta = data.get("metadata", {})
                reqs = data.get("requirements", {})
                summary.append({
                    "filename": fpath.name,
                    "title": meta.get("job_title", "Unknown Title"),
                    "company": meta.get("company_name", "Unknown Company"),
                    "created_at": meta.get("created_at", "N/A"),
                    "must_skills_count": len(reqs.get("skills", {}).get("must_have", [])),
                    "min_experience_years": reqs.get("experience", {}).get("min_years", 0)
                })
        except Exception as e:
            logger.warning(f"Could not read {fpath.name}: {e}")

    return summary


# -----------------------------------------------------------
# ENTRY POINT / UNIT TEST
# -----------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING JD PARSER & LIBRARY MANAGER")
    logger.info("=" * 60)

    sample_hr_text = """
    Job Title: Senior Data Scientist
    Company: Outlier AI
    
    About the role:
    We are seeking a Senior Data Scientist to train advanced AI models.
    
    Requirements:
    • Must be proficient in Python, SQL, and Machine Learning.
    • Bachelor's or Master's degree in Computer Science or related domain.
    • At least 3+ years of professional experience in data analysis and modeling.
    • Experience with AWS, PyTorch, and Git is preferred.
    • Familiarity with Docker is a plus.
    """

    # Test parsing
    structured = parse_job_description(
        raw_or_scraped_text=sample_hr_text,
        role_title="Senior Data Scientist",
        company_name="Outlier AI"
    )

    saved_path = save_jd(structured, custom_name="Senior Data Scientist Outlier")

    saved_library = list_saved_jds()
    logger.info(f"\nCurrent Saved JDs in Library ({len(saved_library)}):")
    for item in saved_library:
        logger.info(f"  📁 {item['filename']} | {item['title']} ({item['company']}) | Created: {item['created_at']}")

    logger.info("\nStructured JSON Preview:")
    logger.info(json.dumps(structured, indent=2))
    logger.info("=" * 60)