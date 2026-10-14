"""
Skill & Entity Extraction Engine (Production)
=============================================
Extracts skills, experience years, and education degrees from any text
using spaCy NLP + custom PhraseMatcher + regex pattern matching.

Used by both the JD pipeline and Resume pipeline for consistent extraction.

Team: AI/ML (Week 1 Core Deliverable)
"""

import spacy
from spacy.matcher import PhraseMatcher
import re
from typing import Dict, List, Any

# -----------------------------------------------------------
# LOAD SPACY MODEL
# -----------------------------------------------------------
nlp = spacy.load("en_core_web_sm")


# -----------------------------------------------------------
# SKILL TAXONOMY (Technical + Soft Skills)
# -----------------------------------------------------------
SKILL_TAXONOMY = [
    # Programming Languages
    "python", "r", "java", "c", "c++", "c#", "javascript", "typescript",
    "php", "ruby", "swift", "kotlin", "scala", "go", "rust", "sql",
    "pl/sql", "bash", "powershell",

    # Data Science / AI / ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "artificial intelligence", "data analysis",
    "data analytics", "data science", "data mining", "predictive modeling",
    "statistics", "statistical analysis", "scikit-learn", "tensorflow",
    "keras", "pytorch", "pandas", "numpy", "scipy", "xgboost", "lightgbm",
    "spacy", "nltk", "hugging face", "transformers", "bert", "llm",
    "large language models", "opencv",

    # Data Visualization & BI
    "tableau", "power bi", "matplotlib", "seaborn", "plotly", "excel",
    "advanced excel", "looker", "qlik", "d3.js",

    # Big Data & Cloud Platforms
    "aws", "azure", "gcp", "google cloud platform", "hadoop", "spark",
    "pyspark", "kafka", "snowflake", "bigquery", "redshift", "hive",
    "docker", "kubernetes", "airflow", "databricks",

    # Web & Databases
    "django", "flask", "fastapi", "react", "node.js", "angular", "vue",
    "html", "css", "postgresql", "mysql", "mongodb", "sqlite", "oracle",
    "nosql", "redis", "elasticsearch",

    # Developer Tools & Methodologies
    "git", "github", "gitlab", "ci/cd", "agile", "scrum", "jira", "linux",
    "rest api", "api development", "unit testing",

    # Business & Soft Skills
    "communication", "team leadership", "leadership", "project management",
    "critical thinking", "problem solving", "stakeholder management",
    "collaboration", "time management", "presentation skills"
]


# -----------------------------------------------------------
# EDUCATION KEYWORD MAPPING
# -----------------------------------------------------------
EDUCATION_KEYWORDS = {
    "Doctorate": ["phd", "ph.d", "doctorate", "doctoral"],
    "Master's": [
        "master", "masters", "msc", "m.sc", "mtech", "m.tech", "mba",
        "m.b.a", "mca", "m.c.a", "ms", "m.s", "post graduate",
        "postgraduate", "pg"
    ],
    "Bachelor's": [
        "bachelor", "bachelors", "bsc", "b.sc", "btech", "b.tech", "bca",
        "b.c.a", "be", "b.e", "bba", "b.b.a", "ba", "b.a", "undergraduate",
        "ug"
    ],
    "Diploma": ["diploma", "associate degree", "polytechnic"]
}


# -----------------------------------------------------------
# INITIALIZE PHRASEMATCHER
# -----------------------------------------------------------
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in SKILL_TAXONOMY]
matcher.add("SKILLS", patterns)


# -----------------------------------------------------------
# SKILL EXTRACTION
# -----------------------------------------------------------
def extract_skills(text: str) -> List[str]:
    """
    Extracts matched skills from cleaned text using spaCy PhraseMatcher.

    Args:
        text: Cleaned text string.

    Returns:
        Unique, title-cased list of detected skills.
    """
    if not text:
        return []

    doc = nlp(text.lower())
    matches = matcher(doc)

    extracted = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        extracted.add(span.text.title())

    return sorted(list(extracted))


# -----------------------------------------------------------
# EXPERIENCE EXTRACTION (WITH REALISTIC VALIDATION)
# -----------------------------------------------------------
def extract_experience_years(text: str) -> float:
    """
    Extracts stated years of experience from text using regex.
    Ignores calendar years (e.g., 1998, 2000, 2016, 2024) and
    caps realistic experience at 45 years.

    Args:
        text: Cleaned text string.

    Returns:
        Maximum validated experience in years, or 0.0 if none found.
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
                # IGNORE calendar years (e.g., 1990-2026) and unrealistic values (>45)
                if 0.5 <= val <= 45.0:
                    found_years.append(val)
            except ValueError:
                continue

    return max(found_years) if found_years else 0.0


# -----------------------------------------------------------
# EDUCATION EXTRACTION
# -----------------------------------------------------------
def extract_education(text: str) -> List[str]:
    """
    Identifies degree tiers (Doctorate, Master's, Bachelor's, Diploma) present in text.

    Args:
        text: Cleaned text string.

    Returns:
        Sorted list of detected degree levels.
    """
    if not text:
        return []

    text_lower = text.lower()
    detected_degrees = set()

    for degree_level, keywords in EDUCATION_KEYWORDS.items():
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                detected_degrees.add(degree_level)
                break

    return sorted(list(detected_degrees))


# -----------------------------------------------------------
# WRAPPER FUNCTION
# -----------------------------------------------------------
def extract_profile(text: str) -> Dict[str, Any]:
    """
    Runs all extraction components on input text.

    Args:
        text: Cleaned text string.

    Returns:
        Dictionary with 'skills', 'experience_years', 'education'.
    """
    return {
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text)
    }


# -----------------------------------------------------------
# ENTRY POINT (UNIT TEST)
# -----------------------------------------------------------
if __name__ == "__main__":
    sample_text = """
    Senior Data Scientist with 4+ years of experience in Machine Learning, Python, and SQL.
    Hold a Master's degree in Computer Science and a Bachelor's in Mathematics.
    Hands-on expertise in PyTorch, Scikit-Learn, Power BI, AWS, and Git.
    Strong communication and Agile team leadership abilities.
    Graduated in 2000. Joined first job in 2005.
    """

    print("=" * 60)
    print("🧠 TESTING SKILL & ENTITY EXTRACTION")
    print("=" * 60)
    print("INPUT TEXT:\n", sample_text.strip())
    print("-" * 60)
    profile = extract_profile(sample_text)
    print("EXTRACTED PROFILE:")
    print(f"✅ Skills ({len(profile['skills'])}): {profile['skills']}")
    print(f"✅ Experience: {profile['experience_years']} years")
    print(f"✅ Education Levels: {profile['education']}")
    print("=" * 60)