import spacy
from spacy.matcher import PhraseMatcher
import re
from typing import Dict, List, Any

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Comprehensive taxonomy of skills (Technical, Analytical, Tools, Soft Skills)
SKILL_TAXONOMY = [
    # Programming Languages
    "python", "r", "java", "c", "c++", "c#", "javascript", "typescript", "php", "ruby", "swift", "kotlin", "scala", "go", "rust", "sql", "pl/sql", "bash", "powershell",
    
    # Data Science / AI / ML
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision", "artificial intelligence", 
    "data analysis", "data analytics", "data science", "data mining", "predictive modeling", "statistics", "statistical analysis",
    "scikit-learn", "tensorflow", "keras", "pytorch", "pandas", "numpy", "scipy", "xgboost", "lightgbm", "spacy", "nltk", "hugging face", "transformers", "bert", "llm", "large language models", "opencv",
    
    # Data Visualization & BI
    "tableau", "power bi", "matplotlib", "seaborn", "plotly", "excel", "advanced excel", "looker", "qlik", "d3.js",
    
    # Big Data & Cloud Platforms
    "aws", "azure", "gcp", "google cloud platform", "hadoop", "spark", "pyspark", "kafka", "snowflake", "bigquery", "redshift", "hive", "docker", "kubernetes", "airflow", "databricks",
    
    # Web & Databases
    "django", "flask", "fastapi", "react", "node.js", "angular", "vue", "html", "css",
    "postgresql", "mysql", "mongodb", "sqlite", "oracle", "nosql", "redis", "elasticsearch",
    
    # Developer Tools & Methodologies
    "git", "github", "gitlab", "ci/cd", "agile", "scrum", "jira", "linux", "rest api", "api development", "unit testing",
    
    # Business & Soft Skills
    "communication", "team leadership", "leadership", "project management", "critical thinking", "problem solving", "stakeholder management", "collaboration", "time management", "presentation skills"
]

# Standard Education Degree Keywords
EDUCATION_KEYWORDS = {
    "Doctorate": ["phd", "ph.d", "doctorate", "doctoral"],
    "Master's": ["master", "masters", "msc", "m.sc", "mtech", "m.tech", "mba", "m.b.a", "mca", "m.c.a", "ms", "m.s", "post graduate", "postgraduate", "pg"],
    "Bachelor's": ["bachelor", "bachelors", "bsc", "b.sc", "btech", "b.tech", "bca", "b.c.a", "be", "b.e", "bba", "b.b.a", "ba", "b.a", "undergraduate", "ug"],
    "Diploma": ["diploma", "associate degree", "polytechnic"]
}

# Initialize PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in SKILL_TAXONOMY]
matcher.add("SKILLS", patterns)


def extract_skills(text: str) -> List[str]:
    """
    Extracts matched skills from cleaned text using spaCy PhraseMatcher.
    Returns a unique, title-cased list of skills.
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


def extract_experience_years(text: str) -> float:
    """
    Extracts total or minimum years of experience mentioned in the text using regex.
    """
    if not text:
        return 0.0
    
    # Patterns like "5+ years", "3 to 5 yrs", "2.5 years", "10 years of experience"
    patterns = [
        r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience',
        r'experience\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)',
        r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)'
    ]
    
    found_years = []
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for m in matches:
            try:
                found_years.append(float(m))
            except ValueError:
                continue
                
    # Return the maximum explicit years number found, or 0.0
    return max(found_years) if found_years else 0.0


def extract_education(text: str) -> List[str]:
    """
    Identifies degree tiers (Doctorate, Master's, Bachelor's, Diploma) present in text.
    """
    if not text:
        return []
    
    text_lower = text.lower()
    detected_degrees = set()
    
    for degree_level, keywords in EDUCATION_KEYWORDS.items():
        for kw in keywords:
            # Match standalone keywords
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                detected_degrees.add(degree_level)
                break
                
    return sorted(list(detected_degrees))


def extract_profile(text: str) -> Dict[str, Any]:
    """
    Wrapper function that runs all extraction components on input text.
    """
    return {
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text)
    }


if __name__ == "__main__":
    sample_text = """
    Senior Data Scientist with 4+ years of experience in Machine Learning, Python, and SQL.
    Hold a Master's degree in Computer Science and a Bachelor's in Mathematics.
    Hands-on expertise in PyTorch, Scikit-Learn, Power BI, AWS, and Git.
    Strong communication and Agile team leadership abilities.
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