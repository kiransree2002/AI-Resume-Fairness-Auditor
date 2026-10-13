"""
Master Resume Screening Pipeline (Production v3 — Semantic Powered)
====================================================================
Orchestrates the full screening workflow with:
1. Three JD input modes: URL scrape, manual text, saved JD library
2. Editable JD review step before screening
3. Configurable gate order and weights (Skills first by default)
4. Sentence-Transformers (BERT) semantic similarity scoring
5. Generates auditable shortlist outputs (CSV & JSON)

Team: Joint (AI/ML Pipeline + DS/DA Audit Input)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

from config import (
    OUTPUT_DIR,
    POSTINGS_CSV,
    SAVED_JDS_DIR,
    SHORTLIST_THRESHOLD,
    MAYBE_THRESHOLD,
    GATE_WEIGHT,
    TFIDF_WEIGHT,
    DEFAULT_GATE_ORDER,
    DEFAULT_GATE_WEIGHTS,
    JD_MODE_URL,
    JD_MODE_TEXT,
    JD_MODE_SAVED
)
from jd_parser import (
    parse_job_description,
    scrape_job_url,
    save_jd,
    load_jd,
    list_saved_jds
)
from resume_parser import parse_all_resumes
from gate_cascade import run_candidate_cascade
from semantic_scorer import SemanticScorer, calculate_semantic_combined_score

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
# 1. JD INPUT HANDLER (3 Modes)
# -----------------------------------------------------------

def obtain_job_description(
    mode: str = JD_MODE_URL,
    url: Optional[str] = None,
    text: Optional[str] = None,
    saved_jd_name: Optional[str] = None,
    role_keyword: Optional[str] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Obtains a structured JD from one of three input modes.
    """
    # MODE 1: Scrape from URL
    if mode == JD_MODE_URL and url:
        logger.info(f"🔗 Mode: URL Scraping → {url}")
        raw_text = scrape_job_url(url)
        if raw_text:
            jd = parse_job_description(raw_text, role_title="Scraped Role", source_url=url)
            return jd, f"Scraped from {url}"
        logger.warning("URL scraping failed. Falling back to dataset search.")

    # MODE 2: Manual text input
    if mode == JD_MODE_TEXT and text:
        logger.info("✍️ Mode: Manual Text Input")
        jd = parse_job_description(text, role_title="Custom Role")
        return jd, "Manual text input by HR"

    # MODE 3: Load from saved JD library
    if mode == JD_MODE_SAVED and saved_jd_name:
        logger.info(f"📚 Mode: Loading Saved JD → '{saved_jd_name}'")
        jd = load_jd(saved_jd_name)
        if jd:
            return jd, f"Loaded from library: {saved_jd_name}"
        logger.warning(f"Saved JD '{saved_jd_name}' not found. Falling back to dataset.")

    # FALLBACK: Search postings.csv by keyword
    keyword = role_keyword or "Data Scientist"
    logger.info(f"🔍 Fallback: Searching '{keyword}' in postings.csv...")

    if POSTINGS_CSV.exists():
        df_postings = pd.read_csv(POSTINGS_CSV, nrows=2000)
        mask = (
            df_postings["title"].str.contains(keyword, case=False, na=False) |
            df_postings["description"].str.contains(keyword, case=False, na=False)
        )
        matches = df_postings[mask]

        if len(matches) > 0:
            row = matches.iloc[0]
            title = str(row["title"])
            company = str(row.get("company_name", "Company"))
            desc = str(row.get("description", ""))
            skills_desc = str(row.get("skills_desc", "")) if pd.notna(row.get("skills_desc")) else ""
            full_text = f"Job Title: {title}\nCompany: {company}\n\n{desc}\n\nSkills: {skills_desc}"
            jd = parse_job_description(full_text, role_title=title, company_name=company)
            return jd, f"Dataset search: '{title}' at '{company}'"

    # Ultimate fallback
    fallback = """
    Senior Data Scientist at Analytics Corp.
    Must have 3+ years in Python, SQL, Machine Learning.
    Bachelor's or Master's degree required.
    Preferred: AWS, Tableau, Git.
    """
    jd = parse_job_description(fallback, role_title="Senior Data Scientist")
    return jd, "Fallback default JD"


# -----------------------------------------------------------
# 2. JD REVIEW DISPLAY
# -----------------------------------------------------------

def display_jd_review(jd: Dict[str, Any], source: str) -> None:
    """
    Displays the parsed JD in a human-readable format for HR review.
    """
    meta = jd.get("metadata", {})
    gates = jd.get("gate_configuration", {})
    reqs = jd.get("requirements", {})
    skills = reqs.get("skills", {})

    logger.info("\n" + "=" * 65)
    logger.info("📋 JD REVIEW SCREEN — Please verify before screening")
    logger.info("=" * 65)
    logger.info(f"  Source: {source}")
    logger.info(f"  Job Title: {meta.get('job_title', 'N/A')}")
    logger.info(f"  Company: {meta.get('company_name', 'N/A')}")
    logger.info(f"  Created: {meta.get('created_at', 'N/A')}")

    logger.info(f"\n  Gate Order: {' → '.join(gates.get('gate_order', []))}")
    logger.info(f"  Gate Weights: {gates.get('gate_weights', {})}")

    logger.info(f"\n  Must-Have Skills ({len(skills.get('must_have', []))}):")
    for s in skills.get("must_have", []):
        logger.info(f"    🔴 {s['skill']} (importance: {s['importance']}) [{s['source']}]")

    logger.info(f"\n  Preferred Skills ({len(skills.get('preferred', []))}):")
    for s in skills.get("preferred", []):
        logger.info(f"    🟡 {s['skill']} (importance: {s['importance']}) [{s['source']}]")

    edu = reqs.get("education", {})
    logger.info(f"\n  Education: {edu.get('required_degrees', [])}")

    exp = reqs.get("experience", {})
    logger.info(f"  Experience: {exp.get('min_years', 0)}+ years")
    logger.info("=" * 65)


# -----------------------------------------------------------
# 3. MASTER PIPELINE EXECUTION
# -----------------------------------------------------------

def run_screening_pipeline(
    mode: str = JD_MODE_URL,
    url: Optional[str] = None,
    text: Optional[str] = None,
    saved_jd_name: Optional[str] = None,
    role_keyword: Optional[str] = "Data Scientist",
    resume_limit: Optional[int] = None,
    save_jd_to_library: bool = False,
    custom_jd_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Executes end-to-end candidate screening using Transformer Semantic embeddings.
    """
    logger.info("=" * 65)
    logger.info("🚀 AI RESUME SCREENING PIPELINE v3 (Semantic-Powered)")
    logger.info("=" * 65)

    # STEP 1: Obtain JD
    jd_structured, source = obtain_job_description(
        mode=mode, url=url, text=text,
        saved_jd_name=saved_jd_name, role_keyword=role_keyword
    )

    # STEP 2: Display JD Review
    display_jd_review(jd_structured, source)

    # STEP 3: Optionally save to library
    if save_jd_to_library:
        save_jd(jd_structured, custom_name=custom_jd_name)

    # STEP 4: Parse real resumes
    cleaned_csv = OUTPUT_DIR / "resumes_cleaned.csv"
    candidate_profiles = parse_all_resumes(cleaned_csv_path=cleaned_csv, limit=resume_limit)

    if not candidate_profiles:
        logger.error("No candidate profiles available.")
        return pd.DataFrame()

    # STEP 5: Run Gate Cascade with HR's config
    gate_config = jd_structured.get("gate_configuration", {})
    gate_order = gate_config.get("gate_order", DEFAULT_GATE_ORDER)
    gate_weights = gate_config.get("gate_weights", DEFAULT_GATE_WEIGHTS)

    logger.info(f"\nRunning Gate Cascade (Order: {gate_order})...")
    cascade_reports = [
        run_candidate_cascade(jd_structured, p, gate_order, gate_weights)
        for p in candidate_profiles
    ]

    # STEP 6: Run Sentence-Transformer Semantic Scorer
    logger.info("\nComputing Sentence-Transformer semantic similarity scores...")
    df_cleaned = pd.read_csv(cleaned_csv)
    if resume_limit:
        df_cleaned = df_cleaned.head(resume_limit)

    resume_texts = df_cleaned["cleaned_text"].tolist()
    candidate_ids = [p["candidate_id"] for p in candidate_profiles]

    semantic_scorer = SemanticScorer(model_name="all-MiniLM-L6-v2")
    semantic_results = semantic_scorer.compute_similarity_scores(
        jd_text=json.dumps(jd_structured.get("requirements", {})),
        resume_texts=resume_texts,
        candidate_ids=candidate_ids
    )
    semantic_map = {r["candidate_id"]: r["semantic_score"] for r in semantic_results}

    # STEP 7: Merge and rank
    logger.info("Computing final scores...")
    records = []
    for rep in cascade_reports:
        cid = rep["candidate_id"]
        gate_score = rep["final_gate_score"]
        sem_score = semantic_map.get(cid, 0.0)
        final_score = calculate_semantic_combined_score(gate_score, sem_score)

        if final_score >= SHORTLIST_THRESHOLD and rep["status"] != "REJECTED":
            verdict = "SHORTLISTED"
        elif final_score >= MAYBE_THRESHOLD and rep["status"] != "REJECTED":
            verdict = "REVIEW"
        else:
            verdict = "REJECTED"

        records.append({
            "Candidate_ID": cid,
            "Category": rep["category"],
            "Gate_Score": gate_score,
            "Semantic_Score": sem_score,
            "Final_Score": final_score,
            "Status": verdict,
            "Rejection_Reason": rep["rejection_reason"] or "N/A",
            "Matched_Skills": ", ".join(rep["matched_skills"]) or "None",
            "Missing_Skills": ", ".join(rep["missing_skills"]) or "None",
        })

    df_results = pd.DataFrame(records)
    df_results.sort_values(by="Final_Score", ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)
    df_results["Rank"] = df_results.index + 1

    # STEP 8: Save outputs
    output_csv = OUTPUT_DIR / "shortlist_results.csv"
    output_json = OUTPUT_DIR / "shortlist_results.json"
    df_results.to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(cascade_reports, f, indent=2)

    # STEP 9: Print summary
    jd_title = jd_structured.get("metadata", {}).get("job_title", "Unknown")
    logger.info(f"\n{'=' * 65}")
    logger.info(f"🏆 TOP 10 CANDIDATES FOR: {jd_title}")
    logger.info(f"{'=' * 65}")
    display_cols = ["Rank", "Candidate_ID", "Category", "Final_Score", "Gate_Score", "Semantic_Score", "Status"]
    logger.info("\n" + df_results[display_cols].head(10).to_string(index=False))

    status_counts = df_results["Status"].value_counts().to_dict()
    logger.info(f"\n📊 Funnel: {status_counts} (Total: {len(df_results)})")
    logger.info(f"💾 Saved: {output_csv}")
    logger.info(f"{'=' * 65}")

    return df_results


# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------

if __name__ == "__main__":
    manual_jd_text = """
    Senior Machine Learning Engineer at TechCorp.
    Must be proficient in Python, PyTorch, and Machine Learning.
    Master's degree in Computer Science required.
    4+ years of experience in deep learning and NLP.
    Preferred: AWS, Docker, Kubernetes, Git.
    """

    results = run_screening_pipeline(
        mode="text",
        text=manual_jd_text,
        resume_limit=None,  # Screen all 2,484 real resumes
        save_jd_to_library=False
    )