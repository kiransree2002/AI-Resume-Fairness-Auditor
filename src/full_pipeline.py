import os
import json
import pandas as pd

from text_cleaner import clean_text
from skill_extractor import extract_profile
from jd_parser import parse_job_description
from resume_parser import parse_single_resume
from gate_cascade import match_all_resumes
from baseline_scorer import score_resumes_tfidf, combined_score

# -------------------------------------------------------------
# 1. LOAD REAL DATA
# -------------------------------------------------------------

def load_real_data():
    """Loads both datasets from the data/ folder."""
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    
    resumes_path = os.path.join(DATA_DIR, "Resume.csv")
    postings_path = os.path.join(DATA_DIR, "postings.csv")
    
    df_resumes = pd.read_csv(resumes_path)
    df_postings = pd.read_csv(postings_path)
    
    return df_resumes, df_postings


# -------------------------------------------------------------
# 2. PICK A REAL JOB DESCRIPTION
# -------------------------------------------------------------

def pick_real_jd(df_postings, keyword="data scientist"):
    """
    Finds a real JD from the LinkedIn dataset that matches a keyword.
    Falls back to the first available posting if no match found.
    """
    # Search for a JD containing the keyword in title or description
    mask = (
        df_postings["title"].str.contains(keyword, case=False, na=False) |
        df_postings["description"].str.contains(keyword, case=False, na=False)
    )
    matches = df_postings[mask]
    
    if len(matches) > 0:
        jd_row = matches.iloc[0]
    else:
        jd_row = df_postings.iloc[0]
        print(f"⚠️ No JD found for '{keyword}'. Using first available posting.")
    
    jd_text = f"Job Title: {jd_row['title']}\nCompany: {jd_row['company_name']}\n\n{jd_row['description']}"
    
    # Use skills_desc if available
    if pd.notna(jd_row.get("skills_desc", None)):
        jd_text += f"\n\nRequired Skills: {jd_row['skills_desc']}"
    
    return jd_text, jd_row["title"]


# -------------------------------------------------------------
# 3. RUN FULL PIPELINE
# -------------------------------------------------------------

def run_full_pipeline(max_resumes=50):
    """
    Connects all modules and runs the complete screening pipeline
    on real data from Kaggle datasets.
    """
    print("=" * 70)
    print("🚀 RUNNING FULL PIPELINE ON REAL KAGGLE DATA")
    print("=" * 70)
    
    # Load data
    df_resumes, df_postings = load_real_data()
    print(f"📄 Loaded {len(df_resumes)} resumes and {len(df_postings)} job postings")
    
    # Pick a real JD
    jd_text, jd_title = pick_real_jd(df_postings, keyword="data")
    print(f"\n💼 Selected JD: '{jd_title}'")
    print(f"   Preview: {clean_text(jd_text)[:200]}...")
    
    # Parse JD into structured JSON
    print("\n⚙️ Parsing Job Description...")
    jd_json = parse_job_description(jd_text, role_title=jd_title)
    must_count = len(jd_json["gates"]["gate_3_must_skills"])
    pref_count = len(jd_json["gates"]["gate_4_preferred_skills"])
    print(f"   ✅ Extracted {must_count} must-have skills, {pref_count} preferred skills")
    print(f"   ✅ Min experience: {jd_json['gates']['gate_2_experience']['min_years']} years")
    
    # Parse resumes (limit for speed)
    df_sample = df_resumes.head(max_resumes)
    print(f"\n⚙️ Parsing {len(df_sample)} resumes...")
    
    resume_profiles = []
    resume_raw_texts = []
    resume_ids = []
    
    for _, row in df_sample.iterrows():
        profile = parse_single_resume(row["ID"], row["Resume_str"], row["Category"])
        resume_profiles.append(profile)
        resume_raw_texts.append(row["Resume_str"])
        resume_ids.append(f"RES_{row['ID']}")
    
    print(f"   ✅ Parsed {len(resume_profiles)} resume profiles")
    
    # Run Gate Cascade
    print("\n⚙️ Running Gate Cascade Matcher...")
    gate_results = match_all_resumes(jd_json, resume_profiles)
    
    # Run TF-IDF Scorer
    print("⚙️ Running TF-IDF Baseline Scorer...")
    tfidf_results = score_resumes_tfidf(jd_text, resume_raw_texts, resume_ids)
    
    # Create TF-IDF lookup
    tfidf_lookup = {r["candidate_id"]: r["tfidf_score"] for r in tfidf_results}
    
    # Combine scores
    for r in gate_results:
        tfidf = tfidf_lookup.get(r["candidate_id"], 0)
        r["tfidf_score"] = tfidf
        r["combined_score"] = combined_score(r["final_score"], tfidf)
    
    # Re-sort by combined score
    gate_results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Print results
    print("\n" + "=" * 70)
    print(f"📊 RESULTS: Top 10 Candidates for '{jd_title}'")
    print("=" * 70)
    print(f"{'Rank':<5}{'Candidate':<18}{'Gate':<7}{'TF-IDF':<8}{'Combined':<9}{'Status':<12}")
    print("-" * 70)
    
    for rank, r in enumerate(gate_results[:10], 1):
        print(f"{rank:<5}{r['name'][:17]:<18}{r['final_score']:<7}{r['tfidf_score']:<8}{r['combined_score']:<9}{r['status']:<12}")
    
    # Summary stats
    shortlisted = sum(1 for r in gate_results if r["status"] == "SHORTLISTED")
    rejected = sum(1 for r in gate_results if r["status"] == "REJECTED")
    maybe = sum(1 for r in gate_results if r["status"] == "MAYBE")
    
    print(f"\n📈 Summary: {shortlisted} Shortlisted | {maybe} Maybe | {rejected} Rejected (out of {len(gate_results)})")
    
    # Save results
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, "full_pipeline_results.json")
    with open(output_path, "w") as f:
        json.dump(gate_results, f, indent=2)
    
    print(f"\n💾 Full results saved to: {output_path}")
    print("=" * 70)
    
    return gate_results


if __name__ == "__main__":
    results = run_full_pipeline(max_resumes=50)