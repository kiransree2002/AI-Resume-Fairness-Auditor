"""
Streamlit Interactive UI Test App (Production v2.2)
=================================================
Provides a web browser interface to visually test:
1. Live JD URL Scraping / Text Parsing / Saved JD Library
2. Editable JD Review Screen (Metadata, Gates, Skills)
3. Candidate Screening Engine (Gate Cascade + Transformers)
4. Candidate Detail Inspector (Matched/Missing Skills & Rejection Reasons)
5. EDA & Model Comparison Chart Gallery

Run via terminal:
  python -m streamlit run app/test_app.py
"""

import sys
import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document

# Add project root and src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from src.config import (
    OUTPUT_DIR,
    SAVED_JDS_DIR,
    SHORTLIST_THRESHOLD,
    MAYBE_THRESHOLD,
    GATE_WEIGHT,
    TFIDF_WEIGHT
)

# Derived directory for EDA charts
EDA_CHARTS_DIR = OUTPUT_DIR / "eda_charts"
EDA_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

from src.jd_parser import (
    parse_job_description,
    scrape_job_url,
    save_jd,
    list_saved_jds,
    load_jd
)
from src.resume_parser import parse_all_resumes, parse_resume
from src.gate_cascade import run_candidate_cascade
from src.semantic_scorer import SemanticScorer, calculate_semantic_combined_score
from src.score_explainer import generate_candidate_explanation  # NEW IMPORT

# -----------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening & Fairness Auditor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Resume Screening & Fairness Auditor")
st.caption("Powered by spaCy NLP, Sentence-Transformers (BERT), and 4-Gate Cascade Matching")

# -----------------------------------------------------------
# TAB NAVIGATION
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📋 1. Job Description & Review",
    "🏆 2. Candidate Screening Results",
    "📊 3. Analytics & Model Charts"
])


# ===========================================================
# TAB 1: JOB DESCRIPTION INPUT & REVIEW SCREEN
# ===========================================================
with tab1:
    st.header("Step 1: Provide & Review Job Description")
    
    input_mode = st.radio(
        "Choose Input Method:",
        ["🔗 Paste Web Link (Scraper)", "✍️ Type / Paste Text Manually", "📚 Load from JD Library"],
        horizontal=True
    )
    
    # --- MODE A: URL SCRAPER ---
    if "Paste Web Link" in input_mode:
        url_input = st.text_input(
            "Paste Job Posting URL:",
            placeholder="e.g. https://en.wikipedia.org/wiki/Data_science"
        )
        if st.button("🔗 Scrape & Parse URL", type="primary"):
            if url_input:
                with st.spinner("Connecting to live web link and pulling job details..."):
                    scraped_text = scrape_job_url(url_input)
                    if scraped_text:
                        jd_structured = parse_job_description(scraped_text, role_title="Scraped Role", source_url=url_input)
                        st.session_state["parsed_jd"] = jd_structured
                        st.session_state["jd_raw_text"] = scraped_text
                        st.success("✅ Successfully scraped and parsed job description!")
                    else:
                        st.error("⚠️ Scraping failed or URL blocked. Please try manual text paste.")
            else:
                st.warning("Please paste a valid URL.")

    # --- MODE B: MANUAL TEXT PASTE ---
    elif "Type / Paste Text" in input_mode:
        text_title = st.text_input("Job Title:", value="Senior Machine Learning Engineer")
        text_company = st.text_input("Company Name:", value="TechCorp AI")
        text_input = st.text_area(
            "Paste Job Description Text:",
            height=200,
            value="""Senior Machine Learning Engineer at TechCorp AI.
Must be proficient in Python, PyTorch, and Machine Learning.
Master's degree in Computer Science or related domain required.
At least 4+ years of professional experience in deep learning and NLP.
Preferred skills include AWS, Docker, Kubernetes, and Git."""
        )
        if st.button("🧠 Parse Job Description", type="primary"):
            if text_input:
                with st.spinner("Analyzing text with spaCy NLP and assigning importance weights..."):
                    jd_structured = parse_job_description(text_input, role_title=text_title, company_name=text_company)
                    st.session_state["parsed_jd"] = jd_structured
                    st.session_state["jd_raw_text"] = text_input
                    st.success("✅ Job Description parsed successfully!")

    # --- MODE C: LOAD FROM LIBRARY ---
    else:
        saved_list = list_saved_jds()
        if saved_list:
            filenames = [item["filename"] for item in saved_list]
            selected_file = st.selectbox("Select Saved Job Description:", filenames)
            if st.button("📚 Load Selected JD", type="primary"):
                jd_structured = load_jd(selected_file)
                st.session_state["parsed_jd"] = jd_structured
                st.session_state["jd_raw_text"] = json.dumps(jd_structured.get("requirements", {}))
                st.success(f"✅ Loaded '{selected_file}' from library!")
        else:
            st.info("No saved JDs found in `saved_jds/` folder. Parse a new JD above to save it.")

    # --- DISPLAY PARSED JD REVIEW SCREEN ---
    if "parsed_jd" in st.session_state:
        jd = st.session_state["parsed_jd"]
        st.divider()
        st.subheader("📋 Parsed JD Review Screen (Verify Before Screening)")
        
        meta = jd.get("metadata", {})
        gates = jd.get("gate_configuration", {})
        reqs = jd.get("requirements", {})
        skills = reqs.get("skills", {})
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Job Title", meta.get("job_title", "N/A"))
        col2.metric("Company", meta.get("company_name", "N/A"))
        col3.metric("Detected Skills", skills.get("total_skills_detected", 0))
        
        st.markdown(f"**Gate Execution Order**: `{' ➔ '.join(gates.get('gate_order', []))}`")
        st.markdown(f"**Gate Weights**: Skills = **{gates.get('gate_weights',{}).get('skills', 0.5)*100:.0f}%**, Education = **{gates.get('gate_weights',{}).get('education', 0.25)*100:.0f}%**, Experience = **{gates.get('gate_weights',{}).get('experience', 0.25)*100:.0f}%**")
        
        c_must, c_pref = st.columns(2)
        
        with c_must:
            st.markdown("### 🔴 Must-Have Core Skills")
            must_list = skills.get("must_have", [])
            if must_list:
                for s in must_list:
                    st.write(f"- **{s['skill']}** (weight: `{s['importance']}`) — *{s['source']}*")
            else:
                st.info("No explicit must-have skills detected.")

        with c_pref:
            st.markdown("### 🟡 Preferred / Nice-to-Have Skills")
            pref_list = skills.get("preferred", [])
            if pref_list:
                for s in pref_list:
                    st.write(f"- **{s['skill']}** (weight: `{s['importance']}`) — *{s['source']}*")
            else:
                st.info("No preferred skills detected.")

        st.markdown("### 🎓 Education & Experience Requirements")
        e_col1, e_col2 = st.columns(2)
        e_col1.write(f"**Required Degrees**: {', '.join(reqs.get('education',{}).get('required_degrees', []))}")
        e_col2.write(f"**Minimum Experience**: {reqs.get('experience',{}).get('min_years', 0)}+ years")
        
        # Save to library option
        st.divider()
        save_col1, save_col2 = st.columns([3, 1])
        save_name = save_col1.text_input("Save this JD to library as:", value=f"{meta.get('job_title', 'JD')}_Saved")
        if save_col2.button("💾 Save to Library"):
            save_jd(jd, custom_name=save_name)
            st.success(f"Saved as `{save_name}.json` in `saved_jds/`!")


# ===========================================================
# TAB 2: CANDIDATE SCREENING & RESULTS
# ===========================================================
with tab2:
    st.header("Step 2: Candidate Screening Engine")

    
    st.subheader("Upload Additional Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload resume files (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload one or more resumes to screen alongside the existing dataset."
    )


    
    if "parsed_jd" not in st.session_state:
        st.warning("⚠️ Please parse or load a Job Description in Tab 1 first!")
    else:
        st.info("Resumes will be screened against the active parsed Job Description from Tab 1.")
        
        limit_option = st.selectbox(
            "Select Resume Sample Size to Screen:",
            [50, 100, 250, 500, "All 2,484 Resumes"],
            index=0
        )
        
        resume_limit = None if limit_option == "All 2,484 Resumes" else limit_option
        
        if st.button("🚀 Run Screening Engine", type="primary"):
            with st.spinner("Parsing resumes, running Gate Cascade, and generating Sentence-Transformer embeddings..."):
                jd_structured = st.session_state["parsed_jd"]
                jd_raw_text = st.session_state.get("jd_raw_text", json.dumps(jd_structured.get("requirements", {})))
                
                cleaned_csv = OUTPUT_DIR / "resumes_cleaned.csv"
                candidate_profiles = parse_all_resumes(cleaned_csv_path=cleaned_csv, limit=resume_limit)


                
                uploaded_resume_texts = []

                if uploaded_resumes:
                    for i, uploaded_file in enumerate(uploaded_resumes, start=1):
                        try:
                            if uploaded_file.name.lower().endswith(".pdf"):
                                reader = PdfReader(uploaded_file)
                                raw_text = "\n".join(
                                    page.extract_text() or ""
                                    for page in reader.pages
                                )
                            else:
                                document = Document(uploaded_file)
                                raw_text = "\n".join(
                                    paragraph.text
                                    for paragraph in document.paragraphs
                                )

                            if not raw_text.strip():
                                st.warning(
                                    f"Could not extract readable text from {uploaded_file.name}."
                                )
                                continue

                            resume_id = f"UPLOAD_{i}_{Path(uploaded_file.name).stem}"
                            profile = parse_resume(
                                resume_id=resume_id,
                                raw_text=raw_text,
                                category="Uploaded"
                            )
                            candidate_profiles.append(profile)
                            uploaded_resume_texts.append(
                                (profile["candidate_id"], raw_text)
                            )

                        except Exception as e:
                            st.warning(
                                f"Could not read {uploaded_file.name}: {e}"
                            )




                
                # Gate Cascade
                gate_config = jd_structured.get("gate_configuration", {})
                cascade_reports = [
                    run_candidate_cascade(jd_structured, p, gate_config.get("gate_order"), gate_config.get("gate_weights"))
                    for p in candidate_profiles
                ]
                
                # Semantic Scorer
                df_cleaned = pd.read_csv(cleaned_csv)
                if resume_limit:
                    df_cleaned = df_cleaned.head(resume_limit)
                
                resume_texts = df_cleaned["cleaned_text"].tolist()
                
                resume_texts.extend(
                    raw_text for _, raw_text in uploaded_resume_texts
                )

                candidate_ids = [p["candidate_id"] for p in candidate_profiles]
                
                semantic_scorer = SemanticScorer(model_name="all-MiniLM-L6-v2")
                semantic_results = semantic_scorer.compute_similarity_scores(
                    jd_text=jd_raw_text,
                    resume_texts=resume_texts,
                    candidate_ids=candidate_ids
                )
                semantic_map = {r["candidate_id"]: r["semantic_score"] for r in semantic_results}
                
                # Combine & Rank
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
                
                # Save
                df_results.to_csv(OUTPUT_DIR / "shortlist_results.csv", index=False)
                with open(OUTPUT_DIR / "shortlist_results.json", "w", encoding="utf-8") as f:
                    json.dump(cascade_reports, f, indent=2)
                
                st.session_state["screening_df"] = df_results
                st.success("🎉 Candidate screening complete!")

        # DISPLAY RESULTS TABLE
        if "screening_df" in st.session_state:
            df = st.session_state["screening_df"]
            
            st.divider()
            st.subheader("📊 Candidate Screening Shortlist")
            
            # Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Candidates", len(df))
            m2.metric("🟢 Shortlisted (≥70)", len(df[df["Status"] == "SHORTLISTED"]))
            m3.metric("🟡 Review (50-69)", len(df[df["Status"] == "REVIEW"]))
            m4.metric("🔴 Rejected (<50)", len(df[df["Status"] == "REJECTED"]))
            
            # Interactive Data Table
            st.dataframe(
                df[["Rank", "Candidate_ID", "Category", "Final_Score", "Gate_Score", "Semantic_Score", "Status", "Rejection_Reason"]],
                use_container_width=True,
                height=350
            )
            
            # CANDIDATE DRILL-DOWN INSPECTOR
            st.divider()
            st.subheader("🔍 Candidate Audit Inspector")
            
            cand_ids = df["Candidate_ID"].tolist()
            selected_cand = st.selectbox("Inspect Candidate Details:", cand_ids)
            
            if selected_cand:
                row = df[df["Candidate_ID"] == selected_cand].iloc[0]
                
                d1, d2, d3 = st.columns(3)
                d1.write(f"**Candidate ID**: `{row['Candidate_ID']}`")
                d2.write(f"**Domain Category**: `{row['Category']}`")
                
                st.divider()
                
                # --- EXPLAINABILITY ENGINE OUTPUT ---
                explanation_markdown = generate_candidate_explanation(row.to_dict())
                st.markdown(explanation_markdown)


# ===========================================================
# TAB 3: CHARTS & ANALYTICS GALLERY
# ===========================================================
with tab3:
    st.header("Visual Analytics & Model Comparisons")
    st.caption("All charts generated from real Kaggle dataset processing in `src/`")
    
    chart1_path = EDA_CHARTS_DIR / "01_category_distribution.png"
    chart2_path = EDA_CHARTS_DIR / "02_top_skills_overall.png"
    chart3_path = EDA_CHARTS_DIR / "03_experience_distribution.png"
    chart4_path = EDA_CHARTS_DIR / "04_model_comparison.png"
    
    c1, c2 = st.columns(2)
    
    with c1:
        if chart1_path.exists():
            st.image(str(chart1_path), caption="1. Category Distribution Across 2,484 Resumes", use_container_width=True)
        if chart3_path.exists():
            st.image(str(chart3_path), caption="3. Stated Experience Distribution (Fixed Calendar Bug)", use_container_width=True)

    with c2:
        if chart2_path.exists():
            st.image(str(chart2_path), caption="2. Top 20 Most Frequent Skills Overall", use_container_width=True)
        if chart4_path.exists():
            st.image(str(chart4_path), caption="4. Model Comparison: TF-IDF vs Sentence-Transformers", use_container_width=True)