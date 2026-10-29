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
import re
import sys
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document
from src.jd_parser import (
    save_jd,
    parse_job_description,
    load_jd,
    list_saved_jds
    )

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



def clean_jd_text(text):
    """Remove website navigation and application-form content from a scraped JD."""
    import re

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b-\u200f\uFEFF]", "", text)

    # Remove the application form and website footer.
    stop_markers = [
        "Instructions for Candidates",
        "Apply for this job",
        "Select Country",
        "Upload Document",
        "I have read and agree to the Privacy Policy",
        "This site uses cookies",
        "Click here to read our",
    ]

    lower_text = text.casefold()
    cut_positions = [
        lower_text.find(marker.casefold())
        for marker in stop_markers
        if lower_text.find(marker.casefold()) != -1
    ]

    if cut_positions:
        text = text[:min(cut_positions)]

    # Remove common navigation labels and normalize spacing.
    unwanted_lines = {
        "home", "about us", "contact us", "privacy policy",
        "terms of use", "terms & conditions", "accept cookies",
        "reject cookies", "sign in", "sign up", "log in",
        "register", "skip to content", "all rights reserved",
        "apply now", "submit", "upload document"
    }

    cleaned_lines = []
    previous_line = None

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            continue

        if line.casefold() in unwanted_lines:
            continue

        if line != previous_line:
            cleaned_lines.append(line)

        previous_line = line

    return "\n".join(cleaned_lines).strip()



# -----------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening & Fairness Auditor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""

<style>

/* Premium navigation tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid rgba(129, 140, 248, 0.25);
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 72, 0.65);
    border: 1px solid rgba(129, 140, 248, 0.22);
    border-radius: 10px 10px 0 0;
    padding: 12px 20px;
    color: #cbd5e1;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #4338ca, #6d28d9);
    color: #ffffff;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: #a5b4fc;
}


/* Premium dark AI theme */
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #111a33 55%, #17142b 100%);
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #0a1020;
    border-right: 1px solid rgba(129, 140, 248, 0.25);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero-card {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(120deg, #18264a, #24204a);
    border: 1px solid rgba(129, 140, 248, 0.35);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.22);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 32px;
    font-weight: 750;
    color: #ffffff;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 15px;
    color: #c7d2fe;
}

div.stButton > button {
    border-radius: 10px;
    border: 1px solid #6366f1;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: #a5b4fc;
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.35);
    color: white;
}

[data-testid="stMetric"] {
    background: rgba(30, 41, 72, 0.75);
    border: 1px solid rgba(129, 140, 248, 0.25);
    padding: 16px;
    border-radius: 14px;
}
</style>

<div class="hero-card">
    <div class="hero-title">🤖 AI Resume Screening</div>
    <div class="hero-subtitle">
        Intelligent Candidate Matching · Explainable AI · Hiring Fairness
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# TAB NAVIGATION
# -----------------------------------------------------------
# PROFESSIONAL DASHBOARD NAVIGATION

tab1, tab2, tab3 = st.tabs([
    "📋 Screening Dashboard",
    "📊 Visualization & Analytics",
    "✉️ HR Email"
])


# ===========================================================
# TAB 1: JOB DESCRIPTION INPUT & REVIEW SCREEN
# ===========================================================
with tab1:
    st.header("Job Description")
    
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
            if url_input.strip():
                st.session_state["jd_scrape_success"] = False
                st.session_state["jd_confirmed"] = False
                with st.spinner("Scraping Job Description..."):
                    raw_scraped_text = scrape_job_url(url_input.strip())
                    scraped_text = clean_jd_text(raw_scraped_text)

                    if scraped_text and scraped_text.strip():
                        st.session_state["jd_raw_text"] = scraped_text
                        st.session_state["editable_jd_preview"] = scraped_text
                        st.session_state["jd_scrape_success"] = True
                        st.session_state["parsed_jd"] = parse_job_description(
                            scraped_text,
                            role_title="Scraped Role",
                            source_url=url_input.strip()
                        )
                        st.session_state["jd_confirmed"] = False
                        st.success("✅ Job Description scraped successfully!")
                    else:
                        st.error(
                            "Scraping failed or the website blocked access. "
                            "Please paste the JD manually."
                        )
            else:
                st.warning("Please paste a valid URL.")

        # Show editable JD immediately below the scrape button
        if st.session_state.get("jd_scrape_success", False):
            st.subheader("📝 Review & Edit Job Description")

            edited_jd_text = st.text_area(
                "Job Description (you can edit this text)",
                value=st.session_state.get("jd_raw_text", ""),
                height=300,
                key="editable_jd_preview"
            )

            
            if st.button(
                "✅ Confirm Job Description",
                type="primary",
                key="confirm_scraped_jd"
                ):
                if edited_jd_text.strip():
                    st.session_state["jd_raw_text"] = edited_jd_text

                    parsed_jd = parse_job_description(
                        edited_jd_text,
                        role_title="HR Confirmed Role",
                        source_url=url_input.strip()
                        )

                    # Preserve the original JD text for future editing
                    parsed_jd["raw_text"] = edited_jd_text

                    # Save the confirmed JD to the library
                    saved_path = save_jd(parsed_jd)

                    st.session_state["parsed_jd"] = parsed_jd
                    st.session_state["jd_confirmed"] = True

                    st.success(f"Job Description confirmed and saved: {saved_path.name}")
                else:
                    st.error("Please enter a Job Description.")


    
    # --- MODE B: MANUAL TEXT PASTE ---
    elif "Type / Paste Text" in input_mode:
        text_input = st.text_area(
            "Paste Job Description Text:",
            height=250,
            placeholder="Paste the complete job description here...",
            key="manual_jd_input"
        )

        if st.button(
            "✅ Confirm Manual Job Description",
            key="confirm_manual_jd"
        ):
            if text_input.strip():
                parsed_jd = parse_job_description(
                    text_input.strip(),
                    role_title="Manually Entered Role"
                )

                # Preserve original JD text for future editing
                parsed_jd["raw_text"] = text_input.strip()

                # Automatically save the confirmed JD
                saved_path = save_jd(parsed_jd)

                st.session_state["jd_raw_text"] = text_input.strip()
                st.session_state["parsed_jd"] = parsed_jd
                st.session_state["jd_confirmed"] = True

                st.success(
                    f"Job Description confirmed and saved: {saved_path.name}"
                )
            else:
                st.session_state["jd_confirmed"] = False
                st.error("Please enter a Job Description.")


   
    # --- MODE C: LOAD FROM LIBRARY ---
    else:
        saved_list = list_saved_jds()

    if saved_list:
        filenames = [item["filename"] for item in saved_list]

        selected_file = st.selectbox(
            "Select Saved Job Description:",
            filenames,
            key="selected_saved_jd"
        )

        if st.button("📚 Load Selected JD", type="primary"):
            jd_structured = load_jd(selected_file)

            if jd_structured is None:
                st.error(f"Could not load '{selected_file}'.")
            else:
                st.session_state["parsed_jd"] = jd_structured
                st.session_state["selected_jd_filename"] = selected_file

                st.session_state["jd_raw_text"] = jd_structured.get(
                    "raw_text",
                    json.dumps(jd_structured, indent=2)
                )

                st.session_state["jd_confirmed"] = False
                st.session_state["edit_loaded_jd"] = True
                st.success("JD loaded. Review and edit the details below.")

    else:
        st.info(
            "No saved JDs found in `saved_jds/` folder. "
            "Parse a new JD above to save it."
        )



    # --- HR REVIEW OF LOADED JD ---
    if st.session_state.get("edit_loaded_jd", False):
        st.subheader("📝 Review & Edit Saved Job Description")

        current_jd = st.session_state.get("parsed_jd", {})
        original_text = current_jd.get("raw_text", "")

        edited_library_jd = st.text_area(
            "Job Description — edit before confirming",
            value=original_text,
            height=300,
            key="editable_library_jd"
        )

        if st.button(
            "✅ Confirm & Save Edited JD",
            type="primary",
            key="confirm_edited_library_jd"
        ):
            if not edited_library_jd.strip():
                st.error("Please enter a Job Description.")
            else:
                updated_jd = parse_job_description(
                    edited_library_jd.strip(),
                    role_title=current_jd.get(
                        "metadata", {}
                    ).get("job_title", "Manually Entered Role"),
                    source_url=current_jd.get(
                        "metadata", {}
                    ).get("source_url")
                )

                updated_jd["raw_text"] = edited_library_jd.strip()

                saved_path = save_jd(
                    updated_jd,
                    custom_name=st.session_state.get(
                        "selected_jd_filename", "custom_job.json"
                    )
                )

                st.session_state["parsed_jd"] = updated_jd
                st.session_state["jd_raw_text"] = edited_library_jd.strip()
                st.session_state["jd_confirmed"] = True
                st.session_state["edit_loaded_jd"] = False

                st.success(
                    f"JD confirmed and saved: {saved_path.name}"
                )




# ===========================================================
with tab1:
    st.header("Upload resumes")

    
    st.subheader("Upload Additional Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload resume files (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload one or more resumes to screen alongside the existing dataset."
    )


           
        
    limit_option = st.selectbox(
            "Select Resume Sample Size to Screen:",
            [0, 50, 100, 250, 500, 1000, "All 2,484 Resumes"],
            index=0,
            format_func=lambda x: "Uploaded resumes only (0 existing resumes)" if x == 0 else str(x)
            )

    resume_limit = None if limit_option == "All 2,484 Resumes" else limit_option


        
    
    
    
    st.subheader("⚙️ HR Screening Thresholds")

    col1, col2, col3 = st.columns([1.3, 1.5, 1.5])

    
    
    
    with col1:
        st.markdown("**Shortlist Range (%)**")

        shortlist_min = st.number_input(
            "Minimum score",
            min_value=1,
            max_value=100,
            value=None,
            step=1,
            key="hr_shortlist_min",
            placeholder="Enter value",
            label_visibility="collapsed"
        )



    with col2:
        st.markdown(
            "<div style='font-weight:600; margin-bottom:0px;'>Review Score Range (%)</div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div style='margin-top:-10px;'></div>",
            unsafe_allow_html=True)

        range_from, range_to = st.columns([1, 1], gap="small")

        
        with range_from:
            review_min = st.number_input(
                "From (%)",
                min_value=0,
                max_value=100,
                value=None,
                step=1,
                placeholder="Enter value",
                key="hr_review_min"
                )

        with range_to:
            review_max = st.number_input(
                "To (%)",
                min_value=0,
                max_value=100,
                value=None,
                step=1,
                placeholder="Enter value",
                key="hr_review_max"
            )

        review_range_valid = (
            review_min is not None
            and review_max is not None
            and review_min <= review_max
        )
    

    
    with col3:
        st.markdown("**Maximum Resumes to Shortlist**")

        top_n = st.number_input(
            "Maximum resumes",
            min_value=1,
            max_value=100000,
            value=None,
            step=1,
            key="hr_top_n",
            placeholder="Enter value",
            help="Leave empty for no limit.",
            label_visibility="collapsed"
        )

        top_n = int(top_n) if top_n is not None else None



    
    
    review_range_valid = (
        review_min is not None
        and review_max is not None
        and 0 <= review_min <= review_max <= 100
    )

    threshold_mode_valid = (
        shortlist_min is not None
        and 1 <= shortlist_min <= 100
        and review_range_valid
        and review_max < shortlist_min
    )

    top_n_only_mode = (
        shortlist_min is None
        and top_n is not None
        and top_n >= 1
        )

    screening_settings_valid = (
        threshold_mode_valid or top_n_only_mode
    )

    if top_n_only_mode:
        st.info(
            f"Top-{top_n} candidates will be ranked "
            "by Final Score without a shortlist threshold."
        )
    elif not screening_settings_valid:
        st.error(
            "Enter a valid Shortlist Minimum, or set "
            "Maximum Resumes to use Top-N mode. "
            "The Review range must be valid."
        )

    st.write("Review From:", review_min)
    st.write("Review To:", review_max)
    st.write("Shortlist Minimum:", shortlist_min)
    st.write("Screening settings valid:", screening_settings_valid)



                
           
           
# DISPLAY RESULTS TABLE
if "screening_df" in st.session_state:
    df = st.session_state["screening_df"]

    st.divider()
    st.subheader("📊 Candidate Screening Shortlist")

    if df.empty:
        st.warning(
            "No resumes were available for screening. "
            "Please upload resumes or select a dataset sample."
        )
    else:
        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Total Candidates", len(df))

        m2.metric(
            f"🟢 Shortlisted (≥{shortlist_min:g})",
            len(df[df["Status"] == "SHORTLISTED"])
        )

        m3.metric(
            f"🟡 Review ({review_min:g}–{review_max:g})",
            len(df[df["Status"] == "REVIEW"])
        )

        m4.metric(
            "🔴 Rejected",
            len(df[df["Status"] == "REJECTED"])
        )

        # HR Manual Status Review
        st.divider()
        st.subheader("📝 Candidate Results")

        if "hr_manual_decisions" not in st.session_state:
            st.session_state["hr_manual_decisions"] = {}

        editable_df = df.copy()

        editable_df["HR_Decision"] = editable_df["Candidate_ID"].apply(
            lambda cid: st.session_state["hr_manual_decisions"].get(
                str(cid),
                editable_df.loc[
                    editable_df["Candidate_ID"] == cid, "Status"
                ].iloc[0]
            )
        )

        edited_df = st.data_editor(
    editable_df[
        [
            "Rank",
            "Candidate_ID",
            "Category",
            "Gate_Score",
            "Semantic_Score",
            "Final_Score",
            "Matched_Skills",
            "Missing_Skills",
            "Status",
            "HR_Decision",
            "Rejection_Reason",
        ]
    ],
    key="candidate_screening_editor",
    use_container_width=True,
    hide_index=True,
)
if st.button(
    "🧠 Parse Resumes & Screen Candidates",
    type="primary",
    key="parse_and_screen_candidates"
    ):


    jd_raw_text = st.session_state.get("jd_raw_text", "")

    if "Type / Paste Text" in input_mode:
        jd_raw_text = text_input.strip()

    if not jd_raw_text:
        st.error("Please provide a Job Description first.")
        st.stop()

    if "Paste Web Link" in input_mode:
        if not st.session_state.get("jd_confirmed", False):
            st.warning("Please review and confirm the Job Description first.")
            st.stop()

    
    
    if not screening_settings_valid:
        st.error(
            "Enter a valid shortlist threshold, or choose a "
            "Maximum Resumes value for Top-N mode. "
            "Check that the Review range is valid."
        )
        st.stop()

    try:
        with st.spinner("Parsing JD and screening resumes..."):
            # 1. Parse the Job Description
            jd_structured = parse_job_description(jd_raw_text)

            st.session_state["parsed_jd"] = jd_structured
            st.session_state["jd_raw_text"] = jd_raw_text

            cleaned_csv = OUTPUT_DIR / "resumes_cleaned.csv"

            # 2. Load dataset resumes
            uploaded_resume_texts = []
            candidate_profiles = []

            if resume_limit != 0:
                candidate_profiles = parse_all_resumes(
                    cleaned_csv_path=cleaned_csv,
                    limit=resume_limit
                )

            # 3. Parse uploaded resumes, even if resume_limit == 0
            if uploaded_resumes:
                for i, uploaded_file in enumerate(
                    uploaded_resumes, start=1
                ):
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
                                f"Could not extract readable text from "
                                f"{uploaded_file.name}."
                            )
                            continue

                        resume_id = (
                            f"UPLOAD_{i}_{Path(uploaded_file.name).stem}"
                        )

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

            if not candidate_profiles:
                st.warning(
                    "No resumes found. Upload resumes or select a "
                    "dataset sample, then try again."
                )
                st.stop()

            # 4. Run the gate cascade for every candidate

            gate_config = jd_structured.get("gate_configuration", {})

            cascade_reports = [
                run_candidate_cascade(
                    jd_structured,
                    profile,
                    gate_config.get("gate_order"),
                    gate_config.get("gate_weights")
                )
                for profile in candidate_profiles
            ]

            # 5. Prepare dataset resume texts and IDs
            resume_texts = []
            candidate_ids = []

            dataset_profiles = [
                profile for profile in candidate_profiles
                if profile.get("category") != "Uploaded"
            ]

            if dataset_profiles:
                df_cleaned = pd.read_csv(cleaned_csv)

                if resume_limit is not None:
                    df_cleaned = df_cleaned.head(resume_limit)

                # Keep dataset IDs and resume texts in the same order
                dataset_count = min(
                    len(dataset_profiles),
                    len(df_cleaned)
                )

                for i in range(dataset_count):
                    resume_texts.append(
                        str(df_cleaned.iloc[i]["cleaned_text"])
                    )
                    candidate_ids.append(
                        dataset_profiles[i]["candidate_id"]
                    )

            # Add uploaded resumes
            for candidate_id, raw_text in uploaded_resume_texts:
                resume_texts.append(raw_text)
                candidate_ids.append(candidate_id)

            # 6. Calculate semantic similarity scores
            semantic_map = {}

            if resume_texts:
                semantic_scorer = SemanticScorer(
                    model_name="all-MiniLM-L6-v2"
                )

                semantic_results = (
                    semantic_scorer.compute_similarity_scores(
                        jd_text=jd_raw_text,
                        resume_texts=resume_texts,
                        candidate_ids=candidate_ids
                    )
                )

                semantic_map = {
                    result["candidate_id"]: result["semantic_score"]
                    for result in semantic_results
                }

            # 7. Combine scores
            records = []

            for rep in cascade_reports:
                candidate_id = rep["candidate_id"]
                gate_score = rep["final_gate_score"]
                semantic_score = semantic_map.get(candidate_id, 0.0)

                final_score = calculate_semantic_combined_score(
                    gate_score,
                    semantic_score
                )

                records.append({
                    "Candidate_ID": candidate_id,
                    "Category": rep["category"],
                    "Gate_Score": gate_score,
                    "Semantic_Score": semantic_score,
                    "Final_Score": final_score,
                    "Status": "PENDING",
                    "Rejection_Reason": (
                        rep.get("rejection_reason") or "N/A"
                    ),
                    "Matched_Skills": ", ".join(
                        rep.get("matched_skills", [])
                    ) or "None",
                    "Missing_Skills": ", ".join(
                        rep.get("missing_skills", [])
                    ) or "None",
                })

            # 8. Rank candidates from highest to lowest score
            df_results = pd.DataFrame(records)

            if df_results.empty:
                st.warning("No candidate results were generated.")
                st.stop()

            df_results.sort_values(
                by="Final_Score",
                ascending=False,
                inplace=True
            )
            df_results.reset_index(drop=True, inplace=True)
            df_results["Rank"] = df_results.index + 1

            # 9. Apply shortlist and review thresholds
            
            statuses = []
            shortlisted_count = 0

            for _, row in df_results.iterrows():
                score = float(row["Final_Score"])

                # Mode 1: Top-N only, without a shortlist threshold
                if shortlist_min is None:
                    if top_n is not None and shortlisted_count < top_n:
                        statuses.append("SHORTLISTED")
                        shortlisted_count += 1
                    else:
                        statuses.append("REJECTED")

                # Mode 2: Threshold only, or threshold + Top-N
                else:
                    if score >= shortlist_min and (
                        top_n is None or shortlisted_count < top_n
                    ):
                        statuses.append("SHORTLISTED")
                        shortlisted_count += 1
                    elif review_min <= score <= review_max:
                        statuses.append("REVIEW")
                    else:
                        statuses.append("REJECTED")

            df_results["Status"] = statuses


            # Show only shortlisted candidates when Top-N is selected
            if top_n is not None:
                df_results = df_results[
                    df_results["Status"] == "SHORTLISTED"
                ].head(top_n).copy()

                df_results.reset_index(drop=True, inplace=True)


            # 10. Save results and make them available to the UI
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            df_results.to_csv(
                OUTPUT_DIR / "shortlist_results.csv",
                index=False
            )

            with open(
                OUTPUT_DIR / "shortlist_results.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(cascade_reports, f, indent=2)

            st.session_state["screening_df"] = df_results


            # Move Rank to the first column
            if "Rank" in df_results.columns:
                cols = ["Rank"] + [
                col for col in df_results.columns
                if col != "Rank"
                ]
            df_results = df_results[cols]
            st.session_state["screening_df"] = df_results


            st.subheader("📊 Candidate Screening Results")
            st.dataframe(
                st.session_state["screening_df"],
                use_container_width=True,
                hide_index=True
                )

            if "hr_manual_decisions" not in st.session_state:
                st.session_state["hr_manual_decisions"] = {}

            st.success(
                f"Screening complete! Ranked {len(df_results)} candidates."
                )

            csv_df = st.session_state["screening_df"].copy()

            csv_data = csv_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="candidate_screening_results.csv",
                mime="text/csv",
            )

            # Download results as PDF
            from io import BytesIO
            import pandas as pd
            import matplotlib.pyplot as plt
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import landscape, A4
            from reportlab.platypus import (
                SimpleDocTemplate,
                Table,
                TableStyle,
                Paragraph,
                Spacer,
            )
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch

            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=landscape(A4),
                rightMargin=20,
                leftMargin=20,
                topMargin=25,
                bottomMargin=25,
            )

            styles = getSampleStyleSheet()
            elements = [
                Paragraph("Candidate Screening Results", styles["Title"]),
                Spacer(1, 0.2 * inch),
            ]

            pdf_df = st.session_state["screening_df"].fillna("")
            


            table_data = [
                [Paragraph(str(col), styles["Normal"]) for col in pdf_df.columns]
            ]

            for row in pdf_df.fillna("").astype(str).values.tolist():
                table_data.append([
                    Paragraph(
                        str(cell)
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"),
                        styles["Normal"]
                    )
                    for cell in row
                ])

            results_table = Table(
                table_data,
                repeatRows=1,
                splitByRow=1
            )

            results_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))


            elements.append(results_table)
            doc.build(elements)
            pdf_buffer.seek(0)

            st.download_button(
                label="📄 Download Results as PDF",
                data=pdf_buffer.getvalue(),
                file_name="candidate_screening_results.pdf",
                mime="application/pdf",
            )

    except Exception as e:
        st.error(f"Screening failed: {e}")
        st.exception(e)



