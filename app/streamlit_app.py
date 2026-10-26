
import io
import sys
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pdfplumber
from docx import Document

# Make the src folder importable when Streamlit runs this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jd_parser import parse_job_description, scrape_job_url
from resume_parser import parse_resume
from gate_cascade import run_candidate_cascade
from semantic_scorer import (
    SemanticScorer,
)


st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide",
)

st.title("AI-Powered Resume Screening")
st.caption(
    "Upload a job description and resumes to compare requirements, "
    "rank candidates and review the screening results."
)


# --------------------------------------------------
# 1. EXTRACT TEXT FROM PDF AND DOCX
# --------------------------------------------------

def extract_resume_text(uploaded_file):
    """Extract text from an uploaded PDF or DOCX file."""
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [
                page.extract_text() or ""
                for page in pdf.pages
            ]
        return "\n".join(pages).strip()

    if filename.endswith(".docx"):
        document = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs]
        return "\n".join(paragraphs).strip()

    return ""


# --------------------------------------------------
# 2. CLASSIFY CANDIDATES
# --------------------------------------------------

def classify_candidate(
    final_score,
    rank,
    required_experience_met,
    classification_mode,
    shortlist_value,
):
    review_score_floor = 50
    if not required_experience_met:
        return "Rejected"

    if classification_mode == "Score thresholds":
        if final_score >= shortlist_value:
            return "Shortlisted"
    elif rank <= shortlist_value:
        return "Shortlisted"

    if final_score >= review_score_floor:
        return "Under Review"
    return "Rejected"


def get_gate_score(report, gate_name):
    """Get a component score from the gate report."""
    gate = report.get("gate_results", {}).get(gate_name, {})
    return float(gate.get("score_pct", 0)) * 100


def get_skill_match_score(jd, report):
    """Return the percentage of listed JD skills matched by the resume."""
    requirements = jd.get("requirements", {}).get("skills", {})
    required_skills = [
        item.get("skill", "")
        for group in ("must_have", "preferred")
        for item in requirements.get(group, [])
        if item.get("skill")
    ]

    if not required_skills:
        return 100.0

    matched_skills = {
        re.sub(r"\s*\(partial\)$", "", skill, flags=re.IGNORECASE).lower()
        for skill in report.get("matched_skills", [])
    }
    matched_count = sum(
        skill.lower() in matched_skills for skill in required_skills
    )
    return matched_count / len(required_skills) * 100


def get_education_match_score(jd, profile):
    """Give partial credit for a related but lower-level degree."""
    required = jd.get("requirements", {}).get("education", {}).get(
        "required_degrees", []
    )
    candidate = profile.get("education", {}).get("degrees", [])
    if not required:
        return 100.0
    if not candidate:
        return 0.0

    degree_levels = {
        "diploma": 1,
        "associate": 1,
        "bachelor": 2,
        "master": 3,
        "doctor": 4,
        "phd": 4,
    }

    def degree_level(degree):
        degree = degree.lower()
        for label, level in degree_levels.items():
            if label in degree:
                return level
        return 0

    best_score = 0.0
    for required_degree in required:
        required_level = degree_level(required_degree)
        for candidate_degree in candidate:
            candidate_level = degree_level(candidate_degree)
            if required_degree.lower() in candidate_degree.lower():
                score = 100.0
            elif candidate_level > required_level > 0:
                score = 85.0
            elif candidate_level and required_level:
                score = 55.0
            else:
                score = 0.0
            best_score = max(best_score, score)
    return best_score


def get_resume_category(uploaded_file):
    """Use an uploaded subfolder as the category when directory paths exist."""
    relative_path = getattr(uploaded_file, "relative_path", None)
    path = Path(str(relative_path or uploaded_file.name).replace("\\", "/"))
    parent = path.parent.name
    return parent if parent not in ("", ".") else "Unspecified"


def count_skills_by_candidate(values):
    """Count how many resumes mention each skill, counting each once per resume."""
    counts = Counter()
    for value in values.fillna(""):
        skills = {
            re.sub(r"\s*\(partial\)$", "", skill, flags=re.IGNORECASE).strip()
            for skill in str(value).split(",")
        }
        counts.update(skill for skill in skills if skill and skill.lower() != "none")
    return counts.most_common(10)


def get_classification_reason(candidate, classification_mode, shortlist_value):
    """Explain the decision using the candidate's measured component scores."""
    skill_score = candidate["Skill_Match_%"]
    experience_score = candidate["Experience_Match_%"]
    education_score = candidate["Education_Match_%"]
    final_score = candidate["Final_Score"]
    matched = candidate["Matched_Skills"] or "none"
    missing = candidate["Missing_Skills"] or "none"
    evidence = (
        f"Skills {skill_score:.1f}%, experience {experience_score:.1f}% "
        f"({candidate['Experience_Years']:.1f} years parsed), and education "
        f"{education_score:.1f}%. Matched skills: {matched}. "
        f"Missing skills: {missing}."
    )

    if not candidate["Required_Experience_Met"]:
        return f"Rejected: {candidate['Experience_Reason']} {evidence}"

    if candidate["Category"] == "Shortlisted":
        if classification_mode == "Score thresholds":
            basis = f"final score {final_score:.1f} met the {shortlist_value} shortlist threshold"
        else:
            basis = f"rank {int(candidate['Rank'])} was within the top {shortlist_value}"
        return f"Shortlisted: {basis}. {evidence}"

    if candidate["Category"] == "Under Review":
        if classification_mode == "Score thresholds":
            basis = (
                f"final score {final_score:.1f} was below the {shortlist_value} "
                "shortlist threshold but met the 50-point review floor"
            )
        else:
            basis = (
                f"rank {int(candidate['Rank'])} was outside the top {shortlist_value} "
                "shortlist ranks, while the score met the 50-point review floor"
            )
        return f"Under Review: {basis}. {evidence}"

    low_components = []
    if skill_score < 50:
        low_components.append(f"skills ({skill_score:.1f}%)")
    if experience_score < 50:
        low_components.append(f"experience ({experience_score:.1f}%)")
    if education_score < 50:
        low_components.append(f"education ({education_score:.1f}%)")
    weak_reason = ", ".join(low_components) or "overall score"
    return (
        f"Rejected: final score {final_score:.1f} was below the 50-point "
        f"review floor; low match in {weak_reason}. {evidence}"
    )


# --------------------------------------------------
# 3. JOB DESCRIPTION INPUT
# --------------------------------------------------

st.header("1. Job Description")

jd_method = st.radio(
    "Choose how to provide the job description",
    ["Paste job description", "LinkedIn job URL"],
    horizontal=True,
)

job_title = st.text_input(
    "Job title",
    placeholder="e.g. Data Analyst",
)

jd_text = ""

if jd_method == "Paste job description":
    jd_text = st.text_area(
        "Paste the full job description",
        height=220,
        placeholder=(
            "Include required skills, education, minimum experience "
            "and preferred qualifications."
        ),
    )

else:
    job_url = st.text_input(
        "LinkedIn job posting URL",
        placeholder="https://www.linkedin.com/jobs/view/...",
    )

    if st.button("Fetch job description"):
        if not job_url.startswith("https://"):
            st.error("Please enter a valid HTTPS URL.")
        else:
            with st.spinner("Attempting to retrieve the job description..."):
                scraped = scrape_job_url(job_url)

            if scraped and len(scraped.strip()) > 100:
                st.session_state["scraped_jd_text"] = scraped
                st.success("Page text retrieved. Review it before screening.")
            else:
                st.warning(
                    "The page could not be retrieved reliably. "
                    "LinkedIn may restrict automated access. "
                    "Paste the job description manually instead."
                )

    jd_text = st.session_state.get("scraped_jd_text", "")

    if jd_text:
        jd_text = st.text_area(
            "Review or edit the retrieved job description",
            value=jd_text,
            height=220,
            key="reviewed_jd_text",
        )


# --------------------------------------------------
# 4. RESUME UPLOAD
# --------------------------------------------------

st.header("2. Upload Resumes")

uploaded_resumes = st.file_uploader(
    "Select a resume folder or multiple PDF/DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files="directory",
    help="Choose a folder to include its PDF and DOCX files, or add files individually.",
)

# --------------------------------------------------
# 5. SCREENING
# --------------------------------------------------

st.header("3. Run Screening")

classification_mode = st.radio(
    "Choose how candidates are classified",
    ["Score thresholds", "Rank ranges"],
    horizontal=True,
)

if classification_mode == "Score thresholds":
    shortlist_value = st.number_input(
        "Shortlist score threshold",
        min_value=50,
        max_value=100,
        value=70,
        help="Candidates at or above this value are shortlisted.",
    )
    st.caption(
        "Below the shortlist score, candidates scoring 50 or higher are "
        "under review; lower scores are rejected."
    )
else:
    shortlist_value = st.number_input(
        "Number of candidates to shortlist",
        min_value=1,
        max_value=10000,
        value=50,
        help="The top candidates by final score, through this rank, are shortlisted.",
    )
    st.caption(
        "Outside the shortlist, candidates scoring 50 or higher are under "
        "review; lower scores are rejected. Candidates failing required "
        "experience are rejected regardless of rank."
    )

if st.button("Screen and Rank Candidates", type="primary"):

    if not jd_text or not jd_text.strip():
        st.error("Provide a job description before screening.")

    elif not uploaded_resumes:
        st.error("Upload at least one PDF or DOCX resume.")

    else:
        with st.spinner("Parsing the JD and evaluating resumes..."):

            jd = parse_job_description(
                jd_text,
                role_title=job_title or "Unspecified Role",
                source_url=(
                    job_url
                    if jd_method == "LinkedIn job URL"
                    else None
                ),
            )

            profiles = []
            raw_texts = []
            filenames = []
            errors = []

            for index, file in enumerate(uploaded_resumes, start=1):
                try:
                    text = extract_resume_text(file)
                    resume_category = get_resume_category(file)

                    if not text:
                        errors.append(
                            f"{file.name}: no readable text found"
                        )
                        continue

                    profile = parse_resume(
                        resume_id=f"UPLOAD_{index}",
                        raw_text=text,
                        category=resume_category,
                    )

                    profiles.append(profile)
                    raw_texts.append(text)
                    filenames.append(file.name)

                except Exception as exc:
                    errors.append(f"{file.name}: {exc}")

            if not profiles:
                st.error(
                    "No readable resumes were extracted. "
                    "Check that the PDFs contain selectable text."
                )

            else:
                gate_reports = [
                    run_candidate_cascade(
                        jd,
                        profile,
                        jd.get("gate_configuration", {}).get("gate_order"),
                        jd.get("gate_configuration", {}).get("gate_weights"),
                        stop_on_failure=False,
                    )
                    for profile in profiles
                ]

                semantic_available = True
                try:
                    scorer = SemanticScorer(
                        model_name="all-MiniLM-L6-v2"
                    )

                    semantic_results = scorer.compute_similarity_scores(
                        jd_text=jd_text,
                        resume_texts=raw_texts,
                        candidate_ids=[
                            p["candidate_id"] for p in profiles
                        ],
                    )

                    semantic_map = {
                        item["candidate_id"]: item["semantic_score"]
                        for item in semantic_results
                    }
                except Exception:
                    semantic_available = False
                    semantic_map = {}
                    st.warning(
                        "Semantic similarity is unavailable; candidates will "
                        "be ranked using skills, education, and experience."
                    )

                rows = []

                for profile, report, filename in zip(
                    profiles, gate_reports, filenames
                ):
                    cid = profile["candidate_id"]

                    skill_score = get_skill_match_score(jd, report)
                    education_score = get_education_match_score(jd, profile)
                    experience_score = get_gate_score(
                        report, "experience"
                    )

                    semantic_score = semantic_map.get(cid, 0.0)

                    required_experience_met = (
                        report.get("gate_results", {})
                        .get("experience", {})
                        .get("status") == "PASS"
                    )
                    relevant_experience_score = (
                        experience_score * skill_score / 100
                    )
                    final_score = round(
                        skill_score * 0.45
                        + relevant_experience_score * 0.40
                        + education_score * 0.15,
                        2,
                    )
                    if not required_experience_met:
                        final_score = min(final_score, 15.0)

                    actual_experience = float(
                        profile.get("experience", {}).get("years", 0)
                    )

                    matched = sorted(set(report.get("matched_skills", [])))
                    missing = sorted(set(report.get("missing_skills", [])))

                    rows.append({
                        "Rank": 0,
                        "Candidate_ID": cid,
                        "Resume": filename,
                        "Resume_Category": profile.get(
                            "category", "Unspecified"
                        ),
                        "Skill_Match_%": round(skill_score, 2),
                        "Education_Match_%": round(education_score, 2),
                        "Experience_Match_%": round(experience_score, 2),
                        "Semantic_Match_%": round(semantic_score, 2),
                        "Final_Score": round(final_score, 2),
                        "Category": "",
                        "Required_Experience_Met": required_experience_met,
                        "Experience_Years": actual_experience,
                        "Matched_Skills": ", ".join(matched),
                        "Missing_Skills": ", ".join(missing),
                        "Experience_Reason": (
                            report.get("gate_results", {})
                            .get("experience", {})
                            .get("reason", "Required experience not met")
                            if not required_experience_met
                            else "Required experience was met."
                        ),
                        "Reason": "",
                    })

                results_df = pd.DataFrame(rows)

                # Sort all candidates by final score, highest first.
                results_df = results_df.sort_values(
                    ["Final_Score", "Semantic_Match_%"],
                    ascending=[False, False],
                ).reset_index(drop=True)

                results_df["Rank"] = range(1, len(results_df) + 1)
                for row_index in range(len(results_df)):
                    row = results_df.iloc[row_index]
                    results_df.at[row_index, "Category"] = classify_candidate(
                        final_score=row["Final_Score"],
                        rank=row["Rank"],
                        required_experience_met=row["Required_Experience_Met"],
                        classification_mode=classification_mode,
                        shortlist_value=shortlist_value,
                    )
                    results_df.at[row_index, "Reason"] = get_classification_reason(
                        results_df.iloc[row_index],
                        classification_mode,
                        shortlist_value,
                    )

                st.session_state["screening_results"] = results_df
                st.session_state["screening_jd"] = jd

                if errors:
                    st.warning(
                        "Some files could not be processed:\n\n"
                        + "\n".join(errors)
                    )

                st.success(
                    f"Screening completed for {len(results_df)} resumes."
                )


# --------------------------------------------------
# 6. DISPLAY RESULTS
# --------------------------------------------------

if "screening_results" in st.session_state:
    results_df = st.session_state["screening_results"]
    if "Resume_Category" not in results_df:
        results_df = results_df.copy()
        results_df["Resume_Category"] = "Unspecified"

    st.header("4. Ranked Screening Results")

    category_filter = st.multiselect(
        "Filter by category",
        ["Shortlisted", "Under Review", "Rejected"],
        default=["Shortlisted", "Under Review", "Rejected"],
    )
    filtered_df = results_df[
        results_df["Category"].isin(category_filter)
    ]
    page_size = 25
    page_count = max(1, (len(filtered_df) + page_size - 1) // page_size)
    current_page = st.number_input(
        "Results page",
        min_value=1,
        max_value=page_count,
        step=1,
    )
    start_index = (current_page - 1) * page_size
    page_df = filtered_df.iloc[start_index:start_index + page_size]
    st.caption(
        f"Showing {len(page_df)} of {len(filtered_df)} matching resumes "
        f"(page {current_page} of {page_count})."
    )

    ranked_header = st.columns([0.45, 2.4, 0.8, 0.8, 0.8, 0.8, 1.0])
    for column, label in zip(
        ranked_header,
        ["Rank", "Resume", "Skills", "Education", "Experience", "Final", "Category"],
    ):
        column.markdown(f"**{label}**")
    for _, row in page_df.iterrows():
        columns = st.columns([0.45, 2.4, 0.8, 0.8, 0.8, 0.8, 1.0])
        columns[0].write(row["Rank"])
        columns[1].write(row["Resume"])
        columns[2].write(f"{row['Skill_Match_%']:.0f}%")
        columns[3].write(f"{row['Education_Match_%']:.0f}%")
        columns[4].write(f"{row['Experience_Match_%']:.0f}%")
        columns[5].write(f"{row['Final_Score']:.1f}")
        columns[6].write(row["Category"])

    st.subheader("Candidate Details")
    selected_resume = st.selectbox(
        "Choose a resume to inspect",
        results_df["Resume"].tolist(),
    )
    candidate = results_df[results_df["Resume"] == selected_resume].iloc[0]
    st.write(f"**Rank:** {candidate['Rank']}")
    st.write(f"**Category:** {candidate['Category']}")
    st.write(f"**Final score:** {candidate['Final_Score']}/100")
    score_cols = st.columns(3)
    score_cols[0].metric("Skills match", f"{candidate['Skill_Match_%']:.1f}%")
    score_cols[1].metric("Experience match", f"{candidate['Experience_Match_%']:.1f}%")
    score_cols[2].metric("Education match", f"{candidate['Education_Match_%']:.1f}%")
    st.write(f"**Matched skills:** {candidate['Matched_Skills'] or 'None'}")
    st.write(f"**Missing skills:** {candidate['Missing_Skills'] or 'None'}")
    st.write(f"**Reason for decision:** {candidate.get('Reason', 'Not available')}")

    st.download_button(
        "Download ranked results as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="ranked_resume_results.csv",
        mime="text/csv",
    )

    st.header("5. Recruiter Insights Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Resumes", len(results_df))
    col2.metric(
        "Shortlisted",
        int((results_df["Category"] == "Shortlisted").sum()),
    )
    col3.metric(
        "Under Review",
        int((results_df["Category"] == "Under Review").sum()),
    )
    col4.metric(
        "Rejected",
        int((results_df["Category"] == "Rejected").sum()),
    )

    status_order = ["Shortlisted", "Under Review", "Rejected"]
    status_colors = {
        "Shortlisted": "#2e8b57",
        "Under Review": "#d99a22",
        "Rejected": "#c85151",
    }
    status_counts = results_df["Category"].value_counts()
    summary_col, score_col = st.columns(2)

    with summary_col:
        st.subheader("Screening outcomes")
        outcome_chart = go.Figure(
            go.Bar(
                x=status_order,
                y=[int(status_counts.get(status, 0)) for status in status_order],
                marker_color=[status_colors[status] for status in status_order],
                text=[int(status_counts.get(status, 0)) for status in status_order],
                textposition="outside",
            )
        )
        outcome_chart.update_layout(
            height=320,
            xaxis_title="",
            yaxis_title="Resumes",
            showlegend=False,
        )
        st.plotly_chart(
            outcome_chart,
            alt="Resume counts for shortlisted, under-review, and rejected candidates",
        )

    with score_col:
        st.subheader("Final score distribution")
        score_chart = go.Figure(
            go.Histogram(
                x=results_df["Final_Score"].tolist(),
                xbins={"start": 0, "end": 100, "size": 10},
                marker_color="#337f88",
            )
        )
        score_chart.update_layout(
            height=320,
            xaxis_title="Final score",
            yaxis_title="Resumes",
            bargap=0.08,
        )
        st.plotly_chart(
            score_chart,
            alt="Distribution of final resume screening scores from zero to one hundred",
        )

    st.subheader("Resume category analysis")
    st.caption(
        "Category labels come from uploaded subfolders when available; "
        "otherwise candidates are shown as Unspecified."
    )
    category_summary = results_df.groupby("Resume_Category").agg(
        Total=("Candidate_ID", "count"),
        Shortlisted=(
            "Category",
            lambda values: int((values == "Shortlisted").sum()),
        ),
    )
    category_summary["Shortlist_Rate"] = (
        category_summary["Shortlisted"] / category_summary["Total"] * 100
    )
    category_names = category_summary.index.tolist()

    category_col, rate_col = st.columns(2)
    with category_col:
        category_chart = go.Figure()
        for status in status_order:
            counts = [
                int(
                    (
                        (results_df["Resume_Category"] == category)
                        & (results_df["Category"] == status)
                    ).sum()
                )
                for category in category_names
            ]
            category_chart.add_trace(
                go.Bar(
                    name=status,
                    x=category_names,
                    y=counts,
                    marker_color=status_colors[status],
                )
            )
        category_chart.update_layout(
            barmode="stack",
            height=350,
            xaxis_title="Resume category",
            yaxis_title="Candidates",
            legend_title="Decision",
        )
        st.plotly_chart(
            category_chart,
            alt="Candidate outcome counts grouped by resume category",
        )

    with rate_col:
        rate_chart = go.Figure(
            go.Bar(
                x=category_names,
                y=category_summary["Shortlist_Rate"].tolist(),
                marker_color="#337f88",
                text=[
                    f"{rate:.1f}%"
                    for rate in category_summary["Shortlist_Rate"]
                ],
                textposition="outside",
            )
        )
        rate_chart.update_layout(
            height=350,
            xaxis_title="Resume category",
            yaxis_title="Shortlist rate (%)",
            yaxis_range=[0, 100],
            showlegend=False,
        )
        st.plotly_chart(rate_chart, alt="Shortlist rate by resume category")

    st.subheader("Most matched and missing skills")
    matched_counts = count_skills_by_candidate(results_df["Matched_Skills"])
    missing_counts = count_skills_by_candidate(results_df["Missing_Skills"])
    matched_col, missing_col = st.columns(2)

    with matched_col:
        st.markdown("**Most commonly matched**")
        if matched_counts:
            skills, counts = zip(*reversed(matched_counts))
            matched_chart = go.Figure(
                go.Bar(
                    x=list(counts),
                    y=list(skills),
                    orientation="h",
                    marker_color="#2e8b57",
                )
            )
            matched_chart.update_layout(
                height=360,
                xaxis_title="Resumes",
                yaxis_title="",
            )
            st.plotly_chart(matched_chart, alt="Most commonly matched job skills")
        else:
            st.info("No matched skills were recorded.")

    with missing_col:
        st.markdown("**Most commonly missing**")
        if missing_counts:
            skills, counts = zip(*reversed(missing_counts))
            missing_chart = go.Figure(
                go.Bar(
                    x=list(counts),
                    y=list(skills),
                    orientation="h",
                    marker_color="#c85151",
                )
            )
            missing_chart.update_layout(
                height=360,
                xaxis_title="Resumes",
                yaxis_title="",
            )
            st.plotly_chart(missing_chart, alt="Most commonly missing job skills")
        else:
            st.info("No missing skills were recorded.")

    st.subheader("Top-ranked candidates")
    top_candidates = results_df.nlargest(10, "Final_Score").sort_values(
        "Final_Score",
        ascending=True,
    )
    ranking_chart = go.Figure()
    candidate_labels = [
        f"{row.Rank}. {row.Resume}"
        for row in top_candidates.itertuples()
    ]
    for label, column in [
        ("Skills", "Skill_Match_%"),
        ("Experience", "Experience_Match_%"),
        ("Education", "Education_Match_%"),
        ("Final score", "Final_Score"),
    ]:
        ranking_chart.add_trace(
            go.Bar(
                name=label,
                y=candidate_labels,
                x=top_candidates[column].tolist(),
                orientation="h",
            )
        )
    ranking_chart.update_layout(
        barmode="group",
        height=max(360, len(top_candidates) * 42),
        xaxis_title="Score (%)",
        yaxis_title="",
        xaxis_range=[0, 100],
        legend_title="Match",
    )
    st.plotly_chart(
        ranking_chart,
        alt="Top candidates compared by skill, experience, education, and final scores",
    )

