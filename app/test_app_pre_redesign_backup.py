"""
AI Resume Screening & Fairness Auditor
Run from the project root:
    python -m streamlit run app/test_app.py

Uses the existing project modules for JD parsing, resume parsing, screening gates,
semantic scoring, and saved JD storage. Email sending is enabled only when SMTP
credentials are configured in Streamlit secrets or environment variables.
"""
import json
import os
import re
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import streamlit as st
import textwrap
from pypdf import PdfReader
from docx import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR
from src.jd_parser import (
    parse_job_description, scrape_job_url, save_jd, list_saved_jds, load_jd
)
from src.resume_parser import parse_all_resumes, parse_resume
from src.gate_cascade import run_candidate_cascade
from src.semantic_scorer import SemanticScorer, calculate_semantic_combined_score

st.set_page_config(
    page_title="AI Resume Screening & Fairness Auditor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --navy: #111827;
    --navy-deep: #0b1020;
    --blue: #4f46e5;
    --violet: #7c3aed;
    --teal: #0d9488;
    --green: #059669;
    --text: #182235;
    --muted: #64748b;
    --border: #e4e9f2;
    --surface: #ffffff;
    --page: #f5f7fc;
}

.stApp {
    background:
        radial-gradient(ellipse at 8% 0%, rgba(99,102,241,.07), transparent 30%),
        radial-gradient(ellipse at 95% 8%, rgba(20,184,166,.06), transparent 26%),
        var(--page);
    color: var(--text);
}

.block-container {
    max-width: 1600px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #11152b 0%, #171b3b 52%, #172e49 100%);
    border-right: 1px solid rgba(255,255,255,.09);
}

[data-testid="stSidebar"] * {
    color: #eef2ff;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 10px;
    padding: 7px 9px;
    transition: background .2s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(129,140,248,.16);
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #a5b4fc;
}

h1, h2, h3, h4 {
    color: #17213b;
    letter-spacing: -.035em;
    font-weight: 750;
}

h1 {
    font-size: clamp(1.8rem, 3vw, 2.5rem);
}

h2 {
    font-size: 1.55rem;
}

h3 {
    font-size: 1.2rem;
}

.hero-card {
    padding: 30px 34px;
    margin-bottom: 25px;
    border-radius: 22px;
    color: #ffffff;
    background: linear-gradient(115deg, #312e81 0%, #4f46e5 43%, #7c3aed 72%, #0f9f9a 115%);
    box-shadow: 0 15px 38px rgba(79,70,229,.20);
    border: 1px solid rgba(255,255,255,.18);
}

.hero-card * {
    color: #ffffff;
}

[data-testid="stMetric"] {
    position: relative;
    background: linear-gradient(145deg, #ffffff, #f9faff);
    border: 1px solid #e5e7f4;
    border-radius: 17px;
    padding: 19px 20px;
    box-shadow: 0 6px 18px rgba(31,41,85,.055);
    transition: transform .18s ease, box-shadow .18s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(79,70,229,.11);
}

[data-testid="stMetricLabel"] {
    color: #667085;
    font-size: .88rem;
    font-weight: 650;
}

[data-testid="stMetricValue"] {
    color: #3730a3;
    font-weight: 800;
    letter-spacing: -.035em;
}

[data-testid="stMetricDelta"] {
    font-weight: 600;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.96);
    border-color: #e4e8f2;
    border-radius: 17px;
    box-shadow: 0 5px 18px rgba(27,39,94,.045);
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 11px;
    border: 1px solid #4f46e5;
    background: linear-gradient(110deg, #4f46e5, #6d28d9);
    color: #ffffff;
    font-weight: 700;
    padding: .58rem 1rem;
    box-shadow: 0 4px 11px rgba(79,70,229,.14);
    transition: all .18s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #4338ca;
    background: linear-gradient(110deg, #4338ca, #5b21b6);
    color: #ffffff;
    transform: translateY(-1px);
    box-shadow: 0 7px 17px rgba(79,70,229,.22);
}

.stButton > button:focus,
.stDownloadButton > button:focus {
    box-shadow: 0 0 0 .2rem rgba(99,102,241,.22);
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"],
.stNumberInput input {
    border-radius: 10px;
    border-color: #d7dced;
    background: #ffffff;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: #818cf8;
    box-shadow: 0 0 0 2px rgba(99,102,241,.12);
}

[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
    border: 1px solid #e0e5f0;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 3px 12px rgba(31,41,85,.035);
}

[data-testid="stExpander"] {
    border: 1px solid #e1e6f1;
    border-radius: 13px;
    background: #ffffff;
}

[data-testid="stExpander"] summary {
    font-weight: 650;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4f46e5, #8b5cf6, #14b8a6);
}

[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    font-weight: 550;
}

hr {
    border-color: #e1e6f0;
    margin-top: 1.25rem;
    margin-bottom: 1.25rem;
}

[data-testid="stCaptionContainer"] {
    color: #69758b;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {
    accent-color: #818cf8;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero-card {
        padding: 21px;
        border-radius: 16px;
    }

    [data-testid="stMetric"] {
        padding: 13px;
    }
}
</style>
<div class="hero-card">
  <div class="hero-title">🤖 AI Resume Screening & Fairness Auditor</div>
  <div class="hero-subtitle">Intelligent candidate matching · Explainable AI · HR review · Recruitment communication</div>
</div>
""", unsafe_allow_html=True)


def clean_jd_text(text):
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b-\u200f\uFEFF]", "", text)
    markers = [
        "Instructions for Candidates", "Apply for this job", "Select Country",
        "Upload Document", "I have read and agree to the Privacy Policy",
        "This site uses cookies", "Click here to read our",
    ]
    lower = text.casefold()
    positions = [lower.find(m.casefold()) for m in markers if lower.find(m.casefold()) >= 0]
    if positions:
        text = text[:min(positions)]
    unwanted = {
        "home", "about us", "contact us", "privacy policy", "terms of use",
        "terms & conditions", "accept cookies", "reject cookies", "sign in",
        "sign up", "log in", "register", "skip to content", "all rights reserved",
        "apply now", "submit", "upload document",
    }
    lines, previous = [], None
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line and line.casefold() not in unwanted and line != previous:
            lines.append(line)
        previous = line
    return "\n".join(lines).strip()


def smtp_setting(name, default=None):
    try:
        value = st.secrets.get(name, default)
        if value is not None:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


def send_email(to_address, subject, body):
    """Send using SMTP settings from Streamlit secrets or environment variables."""
    host = smtp_setting("SMTP_HOST")
    port = int(smtp_setting("SMTP_PORT", 587))
    username = smtp_setting("SMTP_USERNAME")
    password = smtp_setting("SMTP_PASSWORD")
    sender = smtp_setting("SMTP_FROM", username)
    if not all([host, username, password, sender]):
        raise RuntimeError(
            "Email is not configured. Add SMTP_HOST, SMTP_PORT, SMTP_USERNAME, "
            "SMTP_PASSWORD and SMTP_FROM to Streamlit secrets or environment variables."
        )
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=20) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)


def get_value(row, *keys, default=""):
    for key in keys:
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value and value.lower() != "nan":
                return value
    return default


def email_templates(
    candidate_name, position, company, hr_name, hr_designation,
    hr_email="{hr_email}"
):
    signature = (
        f"Best regards,\n"
        f"{hr_name}\n"
        f"{hr_designation}\n"
        f"{company}\n"
        f"{hr_email}"
    )

    shortlisted_subject = f"Interview Opportunity — {position}"
    shortlisted_body = (
        f"Dear {candidate_name},\n\n"
        f"We are pleased to inform you that your profile has been shortlisted "
        f"for the {position} position at {company}. We will contact you "
        f"regarding the next steps in the recruitment process.\n\n"
        f"{signature}"
    )

    review_subject = f"Application Update — {position}"
    review_body = (
        f"Dear {candidate_name},\n\n"
        f"Thank you for your interest in the {position} position at {company}. "
        f"Your application is currently under review. We will update you once "
        f"a decision has been made.\n\n"
        f"{signature}"
    )

    rejected_subject = f"Application Update — {position}"
    rejected_body = (
        f"Dear {candidate_name},\n\n"
        f"Thank you for applying for the {position} position at {company}. "
        f"Unfortunately, we will not be moving forward with your application "
        f"for this position at this time. We appreciate your interest in our "
        f"company and wish you success in your future career.\n\n"
        f"{signature}"
    )

    return {
        "SHORTLISTED": (shortlisted_subject, shortlisted_body),
        "REVIEW": (review_subject, review_body),
        "REJECTED": (rejected_subject, rejected_body),
    }


def parse_uploaded_file(uploaded_file):
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    doc = Document(uploaded_file)
    return "\n".join(p.text for p in doc.paragraphs)


# Session defaults
for key, value in {
    "jd_raw_text": "",
    "jd_confirmed": False,
    "screening_df": pd.DataFrame(),
    "candidate_detail_map": {},
    "hr_manual_decisions": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = value



# --- Welcome screen: shown before the main workspace ---
if not st.session_state.get("welcome_screen_entered", False):
    st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 15% 20%, rgba(59,130,246,.38), transparent 35%),
            radial-gradient(circle at 85% 35%, rgba(139,92,246,.32), transparent 38%),
            radial-gradient(circle at 50% 90%, rgba(6,182,212,.24), transparent 40%),
            #070B16;
        background-size: 200% 200%;
        animation: aurora 8s ease-in-out infinite alternate;
        }

    @keyframes aurora {
        0%   { background-position: 0% 0%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 100%; }
        }

    .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .welcome-shell {
        position: relative;
        overflow: hidden;
        padding: clamp(28px, 5vw, 68px);
        border-radius: 30px;
        border: 1px solid rgba(148,163,184,.20);
        background: linear-gradient(
            135deg,
            rgba(15,23,42,.96),
            rgba(22,25,58,.94) 55%,
            rgba(10,35,53,.94)
        );
        box-shadow: 0 25px 100px rgba(0,0,0,.38);
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
        isolation: isolate;
    }

    .welcome-shell::before {
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        top: -190px;
        right: -90px;
        border-radius: 50%;
        background: rgba(99,102,241,.20);
        filter: blur(65px);
        z-index: -1;
        animation: glow 7s ease-in-out infinite alternate;
    }

    .welcome-shell::after {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        bottom: -200px;
        left: 15%;
        border-radius: 50%;
        background: rgba(6,182,212,.16);
        filter: blur(65px);
        z-index: -1;
    }

    @keyframes glow {
        from { transform: translate(0, 0) scale(1); }
        to { transform: translate(-35px, 30px) scale(1.2); }
    }

    .welcome-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 48px;
    }

    .welcome-logo {
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 16px;
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        box-shadow: 0 8px 30px rgba(59,130,246,.30);
        flex-shrink: 0;
    }

    .welcome-logo svg {
        width: 29px;
        height: 29px;
    }

    .welcome-brand-title {
        font-size: 17px;
        font-weight: 800;
        letter-spacing: -.4px;
        color: #FFFFFF;
    }

    .welcome-brand-subtitle {
        margin-top: 5px;
        color: #94A3B8;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.8px;
    }

    .welcome-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        padding: 9px 14px;
        border-radius: 999px;
        border: 1px solid rgba(129,140,248,.30);
        background: rgba(99,102,241,.12);
        color: #C4B5FD;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.7px;
    }

    .welcome-status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34D399;
        box-shadow: 0 0 12px rgba(52,211,153,.8);
    }

    .welcome-title {
        margin: 27px 0 0;
        color: #FFFFFF;
        font-size: clamp(38px, 6.2vw, 72px);
        font-weight: 900;
        line-height: 1.04;
        letter-spacing: -3px;
    }

    .welcome-gradient {
        display: inline-block;
        padding-bottom: 5px;
        background: linear-gradient(90deg, #93C5FD, #C4B5FD, #67E8F9);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .welcome-tagline {
        margin-top: 25px;
        color: #E2E8F0;
        font-size: clamp(19px, 2.5vw, 26px);
        font-weight: 700;
        letter-spacing: -.5px;
    }

    .welcome-description {
        max-width: 700px;
        margin-top: 15px;
        color: #AAB8CF;
        font-size: 15px;
        line-height: 1.9;
    }

    .welcome-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 24px;
    }

    .welcome-pill {
        padding: 8px 12px;
        border: 1px solid rgba(148,163,184,.17);
        border-radius: 9px;
        background: rgba(255,255,255,.035);
        color: #CBD5E1;
        font-size: 11px;
        font-weight: 600;
    }

    .welcome-pill span {
        color: #67E8F9;
        margin-right: 5px;
    }

    .welcome-features {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin-top: 45px;
    }

    .welcome-feature {
        position: relative;
        padding: 24px 21px;
        border: 1px solid rgba(148,163,184,.17);
        border-radius: 19px;
        background: linear-gradient(
            145deg,
            rgba(255,255,255,.075),
            rgba(255,255,255,.025)
        );
        transition: transform .3s ease, border-color .3s ease,
                    background .3s ease, box-shadow .3s ease;
        animation: rise .75s ease both;
    }

    .welcome-feature:nth-child(2) { animation-delay: .12s; }
    .welcome-feature:nth-child(3) { animation-delay: .24s; }

    .welcome-feature:hover {
        transform: translateY(-7px);
        border-color: rgba(96,165,250,.55);
        background: rgba(59,130,246,.09);
        box-shadow: 0 18px 35px rgba(0,0,0,.20);
    }

    .welcome-feature-icon {
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        margin-bottom: 20px;
    }

    .welcome-feature-icon svg {
        width: 25px;
        height: 25px;
    }

    .icon-blue {
        background: rgba(59,130,246,.16);
        color: #93C5FD;
    }

    .icon-purple {
        background: rgba(139,92,246,.17);
        color: #C4B5FD;
    }

    .icon-teal {
        background: rgba(20,184,166,.15);
        color: #5EEAD4;
    }

    .welcome-feature-title {
        margin-bottom: 10px;
        color: #F8FAFC;
        font-size: 16px;
        font-weight: 800;
    }

    .welcome-feature-description {
        color: #94A3B8;
        font-size: 12px;
        line-height: 1.8;
    }

    .welcome-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 35px;
        padding-top: 23px;
        border-top: 1px solid rgba(148,163,184,.15);
        color: #94A3B8;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
    }

    .welcome-footer-accent {
        color: #67E8F9;
        letter-spacing: .4px;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 700px) {
        .welcome-brand { margin-bottom: 35px; }
        .welcome-title { letter-spacing: -1.5px; }
        .welcome-features { grid-template-columns: 1fr; }
        .welcome-footer { line-height: 1.8; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
        }
    }
    </style>

    <div class="welcome-shell">

        <div class="welcome-brand">
            <div class="welcome-logo">
                <svg viewBox="0 0 24 24" fill="none"
                     stroke="white" stroke-width="1.8"
                     stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>
                    <rect x="6" y="6" width="12" height="12" rx="3"/>
                    <path d="M9 10h.01M15 10h.01M9 14h6"/>
                </svg>
            </div>
            <div>
                <div class="welcome-brand-title">AI Resume Fairness Auditor</div>
                <div class="welcome-brand-subtitle">
                    INTELLIGENT RECRUITMENT PLATFORM
                </div>
            </div>
        </div>

        <div class="welcome-eyebrow">
            <span class="welcome-status-dot"></span>
            NEXT-GEN RECRUITMENT INTELLIGENCE
        </div>

        <div class="welcome-title">
            Screen Smarter.<br>
            <span class="welcome-gradient">Hire Fairer.</span>
        </div>

        <div class="welcome-tagline">
            Better insights. More informed hiring.
        </div>

        <div class="welcome-description">
            Bring resume screening, AI-powered skill matching and hiring
            fairness analysis into one intelligent workspace. Discover
            relevant skills, understand candidate matches and explore
            screening outcomes through a clear, unified experience.
        </div>

        <div class="welcome-pills">
            <div class="welcome-pill"><span>◆</span> AI Skill Intelligence</div>
            <div class="welcome-pill"><span>◆</span> Explainable Matching</div>
            <div class="welcome-pill"><span>◆</span> Fairness Analytics</div>
        </div>

        <div class="welcome-features">

            <div class="welcome-feature">
                <div class="welcome-feature-icon icon-blue">
                    <svg viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="1.7"
                         stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <path d="M14 2v6h6M8 13h8M8 17h8"/>
                    </svg>
                </div>
                <div class="welcome-feature-title">Resume Screening</div>
                <div class="welcome-feature-description">
                    Upload resumes or use available candidate datasets
                    to organize profiles for job-specific screening.
                </div>
            </div>

            <div class="welcome-feature">
                <div class="welcome-feature-icon icon-purple">
                    <svg viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="1.7"
                         stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 3l1.9 5.8L20 11l-6.1 2.2L12 19l-2-5.8L4 11l6-2.2z"/>
                        <path d="M19 3v4M17 5h4M5 17v4M3 19h4"/>
                    </svg>
                </div>
                <div class="welcome-feature-title">AI Skill Matching</div>
                <div class="welcome-feature-description">
                    Compare candidate skills with job requirements
                    and understand relevant or missing skills.
                </div>
            </div>

            <div class="welcome-feature">
                <div class="welcome-feature-icon icon-teal">
                    <svg viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="1.7"
                         stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 3l8 3v5c0 5-3.4 8.5-8 10-4.6-1.5-8-5-8-10V6z"/>
                        <path d="M8 12l2.5 2.5L16 9"/>
                    </svg>
                </div>
                <div class="welcome-feature-title">Fairness Insights</div>
                <div class="welcome-feature-description">
                    Explore available fairness metrics and review
                    screening outcomes to support more transparent hiring.
                </div>
            </div>

        </div>

        <div class="welcome-footer">
            <span>YOUR RECRUITMENT INTELLIGENCE STARTS HERE</span>
            <span class="welcome-footer-accent">BUILT FOR SMARTER HIRING</span>
        </div>

    </div>
    """)

    _, launch_col, _ = st.columns([1, 2, 1])
    with launch_col:
        if st.button(
            "Launch Screening  →",
            key="launch_screening_welcome",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["welcome_screen_entered"] = True
            st.rerun()

    st.stop()


# --- Main AI Command Centre workspace continues below ---

with st.sidebar:
    st.markdown("""
        <div style="padding: 12px 4px 20px 4px;">
            <div style="font-size: 13px; letter-spacing: 2px;
                        color: #5EEAD4; font-weight: 700;">
                AI POWERED
            </div>
            <div style="font-size: 23px; font-weight: 800;
                        line-height: 1.2; margin-top: 6px;">
                Command Centre
            </div>
            <div style="font-size: 12px; opacity: 0.7; margin-top: 8px;">
                Resume Intelligence Platform
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("WORKSPACE NAVIGATION")

    selected_page = st.radio(
        "Navigate",
        ["Screening Workspace", "Analytics Centre", "HR Communications"],
        key="command_centre_navigation",
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("AI Resume Screening & Fairness Auditor")


if selected_page == "Screening Workspace":
    with st.container(border=True):
        st.header("1. Job Description")
        input_mode = st.radio(
            "Choose JD input method",
            ["🔗 Paste Web Link (Scraper)", "✍️ Type / Paste Text Manually", "📚 Load from JD Library"],
            horizontal=True,
            key="jd_input_mode",
        )

        if input_mode.startswith("🔗"):
            url = st.text_input("Job posting URL", placeholder="https://...")
            if st.button("🔗 Scrape Job Description", type="primary"):
                if not url.strip():
                    st.warning("Paste a job posting URL first.")
                else:
                    with st.spinner("Scraping job description..."):
                        try:
                            raw = clean_jd_text(scrape_job_url(url.strip()))
                            if not raw:
                                st.error("Could not extract JD text. Try manual paste.")
                            else:
                                st.session_state["jd_raw_text"] = raw
                                st.session_state.pop("url_jd_editor", None)
                                st.session_state["jd_source_url"] = url.strip()
                                st.session_state["jd_confirmed"] = False
                                st.success("JD scraped. Review it below and confirm.")
                        except Exception as exc:
                            st.error(f"Scraping failed: {exc}")
            jd_text = st.text_area(
                "Review/edit job description",
                value=st.session_state.get("jd_raw_text", ""),
                height=220,
                key="url_jd_editor",
            )
            if st.button("✅ Confirm JD", key="confirm_url_jd"):
                if jd_text.strip():
                    parsed = parse_job_description(
                        jd_text.strip(), role_title="HR Confirmed Role",
                        source_url=st.session_state.get("jd_source_url"),
                    )
                    parsed["raw_text"] = jd_text.strip()
                    saved = save_jd(parsed)
                    st.session_state["parsed_jd"] = parsed
                    st.session_state["jd_raw_text"] = jd_text.strip()
                    st.session_state["jd_confirmed"] = True
                    st.success(f"JD confirmed and saved as {saved.name}")
                else:
                    st.warning("JD cannot be empty.")

        elif input_mode.startswith("✍️"):
            jd_text = st.text_area(
                "Paste the complete job description",
                value=st.session_state.get("jd_raw_text", ""),
                height=250,
                key="manual_jd_editor",
            )
            if st.button("✅ Confirm and Save JD", key="confirm_manual_jd"):
                if jd_text.strip():
                    parsed = parse_job_description(jd_text.strip(), role_title="Manually Entered Role")
                    parsed["raw_text"] = jd_text.strip()
                    saved = save_jd(parsed)
                    st.session_state["parsed_jd"] = parsed
                    st.session_state["jd_raw_text"] = jd_text.strip()
                    st.session_state["jd_confirmed"] = True
                    st.success(f"JD confirmed and saved as {saved.name}")
                else:
                    st.warning("JD cannot be empty.")

        else:
            saved_list = list_saved_jds()
            if saved_list:
                filenames = [item["filename"] for item in saved_list]
                selected_file = st.selectbox("Saved job descriptions", filenames)
                if st.button("📚 Load selected JD"):
                    loaded = load_jd(selected_file)
                    if loaded:
                        st.session_state["parsed_jd"] = loaded
                        st.session_state["jd_raw_text"] = loaded.get("raw_text", "")
                        st.session_state["jd_confirmed"] = True
                        st.success(f"Loaded {selected_file}")
                    else:
                        st.error("Could not load the selected JD.")
            else:
                st.info("No saved JDs found. Add a JD using one of the other options.")

        if st.session_state.get("jd_raw_text"):
            with st.expander("Current JD text", expanded=False):
                st.write(st.session_state["jd_raw_text"])

    st.divider()
    with st.container(border=True):
        st.header("2. Upload Resumes")
        uploaded_resumes = st.file_uploader(
            "Upload PDF or DOCX resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Uploaded resumes are screened alongside the selected dataset sample.",
        )
        sample_choice = st.selectbox(
            "Existing dataset sample size",
            [0, 50, 100, 250, 500, 1000, "All 2,484 Resumes"],
            index=0,
            format_func=lambda x: "Uploaded resumes only (0 dataset resumes)" if x == 0 else str(x),
        )
        resume_limit = None if sample_choice == "All 2,484 Resumes" else sample_choice

    st.divider()
    with st.container(border=True):
        st.header("3. HR Screening Thresholds")
        c1, c2, c3 = st.columns(3)
        with c1:
            shortlist_min = st.number_input(
                "Shortlist minimum score (%)", min_value=1, max_value=100,
                value=70, step=1, key="hr_shortlist_min",
            )
        with c2:
            review_min = st.number_input(
                "Review minimum (%)", min_value=0, max_value=100,
                value=50, step=1, key="hr_review_min",
            )
            review_max = st.number_input(
                "Review maximum (%)", min_value=0, max_value=100,
                value=69, step=1, key="hr_review_max",
            )
        with c3:
            top_n = st.number_input(
                "Maximum shortlisted candidates (0 = unlimited)",
                min_value=0, max_value=100000, value=0, step=1,
                key="hr_top_n",
            )
        threshold_valid = review_min <= review_max < shortlist_min
        if not threshold_valid:
            st.error("Thresholds must satisfy Review minimum ≤ Review maximum < Shortlist minimum.")
        else:
            st.caption("Candidates at/above shortlist minimum are shortlisted; candidates in the review range need HR review; others are rejected.")

        if st.button("🧠 Parse Resumes & Screen Candidates", type="primary", use_container_width=True):
            jd_raw_text = st.session_state.get("jd_raw_text", "").strip()
            if not jd_raw_text:
                st.error("Enter and confirm a job description first.")
            elif not threshold_valid:
                st.error("Fix the threshold ranges before screening.")
            else:
                try:
                    with st.spinner("Parsing resumes and calculating scores..."):
                        jd_structured = parse_job_description(jd_raw_text)
                        jd_structured["raw_text"] = jd_raw_text
                        cleaned_csv = OUTPUT_DIR / "resumes_cleaned.csv"
                        profiles = []
                        uploaded_texts = []

                        if resume_limit != 0:
                            profiles.extend(parse_all_resumes(cleaned_csv_path=cleaned_csv, limit=resume_limit))

                        for i, upload in enumerate(uploaded_resumes or [], start=1):
                            try:
                                raw = parse_uploaded_file(upload)
                                if not raw.strip():
                                    st.warning(f"No readable text found in {upload.name}.")
                                    continue
                                cid = f"UPLOAD_{i}_{Path(upload.name).stem}"
                                profile = parse_resume(resume_id=cid, raw_text=raw, category="Uploaded")
                                profiles.append(profile)
                                uploaded_texts.append((profile.get("candidate_id", cid), raw, upload.name))
                            except Exception as exc:
                                st.warning(f"Could not parse {upload.name}: {exc}")

                        if not profiles:
                            st.error("No resumes found. Upload resumes or choose a dataset sample.")
                            st.stop()

                        gate_cfg = jd_structured.get("gate_configuration", {})
                        reports = [
                            run_candidate_cascade(
                                jd_structured, profile,
                                gate_cfg.get("gate_order"), gate_cfg.get("gate_weights")
                            )
                            for profile in profiles
                        ]

                        resume_texts, candidate_ids = [], []
                        dataset_profiles = [p for p in profiles if p.get("category") != "Uploaded"]
                        if dataset_profiles and cleaned_csv.exists():
                            raw_df = pd.read_csv(cleaned_csv)
                            if resume_limit is not None:
                                raw_df = raw_df.head(resume_limit)
                            count = min(len(dataset_profiles), len(raw_df))
                            for i in range(count):
                                text_col = "cleaned_text" if "cleaned_text" in raw_df.columns else "text"
                                resume_texts.append(str(raw_df.iloc[i].get(text_col, "")))
                                candidate_ids.append(dataset_profiles[i].get("candidate_id"))

                        for cid, raw, _filename in uploaded_texts:
                            resume_texts.append(raw)
                            candidate_ids.append(cid)

                        semantic_map = {}
                        if resume_texts:
                            scorer = SemanticScorer(model_name="all-MiniLM-L6-v2")
                            results = scorer.compute_similarity_scores(
                                jd_text=jd_raw_text, resume_texts=resume_texts,
                                candidate_ids=candidate_ids,
                            )
                            semantic_map = {
                                r["candidate_id"]: float(r["semantic_score"])
                                for r in results
                            }

                        records, detail_map = [], {}
                        for profile, report in zip(profiles, reports):
                            cid = str(report.get("candidate_id", profile.get("candidate_id", "")))
                            gate_score = float(report.get("final_gate_score", 0) or 0)
                            semantic_score = semantic_map.get(cid, 0.0)
                            final_score = float(calculate_semantic_combined_score(gate_score, semantic_score))
                            raw_resume = next((t for c, t, _ in uploaded_texts if str(c) == cid), "")
                            name = profile.get("name") or profile.get("candidate_name") or ""
                            email = profile.get("email") or profile.get("candidate_email") or ""
                            # Fallback extraction for uploaded resumes if parser did not return contact fields.
                            if raw_resume:
                                if not name:
                                    match = re.search(r"(?im)^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$", raw_resume)
                                    name = match.group(1) if match else Path(cid).stem
                                if not email:
                                    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", raw_resume)
                                    email = match.group(0) if match else ""
                            if not name:
                                name = str(cid)
                            record = {
                                "Candidate_ID": cid,
                                "Candidate_Name": str(name),
                                "Email": str(email),
                                "Category": report.get("category", profile.get("category", "")),
                                "Gate_Score": round(gate_score, 2),
                                "Semantic_Score": round(semantic_score, 2),
                                "Final_Score": round(final_score, 2),
                                "Matched_Skills": ", ".join(report.get("matched_skills", []) or []) or "None",
                                "Missing_Skills": ", ".join(report.get("missing_skills", []) or []) or "None",
                                "Rejection_Reason": report.get("rejection_reason") or "N/A",
                            }
                            records.append(record)
                            detail_map[cid] = {
                                **record,
                                "Gate_Results": report.get("gate_results", report.get("gates", {})),
                                "Raw_Resume_Text": raw_resume,
                                "Profile": profile,
                            }

                        result_df = pd.DataFrame(records).sort_values("Final_Score", ascending=False).reset_index(drop=True)
                        result_df["Rank"] = result_df.index + 1
                        statuses, shortlist_count = [], 0
                        for _, row in result_df.iterrows():
                            score = float(row["Final_Score"])
                            if score >= shortlist_min and (top_n == 0 or shortlist_count < top_n):
                                statuses.append("SHORTLISTED")
                                shortlist_count += 1
                            elif review_min <= score <= review_max:
                                statuses.append("REVIEW")
                            else:
                                statuses.append("REJECTED")
                        result_df["Status"] = statuses
                        result_df["HR_Decision"] = result_df["Status"]
                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        result_df.to_csv(OUTPUT_DIR / "shortlist_results.csv", index=False)
                        (OUTPUT_DIR / "shortlist_results.json").write_text(
                            result_df.to_json(orient="records", indent=2), encoding="utf-8"
                        )
                        st.session_state["screening_df"] = result_df
                        st.session_state["candidate_detail_map"] = detail_map
                        st.session_state["jd_position"] = (
                            jd_structured.get("metadata", {}).get("job_title")
                            or jd_structured.get("job_title")
                            or "the advertised position"
                        )
                        st.session_state["screening_completed"] = True
                        st.success(f"Screening completed: {len(result_df)} candidates processed.")
                except Exception as exc:
                    st.error(f"Screening failed: {exc}")
                    st.exception(exc)

        df = st.session_state.get("screening_df", pd.DataFrame())
        if not df.empty:
            st.divider()
            st.header("4. Candidate Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Candidates", len(df))
            m2.metric("Shortlisted", int((df["Status"] == "SHORTLISTED").sum()))
            m3.metric("Review", int((df["Status"] == "REVIEW").sum()))
            m4.metric("Rejected", int((df["Status"] == "REJECTED").sum()))

            decision_options = ["SHORTLISTED", "REVIEW", "REJECTED"]
            edited = st.data_editor(
                df[["Rank", "Candidate_ID", "Candidate_Name", "Email", "Gate_Score",
                    "Semantic_Score", "Final_Score", "Matched_Skills", "Missing_Skills",
                    "Status", "HR_Decision", "Rejection_Reason"]],
                column_config={
                    "HR_Decision": st.column_config.SelectboxColumn(
                        "HR Decision", options=decision_options, required=True
                    ),
                    "Final_Score": st.column_config.NumberColumn("Final Score (%)", format="%.2f"),
                    "Gate_Score": st.column_config.NumberColumn("Gate Score", format="%.2f"),
                    "Semantic_Score": st.column_config.NumberColumn("Semantic Score", format="%.2f"),
                },
                disabled=["Rank", "Candidate_ID", "Candidate_Name", "Email", "Gate_Score",
                          "Semantic_Score", "Final_Score", "Matched_Skills", "Missing_Skills",
                          "Status", "Rejection_Reason"],
                use_container_width=True,
                hide_index=True,
                key="candidate_results_editor",
            )
            updated = df.copy()
            if "HR_Decision" in edited.columns:
                decisions = dict(zip(edited["Candidate_ID"].astype(str), edited["HR_Decision"]))
                updated["HR_Decision"] = updated["Candidate_ID"].astype(str).map(decisions).fillna(updated["Status"])
                st.session_state["screening_df"] = updated

            candidate_ids = updated["Candidate_ID"].astype(str).tolist()
            selected_id = st.selectbox(
                "Inspect a candidate's full details",
                candidate_ids,
                format_func=lambda cid: (
                    f"{updated.loc[updated['Candidate_ID'].astype(str) == cid, 'Candidate_Name'].iloc[0]} "
                    f"— {cid}"
                ),
            )
            details = st.session_state.get("candidate_detail_map", {}).get(selected_id, {})
            selected_row = updated[updated["Candidate_ID"].astype(str) == selected_id].iloc[0]
            d1, d2, d3 = st.columns(3)
            d1.metric("Final Score", f"{float(selected_row['Final_Score']):.2f}%")
            d2.metric("Semantic Score", f"{float(selected_row['Semantic_Score']):.2f}%")
            d3.metric("Gate Score", f"{float(selected_row['Gate_Score']):.2f}%")
            st.markdown(f"**Name:** {selected_row['Candidate_Name']}  \n**Email:** {selected_row['Email'] or 'Not found in resume'}  \n**Decision:** {selected_row['HR_Decision']}")
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("#### ✅ Matched skills")
                st.write(selected_row["Matched_Skills"])
            with s2:
                st.markdown("#### ⚠️ Missing skills")
                st.write(selected_row["Missing_Skills"])
            st.markdown("#### Screening / rejection reason")
            st.write(selected_row["Rejection_Reason"])
            if details.get("Gate_Results"):
                with st.expander("View gate results"):
                    st.json(details["Gate_Results"])
            if details.get("Raw_Resume_Text"):
                with st.expander("View extracted resume text"):
                    st.text(details["Raw_Resume_Text"])

            csv = updated.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Download candidate results (CSV)", csv,
                               "candidate_screening_results.csv", "text/csv")


elif selected_page == "Analytics Centre":
    with st.container(border=True):
        st.header("📊 Visualization & Analytics")
        df = st.session_state.get("screening_df", pd.DataFrame())
        if df.empty:
            st.info("Run candidate screening in the Screening Dashboard to see analytics here.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("Candidates", len(df))
            b.metric("Average final score", f"{df['Final_Score'].mean():.1f}%")
            c.metric("Highest final score", f"{df['Final_Score'].max():.1f}%")
            d.metric("Average semantic score", f"{df['Semantic_Score'].mean():.1f}%")
            st.subheader("Candidate decision distribution")
            status_counts = df["Status"].value_counts().reindex(
                ["SHORTLISTED", "REVIEW", "REJECTED"], fill_value=0
            )
            st.bar_chart(status_counts)
            st.subheader("Final score by candidate")
            chart_df = df[["Candidate_Name", "Final_Score"]].copy().set_index("Candidate_Name")
            st.bar_chart(chart_df)
            st.subheader("Matched vs missing skills")
            skill_counts = {}
            for col in ("Matched_Skills", "Missing_Skills"):
                for value in df[col].fillna(""):
                    for skill in str(value).split(","):
                        skill = skill.strip()
                        if skill and skill.lower() != "none":
                            skill_counts[(col, skill)] = skill_counts.get((col, skill), 0) + 1
            if skill_counts:
                skills_df = pd.DataFrame(
                    [{"Skill": skill, "Type": kind.replace("_Skills", ""), "Candidates": count}
                     for (kind, skill), count in skill_counts.items()]
                )
                st.dataframe(skills_df.sort_values("Candidates", ascending=False),
                             use_container_width=True, hide_index=True)
            st.subheader("Score distribution by decision")
            st.bar_chart(df.groupby("Status")["Final_Score"].mean())





elif selected_page == "HR Communications":
    with st.container(border=True):
        st.header("✉️ HR Email Centre")

        df = st.session_state.get("screening_df", pd.DataFrame()).copy()

        if df.empty:
            st.info("Screen candidates first, then review candidates and prepare emails.")
        else:
            valid_statuses = ["SHORTLISTED", "REVIEW", "REJECTED"]
            st.session_state.setdefault("hr_picker_version", 0)
            decision_labels = {
                "SHORTLISTED": "Shortlisted",
                "REVIEW": "Review",
                "REJECTED": "Rejected",
            }

            if "HR_Decision" not in df.columns:
                df["HR_Decision"] = df["Status"].astype(str).str.upper()

            df["HR_Decision"] = df.apply(
                lambda r: (
                    str(r.get("HR_Decision", "")).upper()
                    if str(r.get("HR_Decision", "")).upper() in valid_statuses
                    else str(r.get("Status", "REVIEW")).upper()
                ),
                axis=1,
            )
            st.session_state["screening_df"] = df

            st.subheader("1. Review candidates by HR decision")

            selected_items = []

            status_tabs = st.tabs(["✅ Shortlisted", "⏳ Review", "❌ Rejected"])

            for status, status_tab in zip(valid_statuses, status_tabs):
                with status_tab:
                    status_df = df[df["HR_Decision"] == status].copy()
                    st.metric(f"{decision_labels[status]} candidates", len(status_df))

                    if status_df.empty:
                        st.info("No candidates in this category.")
                        continue

                    display_cols = [
                        c for c in [
                            "Candidate_Name", "Candidate_ID", "Email",
                            "Final_Score", "Matched_Skills", "Missing_Skills",
                        ] if c in status_df.columns
                    ]
                    st.dataframe(
                        status_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                    )

                    picker_key = f"hr_picker_{status.lower()}"

                    options = status_df.apply(
                        lambda r: (
                            f"{r.get('Candidate_Name', 'Unknown')} | "
                            f"{r.get('Candidate_ID', '')}"
                        ),
                        axis=1,
                    ).tolist()


                    selected_options = st.multiselect(
                        f"Select {decision_labels[status]} candidates",
                        options=options,
                        key=f"hr_picker_{status.lower()}_{st.session_state['hr_picker_version']}",
                    )

                    selected_items.extend(selected_options)
            st.divider()

            if st.session_state.get("hr_decision_message"):
                st.success(st.session_state["hr_decision_message"])
                st.session_state["hr_decision_message"] = ""

            st.subheader("2. Change HR decision")

            selected_ids = list(dict.fromkeys(
                item.rsplit(" | ", 1)[-1] for item in selected_items
            ))

            st.write(f"Total selected candidates: **{len(selected_ids)}**")

            new_decision = st.selectbox(
                "Set new decision",
                valid_statuses,
                format_func=lambda x: decision_labels[x],
                key="hr_new_decision",
            )

            if st.button("Apply HR Decision", use_container_width=True):
                if not selected_ids:
                    st.warning("Select at least one candidate first.")
                else:
                    df.loc[
                        df["Candidate_ID"].astype(str).isin(selected_ids),
                        "HR_Decision",
                    ] = new_decision

                    st.session_state["screening_df"] = df
                    st.session_state["hr_decision_message"] = (
                        f"Successfully updated {len(selected_ids)} candidate(s) "
                        f"to {decision_labels[new_decision]}."
                        )
                    st.rerun()


            st.divider()
            st.subheader("3. Job and HR details")

            parsed_jd = st.session_state.get("parsed_jd", {})
            jd_text = st.session_state.get("jd_raw_text", "")

            # Prefer the parsed title, then a conservative title extraction.
            metadata = parsed_jd.get("metadata", {}) if isinstance(parsed_jd, dict) else {}
            detected_position = (
                metadata.get("job_title")
                or parsed_jd.get("job_title", "")
                or ""
            )

            if not detected_position or detected_position.lower() in {
                "hr confirmed role", "manually entered role",
                "the advertised position",
            }:
                title_match = re.search(
                    r"(?im)^\s*(?:job title|position|role)\s*[:\-]\s*(.+?)\s*$",
                    jd_text,
                )
                if title_match:
                    detected_position = title_match.group(1).strip()

            company_match = re.search(
                r"(?im)^\s*(?:company name|company|employer)\s*[:\-]\s*(.+?)\s*$",
                jd_text,
            )
            detected_company = company_match.group(1).strip() if company_match else ""

            c1, c2 = st.columns(2)

            with c1:
                position = st.text_input(
                    "Job position",
                    value=st.session_state.get("mail_position", detected_position),
                    key="mail_position",
                )
                company = st.text_input(
                    "Company name",
                    value=st.session_state.get("mail_company", detected_company),
                    key="mail_company",
                )

            with c2:
                hr_name = st.text_input("HR full name", key="mail_hr_name")
                hr_designation = st.text_input("HR designation", key="mail_hr_designation")
                hr_email = st.text_input("HR email address", key="mail_hr_email")

            st.caption(
                "Job title and company are auto-filled when detectable. "
                "You can edit both fields manually.")
            st.divider()
            st.subheader("4. Email content")
            template_status = st.selectbox(
                "Choose email template",
                valid_statuses,
                format_func=lambda x: decision_labels[x],
                key="mail_template_status",
            )

            # Get the latest values from Section 3
            position_value = position.strip() or "[Job Position]"
            company_value = company.strip() or "[Company Name]"
            hr_name_value = hr_name.strip() or "[HR Name]"
            hr_designation_value = (
                hr_designation.strip() or "[HR Designation]"
            )
            hr_email_value = hr_email.strip() or "[HR Email]"

            # Prepare the email automatically
            defaults = email_templates(
                "{candidate}",
                position_value,
                company_value,
                hr_name_value,
                hr_designation_value,
                hr_email_value,
            )

            default_subject, default_body = defaults[template_status]

            # Use a new editor key when any HR/job detail changes
            editor_signature = (
                f"{template_status}|{position_value}|{company_value}|"
                f"{hr_name_value}|{hr_designation_value}|{hr_email_value}"
            )
            editor_key = str(abs(hash(editor_signature)))

            edited_subject = st.text_input(
                "Email subject",
                value=default_subject,
                key=f"mail_subject_{editor_key}",
            )

            edited_body = st.text_area(
                "Email message body",
                value=default_body,
                key=f"mail_body_{editor_key}",
                height=260,
            )

            template_key = f"mail_custom_{template_status.lower()}"
            st.session_state[template_key] = {
                "subject": edited_subject,
                "body": edited_body,
            }

            def render_email(text, candidate_name):
                replacements = {
                    "{candidate}": candidate_name,
                    "{position}": position_value,
                    "{company}": company_value,
                    "{hr_name}": hr_name_value,
                    "{hr_designation}": hr_designation_value,
                    "{hr_email}": hr_email_value,
                }

                for token, value in replacements.items():
                    text = text.replace(token, value)

                return text


            st.divider()
            st.subheader("5. Preview and send")

            email_candidates = df[
                df["Candidate_ID"].astype(str).isin(selected_ids)
                & (df["HR_Decision"] == template_status)
            ]

            if not email_candidates.empty:
                for _, row in email_candidates.iterrows():
                    name = str(row.get("Candidate_Name", "Candidate"))
                    with st.expander(f"Preview: {name}", expanded=False):
                        st.write("**To:**", row.get("Email", ""))
                        st.write("**Subject:**", render_email(edited_subject, name))
                        st.text(render_email(edited_body, name))
            else:
                st.info("Select candidates whose HR decision matches the chosen template.")

            confirm_send = st.checkbox(
                "I checked the recipients and email content.",
                key="mail_confirm_send",
            )

            if st.button(
                "📨 Send Email to Selected Candidates",
                type="primary",
                use_container_width=True,
            ):
                required = [
                    position.strip(), company.strip(), hr_name.strip(),
                    hr_designation.strip(), hr_email.strip(),
                ]

                if not selected_ids:
                    st.error("Select at least one candidate.")
                elif email_candidates.empty:
                    st.error("No selected candidates match this template.")
                elif not all(required):
                    st.error("Complete all job and HR detail fields.")
                elif not edited_subject.strip() or not edited_body.strip():
                    st.error("Email subject and body cannot be empty.")
                elif not confirm_send:
                    st.error("Confirm the recipients before sending.")
                else:
                    sent, failed = 0, []

                    for _, row in email_candidates.iterrows():
                        name = str(row.get("Candidate_Name", "Candidate"))
                        address = str(row.get("Email", "")).strip()

                        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", address):
                            failed.append(f"{name}: missing or invalid email")
                            continue

                        try:
                            send_email(
                                address,
                                render_email(edited_subject, name),
                                render_email(edited_body, name),
                            )
                            sent += 1
                        except Exception as exc:
                            failed.append(f"{name} ({address}): {exc}")

                    if sent:
                        st.success(f"Email sent to {sent} candidate(s).")
                    if failed:
                        st.error("Some emails were not sent:")
                        for item in failed:
                            st.write(f"- {item}")
