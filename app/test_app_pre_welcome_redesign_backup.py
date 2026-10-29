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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

:root {
    --navy: #0B1120;
    --navy-deep: #070B16;
    --blue: #4F46E5;
    --violet: #7C3AED;
    --teal: #0D9488;
    --green: #059669;
    --text: #1E293B;
    --muted: #64748B;
    --border: #E2E8F0;
    --surface: #FFFFFF;
    --page: #F8FAFC;
}

body, html, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 8% 0%, rgba(99, 102, 241, 0.08), transparent 32%),
        radial-gradient(ellipse at 95% 8%, rgba(20, 184, 166, 0.07), transparent 28%),
        radial-gradient(ellipse at 50% 100%, rgba(139, 92, 246, 0.05), transparent 30%),
        var(--page);
    color: var(--text);
}

.block-container {
    max-width: 1560px;
    padding-top: 1.8rem;
    padding-bottom: 3.2rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0B1020 0%, #11182E 50%, #152238 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] * {
    color: #EEF2FF;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 12px;
    padding: 9px 12px;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    border: 1px solid transparent;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(99, 102, 241, 0.18);
    border-color: rgba(129, 140, 248, 0.25);
    transform: translateX(3px);
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #94A3B8;
    font-size: 11px;
    letter-spacing: 1.2px;
    font-weight: 700;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
    color: #0F172A;
    letter-spacing: -0.03em;
    font-weight: 750;
}

h1 { font-size: clamp(1.85rem, 3.2vw, 2.6rem); }
h2 { font-size: 1.55rem; }
h3 { font-size: 1.25rem; }

/* Hero Header Card */
.hero-card {
    padding: 26px 32px;
    margin-bottom: 24px;
    border-radius: 20px;
    color: #FFFFFF;
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 25%, #4F46E5 60%, #7C3AED 90%, #06B6D4 125%);
    box-shadow: 0 16px 36px rgba(79, 70, 229, 0.22);
    border: 1px solid rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
}

.hero-card::after {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    top: -100px;
    right: -50px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.3), transparent 70%);
    filter: blur(40px);
    pointer-events: none;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
    color: #FFFFFF;
}

.hero-subtitle {
    font-size: 14px;
    color: rgba(248, 250, 252, 0.9);
    letter-spacing: 0.2px;
}

/* Metric Cards */
[data-testid="stMetric"] {
    position: relative;
    background: linear-gradient(155deg, #FFFFFF, #F8FAFC);
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(79, 70, 229, 0.12);
    border-color: rgba(99, 102, 241, 0.35);
}

[data-testid="stMetricLabel"] {
    color: #64748B;
    font-size: 0.86rem;
    font-weight: 650;
    letter-spacing: 0.2px;
}

[data-testid="stMetricValue"] {
    color: #1E1B4B;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.03em;
}

/* Container Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.98);
    border-color: #E2E8F0;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.035);
}

/* Modern Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid #4F46E5;
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #7C3AED 100%);
    color: #FFFFFF;
    font-weight: 700;
    padding: 0.62rem 1.25rem;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.2);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #4338CA;
    background: linear-gradient(135deg, #4338CA 0%, #4F46E5 50%, #6D28D9 100%);
    color: #FFFFFF;
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.32);
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"],
.stNumberInput input {
    border-radius: 12px;
    border-color: #CBD5E1;
    background: #FFFFFF;
    transition: all 0.2s ease;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: #6366F1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
}

/* Tables and DataFrames */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    background: #FFFFFF;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.025);
}

[data-testid="stExpander"] summary {
    font-weight: 700;
    color: #1E293B;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #E2E8F0;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    font-weight: 700;
    padding: 10px 18px;
    transition: all 0.2s ease;
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4F46E5, #8B5CF6, #06B6D4);
    border-radius: 8px;
}

hr {
    border-color: #E2E8F0;
    margin: 1.5rem 0;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }
    .hero-card {
        padding: 20px 22px;
    }
}
</style>
<div class="hero-card">
  <div class="hero-title">🤖 AI Resume Screening & Fairness Auditor</div>
  <div class="hero-subtitle">Intelligent Candidate Evaluation · Explainable Matching Gates · Demographic Fairness · HR Communications</div>
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
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@600;700&display=swap');

    .stApp {
        background:
            radial-gradient(ellipse at 20% 15%, rgba(79, 70, 229, 0.42), transparent 42%),
            radial-gradient(ellipse at 80% 25%, rgba(147, 51, 234, 0.38), transparent 46%),
            radial-gradient(ellipse at 50% 85%, rgba(6, 182, 212, 0.28), transparent 50%),
            #070B16;
        background-size: 160% 160%;
        animation: auroraBG 14s ease-in-out infinite alternate;
        color: #F8FAFC;
    }

    @keyframes auroraBG {
        0%   { background-position: 0% 0%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 100%; }
    }

    .block-container {
        max-width: 1220px;
        padding-top: 2.5rem;
        padding-bottom: 3.5rem;
    }

    /* Ambient floating particles canvas simulation */
    .welcome-canvas {
        position: relative;
        overflow: hidden;
    }

    .welcome-shell {
        position: relative;
        overflow: hidden;
        padding: clamp(34px, 5.5vw, 76px);
        border-radius: 32px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        background: linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.88) 0%,
            rgba(20, 22, 54, 0.86) 50%,
            rgba(10, 30, 48, 0.88) 100%
        );
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        box-shadow: 
            0 25px 90px rgba(0, 0, 0, 0.65),
            inset 0 1px 0 rgba(255, 255, 255, 0.16);
        color: #F8FAFC;
        font-family: 'Plus Jakarta Sans', sans-serif;
        isolation: isolate;
    }

    /* Ambient Glow Blobs */
    .welcome-shell::before {
        content: "";
        position: absolute;
        width: 440px;
        height: 440px;
        top: -160px;
        right: -80px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(124, 58, 237, 0.35) 0%, rgba(79, 70, 229, 0.1) 70%, transparent 100%);
        filter: blur(75px);
        z-index: -1;
        animation: glowFloat 9s ease-in-out infinite alternate;
    }

    .welcome-shell::after {
        content: "";
        position: absolute;
        width: 380px;
        height: 380px;
        bottom: -150px;
        left: 5%;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.28) 0%, rgba(59, 130, 246, 0.08) 70%, transparent 100%);
        filter: blur(80px);
        z-index: -1;
        animation: glowFloat 11s ease-in-out infinite alternate-reverse;
    }

    @keyframes glowFloat {
        0%   { transform: translate(0, 0) scale(1); }
        100% { transform: translate(-30px, 35px) scale(1.18); }
    }

    /* Floating Particles */
    .particle {
        position: absolute;
        border-radius: 50%;
        pointer-events: none;
        z-index: -1;
    }
    .p1 { width: 5px; height: 5px; background: #67E8F9; top: 18%; left: 12%; box-shadow: 0 0 10px #67E8F9; animation: floatP 6s infinite ease-in-out; }
    .p2 { width: 4px; height: 4px; background: #C4B5FD; top: 40%; left: 88%; box-shadow: 0 0 8px #C4B5FD; animation: floatP 8s infinite ease-in-out 1s; }
    .p3 { width: 6px; height: 6px; background: #93C5FD; top: 75%; left: 22%; box-shadow: 0 0 12px #93C5FD; animation: floatP 7s infinite ease-in-out 2s; }
    .p4 { width: 3px; height: 3px; background: #F472B6; top: 28%; left: 74%; box-shadow: 0 0 8px #F472B6; animation: floatP 9s infinite ease-in-out 0.5s; }
    .p5 { width: 5px; height: 5px; background: #34D399; top: 82%; left: 78%; box-shadow: 0 0 10px #34D399; animation: floatP 6.5s infinite ease-in-out 1.5s; }

    @keyframes floatP {
        0%, 100% { transform: translateY(0px) scale(1); opacity: 0.35; }
        50%      { transform: translateY(-22px) scale(1.4); opacity: 0.95; }
    }

    /* Top Brand & AI Orb Header */
    .welcome-top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 40px;
        flex-wrap: wrap;
    }

    .welcome-brand {
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .welcome-brand-logo {
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 17px;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.45), inset 0 1px 0 rgba(255,255,255,0.4);
        flex-shrink: 0;
    }

    .welcome-brand-logo svg {
        width: 28px;
        height: 28px;
    }

    .welcome-brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 19px;
        font-weight: 700;
        letter-spacing: -0.3px;
        color: #FFFFFF;
    }

    .welcome-brand-subtitle {
        margin-top: 3px;
        color: #94A3B8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.8px;
    }

    /* GLOWING AI ORB */
    .ai-orb-container {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 8px 16px 8px 10px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(12px);
    }

    .ai-orb {
        position: relative;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #E0E7FF 0%, #6366F1 50%, #7C3AED 85%, #0F172A 100%);
        box-shadow:
            0 0 20px rgba(99, 102, 241, 0.85),
            0 0 35px rgba(139, 92, 246, 0.55),
            inset 0 0 10px rgba(255, 255, 255, 0.7);
        animation: orbPulse 3s infinite ease-in-out;
    }

    .ai-orb::before {
        content: "";
        position: absolute;
        inset: -6px;
        border-radius: 50%;
        border: 1.5px solid rgba(147, 197, 253, 0.45);
        animation: orbSpin 6s linear infinite;
    }

    .ai-orb::after {
        content: "";
        position: absolute;
        inset: -11px;
        border-radius: 50%;
        border: 1px dashed rgba(196, 181, 253, 0.35);
        animation: orbSpinRev 9s linear infinite;
    }

    @keyframes orbPulse {
        0%, 100% { transform: scale(1); filter: brightness(1); }
        50%      { transform: scale(1.08); filter: brightness(1.25); }
    }

    @keyframes orbSpin {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes orbSpinRev {
        0%   { transform: rotate(360deg); }
        100% { transform: rotate(0deg); }
    }

    .ai-orb-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        color: #E2E8F0;
    }

    /* Eyebrow badge */
    .welcome-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        padding: 8px 16px;
        border-radius: 999px;
        border: 1px solid rgba(129, 140, 248, 0.35);
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.16), rgba(168, 85, 247, 0.12));
        color: #DDD6FE;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.6px;
        margin-bottom: 22px;
    }

    .welcome-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34D399;
        box-shadow: 0 0 12px #34D399;
        animation: pulseDot 2s infinite ease-in-out;
    }

    @keyframes pulseDot {
        0%, 100% { transform: scale(1); opacity: 1; }
        50%      { transform: scale(1.3); opacity: 0.6; }
    }

    /* Headline */
    .welcome-title {
        font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
        margin: 0;
        color: #FFFFFF;
        font-size: clamp(42px, 6.8vw, 76px);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -2.5px;
    }

    .welcome-gradient {
        display: inline-block;
        background: linear-gradient(90deg, #93C5FD 0%, #C4B5FD 45%, #67E8F9 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .welcome-tagline {
        margin-top: 20px;
        color: #E2E8F0;
        font-size: clamp(19px, 2.4vw, 25px);
        font-weight: 600;
        letter-spacing: -0.4px;
    }

    .welcome-description {
        max-width: 740px;
        margin-top: 15px;
        color: #94A3B8;
        font-size: 15.5px;
        line-height: 1.85;
    }

    /* Badges / Pills */
    .welcome-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 26px;
    }

    .welcome-pill {
        padding: 8px 15px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(8px);
        color: #E2E8F0;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 7px;
    }

    .welcome-pill span {
        color: #67E8F9;
        font-size: 10px;
    }

    /* 3 FEATURE CARDS - GLASSMORPHISM */
    .welcome-features {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 22px;
        margin-top: 48px;
    }

    .welcome-feature {
        position: relative;
        padding: 30px 24px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 24px;
        background: linear-gradient(
            155deg,
            rgba(255, 255, 255, 0.08) 0%,
            rgba(255, 255, 255, 0.02) 100%
        );
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 
            0 14px 40px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.12);
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        overflow: hidden;
    }

    .welcome-feature::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 24px;
        padding: 1.5px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.22), transparent 60%);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }

    .welcome-feature:hover {
        transform: translateY(-8px);
        border-color: rgba(147, 197, 253, 0.5);
        background: linear-gradient(
            155deg,
            rgba(59, 130, 246, 0.14) 0%,
            rgba(139, 92, 246, 0.06) 100%
        );
        box-shadow: 
            0 24px 50px rgba(15, 23, 42, 0.6),
            0 0 25px rgba(99, 102, 241, 0.25);
    }

    .welcome-feature-icon {
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 16px;
        margin-bottom: 22px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }

    .welcome-feature-icon svg {
        width: 26px;
        height: 26px;
    }

    .icon-blue {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.28), rgba(37, 99, 235, 0.1));
        border: 1px solid rgba(147, 197, 253, 0.35);
        color: #93C5FD;
    }

    .icon-purple {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.28), rgba(124, 58, 237, 0.1));
        border: 1px solid rgba(216, 180, 254, 0.35);
        color: #C4B5FD;
    }

    .icon-teal {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.28), rgba(13, 148, 136, 0.1));
        border: 1px solid rgba(94, 234, 212, 0.35);
        color: #5EEAD4;
    }

    .welcome-feature-title {
        margin-bottom: 10px;
        color: #F8FAFC;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.2px;
    }

    .welcome-feature-description {
        color: #94A3B8;
        font-size: 13.5px;
        line-height: 1.75;
    }

    .welcome-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 14px;
        margin-top: 45px;
        padding-top: 26px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #94A3B8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
    }

    .welcome-footer-accent {
        color: #67E8F9;
        letter-spacing: 0.8px;
    }

    /* Style the launch button inside welcome */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #06B6D4 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        padding: 16px 36px !important;
        border-radius: 16px !important;
        box-shadow: 
            0 12px 35px rgba(99, 102, 241, 0.45),
            0 0 20px rgba(124, 58, 237, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 
            0 18px 45px rgba(99, 102, 241, 0.65),
            0 0 30px rgba(6, 182, 212, 0.45) !important;
    }

    @media (max-width: 860px) {
        .welcome-features { grid-template-columns: 1fr; }
        .welcome-top-bar { flex-direction: column; align-items: flex-start; }
    }
    </style>

    <div class="welcome-shell">
        <!-- Floating Particles -->
        <div class="particle p1"></div>
        <div class="particle p2"></div>
        <div class="particle p3"></div>
        <div class="particle p4"></div>
        <div class="particle p5"></div>

        <div class="welcome-top-bar">
            <div class="welcome-brand">
                <div class="welcome-brand-logo">
                    <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                    </svg>
                </div>
                <div>
                    <div class="welcome-brand-title">AI Resume Fairness Auditor</div>
                    <div class="welcome-brand-subtitle">INTELLIGENT RECRUITMENT PLATFORM</div>
                </div>
            </div>

            <!-- Glowing AI Orb Element -->
            <div class="ai-orb-container">
                <div class="ai-orb"></div>
                <div class="ai-orb-label">AI AUDITOR ACTIVE</div>
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
            Precision Candidate Evaluation & Unbiased Screening
        </div>

        <div class="welcome-description">
            Bring resume screening, AI-powered skill matching, explainable qualification gates, and hiring
            fairness analysis into one unified command center. Discover qualified talent with complete auditability, 
            zero guesswork, and rapid HR communication workflows.
        </div>

        <div class="welcome-pills">
            <div class="welcome-pill"><span>◆</span> AI Semantic Matching</div>
            <div class="welcome-pill"><span>◆</span> Explainable Gate Cascades</div>
            <div class="welcome-pill"><span>◆</span> Demographic Fairness Audits</div>
            <div class="welcome-pill"><span>◆</span> Automated HR Email Workflow</div>
        </div>

        <!-- Three Glassmorphic Feature Cards -->
        <div class="welcome-features">
            <div class="welcome-feature">
                <div class="welcome-feature-icon icon-blue">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <path d="M14 2v6h6M8 13h8M8 17h8"/>
                    </svg>
                </div>
                <div class="welcome-feature-title">Resume Screening</div>
                <div class="welcome-feature-description">
                    Scrape or enter job descriptions, upload candidate resumes, configure qualification thresholds, and generate ranked match scores.
                </div>
            </div>

            <div class="welcome-feature">
                <div class="welcome-feature-icon icon-purple">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10"/>
                        <line x1="12" y1="20" x2="12" y2="4"/>
                        <line x1="6" y1="20" x2="6" y2="14"/>
                    </svg>
                </div>
                <div class="welcome-feature-title">Visualization & Analytics</div>
                <div class="welcome-feature-description">
                    Explore candidate score distributions, matched versus missing skill breakdowns, and transparent demographic parity metrics.
                </div>
            </div>

            <div class="welcome-feature">
                <div class="welcome-feature-icon icon-teal">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                        <polyline points="22,6 12,13 2,6"/>
                    </svg>
                </div>
                <div class="welcome-feature-title">HR Email Centre</div>
                <div class="welcome-feature-description">
                    Review candidate recommendations, adjust decisions (Shortlist, Review, Reject), and send personalized HR communications in bulk.
                </div>
            </div>
        </div>

        <div class="welcome-footer">
            <span>AUDITABLE HIRING INTELLIGENCE PLATFORM</span>
            <span class="welcome-footer-accent">ENTERPRISE FAIRNESS STANDARD</span>
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
        <div style="padding: 14px 6px 20px 6px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #34D399; box-shadow: 0 0 10px #34D399;"></span>
                <span style="font-size: 11px; letter-spacing: 2px; color: #5EEAD4; font-weight: 800;">COMMAND CENTRE</span>
            </div>
            <div style="font-size: 21px; font-weight: 800; line-height: 1.25; color: #FFFFFF; font-family: 'Space Grotesk', sans-serif;">
                AI Fairness Auditor
            </div>
            <div style="font-size: 12px; color: #94A3B8; margin-top: 6px;">
                Intelligent Recruitment Platform
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("WORKSPACE NAVIGATION")

    PAGE_SCREENING = "Screening"
    PAGE_ANALYTICS = "Visualization & Analytics"
    PAGE_HR = "HR Email Centre"

    selected_page = st.radio(
        "Navigate",
        [PAGE_SCREENING, PAGE_ANALYTICS, PAGE_HR],
        key="command_centre_navigation",
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("AI Resume Screening & Fairness Auditor")


if selected_page in (PAGE_SCREENING, "Screening Workspace"):
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
            col_csv, _, col_next = st.columns([4, 2, 3])
            with col_csv:
                st.download_button("📥 Download candidate results (CSV)", csv,
                                   "candidate_screening_results.csv", "text/csv")
            with col_next:
                if st.button("Next: Visualization & Analytics →", key="btn_next_to_analytics", type="primary", use_container_width=True):
                    st.session_state["command_centre_navigation"] = PAGE_ANALYTICS
                    st.rerun()


elif selected_page in (PAGE_ANALYTICS, "Analytics Centre"):
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

        st.divider()
        _, col_next_hr = st.columns([7, 3])
        with col_next_hr:
            if st.button("Next: HR Email Centre →", key="btn_next_to_hr", type="primary", use_container_width=True):
                st.session_state["command_centre_navigation"] = PAGE_HR
                st.rerun()


elif selected_page in (PAGE_HR, "HR Communications"):
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
