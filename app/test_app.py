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
    host = str(smtp_setting("SMTP_HOST") or "").strip()
    port = int(smtp_setting("SMTP_PORT", 587))
    username = str(smtp_setting("SMTP_USERNAME") or "").strip()
    password = str(smtp_setting("SMTP_PASSWORD") or "").strip()
    sender = str(smtp_setting("SMTP_FROM", username) or "").strip()

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

    server = None
    try:
        server = smtplib.SMTP(host, port, timeout=30)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(username, password)
        server.send_message(msg)
    except smtplib.SMTPAuthenticationError as err:
        raise RuntimeError("SMTP authentication failed. Please verify your Gmail username and 16-character App Password in .streamlit/secrets.toml.")
    except smtplib.SMTPServerDisconnected as err:
        raise RuntimeError(
            "Connection closed by Gmail SMTP server during login. "
            "Note: Gmail automatically drops the connection if your 16-character App Password is incorrect, revoked, or still set to a placeholder in .streamlit/secrets.toml."
        )
    except smtplib.SMTPConnectError as err:
        raise RuntimeError(f"Could not connect to SMTP server ({host}:{port}): {err.smtp_error or err}")
    except smtplib.SMTPException as err:
        raise RuntimeError(f"SMTP error: {err}")
    except TimeoutError:
        raise RuntimeError(f"Timed out connecting to SMTP server ({host}:{port}). Please check your internet connection.")
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass


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
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 10% 12%, rgba(20, 184, 166, 0.08), transparent 30%),
            radial-gradient(circle at 90% 18%, rgba(59, 130, 246, 0.09), transparent 32%),
            radial-gradient(circle at 50% 90%, rgba(99, 102, 241, 0.06), transparent 35%),
            #F8FAFC;
        background-size: 100% 100%;
        color: #0F172A;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2.2rem;
        padding-bottom: 3.5rem;
    }

    .welcome-card-shell {
        position: relative;
        overflow: hidden;
        padding: clamp(34px, 5.2vw, 68px);
        border-radius: 32px;
        border: 1px solid rgba(226, 232, 240, 0.9);
        background: linear-gradient(
            145deg,
            rgba(255, 255, 255, 0.96) 0%,
            rgba(248, 250, 252, 0.92) 50%,
            rgba(241, 245, 249, 0.95) 100%
        );
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 
            0 24px 60px rgba(15, 23, 42, 0.06),
            0 4px 16px rgba(15, 23, 42, 0.03),
            inset 0 1px 0 rgba(255, 255, 255, 1);
        color: #0F172A;
        font-family: 'Plus Jakarta Sans', sans-serif;
        isolation: isolate;
    }

    /* Ambient soft decorative glows */
    .welcome-card-shell::before {
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        top: -150px;
        right: -90px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.12) 0%, rgba(13, 148, 136, 0.04) 65%, transparent 100%);
        filter: blur(65px);
        z-index: -1;
        animation: lightFloat 10s ease-in-out infinite alternate;
    }

    .welcome-card-shell::after {
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        bottom: -130px;
        left: 5%;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(13, 148, 136, 0.11) 0%, rgba(79, 70, 229, 0.04) 65%, transparent 100%);
        filter: blur(65px);
        z-index: -1;
        animation: lightFloat 12s ease-in-out infinite alternate-reverse;
    }

    @keyframes lightFloat {
        0%   { transform: translate(0, 0) scale(1); }
        100% { transform: translate(-25px, 25px) scale(1.15); }
    }

    /* Top Brand & AI Status Header */
    .welcome-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 34px;
        flex-wrap: wrap;
    }

    .welcome-brand-group {
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .welcome-brand-avatar {
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 16px;
        background: linear-gradient(135deg, #0284C7 0%, #0D9488 100%);
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.4);
        flex-shrink: 0;
    }

    .welcome-brand-avatar svg {
        width: 27px;
        height: 27px;
    }

    .welcome-brand-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.4px;
        color: #0F172A;
    }

    .welcome-brand-sub {
        margin-top: 3px;
        color: #0D9488;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 1.6px;
    }

    /* Refined Live Status Badge */
    .welcome-status-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 18px;
        border-radius: 999px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }

    .welcome-status-orb {
        position: relative;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #0D9488;
        box-shadow: 0 0 10px rgba(13, 148, 136, 0.6);
    }

    .welcome-status-orb::after {
        content: "";
        position: absolute;
        inset: -4px;
        border-radius: 50%;
        border: 1.5px solid rgba(13, 148, 136, 0.35);
        animation: pulseRing 2s infinite ease-out;
    }

    @keyframes pulseRing {
        0%   { transform: scale(0.9); opacity: 1; }
        100% { transform: scale(1.6); opacity: 0; }
    }

    .welcome-status-text {
        font-size: 11.5px;
        font-weight: 750;
        letter-spacing: 1.2px;
        color: #334155;
    }

    /* Eyebrow Badge */
    .welcome-eyebrow-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 16px;
        border-radius: 999px;
        border: 1px solid rgba(13, 148, 136, 0.25);
        background: rgba(13, 148, 136, 0.08);
        color: #0F766E;
        font-size: 11.5px;
        font-weight: 800;
        letter-spacing: 1.3px;
        margin-bottom: 20px;
    }

    /* Top Preserved Heading & Tagline */
    .welcome-top-tagline {
        font-family: 'Space Grotesk', sans-serif;
        color: #0F172A;
        font-size: clamp(23px, 3.4vw, 36px);
        font-weight: 800;
        letter-spacing: -0.8px;
        line-height: 1.28;
        margin-bottom: 8px;
    }

    .welcome-top-tagline .highlight-teal {
        background: linear-gradient(135deg, #0284C7 0%, #0D9488 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .welcome-hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(38px, 5.8vw, 68px);
        font-weight: 900;
        letter-spacing: -2px;
        line-height: 1.05;
        color: #0F172A;
        margin: 18px 0 14px 0;
    }

    .welcome-hero-gradient {
        background: linear-gradient(135deg, #0284C7 0%, #0D9488 50%, #4F46E5 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .welcome-body-lead {
        max-width: 760px;
        color: #475569;
        font-size: 16.5px;
        line-height: 1.75;
        margin-top: 14px;
    }

    /* Capability Badges */
    .welcome-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 26px;
    }

    .welcome-badge-item {
        padding: 8px 16px;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        background: #FFFFFF;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        color: #334155;
        font-size: 12.5px;
        font-weight: 650;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .welcome-badge-item:hover {
        transform: translateY(-2px);
        border-color: #0D9488;
    }

    .welcome-badge-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #0D9488;
    }

    /* Three Feature Cards with Refined Light Styling */
    .welcome-cards-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 22px;
        margin-top: 45px;
    }

    .welcome-card-box {
        position: relative;
        padding: 30px 24px;
        border-radius: 22px;
        border: 1px solid #E2E8F0;
        background: #FFFFFF;
        box-shadow: 
            0 10px 28px rgba(15, 23, 42, 0.04),
            0 2px 6px rgba(15, 23, 42, 0.02);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .welcome-card-box:hover {
        transform: translateY(-7px);
        border-color: rgba(13, 148, 136, 0.45);
        box-shadow: 
            0 20px 42px rgba(13, 148, 136, 0.12),
            0 4px 14px rgba(15, 23, 42, 0.05);
    }

    .welcome-card-icon {
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 16px;
        margin-bottom: 22px;
    }

    .welcome-card-icon svg {
        width: 26px;
        height: 26px;
    }

    .icon-blue-card {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.12), rgba(59, 130, 246, 0.06));
        border: 1px solid rgba(2, 132, 199, 0.2);
        color: #0284C7;
    }

    .icon-teal-card {
        background: linear-gradient(135deg, rgba(13, 148, 136, 0.12), rgba(20, 184, 166, 0.06));
        border: 1px solid rgba(13, 148, 136, 0.2);
        color: #0D9488;
    }

    .icon-indigo-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.12), rgba(99, 102, 241, 0.06));
        border: 1px solid rgba(79, 70, 229, 0.2);
        color: #4F46E5;
    }

    .welcome-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 19px;
        font-weight: 750;
        color: #0F172A;
        letter-spacing: -0.3px;
        margin-bottom: 10px;
    }

    .welcome-card-desc {
        color: #64748B;
        font-size: 13.8px;
        line-height: 1.7;
    }

    .welcome-bottom-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 14px;
        margin-top: 45px;
        padding-top: 24px;
        border-top: 1px solid #E2E8F0;
        color: #64748B;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 1.2px;
    }

    .welcome-footer-tag {
        color: #0D9488;
        font-weight: 800;
    }

    /* Polished Primary Launch Button */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0D9488 50%, #0F766E 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 750 !important;
        font-size: 18px !important;
        padding: 16px 38px !important;
        border-radius: 16px !important;
        box-shadow: 
            0 12px 30px rgba(13, 148, 136, 0.35),
            0 4px 12px rgba(2, 132, 199, 0.2) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 
            0 18px 40px rgba(13, 148, 136, 0.45),
            0 6px 16px rgba(2, 132, 199, 0.28) !important;
    }

    @media (max-width: 860px) {
        .welcome-cards-grid { grid-template-columns: 1fr; }
        .welcome-header-bar { flex-direction: column; align-items: flex-start; }
    }
    </style>

    <div class="welcome-card-shell">
        <div class="welcome-header-bar">
            <div class="welcome-brand-group">
                <div class="welcome-brand-avatar">
                    <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                    </svg>
                </div>
                <div>
                    <div class="welcome-brand-text">AI Resume Fairness Auditor</div>
                    <div class="welcome-brand-sub">INTELLIGENT RECRUITMENT SUITE</div>
                </div>
            </div>

            <!-- Live Status Pill -->
            <div class="welcome-status-pill">
                <div class="welcome-status-orb"></div>
                <div class="welcome-status-text">ENTERPRISE AUDIT READY</div>
            </div>
        </div>

        <div class="welcome-eyebrow-badge">
            ◆ NEXT-GEN RECRUITMENT INTELLIGENCE
        </div>

        <!-- Preserved Top Heading & Tagline -->
        <div class="welcome-top-tagline">
            AI Resume Screening, <span class="highlight-teal">Fairness Auditor</span>, End-to-End Candidate Evaluation
        </div>

        <!-- Hero Headline -->
        <div class="welcome-hero-title">
            Screen Smarter.<br>
            <span class="welcome-hero-gradient">Hire Fairer.</span>
        </div>

        <div class="welcome-body-lead">
            Empower talent acquisition teams with transparent skill matching, explainable qualification gates, 
            and demographic audit metrics. Transition seamlessly from job parsing to ranked candidate shortlists 
            and personalized HR communications.
        </div>

        <div class="welcome-badges">
            <div class="welcome-badge-item"><span class="welcome-badge-dot"></span> AI Semantic Scorer</div>
            <div class="welcome-badge-item"><span class="welcome-badge-dot"></span> Explainable Gate Cascades</div>
            <div class="welcome-badge-item"><span class="welcome-badge-dot"></span> Demographic Parity Auditing</div>
            <div class="welcome-badge-item"><span class="welcome-badge-dot"></span> One-Click HR Email Centre</div>
        </div>

        <!-- Refined Feature Cards -->
        <div class="welcome-cards-grid">
            <div class="welcome-card-box">
                <div class="welcome-card-icon icon-blue-card">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <path d="M14 2v6h6M8 13h8M8 17h8"/>
                    </svg>
                </div>
                <div>
                    <div class="welcome-card-title">Resume Screening</div>
                    <div class="welcome-card-desc">
                        Scrape or paste job descriptions, upload candidate resumes, set threshold parameters, and review ranked scoring outputs.
                    </div>
                </div>
            </div>

            <div class="welcome-card-box">
                <div class="welcome-card-icon icon-teal-card">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10"/>
                        <line x1="12" y1="20" x2="12" y2="4"/>
                        <line x1="6" y1="20" x2="6" y2="14"/>
                    </svg>
                </div>
                <div>
                    <div class="welcome-card-title">Visualization & Analytics</div>
                    <div class="welcome-card-desc">
                        Analyze candidate score distributions, compare matched vs. missing skills, and monitor demographic breakdown charts.
                    </div>
                </div>
            </div>

            <div class="welcome-card-box">
                <div class="welcome-card-icon icon-indigo-card">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                        <polyline points="22,6 12,13 2,6"/>
                    </svg>
                </div>
                <div>
                    <div class="welcome-card-title">HR Email Centre</div>
                    <div class="welcome-card-desc">
                        Categorize decisions into Shortlisted, Review, or Rejected cohorts and generate customized correspondence ready to dispatch.
                    </div>
                </div>
            </div>
        </div>

        <div class="welcome-bottom-footer">
            <span>AUDITABLE HIRING INTELLIGENCE PLATFORM</span>
            <span class="welcome-footer-tag">ENTERPRISE FAIRNESS STANDARD</span>
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

st.markdown("""
<div class="hero-card">
  <div class="hero-title">🤖 AI Resume Screening & Fairness Auditor</div>
  <div class="hero-subtitle">Intelligent Candidate Evaluation · Explainable Matching Gates · Demographic Fairness · HR Communications</div>
</div>
""", unsafe_allow_html=True)

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

    def navigate_to(page_name):
        st.session_state["command_centre_navigation"] = page_name

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
                st.button(
                    "Next: Visualization & Analytics →",
                    key="btn_next_to_analytics",
                    type="primary",
                    use_container_width=True,
                    on_click=navigate_to,
                    args=(PAGE_ANALYTICS,),
                )


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
            st.button(
                "Next: HR Email Centre →",
                key="btn_next_to_hr",
                type="primary",
                use_container_width=True,
                on_click=navigate_to,
                args=(PAGE_HR,),
            )


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
