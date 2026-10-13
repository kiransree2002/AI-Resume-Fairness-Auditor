"""
Exploratory Data Analysis (EDA) & Skill Frequency Analyzer
==========================================================
Analyzes the cleaned Kaggle Resume dataset to extract:
1. Category distribution and percentages
2. Skill frequencies across all 2,484 real resumes
3. Category-specific top skills (e.g., top skills in IT vs Finance vs HR)
4. Experience distribution estimates

Team: DS/DA (Week 1 Core Deliverable)
"""

import os
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR
from skill_extractor import extract_skills, extract_experience_years

# -----------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Output folder for EDA visual charts
EDA_CHARTS_DIR = OUTPUT_DIR / "eda_charts"
EDA_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Set clean visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


# -----------------------------------------------------------
# 1. CATEGORY DISTRIBUTION
# -----------------------------------------------------------

def analyze_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes count and percentage distribution of resumes across job categories.
    Saves a horizontal bar chart of category distribution.
    """
    logger.info("Analyzing category distribution...")
    cat_counts = df["category_clean"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    cat_counts["Percentage"] = (cat_counts["Count"] / len(df) * 100).round(2)

    # Plot Category Distribution Chart
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        data=cat_counts,
        y="Category",
        x="Count",
        palette="viridis",
        hue="Category",
        legend=False
    )
    plt.title("Resume Distribution Across Job Categories (N = 2,484)", fontsize=14, weight="bold")
    plt.xlabel("Number of Resumes")
    plt.ylabel("Job Category")

    # Annotate bar values
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{int(width)}",
            (width + 1, p.get_y() + p.get_height() / 2),
            va="center",
            fontsize=9
        )

    chart_path = EDA_CHARTS_DIR / "01_category_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f"Saved category distribution chart to: {chart_path}")

    return cat_counts


# -----------------------------------------------------------
# 2. SKILL FREQUENCY ANALYSIS (REAL DATA)
# -----------------------------------------------------------

def analyze_skills_and_experience(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts skills and experience levels across every real resume.
    Calculates overall top skills and top skills per job category.
    """
    logger.info("Extracting skills and experience from all 2,484 resumes (takes ~30-45s)...")

    all_skills_list = []
    category_skills_map: Dict[str, List[str]] = {}
    experience_list = []

    for idx, row in df.iterrows():
        text = row["cleaned_text"]
        cat = row["category_clean"]

        if not isinstance(text, str) or not text.strip():
            continue

        # Extract real skills and experience using our NLP engine
        skills = extract_skills(text)
        exp_years = extract_experience_years(text)

        all_skills_list.extend(skills)
        experience_list.append(exp_years)

        if cat not in category_skills_map:
            category_skills_map[cat] = []
        category_skills_map[cat].extend(skills)

    # 1. Overall Skill Frequency
    skill_counts = Counter(all_skills_list)
    top_25_skills = pd.DataFrame(skill_counts.most_common(25), columns=["Skill", "Frequency"])

    # Plot Overall Top 20 Skills
    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=top_25_skills.head(20),
        y="Skill",
        x="Frequency",
        palette="magma",
        hue="Skill",
        legend=False
    )
    plt.title("Top 20 Most Frequent Skills Across All Resumes", fontsize=14, weight="bold")
    plt.xlabel("Frequency Count")
    plt.ylabel("Skill")
    chart_path = EDA_CHARTS_DIR / "02_top_skills_overall.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f"Saved overall skills chart to: {chart_path}")

    # 2. Top 5 Skills per Job Category
    top_per_category = {}
    for cat, skills in category_skills_map.items():
        c_counts = Counter(skills)
        top_per_category[cat] = [s for s, _ in c_counts.most_common(5)]

    # 3. Experience Distribution
    exp_series = pd.Series(experience_list)
    plt.figure(figsize=(8, 5))
    sns.histplot(exp_series[exp_series > 0], bins=20, kde=True, color="#2b5c8f")
    plt.title("Distribution of Stated Experience (Years)", fontsize=13, weight="bold")
    plt.xlabel("Years of Experience")
    plt.ylabel("Candidate Count")
    chart_path = EDA_CHARTS_DIR / "03_experience_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f"Saved experience distribution chart to: {chart_path}")

    return {
        "top_25_skills": top_25_skills,
        "top_per_category": top_per_category,
        "experience_stats": {
            "mean": round(exp_series.mean(), 2),
            "median": round(exp_series.median(), 2),
            "max": round(exp_series.max(), 2),
        }
    }


# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------

def run_eda_pipeline():
    cleaned_csv = OUTPUT_DIR / "resumes_cleaned.csv"
    if not cleaned_csv.exists():
        logger.error(f"Cleaned data not found at {cleaned_csv}. Run data_cleaner.py first.")
        return

    df = pd.read_csv(cleaned_csv)
    logger.info(f"Loaded {len(df)} cleaned resumes for EDA.")

    # 1. Run Category Breakdown
    cat_df = analyze_categories(df)
    cat_summary_path = OUTPUT_DIR / "category_summary.csv"
    cat_df.to_csv(cat_summary_path, index=False)

    # 2. Run Skill and Experience Breakdown
    analytics = analyze_skills_and_experience(df)
    
    # Save Top Skills Overall
    skills_path = OUTPUT_DIR / "top_skills_overall.csv"
    analytics["top_25_skills"].to_csv(skills_path, index=False)

    # Save Top Skills Per Category JSON
    category_skills_path = OUTPUT_DIR / "top_skills_per_category.json"
    with open(category_skills_path, "w", encoding="utf-8") as f:
        json.dump(analytics["top_per_category"], f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("📊 EDA ANALYSIS SUMMARY HIGHLIGHTS")
    logger.info("=" * 60)
    logger.info(f"Total Unique Job Categories: {len(cat_df)}")
    logger.info(f"Top 5 Categories by volume:\n{cat_df.head(5).to_string(index=False)}")
    logger.info(f"\nTop 10 Most Common Skills:\n{analytics['top_25_skills'].head(10).to_string(index=False)}")
    logger.info(f"\nExperience Stats: {analytics['experience_stats']}")
    logger.info(f"All visual charts saved inside: {EDA_CHARTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_eda_pipeline()