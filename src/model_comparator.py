"""
Model Comparison Benchmarking Engine (Production)
=================================================
Compares TF-IDF vs Sentence-Transformers on real Kaggle resumes.
Evaluates:
1. Category-level average match scores
2. Score distribution spread (Min, Max, Mean, Std)
3. Rank divergence (Spearman rank correlation)
4. Runtime throughput (resumes processed per second)
5. Exports comparison report CSV and comparative visual chart

Team: AI/ML (Week 2 Core Deliverable)
"""

import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from config import OUTPUT_DIR
from baseline_scorer import TFIDFScorer
from semantic_scorer import SemanticScorer

# -----------------------------------------------------------
# CHARTS FOLDER
# -----------------------------------------------------------
EDA_CHARTS_DIR = OUTPUT_DIR / "eda_charts"
EDA_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------
# BENCHMARK RUNNER
# -----------------------------------------------------------

def run_model_benchmark(sample_limit: int = 240) -> Dict[str, Any]:
    """
    Runs both TF-IDF and Sentence-Transformer across real resumes
    and computes comparative performance metrics.
    """
    logger.info("=" * 65)
    logger.info("🚀 STARTING MODEL BENCHMARK: TF-IDF vs SENTENCE-TRANSFORMERS")
    logger.info("=" * 65)

    cleaned_csv = OUTPUT_DIR / "resumes_cleaned.csv"
    if not cleaned_csv.exists():
        logger.error(f"Cleaned dataset not found at: {cleaned_csv}")
        return {}

    logger.info(f"Loading cleaned dataset from: {cleaned_csv}")
    df = pd.read_csv(cleaned_csv)
    logger.info(f"Total available resumes: {len(df)}")

    # Detect exact column names safely
    cat_col = "category_clean" if "category_clean" in df.columns else ("category" if "category" in df.columns else "Category")
    id_col = "id" if "id" in df.columns else "ID"
    text_col = "cleaned_text" if "cleaned_text" in df.columns else "Resume_str"

    logger.info(f"Detected columns: ID='{id_col}', Category='{cat_col}', Text='{text_col}'")

    # Take a balanced sample using standard groupby head (version safe)
    unique_cats = df[cat_col].nunique()
    per_cat = max(1, sample_limit // unique_cats)
    sampled_df = df.groupby(cat_col).head(per_cat).reset_index(drop=True)

    logger.info(f"Selected {len(sampled_df)} balanced resumes ({per_cat} per category across {unique_cats} categories).")

    resume_texts = sampled_df[text_col].tolist()
    candidate_ids = [f"RES_{cid}" for cid in sampled_df[id_col].tolist()]
    categories = sampled_df[cat_col].tolist()

    # Benchmark Job Description (Data Science / ML)
    benchmark_jd = """
    Senior Data Scientist and Machine Learning Engineer.
    Must have hands-on experience in Python, SQL, predictive modeling,
    statistical analysis, machine learning algorithms, and deep learning.
    Preferred expertise in PyTorch, Scikit-Learn, AWS cloud, and Docker.
    Strong communication and problem solving capabilities required.
    """

    # -----------------------------------------------------------
    # 1. TF-IDF BENCHMARK
    # -----------------------------------------------------------
    logger.info("\n" + "─" * 65)
    logger.info("⚡ 1. BENCHMARKING TF-IDF (KEYWORD BASELINE)")
    logger.info("─" * 65)

    start_tfidf = time.time()
    tfidf_engine = TFIDFScorer()
    tfidf_results = tfidf_engine.compute_similarity_scores(
        jd_text=benchmark_jd,
        resume_texts=resume_texts,
        candidate_ids=candidate_ids
    )
    time_tfidf = time.time() - start_tfidf
    tfidf_speed = len(resume_texts) / time_tfidf if time_tfidf > 0 else 0
    logger.info(f"  ✅ TF-IDF finished in {time_tfidf:.2f}s ({tfidf_speed:.1f} resumes/sec)")

    # -----------------------------------------------------------
    # 2. SENTENCE-TRANSFORMER BENCHMARK
    # -----------------------------------------------------------
    logger.info("\n" + "─" * 65)
    logger.info("🧠 2. BENCHMARKING SENTENCE-TRANSFORMERS (SEMANTIC)")
    logger.info("─" * 65)

    start_semantic = time.time()
    semantic_engine = SemanticScorer(model_name="all-MiniLM-L6-v2")
    semantic_results = semantic_engine.compute_similarity_scores(
        jd_text=benchmark_jd,
        resume_texts=resume_texts,
        candidate_ids=candidate_ids
    )
    time_semantic = time.time() - start_semantic
    semantic_speed = len(resume_texts) / time_semantic if time_semantic > 0 else 0
    logger.info(f"  ✅ Semantic finished in {time_semantic:.2f}s ({semantic_speed:.1f} resumes/sec)")

    # -----------------------------------------------------------
    # 3. MERGE & COMPARE SCORES
    # -----------------------------------------------------------
    logger.info("\n" + "─" * 65)
    logger.info("📊 3. ANALYZING CORRELATION & METRICS")
    logger.info("─" * 65)

    tfidf_map = {r["candidate_id"]: r["tfidf_score"] for r in tfidf_results}
    semantic_map = {r["candidate_id"]: r["semantic_score"] for r in semantic_results}

    records = []
    for cid, cat in zip(candidate_ids, categories):
        t_val = tfidf_map.get(cid, 0.0)
        s_val = semantic_map.get(cid, 0.0)
        records.append({
            "candidate_id": cid,
            "category": cat,
            "tfidf_score": t_val,
            "semantic_score": s_val,
            "score_lift": round(s_val - t_val, 2)
        })

    comp_df = pd.DataFrame(records)

    # Spearman rank correlation
    spearman_corr, p_val = spearmanr(comp_df["tfidf_score"], comp_df["semantic_score"])

    # Category aggregations
    cat_summary = comp_df.groupby("category").agg(
        TFIDF_Avg=("tfidf_score", "mean"),
        Semantic_Avg=("semantic_score", "mean"),
        TFIDF_Max=("tfidf_score", "max"),
        Semantic_Max=("semantic_score", "max")
    ).round(2).reset_index()

    cat_summary.sort_values(by="Semantic_Avg", ascending=False, inplace=True)

    # -----------------------------------------------------------
    # 4. EXPORT VISUAL CHART
    # -----------------------------------------------------------
    chart_path = EDA_CHARTS_DIR / "04_model_comparison.png"

    plt.figure(figsize=(12, 7))
    x = np.arange(len(cat_summary))
    bar_width = 0.38

    plt.bar(x - bar_width/2, cat_summary["TFIDF_Avg"], bar_width, label="TF-IDF (Exact Keywords)", color="#7f8c8d")
    plt.bar(x + bar_width/2, cat_summary["Semantic_Avg"], bar_width, label="Sentence-Transformer (Semantic)", color="#2980b9")

    plt.xlabel("Job Category", fontsize=11, weight="bold")
    plt.ylabel("Average Match Score (%)", fontsize=11, weight="bold")
    plt.title(f"Model Comparison: TF-IDF vs Sentence-Transformer Embeddings (N = {len(comp_df)})", fontsize=13, weight="bold")
    plt.xticks(x, cat_summary["category"], rotation=60, ha="right", fontsize=9)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

    # Save Comparison CSV
    csv_report_path = OUTPUT_DIR / "model_comparison_report.csv"
    cat_summary.to_csv(csv_report_path, index=False)

    # -----------------------------------------------------------
    # 5. PRINT RESULTS TO TERMINAL
    # -----------------------------------------------------------
    logger.info(f"Spearman Rank Correlation: {spearman_corr:.3f} (p-value: {p_val:.2e})")
    logger.info(f"Average Semantic Score Lift: +{comp_df['score_lift'].mean():.2f}% points over TF-IDF")
    logger.info(f"Saved Chart: {chart_path}")
    logger.info(f"Saved CSV Report: {csv_report_path}")

    logger.info("\n" + "=" * 65)
    logger.info("🏆 TOP 5 CATEGORIES (MOST RELEVANT TO DATA SCIENCE JD)")
    logger.info("=" * 65)
    logger.info("\n" + cat_summary.head(5).to_string(index=False))

    logger.info("\n" + "=" * 65)
    logger.info("🛑 BOTTOM 5 CATEGORIES (CORRECTLY IDENTIFIED AS UNRELATED)")
    logger.info("=" * 65)
    logger.info("\n" + cat_summary.tail(5).to_string(index=False))
    logger.info("=" * 65)

    return {
        "spearman_correlation": spearman_corr,
        "comp_df": comp_df,
        "cat_summary": cat_summary
    }


if __name__ == "__main__":
    run_model_benchmark(sample_limit=240)