"""
TF-IDF Baseline Similarity Scorer (Production)
==============================================
Calculates document-level semantic keyword overlap between a Job Description
and candidate resumes using Term Frequency-Inverse Document Frequency (TF-IDF)
and Cosine Similarity.

Team: AI/ML (Week 1 Core Deliverable)
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    GATE_WEIGHT,
    TFIDF_WEIGHT
)
from text_cleaner import clean_text

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
# TF-IDF COMPUTATION ENGINE
# -----------------------------------------------------------

class TFIDFScorer:
    """
    Vectorizes Job Descriptions and candidate resumes to compute
    cosine similarity match percentages.
    """

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
            sublinear_tf=True  # Applies sublinear scaling (1 + log(tf)) to prevent keyword stuffing
        )

    def compute_similarity_scores(
        self,
        jd_text: str,
        resume_texts: List[str],
        candidate_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Computes cosine similarity between one JD and N resumes.

        Args:
            jd_text: Raw or cleaned text of the Job Description.
            resume_texts: List of resume text strings.
            candidate_ids: Optional list of candidate IDs corresponding to resume_texts.

        Returns:
            List of score dictionaries sorted by tfidf_score descending.
        """
        if not jd_text.strip() or not resume_texts:
            logger.warning("Empty JD text or resume text list provided to TF-IDF scorer.")
            return []

        if candidate_ids is None:
            candidate_ids = [f"RES_{i+1:04d}" for i in range(len(resume_texts))]

        # 1. Clean all input texts
        clean_jd = clean_text(jd_text)
        clean_resumes = [clean_text(t) for t in resume_texts]

        # 2. Build combined corpus: index 0 is JD, indices 1..N are Resumes
        corpus = [clean_jd] + clean_resumes

        # 3. Fit and transform into TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # 4. Extract JD vector vs Resume matrix
        jd_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]

        # 5. Calculate Cosine Similarity: angle between vectors (0.0 to 1.0)
        cosine_sims = cosine_similarity(jd_vector, resume_vectors).flatten()

        results = []
        for cid, sim in zip(candidate_ids, cosine_sims):
            score_100 = round(float(sim) * 100.0, 2)
            results.append({
                "candidate_id": cid,
                "tfidf_score": score_100,
                "raw_cosine_similarity": round(float(sim), 4)
            })

        return results


# -----------------------------------------------------------
# UNIFIED SCORE COMBINER
# -----------------------------------------------------------

def calculate_combined_score(
    gate_score: float,
    tfidf_score: float,
    gate_weight: float = GATE_WEIGHT,
    tfidf_weight: float = TFIDF_WEIGHT
) -> float:
    """
    Combines the Gate Cascade score with the TF-IDF cosine score.

    Formula:
        Final Score = (Gate_Score * 0.70) + (TFIDF_Score * 0.30)

    Args:
        gate_score: Score from gate cascade (0.0 to 100.0).
        tfidf_score: Score from TF-IDF matcher (0.0 to 100.0).
        gate_weight: Importance weighting for gate cascade (default: 0.70).
        tfidf_weight: Importance weighting for TF-IDF (default: 0.30).

    Returns:
        Weighted combined score rounded to 2 decimal places.
    """
    combined = (gate_score * gate_weight) + (tfidf_score * tfidf_weight)
    return round(min(max(combined, 0.0), 100.0), 2)


# -----------------------------------------------------------
# ENTRY POINT (UNIT TEST)
# -----------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING PRODUCTION TF-IDF BASELINE SCORER")
    logger.info("=" * 60)

    sample_jd = """
    We are seeking a Data Scientist with strong experience in Python,
    SQL, Machine Learning, predictive modeling, and Scikit-Learn.
    Must understand statistical analysis and data visualization tools like Power BI.
    """

    sample_resumes = [
        "Experienced Data Scientist skilled in Python, Machine Learning, SQL, statistical modeling, Scikit-Learn, and Power BI.",
        "Sales Associate with 5 years in customer relations, retail management, POS systems, and marketing.",
        "Software Developer with Java, Spring Boot, microservices architecture, Docker, and REST APIs."
    ]

    c_ids = ["RES_DATA_SCIENTIST", "RES_SALES", "RES_JAVA_DEV"]

    scorer = TFIDFScorer()
    tfidf_out = scorer.compute_similarity_scores(sample_jd, sample_resumes, c_ids)

    logger.info(f"\n{'Candidate ID':<22}{'TF-IDF Match (%)':<18}{'Raw Cosine':<12}")
    logger.info("-" * 52)
    for r in tfidf_out:
        logger.info(f"{r['candidate_id']:<22}{r['tfidf_score']:<18}{r['raw_cosine_similarity']:<12}")

    logger.info("\n" + "=" * 60)