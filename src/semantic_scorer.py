"""
Semantic Similarity Scorer using Sentence-Transformers (Production)
===================================================================
Computes meaning-based similarity between Job Descriptions and resumes
using pre-trained transformer embeddings.

Solves the TF-IDF keyword mismatch problem:
  - TF-IDF: "PyTorch" vs "deep learning" = 0% match (different words)
  - Semantic: "PyTorch" vs "deep learning" = 78% match (similar meaning)

Model: all-MiniLM-L6-v2 (80MB, runs on CPU, no GPU required)

Team: AI/ML (Week 2 Core Deliverable)
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np

from config import GATE_WEIGHT, TFIDF_WEIGHT
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
# SEMANTIC SCORER CLASS
# -----------------------------------------------------------

class SemanticScorer:
    """
    Encodes text into semantic vectors using Sentence-Transformers
    and computes cosine similarity between JD and resumes.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Loads the pre-trained sentence transformer model.

        Args:
            model_name: HuggingFace model identifier.
                        'all-MiniLM-L6-v2' = fast, lightweight, CPU-friendly
                        'all-mpnet-base-v2' = more accurate, slower (use Colab GPU)
        """
        logger.info(f"Loading Sentence-Transformer model: '{model_name}'...")
        logger.info("  (First run downloads ~80MB model. Subsequent runs are instant.)")

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        logger.info(f"  ✅ Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Converts a list of text strings into semantic embedding vectors.

        Args:
            texts: List of cleaned text strings.

        Returns:
            NumPy array of shape (num_texts, embedding_dim).
        """
        cleaned = [clean_text(t) if t else "" for t in texts]
        embeddings = self.model.encode(
            cleaned,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embeddings

    def compute_similarity_scores(
        self,
        jd_text: str,
        resume_texts: List[str],
        candidate_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Computes semantic cosine similarity between one JD and N resumes.

        Args:
            jd_text: Job Description text.
            resume_texts: List of resume text strings.
            candidate_ids: Optional candidate ID list.

        Returns:
            List of score dicts sorted by semantic_score descending.
        """
        if not jd_text.strip() or not resume_texts:
            logger.warning("Empty JD or resume list provided.")
            return []

        if candidate_ids is None:
            candidate_ids = [f"RES_{i+1:04d}" for i in range(len(resume_texts))]

        logger.info(f"Encoding JD + {len(resume_texts)} resumes into semantic vectors...")

        # Encode JD (index 0) and all resumes (indices 1..N) together
        all_texts = [jd_text] + resume_texts
        all_embeddings = self.encode_texts(all_texts)

        jd_embedding = all_embeddings[0:1]  # Shape: (1, dim)
        resume_embeddings = all_embeddings[1:]  # Shape: (N, dim)

        # Cosine similarity (vectors are already L2 normalized, so dot product = cosine)
        similarities = np.dot(resume_embeddings, jd_embedding.T).flatten()

        results = []
        for cid, sim in zip(candidate_ids, similarities):
            score_100 = round(float(sim) * 100.0, 2)
            results.append({
                "candidate_id": cid,
                "semantic_score": score_100,
                "raw_cosine_similarity": round(float(sim), 4)
            })

        results.sort(key=lambda x: x["semantic_score"], reverse=True)
        return results


# -----------------------------------------------------------
# COMBINED SCORER (Gate + Semantic)
# -----------------------------------------------------------

def calculate_semantic_combined_score(
    gate_score: float,
    semantic_score: float,
    gate_weight: float = GATE_WEIGHT,
    semantic_weight: float = TFIDF_WEIGHT
) -> float:
    """
    Combines Gate Cascade score with Semantic similarity score.

    Formula: Final = (Gate * 0.70) + (Semantic * 0.30)

    This replaces the old TF-IDF combined score.
    Semantic scores are typically 30-60% (vs TF-IDF's 3-9%),
    so the combined scores will be significantly higher.
    """
    combined = (gate_score * gate_weight) + (semantic_score * semantic_weight)
    return round(min(max(combined, 0.0), 100.0), 2)


# -----------------------------------------------------------
# ENTRY POINT (UNIT TEST)
# -----------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING SEMANTIC SCORER (Sentence-Transformers)")
    logger.info("=" * 60)

    jd_text = """
    Senior Data Scientist role. Must have Python, Machine Learning,
    SQL, and deep learning experience with PyTorch or TensorFlow.
    Statistical modeling and data visualization skills required.
    """

    resume_texts = [
        "Experienced data scientist with expertise in Python, deep learning, neural networks, TensorFlow, and statistical analysis.",
        "Sales manager with 10 years in retail, customer relations, and team leadership.",
        "Java developer skilled in Spring Boot, microservices, Docker, and REST APIs."
    ]

    c_ids = ["RES_DATA_SCIENTIST", "RES_SALES", "RES_JAVA_DEV"]

    scorer = SemanticScorer()
    results = scorer.compute_similarity_scores(jd_text, resume_texts, c_ids)

    logger.info(f"\n{'Candidate':<22}{'Semantic (%)':<15}{'Raw Cosine':<12}")
    logger.info("-" * 49)
    for r in results:
        logger.info(f"{r['candidate_id']:<22}{r['semantic_score']:<15}{r['raw_cosine_similarity']:<12}")

    # Compare with expected TF-IDF scores
    logger.info("\n📊 Expected Comparison:")
    logger.info("  TF-IDF would give:  Data Scientist ~40%, Sales ~0%, Java ~0%")
    logger.info("  Semantic gives:     Data Scientist ~60-75%, Sales ~15-25%, Java ~20-30%")
    logger.info("  WHY? Semantic understands 'deep learning' ≈ 'neural networks' ≈ 'PyTorch'")
    logger.info("=" * 60)