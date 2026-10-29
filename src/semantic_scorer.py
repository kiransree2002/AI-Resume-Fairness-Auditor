"""
Semantic Similarity Scorer using Sentence-Transformers (Production v2)
===================================================================
Computes meaning-based similarity between Job Descriptions and resumes
using pre-trained transformer embeddings.

Solves the TF-IDF keyword mismatch problem:
  - TF-IDF: "PyTorch" vs "deep learning" = 0% match (different words)
  - Semantic: "PyTorch" vs "deep learning" = 78% match (similar meaning)

Model: all-MiniLM-L6-v2 (80MB, runs on CPU, no GPU required)
Uses local cache to prevent HuggingFace network pings in Streamlit.

Team: AI/ML (Week 2 Core Deliverable)
"""

import os
import logging
from typing import Dict, List, Any, Optional

import numpy as np

# Suppress HuggingFace parallelization and symlink warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.config import GATE_WEIGHT, TFIDF_WEIGHT
from src.text_cleaner import clean_text

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
# GLOBAL MODEL CACHE (Prevents re-instantiating HTTP clients)
# -----------------------------------------------------------
_MODEL_CACHE = {}


def _get_transformer_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Loads and caches the SentenceTransformer model singleton.
    Tries loading from local disk cache first to prevent network calls.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading Sentence-Transformer model: '{model_name}'...")

    try:
        # Try local cache first (offline, fast, no HTTP calls)
        model = SentenceTransformer(model_name, local_files_only=True)
        logger.info("  ✅ Loaded model from local cache (Offline Mode).")
    except Exception:
        # Fallback to Hub download if local cache miss
        logger.info("  Connecting to HuggingFace Hub to download model...")
        model = SentenceTransformer(model_name)
        logger.info("  ✅ Loaded model from Hugging Face Hub.")

    _MODEL_CACHE[model_name] = model
    return model


# -----------------------------------------------------------
# SEMANTIC SCORER CLASS
# -----------------------------------------------------------

class SemanticScorer:
    """
    Encodes text into semantic vectors using Sentence-Transformers
    and computes cosine similarity between JD and resumes.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = _get_transformer_model(model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Converts a list of text strings into semantic embedding vectors.
        """
        cleaned = [clean_text(t) if t else "" for t in texts]
        embeddings = self.model.encode(
            cleaned,
            batch_size=64,
            show_progress_bar=False,
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

        jd_embedding = all_embeddings[0:1]       # Shape: (1, dim)
        resume_embeddings = all_embeddings[1:]   # Shape: (N, dim)

        # Cosine similarity (L2 normalized vectors -> dot product = cosine)
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
    """
    combined = (gate_score * gate_weight) + (semantic_score * semantic_weight)
    return round(min(max(combined, 0.0), 100.0), 2)


# -----------------------------------------------------------
# ENTRY POINT (UNIT TEST)
# -----------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING OFFLINE SEMANTIC SCORER")
    logger.info("=" * 60)

    jd_text = "Senior Data Scientist role. Python, SQL, Machine Learning, PyTorch."
    resume_texts = [
        "Data Scientist skilled in Python, Deep Learning, PyTorch, SQL.",
        "Sales Associate with experience in retail management."
    ]

    scorer = SemanticScorer()
    res = scorer.compute_similarity_scores(jd_text, resume_texts)
    for r in res:
        logger.info(f"  {r['candidate_id']}: {r['semantic_score']}%")
    logger.info("=" * 60)