"""
Gate Cascade Matching Engine (Production v2 — Configurable)
============================================================
Executes a configurable N-gate waterfall evaluation comparing
structured Job Descriptions against candidate resume profiles.

Key Changes (Post Interviewer Feedback):
- Default gate order: Skills → Education → Experience
- Gate order and weights are configurable per job role by HR
- Dynamic gate count (2, 3, or more gates)
- Score calculation uses HR-defined weights

Team: AI/ML (Week 1 Rebuild Deliverable)
"""

import logging
from typing import Dict, List, Any, Tuple, Optional

from src.config import (
    DEFAULT_GATE_ORDER,
    DEFAULT_GATE_WEIGHTS,
    SHORTLIST_THRESHOLD,
    MAYBE_THRESHOLD
)

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
# INDIVIDUAL GATE EVALUATORS
# -----------------------------------------------------------

def evaluate_skills_gate(
    jd_skills: Dict[str, Any],
    candidate_skills: Dict[str, Any]
) -> Tuple[str, float, str, List[str], List[str]]:
    """
    Evaluates the SKILLS gate (now runs FIRST by default).

    Returns:
        status: 'PASS' | 'PARTIAL' | 'FAIL'
        score_pct: 0.0 to 1.0 (fraction of max points earned)
        reason: Human-readable explanation
        matched: List of matched skill names
        missing: List of missing skill names
    """
    must_skills = jd_skills.get("must_have", [])
    preferred_skills = jd_skills.get("preferred", [])


    print("DEBUG JD skills structure:", jd_skills)
    print("DEBUG Candidate skills structure:", candidate_skills)

    if not must_skills and not preferred_skills:
        return "PASS", 1.0, "No skill requirements specified.", [], []

    # Combine all candidate skills
    all_candidate = [
        s.lower() for s in candidate_skills.get("technical", []) + candidate_skills.get("soft", [])
    ]

    # --- Check Must-Have Skills ---
    matched_must = []
    missing_must = []

    for item in must_skills:
        skill_lower = item["skill"].lower()
        if skill_lower in all_candidate:
            matched_must.append(item["skill"])
        elif any(skill_lower in s or s in skill_lower for s in all_candidate):
            matched_must.append(f"{item['skill']} (Partial)")
        else:
            missing_must.append(item["skill"])

    # --- Check Preferred Skills (bonus) ---
    # --- Check Preferred Skills (bonus) ---
    matched_pref = []
    missing_pref = []

    for item in preferred_skills:
        skill_lower = item["skill"].lower()

        if (
            skill_lower in all_candidate
            or any(skill_lower in s for s in all_candidate)
        ):
            matched_pref.append(item["skill"])
        else:
            missing_pref.append(item["skill"])

    all_matched = matched_must + matched_pref
    total_must = len(must_skills)
    matched_must_count = len(matched_must)

    # Determine status
    if total_must == 0:
        status = "PASS"
        score_pct = 0.5 + (len(matched_pref) * 0.1)  # Bonus for preferred
        score_pct = min(score_pct, 1.0)
    elif matched_must_count == total_must:
        status = "PASS"
        score_pct = 0.8 + (len(matched_pref) * 0.05)
        score_pct = min(score_pct, 1.0)
    elif matched_must_count >= (total_must / 2):
        status = "PARTIAL"
        score_pct = 0.4 + (matched_must_count / total_must) * 0.3
    else:
        status = "FAIL"
        score_pct = (matched_must_count / total_must) * 0.3

    reason = f"Matched {matched_must_count}/{total_must} must-have + {len(matched_pref)} preferred skills."
    all_missing = missing_must + missing_pref

    print("DEBUG must skills:", must_skills)
    print("DEBUG preferred skills:", preferred_skills)
    print("DEBUG missing must:", missing_must)
    print("DEBUG missing preferred:", missing_pref)

    return status, round(score_pct, 3), reason, all_matched, all_missing


def evaluate_education_gate(
    jd_education: Dict[str, Any],
    candidate_education: Dict[str, Any]
) -> Tuple[str, float, str]:
    """
    Evaluates the EDUCATION gate.

    Returns:
        status: 'PASS' | 'FAIL'
        score_pct: 0.0 to 1.0
        reason: Human-readable explanation
    """
    required = [d.lower() for d in jd_education.get("required_degrees", ["Bachelor's"])]
    candidate = [d.lower() for d in candidate_education.get("degrees", [])]

    if not required:
        return "PASS", 1.0, "No specific degree requirements."

    if not candidate:
        return "FAIL", 0.0, f"No degree detected. Required: {', '.join(required)}."

    for c_deg in candidate:
        for r_deg in required:
            if r_deg in c_deg or c_deg in r_deg:
                return "PASS", 1.0, f"Degree match: {c_deg.title()} satisfies {r_deg.title()}."

    return "FAIL", 0.0, f"Education mismatch: has [{', '.join(candidate)}], needs [{', '.join(required)}]."


def evaluate_experience_gate(
    jd_experience: Dict[str, Any],
    candidate_experience: Dict[str, Any]
) -> Tuple[str, float, str]:
    """
    Evaluates the EXPERIENCE gate.

    Returns:
        status: 'PASS' | 'FAIL'
        score_pct: 0.0 to 1.0
        reason: Human-readable explanation
    """
    min_years = float(jd_experience.get("min_years", 0.0))
    candidate_years = float(candidate_experience.get("years", 0.0))
    required_skill_years = jd_experience.get("skill_years", {})
    candidate_skill_years = {
        skill.lower(): float(years)
        for skill, years in candidate_experience.get("skill_years", {}).items()
    }

    if required_skill_years:
        skill_scores = []
        unmet_skills = []
        for skill, required_years in required_skill_years.items():
            required_years = float(required_years)
            if required_years <= 0:
                continue
            actual_years = candidate_skill_years.get(skill.lower(), 0.0)
            if actual_years >= required_years:
                ratio = min(actual_years / required_years, 2.0)
                skill_scores.append(0.7 + (ratio - 1.0) * 0.3)
            else:
                skill_scores.append(
                    (actual_years / required_years) * 0.5
                )
                unmet_skills.append(skill)

        if skill_scores:
            score_pct = sum(skill_scores) / len(skill_scores)
            if unmet_skills:
                return (
                    "FAIL",
                    round(score_pct, 3),
                    "Skill-specific experience not met for: "
                    + ", ".join(unmet_skills)
                    + ".",
                )
            return (
                "PASS",
                round(score_pct, 3),
                "Required experience met for all specified skills.",
            )

    if min_years == 0:
        return "PASS", 1.0, "No minimum experience required."

    if candidate_years >= min_years:
        # Scale: exactly meeting = 0.7, double = 1.0
        ratio = min(candidate_years / min_years, 2.0)
        score_pct = 0.7 + (ratio - 1.0) * 0.3
        return "PASS", round(min(score_pct, 1.0), 3), f"Experience met: {candidate_years}yrs >= {min_years}yrs."
    else:
        # Partial credit for being close
        score_pct = (candidate_years / min_years) * 0.5
        return "FAIL", round(score_pct, 3), f"Low experience: {candidate_years}yrs < {min_years}yrs."


# -----------------------------------------------------------
# GATE DISPATCHER (Maps gate name → evaluator function)
# -----------------------------------------------------------

GATE_EVALUATORS = {
    "skills": lambda jd, cand: evaluate_skills_gate(
        jd.get("requirements", {}).get("skills", {}),
        cand.get("skills", {})
    ),
    "education": lambda jd, cand: evaluate_education_gate(
        jd.get("requirements", {}).get("education", {}),
        cand.get("education", {})
    ),
    "experience": lambda jd, cand: evaluate_experience_gate(
        jd.get("requirements", {}).get("experience", {}),
        cand.get("experience", {})
    )
}


# -----------------------------------------------------------
# CONFIGURABLE CASCADE RUNNER
# -----------------------------------------------------------

def run_candidate_cascade(
    jd_json: Dict[str, Any],
    candidate_profile: Dict[str, Any],
    gate_order: Optional[List[str]] = None,
    gate_weights: Optional[Dict[str, float]] = None,
    stop_on_failure: bool = True
) -> Dict[str, Any]:
    """
    Executes the configurable gate cascade for a single candidate.

    Args:
        jd_json: Structured JD from jd_parser.py
        candidate_profile: Structured resume from resume_parser.py
        gate_order: HR-defined gate execution order (default: skills → education → experience)
        gate_weights: HR-defined weight per gate (default: 0.50, 0.25, 0.25)
        stop_on_failure: Whether a failed gate should prevent evaluating later gates.

    Returns:
        Full evaluation report with per-gate results and final score.
    """
    # Use HR config from JD if available, otherwise use defaults
    jd_gate_config = jd_json.get("gate_configuration", {})
    active_gate_order = gate_order or jd_gate_config.get("gate_order", DEFAULT_GATE_ORDER)
    active_gate_weights = gate_weights or jd_gate_config.get("gate_weights", DEFAULT_GATE_WEIGHTS)

    candidate_id = candidate_profile.get("candidate_id", "UNKNOWN")

    report = {
        "candidate_id": candidate_id,
        "category": candidate_profile.get("category", "Unspecified"),
        "gate_order_used": active_gate_order,
        "gate_results": {},
        "final_gate_score": 0.0,
        "status": "REJECTED",
        "rejection_reason": None,
        "matched_skills": [],
        "missing_skills": []
    }

    total_weighted_score = 0.0

    for gate_name in active_gate_order:
        evaluator = GATE_EVALUATORS.get(gate_name)
        if not evaluator:
            logger.warning(f"Unknown gate '{gate_name}' — skipping.")
            continue

        weight = active_gate_weights.get(gate_name, 0.25)
        result = evaluator(jd_json, candidate_profile)

        print("DEBUG SKILLS RESULT:", result)

        # Skills gate returns 5 values (with matched/missing lists)
        if gate_name == "skills":
            status, score_pct, reason, matched, missing = result
            report["gate_results"][gate_name] = {
                "status": status,
                "score_pct": score_pct,
                "weight": weight,
                "weighted_score": round(score_pct * weight * 100, 2),
                "reason": reason,
                "matched": matched,
                "missing": missing
            }
            report["matched_skills"].extend(matched)
            report["missing_skills"].extend(missing)
        else:
            status, score_pct, reason = result
            report["gate_results"][gate_name] = {
                "status": status,
                "score_pct": score_pct,
                "weight": weight,
                "weighted_score": round(score_pct * weight * 100, 2),
                "reason": reason}
        total_weighted_score += score_pct * weight * 100
        # Record gate failures and respect stop_on_failure.
        if status == "FAIL":
            if not report["rejection_reason"]:
                report["rejection_reason"] = (
                    f"Gate '{gate_name}' failed: {reason}"
                )

            if stop_on_failure:
                report["final_gate_score"] = round(
                    total_weighted_score, 2
                )
                report["status"] = "REJECTED"
                return report
    # Final score and status
    final_score = min(round(total_weighted_score, 2), 100.0)
    report["final_gate_score"] = final_score

    skills_result = report["gate_results"].get("skills", {})
    matched_skills = report["matched_skills"]
    skills_score = skills_result.get("score_pct", 0.0)

    if final_score >= SHORTLIST_THRESHOLD:
        report["status"] = "SHORTLISTED"
    elif final_score >= MAYBE_THRESHOLD:
        report["status"] = "REVIEW"
    elif matched_skills or skills_score > 0:
        # Don't automatically reject a candidate who has relevant skills.
        report["status"] = "REVIEW"
        report["rejection_reason"] = (
            "Relevant skills detected; manual review recommended. "
            + (report["rejection_reason"] or "")
        ).strip()
    else:
        report["status"] = "REJECTED"
        if not report["rejection_reason"]:
            report["rejection_reason"] = (
                f"Score ({final_score}) below threshold "
                f"({SHORTLIST_THRESHOLD}) and no relevant skills matched."
            )

    return report



def run_batch_cascade(
    jd_json: Dict[str, Any],
    candidate_profiles: List[Dict[str, Any]],
    gate_order: Optional[List[str]] = None,
    gate_weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Runs the cascade for all candidates and sorts by final score descending.
    """
    results = [
        run_candidate_cascade(jd_json, p, gate_order, gate_weights)
        for p in candidate_profiles
    ]
    results.sort(key=lambda x: x["final_gate_score"], reverse=True)
    return results


# -----------------------------------------------------------
# ENTRY POINT (TEST WITH NEW GATE ORDER)
# -----------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TESTING CONFIGURABLE GATE CASCADE (Skills First)")
    logger.info("=" * 60)

    sample_jd = {
        "metadata": {"job_title": "Data Scientist", "company_name": "Test Corp"},
        "gate_configuration": {
            "gate_order": ["skills", "education", "experience"],
            "gate_weights": {"skills": 0.50, "education": 0.25, "experience": 0.25}
        },
        "requirements": {
            "skills": {
                "must_have": [
                    {"skill": "Python", "importance": 0.95},
                    {"skill": "Machine Learning", "importance": 0.90}
                ],
                "preferred": [
                    {"skill": "Aws", "importance": 0.65}
                ]
            },
            "education": {"required_degrees": ["Bachelor's", "Master's"], "importance": 0.85},
            "experience": {"min_years": 3.0, "importance": 0.80}
        }
    }

    test_candidates = [
        {
            "candidate_id": "RES_A",
            "category": "IT",
            "education": {"degrees": ["Master's"]},
            "experience": {"years": 5.0},
            "skills": {"technical": ["Python", "Machine Learning", "Aws"], "soft": ["Agile"]}
        },
        {
            "candidate_id": "RES_B",
            "category": "HR",
            "education": {"degrees": ["Bachelor's"]},
            "experience": {"years": 8.0},
            "skills": {"technical": ["Excel"], "soft": ["Communication"]}
        },
        {
            "candidate_id": "RES_C",
            "category": "IT",
            "education": {"degrees": ["Diploma"]},
            "experience": {"years": 2.0},
            "skills": {"technical": ["Python", "Sql"], "soft": []}
        }
    ]

    results = run_batch_cascade(sample_jd, test_candidates)

    for r in results:
        logger.info(f"\n{'─' * 55}")
        logger.info(f"Candidate: {r['candidate_id']} | Score: {r['final_gate_score']}/100 | Status: {r['status']}")
        logger.info(f"Gate Order Used: {r['gate_order_used']}")
        for gname, gdata in r["gate_results"].items():
            icon = "✅" if gdata["status"] in ["PASS"] else ("⚠️" if gdata["status"] == "PARTIAL" else "❌")
            logger.info(f"  {icon} {gname.upper()} (wt:{gdata['weight']}): {gdata['reason']} → {gdata['weighted_score']}pts")
        if r["rejection_reason"]:
            logger.info(f"  🛑 {r['rejection_reason']}")

    logger.info(f"\n{'=' * 60}")