"""
Score Explainability Engine
===========================
Translates raw numerical scores and gate cascade logs into 
human-readable, plain-English narratives for HR and compliance audits.
"""

from typing import Dict

def generate_candidate_explanation(candidate_record: Dict) -> str:
    """
    Takes a candidate's screening results dictionary and generates
    a formatted markdown explanation of their score.
    """
    cid = candidate_record.get("Candidate_ID", "Unknown")
    status = candidate_record.get("Status", "REJECTED")
    
    # Safe float conversion
    final_score = float(candidate_record.get("Final_Score", 0.0))
    gate_score = float(candidate_record.get("Gate_Score", 0.0))
    semantic_score = float(candidate_record.get("Semantic_Score", 0.0))
    
    matched = candidate_record.get("Matched_Skills", "None")
    missing = candidate_record.get("Missing_Skills", "None")
    reason = candidate_record.get("Rejection_Reason", "N/A")

    # 1. Headline
    status_icon = "🟢" if status == "SHORTLISTED" else "🟡" if status == "REVIEW" else "🔴"
    lines = [
        f"### {status_icon} Verdict: {status} (Final Score: **{final_score:.2f}/100**)"
    ]

    # 2. Rejection Logic (If applicable)
    if status == "REJECTED":
        lines.append("#### 🛑 Why was this candidate rejected?")
        lines.append(f"> **{reason}**")
        lines.append("")
        lines.append(f"- **Skills matched before rejection:** {matched}")
        if missing and missing != "None":
            lines.append(f"- **Critical skills missing:** {missing}")
        
    # 3. Success / Review Logic
    else:
        lines.append("#### ✅ Gate Cascade Breakdown (Hard Requirements)")
        lines.append(f"**Gate Score:** `{gate_score:.2f}/100`")
        lines.append(f"- **Matched Skills:** {matched}")
        if missing and missing != "None":
            lines.append(f"- **Missing Skills (Penalty applied):** {missing}")
        else:
            lines.append("- **Missing Skills:** None (Perfect keyword match!)")
            
    # 4. Semantic Context Logic
    lines.append("\n#### 🧠 AI Semantic Context (Soft Matching)")
    lines.append(f"**Semantic Score:** `{semantic_score:.2f}/100`")
    
    if semantic_score >= 35:
        lines.append("> *High contextual alignment. The candidate's overall experience closely mirrors the job description.*")
    elif semantic_score >= 20:
        lines.append("> *Moderate contextual alignment. The AI found some overlap in domain language, but not a perfect match.*")
    else:
        lines.append("> *Low contextual alignment. The resume language differs significantly from the actual day-to-day responsibilities of this job.*")

    return "\n".join(lines)