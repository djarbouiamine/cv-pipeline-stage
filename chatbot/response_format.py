"""Format de réponse concis pour les recherches sémantiques de profils."""

from __future__ import annotations

import re
from typing import List, Optional

SEARCH_CRITERIA_PATTERNS = {
    "Intelligence Artificielle": [
        r"\bia\b", r"\bai\b", "intelligence artificielle", "artificial intelligence",
        "machine learning", "deep learning", "llm", "nlp", "computer vision",
        "réseau de neurones", "reseau de neurones", "mlops",
    ],
    "Embarqué": [
        "embarqu", "embedded", "iot", "adas", "microcontr", "fpga", "rtos",
        "système embarqué", "systeme embarque", "embarque",
    ],
    "Cybersécurité": [
        "cybersécurité", "cybersecurite", "cyber", "sécurité informatique",
        "pentest", "malware",
    ],
    "Data / Analytics": [
        "data science", "data engineering", "data engineer", "big data", "analytics",
    ],
    "Cloud & DevOps": [
        "devops", "cloud", "kubernetes", "docker", "ci/cd", "terraform",
    ],
    "Développement Logiciel": [
        "développement", "developpement", "software", "full stack", "fullstack",
        "backend", "frontend",
    ],
}

SEMANTIC_RESPONSE_INSTRUCTIONS = """
MODE-SPECIFIC FORMAT (recommendations / profile search):

Follow the OUTPUT FORMAT defined in the system prompt. Use medals 🥇🥈🥉 for top 3.

Sections finales (courtes, puces) :

🔍 **Search Summary** — intent, method, CVs analyzed, ranking method (from context).

🛡️ **Confidence:** High / Medium / Low (XX%) — **Reason:** …

🚫 **Rejected profiles** — name + reason (do not recommend).

**Conclusion** — 1–2 sentences max.

**You can also ask:** — 3–5 follow-up suggestions from context.
"""

COMPARISON_RESPONSE_INSTRUCTIONS = """
COMPARISON FORMAT (mandatory):

1. Markdown table **Feature | Candidate A | Candidate B | …** covering:
   Skills, Languages, Frameworks, Projects, Certifications, Experience,
   Education, Strengths, Weaknesses, Risks, Best role.
   Use ✅ / ✗ for key skills and ⭐ (1–5) for domain ratings based on
   scores_categories / actual content — no invention.

2. Label both scores when present:
   **Job Fit:** X% (computed for this query) | **CV Quality:** Y/100 (stored in database)

3. **Best for …** — one conclusion per axis (e.g. "Best for Computer Vision: …").

4. End with **Overall recommendation.**

Professional tone. Short evidence (project or technology cited).
"""

STATS_RESPONSE_INSTRUCTIONS = """
STATISTICS FORMAT:

Never invent numbers. Use only database aggregations provided in context.

Cover when available: average score, category distribution, most common
technology/framework, years of experience, technology frequency,
programming language frequency, project counts.

For each figure, one line **How calculated:** (e.g. average over all indexed CVs).

Summarize histograms / top technologies / languages / categories in bullets
with a recruiter-oriented reading (distribution, trends).

Do not invent charts not present in the data.
"""

NO_EXACT_MATCH_INSTRUCTIONS = """
NO EXACT MATCH — mandatory structure:

1. **No exact match found.** (or **Aucune correspondance exacte trouvée.**)
2. Explain what is missing (mandatory criterion).
3. **Closest matching profiles** — related skills only; do not claim they meet the mandatory criterion.
4. **Rejected profiles** (name + reason) if listed in context.
5. State clearly which required skills are missing.
6. Prudent **Conclusion** for the recruiter.
"""


def detect_search_criteria(question: str) -> List[str]:
    q = question.lower()
    found: List[str] = []
    for label, patterns in SEARCH_CRITERIA_PATTERNS.items():
        for pattern in patterns:
            if pattern.startswith(r"\b"):
                if re.search(pattern, q):
                    found.append(label)
                    break
            elif pattern in q:
                found.append(label)
                break
    return found


def build_semantic_answer_instructions(
    question: str,
    n_docs: int,
    total_cvs: int | None = None,
    no_exact_match: bool = False,
    mandatory_label: Optional[str] = None,
    is_recommendation: bool = False,
    confidence_label: Optional[str] = None,
    confidence_pct: Optional[float] = None,
    confidence_reasons: Optional[List[str]] = None,
    follow_ups: Optional[List[str]] = None,
    rejected: Optional[List[dict]] = None,
    search_process: Optional[List[str]] = None,
    interview_questions: Optional[List[str]] = None,
) -> str:
    criteria = detect_search_criteria(question)
    lines = [SEMANTIC_RESPONSE_INSTRUCTIONS.strip()]

    if no_exact_match and mandatory_label:
        lines.append(NO_EXACT_MATCH_INSTRUCTIONS.strip())
        lines.append(
            f"\n**Critère obligatoire non satisfait par les CVs fournis** : {mandatory_label}"
        )

    if is_recommendation:
        lines.append(
            "\n**Mode recommandation** : le classement utilise le **Job Fit Score** "
            "(calculé pour cette recherche : 40% compétences, 25% sémantique, "
            "15% qualité CV, 10% expérience, 5% certifs, 5% projets). "
            "Le **CV Quality Score** reste la note stockée en base — ne pas l'utiliser "
            "seul pour classer les candidats."
        )

    if criteria:
        crit_str = ", ".join(criteria)
        lines.append(
            f"\n**Critères de recherche détectés** : {crit_str}\n"
            f"Pour CHAQUE critère, ✅ preuve explicite ou ⚠️ absence explicite."
        )

    if confidence_label and confidence_pct is not None:
        reasons = " ; ".join(confidence_reasons or [])
        lines.append(
            f"\n**Confiance à afficher** : {confidence_label} ({confidence_pct}%). "
            f"Raisons suggérées : {reasons or 'voir contexte'}."
        )

    if search_process:
        lines.append(
            "\n**Search Summary (recopy as bullets ✓):**\n"
            + "\n".join(f"- {s}" for s in search_process)
        )

    if rejected:
        rej_lines = [f"- {r.get('nom', '?')} : {r.get('reason', '')}" for r in rejected[:8]]
        lines.append("\n**Profils écartés (ne pas recommander comme match exact)** :\n" + "\n".join(rej_lines))

    if follow_ups:
        lines.append(
            "\n**Suggestions de questions de suivi (section « Vous pouvez aussi demander »)** :\n"
            + "\n".join(f"- {s}" for s in follow_ups)
        )

    if interview_questions:
        lines.append(
            "\n**Questions d'entretien suggérées (si pertinent)** :\n"
            + "\n".join(f"{i+1}. {q}" for i, q in enumerate(interview_questions))
        )

    if total_cvs is not None and n_docs < total_cvs:
        lines.append(
            f"\n({n_docs} CVs fournis sur {total_cvs} indexés — ne citer que ceux-ci.)"
        )
    else:
        lines.append(f"\n({n_docs} CV(s) dans le contexte.)")

    return "\n".join(lines)
