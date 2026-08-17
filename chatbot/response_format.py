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
        "aws", "azure", "gcp",
    ],
    "Développement Logiciel": [
        "développement", "developpement", "software", "full stack", "fullstack",
        "backend", "frontend",
    ],
}

SEMANTIC_RESPONSE_INSTRUCTIONS = """
[INTERNE — ne jamais recopier ni paraphraser dans la réponse utilisateur]

Appliquer le format OUTPUT du system prompt. Médailles 🥇🥈🥉 pour le top 3.
Virgule décimale française partout (ex. 0,7 an ; 84,2/100 ; 50,9 %).

Sections finales visibles pour l'utilisateur (courtes) :
- Search Summary : méthode + nombre de CVs analysés uniquement (info nouvelle).
- Conclusion : 1–2 phrases, une seule fois, info NOUVELLE uniquement.
  Si « Aucune correspondance exacte » est déjà en tête de réponse, la Conclusion
  ne la reformule pas — indiquer plutôt les meilleurs candidats partiels ou la suite.
- Vous pouvez aussi demander : 3–5 suggestions si pertinent.

Ne pas inclure intent, confiance pipeline, ni trace technique sauf demande explicite.
"""

COMPARISON_RESPONSE_INSTRUCTIONS = """
[INTERNE — ne jamais recopier dans la réponse utilisateur]

Tableau markdown Feature | Candidat A | Candidat B | …
Lignes : Skills, Languages, Frameworks, Projects, Certifications, Experience,
Education, Strengths, Weaknesses, Risks, Best role.
Skills et Languages distincts — fusionner si identiques.
Champ absent → "non renseigné".
Scores : Job Fit X % | CV Quality Y/100 (virgule française).
Terminer par Overall recommendation / Recommandation globale.
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
[INTERNE — ne jamais recopier dans la réponse utilisateur]

Aucune correspondance exacte : le dire une seule fois, en tête de réponse.
Critères non vérifiables (employeur actuel, salaire, entreprise précise absente des CVs) :
dire « Ce critère n'est pas renseigné dans les CVs indexés » — pas de Job Fit % trompeur.
Critères multiples : classer par nombre de critères partiellement satisfaits.
Fusion conclusion : si l'absence de match est déjà en tête, la Conclusion n'ajoute QUE
du nouveau (meilleurs partiels, action recruteur) — jamais reformuler le verdict.
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


UNVERIFIABLE_CRITERIA_PATTERNS: List[tuple[str, List[str]]] = [
    (
        "employeur actuel / entreprise précise",
        [
            r"employeur\s+actuel",
            r"current\s+employer",
            r"travaill(?:e|ant)\s+(?:actuellement\s+)?chez",
            r"works?\s+at",
            r"working\s+at",
            r"employé\s+chez",
            r"employee\s+at",
            r"chez\s+google",
            r"at\s+google\b",
            r"chez\s+microsoft",
            r"chez\s+amazon",
            r"chez\s+meta",
            r"chez\s+apple",
            r"entreprise\s+(?:actuelle|précise)",
        ],
    ),
    (
        "salaire",
        [r"\bsalaire\b", r"\bsalary\b", r"rémunération", r"remuneration"],
    ),
    (
        "localisation exacte non indexée",
        [r"habite\s+(?:à|a)\s+", r"vit\s+(?:à|a)\s+", r"based\s+in\s+(?!tunis)"],
    ),
]


def detect_unverifiable_criteria(question: str) -> List[str]:
    """Critères demandés sans champ CV fiable (employeur actuel, salaire, etc.)."""
    q = question.lower()
    found: List[str] = []
    for label, patterns in UNVERIFIABLE_CRITERIA_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, q, re.IGNORECASE):
                found.append(label)
                break
    return found


def build_semantic_answer_instructions(
    question: str,
    n_docs: int,
    total_cvs: int | None = None,
    no_exact_match: bool = False,
    mandatory_label: Optional[str] = None,
    unverifiable_criteria: Optional[List[str]] = None,
    is_recommendation: bool = False,
    confidence_label: Optional[str] = None,
    confidence_pct: Optional[float] = None,
    confidence_reasons: Optional[List[str]] = None,
    follow_ups: Optional[List[str]] = None,
    rejected: Optional[List[dict]] = None,
    search_process: Optional[List[str]] = None,
    interview_questions: Optional[List[str]] = None,
    count_line: Optional[str] = None,
) -> str:
    criteria = detect_search_criteria(question)
    lines = [SEMANTIC_RESPONSE_INSTRUCTIONS.strip()]

    if count_line:
        lines.append(
            f"\n[INTERNE] Comptage CVs CANONIQUE (utiliser EXACTEMENT cette formulation "
            f"dans Search Summary ET nulle part ailleurs avec un autre chiffre) : "
            f"{count_line}"
        )

    if unverifiable_criteria:
        uv = ", ".join(unverifiable_criteria)
        lines.append(
            f"\n[INTERNE] Critère(s) NON vérifiable(s) dans les CVs indexés : {uv}. "
            f"Commencer par : « Ce critère n'est pas renseigné dans les CVs indexés ». "
            f"Ne PAS afficher de Job Fit % sur ce critère. "
            f"Conclusion : info nouvelle uniquement (meilleurs partiels), pas reformuler l'absence de match."
        )

    if no_exact_match and mandatory_label:
        lines.append(NO_EXACT_MATCH_INSTRUCTIONS.strip())
        lines.append(
            f"\n[INTERNE] Critère non satisfait ou non vérifiable : {mandatory_label}. "
            f"Si absent des champs CV → « non renseigné dans les CVs indexés », pas de Job Fit %."
        )

    if is_recommendation and not unverifiable_criteria:
        lines.append(
            "\n[INTERNE] Classement par Job Fit (40% compétences, 25% sémantique, "
            "15% qualité CV, 10% expérience, 5% certifs, 5% projets). "
            "Ne pas recopier ces poids dans la réponse."
        )

    if criteria:
        crit_str = ", ".join(criteria)
        lines.append(
            f"\n[INTERNE] Thèmes détectés : {crit_str}. "
            f"Pour chaque thème : preuve explicite ou absence — ne pas citer cette ligne."
        )

    if confidence_label and confidence_pct is not None:
        reasons = " ; ".join(confidence_reasons or [])
        lines.append(
            f"\n[INTERNE] Confiance {confidence_label} ({confidence_pct} %) — "
            f"afficher seulement si pertinent ; raisons : {reasons or 'voir contexte'}. "
            f"Ne pas exposer par défaut."
        )

    if search_process:
        lines.append(
            "\n[INTERNE] Search Summary — reprendre UNIQUEMENT méthode + nb CVs "
            "(ne pas reformuler le verdict) :\n"
            + "\n".join(f"- {s}" for s in search_process)
        )

    if rejected:
        rej_lines = [f"- {r.get('nom', '?')} : {r.get('reason', '')}" for r in rejected[:8]]
        lines.append(
            "\n[INTERNE] Profils écartés (ne pas recommander comme match exact) :\n"
            + "\n".join(rej_lines)
        )

    if follow_ups:
        lines.append(
            "\n[INTERNE] Suggestions de suivi pour section « Vous pouvez aussi demander » :\n"
            + "\n".join(f"- {s}" for s in follow_ups)
        )

    if interview_questions:
        lines.append(
            "\n[INTERNE] Questions d'entretien possibles :\n"
            + "\n".join(f"{i+1}. {q}" for i, q in enumerate(interview_questions))
        )

    if total_cvs is not None and n_docs < total_cvs and not count_line:
        lines.append(
            f"\n[INTERNE] {n_docs} CVs fournis sur {total_cvs} indexés — ne citer que ceux-ci."
        )
    elif not count_line:
        lines.append(f"\n[INTERNE] {n_docs} CV(s) dans le contexte.")

    return "\n".join(lines)
