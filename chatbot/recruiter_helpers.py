"""Preuves, confiance, filtres recruteur, drapeaux rouges, questions d'entretien."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from chatbot.final_ranking import extract_query_skills, missing_skills


def build_evidence_lines(doc: Dict, highlight_skills: Optional[List[str]] = None) -> List[str]:
    """Lignes de preuves structurées pour le contexte LLM."""
    highlight_skills = highlight_skills or []
    lines: List[str] = []

    def _pick(items: List[str], limit: int = 6) -> List[str]:
        if not highlight_skills:
            return items[:limit]
        out = []
        for it in items:
            mark = "✅ " if any(h.lower() in it.lower() for h in highlight_skills) else "• "
            out.append(f"{mark}{it}")
        return out[:limit]

    langs = doc.get("langages") or []
    techs = doc.get("technologies") or []
    fws = doc.get("frameworks") or []
    if isinstance(langs, list) and langs:
        lines.append("Skills / langages : " + ", ".join(_pick([str(x) for x in langs], 8)))
    if isinstance(techs, list) and techs:
        lines.append("Technologies : " + ", ".join(_pick([str(x) for x in techs], 8)))
    if isinstance(fws, list) and fws:
        lines.append("Frameworks : " + ", ".join(_pick([str(x) for x in fws], 6)))

    proj = doc.get("projets") or []
    if isinstance(proj, list) and proj:
        proj_lines = _pick([str(p) for p in proj[:5]], 5)
        lines.append("Projets : " + " | ".join(proj_lines))

    certs = doc.get("certifications") or []
    if isinstance(certs, list) and certs:
        lines.append("Certifications : " + ", ".join(str(c) for c in certs[:4]))

    return lines


def recruiter_dimension_bars(doc: Dict) -> List[str]:
    """Barres texte à partir des scores catégories ou heuristiques."""
    nested = doc.get("scores_categories_ponderes") or doc.get("scores_categories") or []
    bars: List[str] = []
    if isinstance(nested, list):
        for item in nested[:8]:
            if isinstance(item, dict):
                dom = item.get("domaine", "?")
                sc = item.get("score")
                if isinstance(sc, (int, float)):
                    pct = min(100, max(0, float(sc)))
                    if pct <= 10:
                        pct *= 10
                    filled = int(pct / 10)
                    bar = "█" * filled + "░" * (10 - filled)
                    bars.append(f"  {dom:<22} {bar} {pct:.0f}%")
    if bars:
        return bars

    qual = doc.get("score_qualite_globale")
    if isinstance(qual, (int, float)):
        filled = int(min(100, qual) / 10)
        bar = "█" * filled + "░" * (10 - filled)
        bars.append(f"  Qualité globale       {bar} {qual:.0f}%")
    return bars


def detect_cv_red_flags(doc: Dict) -> List[str]:
    flags: List[str] = []
    if not (doc.get("email") or doc.get("telephone")):
        flags.append("coordonnées incomplètes")
    proj = doc.get("projets") or []
    if not proj or (isinstance(proj, list) and len(proj) == 0):
        flags.append("aucun projet listé")
    if doc.get("annees_experience") in (None, 0, "N/A"):
        flags.append("expérience professionnelle non renseignée ou nulle")
    alertes = doc.get("alertes_parcours") or []
    if isinstance(alertes, list):
        flags.extend(str(a) for a in alertes[:4])
    elif alertes:
        flags.append(str(alertes))
    # Duplicate skills heuristic
    all_sk = []
    for key in ("technologies", "langages", "frameworks"):
        v = doc.get(key) or []
        if isinstance(v, list):
            all_sk.extend(str(x).lower() for x in v)
    if len(all_sk) != len(set(all_sk)) and all_sk:
        flags.append("compétences possiblement dupliquées")
    dipl = doc.get("diplomes")
    if not dipl or dipl == "N/A":
        flags.append("diplôme / formation peu documenté")
    return flags[:8]


def suggest_interview_questions(doc: Dict, max_q: int = 5) -> List[str]:
    questions: List[str] = []
    proj = doc.get("projets") or []
    if isinstance(proj, list):
        for p in proj[:3]:
            pname = str(p)[:80]
            questions.append(f"Présentez le projet « {pname} » : rôle, stack, résultats.")
    techs = doc.get("technologies") or doc.get("langages") or []
    if isinstance(techs, list):
        for t in techs[:2]:
            questions.append(f"Quelle est votre expérience concrète avec {t} ?")
    certs = doc.get("certifications") or []
    if isinstance(certs, list) and certs:
        questions.append(f"Comment la certification « {certs[0]} » s'applique à ce poste ?")
    if len(questions) < max_q:
        questions.append("Comment évalueriez-vous et déployeriez-vous un modèle en production ?")
    return questions[:max_q]


def compute_answer_confidence(
    semantic_meta: Dict[str, Any],
    docs: List[Dict],
    question: str,
    route_confidence: float = 0.8,
) -> Tuple[str, float, List[str]]:
    """Retourne (label Haute/Moyenne/Faible, pct, raisons)."""
    reasons: List[str] = []
    pct = route_confidence * 100
    skills = extract_query_skills(question)

    if semantic_meta.get("no_exact_match"):
        pct = min(pct, 55)
        reasons.append("aucune correspondance exacte pour le critère obligatoire")
    elif docs and skills:
        top = docs[0]
        miss = top.get("_missing_skills") or missing_skills(top, skills)
        if not miss:
            pct = min(100, pct + 12)
            reasons.append("correspondance exacte sur les compétences")
        else:
            pct = max(40, pct - 15)
            reasons.append("compétences partiellement couvertes")

    if docs and docs[0].get("_retrieval_score"):
        reasons.append("similarité sémantique")
    if semantic_meta.get("is_recommendation") and docs and docs[0].get("_match_score"):
        reasons.append("classement multi-critères appliqué")
    if len(docs or []) >= 2:
        reasons.append("plusieurs profils comparables dans le contexte")

    pct = max(25, min(98, pct))
    if pct >= 80:
        label = "Haute"
    elif pct >= 55:
        label = "Moyenne"
    else:
        label = "Faible"
    return label, round(pct, 1), reasons[:5]


def build_search_process_lines(
    question: str,
    semantic_meta: Dict[str, Any],
    mode: str,
) -> List[str]:
    lines: List[str] = []
    from chatbot.response_format import detect_search_criteria

    for c in detect_search_criteria(question):
        lines.append(f"Thème détecté : {c}")
    skills = extract_query_skills(question)
    if skills:
        lines.append(f"Compétences ciblées : {', '.join(skills)}")
    if mode == "semantic":
        lines.append("Recherche sémantique (kNN)")
    if semantic_meta.get("no_exact_match"):
        lines.append("Aucun match exact → profils les plus proches")
    else:
        lines.append("Re-classement par adéquation poste (40% skills, 25% sémantique, …)")
    if semantic_meta.get("rejected"):
        lines.append(f"{len(semantic_meta['rejected'])} profil(s) écarté(s) (critères non remplis)")
    lines.append(f"Top {min(8, semantic_meta.get('returned', 5))} retenu(s) pour la réponse")
    return lines


def follow_up_suggestions(
    question: str,
    docs: List[Dict],
    mode: str,
) -> List[str]:
    sugs: List[str] = []
    if not docs:
        return [
            "Élargir la recherche (autre compétence ou catégorie)",
            "Afficher tous les CVs par score qualité",
        ]
    names = [d.get("nom") for d in docs[:3] if d.get("nom")]
    if len(names) >= 2 and mode != "comparison":
        sugs.append(f"Comparer {names[0]} avec {names[1]}")
    if names:
        sugs.append(f"Montrer les projets de {names[0]}")
        sugs.append(f"Pourquoi {names[0]} a été sélectionné ?")
        sugs.append(f"Questions d'entretien pour {names[0]}")
    if extract_query_skills(question):
        sugs.append("Trouver des candidats avec des compétences plus complètes")
    sugs.append("Résumé recruteur (forces / faiblesses / risque)")
    return sugs[:5]


def parse_recruiter_filters(question: str) -> Dict[str, Any]:
    """Filtres simples AND / NOT / expérience min / master."""
    q = question.lower()
    filt: Dict[str, Any] = {
        "min_years": None,
        "require_master": False,
        "must_skills": [],
        "must_not": [],
        "any_skills": [],
    }
    m = re.search(r"(?:au moins|at least|minimum)\s+(\d+)\s*(?:an|ans|year)", q)
    if m:
        filt["min_years"] = int(m.group(1))
    m2 = re.search(r"(\d+)\s*(?:years?\s+of\s+experience|ans?\s+d['']?exp[eé]rience)", q)
    if m2 and filt["min_years"] is None:
        filt["min_years"] = int(m2.group(1))
    if re.search(r"master|mastère|m2\b|bac\+5", q) and re.search(r"uniquement|only|seulement", q):
        filt["require_master"] = True
    if " but not " in q or " mais pas " in q:
        parts = re.split(r" but not | mais pas ", q, maxsplit=1)
        if len(parts) == 2:
            filt["must_not"] = extract_query_skills(parts[1])
    if " and " in q or " et " in q:
        filt["must_skills"] = extract_query_skills(question)
    if " or " in q or " ou " in q:
        filt["any_skills"] = extract_query_skills(question)
    return filt


def apply_recruiter_filters(docs: List[Dict], filt: Dict[str, Any]) -> List[Dict]:
    if not docs:
        return docs
    out: List[Dict] = []
    for d in docs:
        if filt.get("min_years") is not None:
            exp = d.get("annees_experience")
            if not isinstance(exp, (int, float)) or exp < filt["min_years"]:
                continue
        if filt.get("require_master"):
            hay = str(d.get("diplomes", "")).lower()
            if not re.search(r"master|mastère|m2|bac\+5|ingénieur|ingenieur", hay):
                continue
        must = filt.get("must_skills") or []
        if must and missing_skills(d, must):
            continue
        any_sk = filt.get("any_skills") or []
        if any_sk:
            h = " ".join(
                str(x) for x in (d.get("technologies") or []) + (d.get("langages") or [])
            ).lower()
            if not any(s.lower() in h for s in any_sk):
                continue
        skip = False
        hay_not = str(d.get("text", "")).lower()
        for key in ("technologies", "langages", "frameworks", "projets"):
            v = d.get(key) or []
            if isinstance(v, list):
                hay_not += " " + " ".join(str(x).lower() for x in v)
        for neg in filt.get("must_not") or []:
            if neg.lower() in hay_not:
                skip = True
                break
        if skip:
            continue
        out.append(d)
    return out if out else docs
