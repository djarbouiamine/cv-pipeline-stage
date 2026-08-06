"""Détection d'intention : question générale vs question sur les CVs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Patterns généralistes → conversation (priorité la plus haute)
GENERAL_PATTERNS = [
    r"^what is\b",
    r"^what's\b",
    r"^what are\b",
    r"^explain\b",
    r"^define\b",
    r"^how does\b",
    r"^how do\b",
    r"^can you explain\b",
    r"^tell me about\b(?!.*\b(candidat|candidate|cv|profil|profile)\b)",
    r"^hello\b",
    r"^hi\b",
    r"^hey\b",
    r"^how are you\b",
    r"^good morning\b",
    r"^good afternoon\b",
    r"qu'est-ce que\b",
    r"qu est ce que\b",
    r"c'est quoi\b",
    r"c est quoi\b",
    r"^explique\b(?!.*\b(candidat|candidate|cv|profil)\b)",
    r"^explique-moi\b",
    r"^explique moi\b",
    r"^bonjour\b",
    r"^salut\b",
    r"^merci\b",
    r"comment ça va\b",
    r"comment ca va\b",
]

# Patterns recrutement / base CV → mode CV
CV_PATTERNS = [
    r"\bfind\b.*\b(candidat|candidate|engineer|developer|profil|profile|specialist|analyst|intern)\b",
    r"\bfind\b.*\b(ai|iot|nlp|backend|frontend|devops|embedded|cybersecurity|blockchain)\b",
    r"\bwho knows\b",
    r"\bwho has\b.*\b(certification|certified|degree|project|experience)\b",
    r"\bwho should i hire\b",
    r"\brecommend\b.*\b(candidat|candidate|developer|engineer|profil|profile|someone|intern)\b",
    r"\brecommend\b",
    r"\bhow many\b.*\b(candidat|candidate|cv|cvs|profil|profile|embedded|ai)\b",
    r"\bhow many candidates\b",
    r"\bwhich candidate\b",
    r"\bwhich profile\b",
    r"\bwhich technology\b",
    r"\bwhich programming language\b",
    r"\bwhich framework\b",
    r"\bshow the top\b",
    r"\bshow the highest\b",
    r"\brank\b.*\b(candidat|candidate|by|ai)\b",
    r"\bcompare\b.*\b(candidat|candidate|and|et|vs|versus|top)\b",
    r"\bcompare the\b",
    r"\bcompare\b.*\band\b",
    r"\bwho is stronger\b",
    r"\bwho has more\b",
    r"\bwhich candidate has\b",
    r"\bdistribution of categories\b",
    r"\baverage quality score\b",
    r"\baverage years of experience\b",
    r"\bmost common\b",
    r"\bappears most often\b",
    r"\btrouve\b.*\b(candidat|profil|ingénieur|ingenieur|développeur|developpeur)\b",
    r"\btrouve\b.*\b(ia|ai|iot|nlp|backend|frontend|devops|embarqu)\b",
    r"\bqui maîtrise\b",
    r"\bqui maitrise\b",
    r"\bqui connaît\b",
    r"\bqui connait\b",
    r"\brecommande\b",
    r"\bcombien de\b.*\b(candidat|cv|cvs|profil)\b",
    r"\bquelle technologie\b",
    r"\bquel langage\b",
    r"\bquel framework\b",
    r"\bclassement\b",
    r"\btop \d+\b.*\b(candidat|candidate|cv|ai|ia)\b",
    r"\bcandidates with\b",
    r"\bcandidates who\b",
    r"\bfind candidates\b",
    r"\bbest suited for\b",
    r"\bbest ai candidate\b",
    r"\bwithout java\b",
    r"\bbut without\b",
    r"\branked by experience\b",
    r"\bstudied at\b",
    r"\bengineering degrees\b",
    r"\bspeaks japanese\b",
    r"\byears of experience\b",
    r"\bsenior devops\b",
    r"\bphd\b",
    r"\bgo developer\b",
    r"\brust developer\b",
    r"\bblockchain developer\b",
    r"\baws-certified\b",
    r"\bazure expert\b",
    r"\bgcp expert\b",
]

# Mots-clés forts → mode CV (sans termes tech ambigus type kubernetes/ml)
CV_KEYWORDS = [
    "cv", "cvs", "candidat", "candidats", "candidature", "recrutement",
    "profil", "profils", "embauche", "hire",
    "score", "scores", "classement", "classer", "top 3", "top 5", "top 10",
    "compare", "comparer", "comparaison", "versus", " vs ",
    "combien de cv", "nombre de cv", "liste des", "tous les cv", "all cvs",
    "meilleur cv", "meilleure cv", "meilleur candidat", "meilleure candidate",
    "best candidate", "strongest overall",
    "data science", "devops", "cybersécurité", "cybersecurite",
    "technologies", "compétence", "competence", "skills search",
    "diplome", "diplôme", "certification",
    "how many", "who knows", "recommend", "recommande",
    "rank candidates", "show the top", "find an", "find a", "find the",
    "find candidates", "which candidates",
]

# Formulations typiques → mode général
GENERAL_KEYWORDS = [
    "merci", "au revoir", "python programming", "javascript", "html", "css",
    "math", "maths", "histoire", "géographie", "geographie", "recette",
    "météo", "meteo", "help me write", "translate", "traduis",
    "what is rag", "explain rag", "what is elasticsearch",
]


def _contains_keyword(text: str, keywords: List[str]) -> bool:
    q = text.lower()
    for kw in keywords:
        if " " in kw or "'" in kw:
            if kw in q:
                return True
        elif re.search(r"\b" + re.escape(kw) + r"\b", q):
            return True
    return False


def _matches_patterns(text: str, patterns: List[str]) -> bool:
    q = text.lower().strip()
    for pat in patterns:
        if re.search(pat, q, re.IGNORECASE):
            return True
    return False


def _mentions_known_name(question: str, known_names: List[str]) -> bool:
    q_lower = question.lower()
    for name in known_names:
        name_lower = name.strip().lower()
        if not name_lower:
            continue
        if name_lower in q_lower:
            return True
        for part in name_lower.split():
            if len(part) >= 3 and re.search(r"\b" + re.escape(part) + r"\b", q_lower):
                return True
    return False


def classify_intent_heuristic(
    question: str,
    known_names: Optional[List[str]] = None,
) -> Optional[str]:
    """Retourne 'cv', 'general', ou None si ambigu."""
    known_names = known_names or []
    q = question.strip()

    # 1. Questions de définition / conversation → général (priorité sur les mots tech)
    if _matches_patterns(q, GENERAL_PATTERNS):
        return "general"
    if _contains_keyword(q, GENERAL_KEYWORDS):
        return "general"

    # 2. Noms de candidats connus → CV
    if _mentions_known_name(q, known_names):
        return "cv"

    # 3. Formulations recrutement explicites → CV
    if _matches_patterns(q, CV_PATTERNS):
        return "cv"
    if _contains_keyword(q, CV_KEYWORDS):
        return "cv"

    return None


def classify_intent(
    question: str,
    known_names: Optional[List[str]] = None,
    call_llm_structured=None,
) -> Dict[str, Any]:
    """
    Détermine si la question concerne la base de CVs ou une conversation générale.

    Retourne {"intent": "cv"|"general", "confidence": float, "source": str}
    """
    known_names = known_names or []
    heuristic = classify_intent_heuristic(question, known_names)
    if heuristic == "cv":
        return {"intent": "cv", "confidence": 0.9, "source": "heuristic"}
    if heuristic == "general":
        return {"intent": "general", "confidence": 0.85, "source": "heuristic"}

    if call_llm_structured is None:
        # Formulation courte avec verbe de recherche → plutôt CV que général
        q = question.lower()
        if re.search(r"\b(find|show|list|rank|compare|recommend|who)\b", q):
            return {"intent": "cv", "confidence": 0.55, "source": "default_cv"}
        return {"intent": "general", "confidence": 0.5, "source": "default"}

    names_preview = ", ".join(known_names[:20]) if known_names else "aucun"
    prompt = f"""Tu es un classifieur d'intention pour un assistant à double mode.

Mode "cv" : la question porte sur les candidats, CVs, scores, compétences des
personnes indexées, comparaisons entre candidats, statistiques de la base CV,
recrutement, profils ingénieurs dans la base.

Mode "general" : question générale (culture, programmation, définitions,
conversation, aide rédaction, traduction, maths, etc.) SANS lien avec une
base de CVs ou des candidats nommés.

Noms de candidats connus dans la base (extrait) : {names_preview}

Exemples :
Q: "Quel candidat a le meilleur score ?" → {{"intent": "cv", "confidence": 0.95}}
Q: "Compare Rim et Mohamed" → {{"intent": "cv", "confidence": 0.95}}
Q: "Find the best AI candidate" → {{"intent": "cv", "confidence": 0.95}}
Q: "Who knows Python?" → {{"intent": "cv", "confidence": 0.95}}
Q: "Combien de CVs en IA ?" → {{"intent": "cv", "confidence": 0.95}}
Q: "Qu'est-ce que Python ?" → {{"intent": "general", "confidence": 0.95}}
Q: "What is Python?" → {{"intent": "general", "confidence": 0.95}}
Q: "Explain machine learning." → {{"intent": "general", "confidence": 0.9}}
Q: "What is Kubernetes?" → {{"intent": "general", "confidence": 0.9}}
Q: "Bonjour, comment ça va ?" → {{"intent": "general", "confidence": 0.95}}

Règle : "What is X" / "Explain X" = general. "Find X candidate" / "Who knows X" = cv.

Retourne UNIQUEMENT un JSON : {{"intent": "cv"|"general", "confidence": 0.0-1.0}}

Question : {question}"""

    try:
        raw = call_llm_structured(prompt, provider_order=["groq", "openrouter", "gemini"])
        intent = raw.get("intent", "general")
        if intent not in ("cv", "general"):
            intent = "general"
        confidence = float(raw.get("confidence", 0.6))
        confidence = max(0.0, min(1.0, confidence))
        return {"intent": intent, "confidence": confidence, "source": "llm"}
    except Exception:
        q = question.lower()
        if re.search(r"\b(find|show|list|rank|compare|recommend|who knows|how many)\b", q):
            return {"intent": "cv", "confidence": 0.5, "source": "fallback_cv"}
        return {"intent": "general", "confidence": 0.5, "source": "fallback"}
