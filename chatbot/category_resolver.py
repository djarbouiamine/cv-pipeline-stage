"""Résolution des catégories CV : alias utilisateur → valeurs réelles dans Elasticsearch."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# Termes dans la question → libellés canoniques (peuvent ne pas exister tels quels dans l'index)
TOPIC_ALIASES: Dict[str, str] = {
    "ia": "Intelligence Artificielle",
    "ai": "Intelligence Artificielle",
    "intelligence artificielle": "Intelligence Artificielle",
    "artificial intelligence": "Intelligence Artificielle",
    "machine learning": "Intelligence Artificielle",
    "deep learning": "Intelligence Artificielle",
    "nlp": "Intelligence Artificielle",
    "computer vision": "Intelligence Artificielle",
    "llm": "Intelligence Artificielle",
    "mlops": "Intelligence Artificielle",
    "cybersécurité": "Cybersécurité",
    "cybersecurite": "Cybersécurité",
    "cyber": "Cybersécurité",
    "sécurité informatique": "Cybersécurité",
    "malware": "Cybersécurité",
    "data science": "Data Science",
    "data engineering": "Data Engineering",
    "data engineer": "Data Engineering",
    "big data": "Data Engineering",
    "réseau": "Réseaux",
    "reseaux": "Réseaux",
    "réseaux": "Réseaux",
    "network": "Réseaux",
    "networking": "Réseaux",
    "développement": "Développement Logiciel",
    "developpement": "Développement Logiciel",
    "dev": "Développement Logiciel",
    "software": "Développement Logiciel",
    "backend": "Développement Logiciel",
    "frontend": "Développement Logiciel",
    "full stack": "Développement Logiciel",
    "fullstack": "Développement Logiciel",
    "cloud": "Cloud & DevOps",
    "devops": "Cloud & DevOps",
    "kubernetes": "Cloud & DevOps",
    "docker": "Cloud & DevOps",
    "terraform": "Cloud & DevOps",
    "ci/cd": "Cloud & DevOps",
    "aws": "Cloud & DevOps",
    "azure": "Cloud & DevOps",
    "gcp": "Cloud & DevOps",
    "iot": "IoT & Embarqué",
    "embarqué": "IoT & Embarqué",
    "embarque": "IoT & Embarqué",
    "embedded": "IoT & Embarqué",
    "embedded systems": "Embedded Systems",
}

# Si la catégorie canonique n'existe pas dans l'index, recherche par mots-clés dans les CVs
TOPIC_KEYWORD_FALLBACK: Dict[str, List[str]] = {
    "Cloud & DevOps": [
        "devops", "docker", "kubernetes", "terraform", "ansible", "jenkins",
        "ci/cd", "aws", "azure", "gcp", "cloud", "gitlab", "prometheus",
    ],
    "Data Science": ["data science", "pandas", "numpy", "scikit", "analytics", "jupyter"],
    "IoT & Embarqué": ["iot", "embedded", "embarqu", "arduino", "fpga", "rtos", "microcontr"],
    "Réseaux": ["réseau", "reseau", "network", "cisco", "routing", "switch"],
}


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower().strip())


def get_known_categories(es) -> List[str]:
    try:
        res = es.search(
            index="cvs",
            size=0,
            aggs={"categories": {"terms": {"field": "categorie_principale", "size": 50}}},
        )
        return [
            b["key"]
            for b in res["aggregations"]["categories"]["buckets"]
            if b.get("key")
        ]
    except Exception:
        return []


def match_category_in_index(canonical: str, known: List[str]) -> Optional[str]:
    """Retourne le libellé exact présent dans l'index, ou None."""
    if not canonical or not known:
        return None
    cn = _normalize(canonical)
    for k in known:
        if _normalize(k) == cn:
            return k
    for k in known:
        kn = _normalize(k)
        if cn in kn or kn in cn:
            return k
    # Data Engineering ↔ Data Science proximité
    if "data" in cn:
        for k in known:
            if "data" in _normalize(k):
                return k
    if "embed" in cn or "iot" in cn:
        for k in known:
            kn = _normalize(k)
            if "embed" in kn or "iot" in kn:
                return k
    return None


def detect_topic_in_question(question: str) -> Optional[str]:
    q = question.lower()
    for term in sorted(TOPIC_ALIASES.keys(), key=len, reverse=True):
        if " " in term or "'" in term:
            if term in q:
                return TOPIC_ALIASES[term]
        elif re.search(r"\b" + re.escape(term) + r"\b", q):
            return TOPIC_ALIASES[term]
    return None


def resolve_category(question: str, es, hint: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Retourne (categorie_principale ES exacte, libellé thème pour l'UI).
    """
    known = get_known_categories(es)
    topic = detect_topic_in_question(question) or hint
    if not topic:
        return None, None

    matched = match_category_in_index(topic, known)
    if matched:
        return matched, topic
    return None, topic


def topic_fallback_keywords(topic: str) -> List[str]:
    return TOPIC_KEYWORD_FALLBACK.get(topic, [])


def build_topic_keyword_query(keywords: List[str]) -> dict:
    should = []
    for kw in keywords:
        for field in ("technologies", "langages", "frameworks", "outils_devops"):
            should.append({"term": {field: kw}})
            if kw != kw.title():
                should.append({"term": {field: kw.title()}})
        should.append({"match": {"text": {"query": kw, "operator": "and"}}})
    return {"bool": {"should": should, "minimum_should_match": 1}}
