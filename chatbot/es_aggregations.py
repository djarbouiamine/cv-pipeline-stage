"""Requêtes agrégatives Elasticsearch pour le chatbot (stats enrichies)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

AI_PROJECT_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "nlp", "natural language", "computer vision", "llm", "neural",
    "tensorflow", "pytorch", "transformer", "classification", "réseau de neurones",
]

CLOUD_PROJECT_KEYWORDS = [
    "cloud", "devops", "docker", "kubernetes", "terraform", "aws", "azure",
    "gcp", "ci/cd", "infrastructure",
]


def terms_aggregation(es, field: str, size: int = 30) -> List[Tuple[str, int]]:
    try:
        res = es.search(
            index="cvs",
            size=0,
            aggs={"vals": {"terms": {"field": field, "size": size}}},
        )
        buckets = res["aggregations"]["vals"]["buckets"]
        return [(b["key"], b["doc_count"]) for b in buckets if b.get("key")]
    except Exception:
        return []


def count_docs_with_term(es, field: str, term: str) -> int:
    try:
        res = es.search(
            index="cvs",
            size=0,
            query={"term": {field: term}},
        )
        total = res["hits"]["total"]
        if isinstance(total, dict):
            return int(total.get("value", 0))
        return int(total)
    except Exception:
        return 0


def count_docs_matching_text(es, keywords: List[str]) -> int:
    should = []
    for kw in keywords:
        should.append({"match_phrase": {"projets": kw}})
        should.append({"match_phrase": {"text": kw}})
    if not should:
        return 0
    try:
        res = es.search(
            index="cvs",
            size=0,
            query={"bool": {"should": should, "minimum_should_match": 1}},
        )
        total = res["hits"]["total"]
        if isinstance(total, dict):
            return int(total.get("value", 0))
        return int(total)
    except Exception:
        return 0


def university_terms(es, size: int = 20) -> List[Tuple[str, int]]:
    """Approximation via termes fréquents dans diplomes (text field → match agrégé sur keyword si absent)."""
    try:
        res = es.search(
            index="cvs",
            size=50,
            _source=["diplomes"],
        )
        counts: Dict[str, int] = {}
        for hit in res["hits"]["hits"]:
            dipl = hit["_source"].get("diplomes") or ""
            if isinstance(dipl, list):
                dipl = " ".join(str(d) for d in dipl)
            dipl = str(dipl).strip()
            if not dipl:
                continue
            # Regrouper par ligne / diplôme court
            key = dipl[:80].strip()
            counts[key] = counts.get(key, 0) + 1
        return sorted(counts.items(), key=lambda x: -x[1])[:size]
    except Exception:
        return []


def detect_aggregation_intent(question: str) -> Optional[str]:
    q = question.lower()
    if re.search(r"(framework|frameworks).*(plus|common|souvent|fréquent|frequent|most)", q):
        return "top_frameworks"
    if re.search(r"(technolog|technolog).*(plus|common|souvent|fréquent|frequent|most|often|appears)", q):
        return "top_technologies"
    if re.search(r"(langage|language|programming language).*(plus|common|souvent|fréquent|frequent|most)", q):
        return "top_languages"
    if re.search(r"how many.*\b(ai|embedded|embed|iot)\b", q):
        return "count_category_topic"
    if re.search(r"combien.*\b(ia|ai|embarqu|embed|iot)\b", q):
        return "count_category_topic"
    if "python" in q and re.search(r"combien|how many|nombre|count", q):
        return "count_python"
    if re.search(r"projet(s)?\s+(ia|ai|intelligence artificielle|machine learning)", q):
        return "count_ai_projects"
    if re.search(r"projet(s)?\s+(cloud|devops)", q):
        return "count_cloud_projects"
    if re.search(r"universit|école|ecole|school", q) and re.search(r"plus|common|souvent|often|most", q):
        return "top_universities"
    if re.search(r"\b(tensorflow|pytorch|docker|kubernetes|react|flutter)\b", q) and re.search(
        r"combien|how many|nombre|count", q
    ):
        return "count_named_skill"
    return None


def count_by_category(es, category_term: str) -> int:
    try:
        res = es.search(
            index="cvs",
            size=0,
            query={"term": {"categorie_principale": category_term}},
        )
        total = res["hits"]["total"]
        if isinstance(total, dict):
            return int(total.get("value", 0))
        return int(total)
    except Exception:
        return 0


def detect_category_count_topic(question: str) -> Optional[str]:
    q = question.lower()
    if re.search(r"\b(ai|ia|intelligence artificielle|machine learning)\b", q):
        return "Intelligence Artificielle"
    if re.search(r"\b(embedded|embed|iot|embarqu)\b", q):
        return "Embedded Systems"
    return None


def extract_named_skill_for_count(question: str) -> Optional[str]:
    q = question.lower()
    for name in (
        "tensorflow", "pytorch", "docker", "kubernetes", "react", "flutter",
        "java", "javascript", "sql", "linux", "angular", "terraform",
        "mongodb", "postgresql", "postgres", "python", "c++",
    ):
        if re.search(r"\b" + re.escape(name) + r"\b", q):
            if name == "sql":
                return "SQL"
            if name == "c++":
                return "C++"
            if name in ("postgresql", "postgres"):
                return "PostgreSQL"
            return name.capitalize()
    return None


def run_special_aggregation(es, intent: str, question: str) -> Dict[str, Any]:
    if intent == "top_frameworks":
        rows = terms_aggregation(es, "frameworks", 25)
        return {"type": intent, "rows": rows, "field": "frameworks"}
    if intent == "top_technologies":
        rows = terms_aggregation(es, "technologies", 25)
        return {"type": intent, "rows": rows, "field": "technologies"}
    if intent == "top_languages":
        rows = terms_aggregation(es, "langages", 25)
        return {"type": intent, "rows": rows, "field": "langages"}
    if intent == "count_python":
        n = count_docs_with_term(es, "langages", "Python")
        return {"type": intent, "count": n, "label": "Python (langages)"}
    if intent == "count_ai_projects":
        n = count_docs_matching_text(es, AI_PROJECT_KEYWORDS)
        return {"type": intent, "count": n, "label": "CVs avec projets IA / ML"}
    if intent == "count_cloud_projects":
        n = count_docs_matching_text(es, CLOUD_PROJECT_KEYWORDS)
        return {"type": intent, "count": n, "label": "CVs avec projets cloud / DevOps"}
    if intent == "top_universities":
        rows = university_terms(es)
        return {"type": intent, "rows": rows, "field": "diplomes"}
    if intent == "count_category_topic":
        cat = detect_category_count_topic(question)
        if cat:
            n = count_by_category(es, cat)
            if n == 0:
                # Fallback keyword count for categories not exactly matching index
                from chatbot.category_resolver import topic_fallback_keywords, build_topic_keyword_query
                kw = topic_fallback_keywords(cat) or [cat.lower()]
                try:
                    res = es.search(index="cvs", size=0, query=build_topic_keyword_query(kw))
                    total = res["hits"]["total"]
                    n = int(total.get("value", 0) if isinstance(total, dict) else total)
                except Exception:
                    n = 0
            return {"type": intent, "count": n, "label": f"CVs catégorie / thème « {cat} »"}
    if intent == "count_named_skill":
        skill = extract_named_skill_for_count(question)
        if skill:
            n_tech = count_docs_with_term(es, "technologies", skill)
            n_lang = count_docs_with_term(es, "langages", skill)
            n_fw = count_docs_with_term(es, "frameworks", skill)
            return {
                "type": intent,
                "skill": skill,
                "count_technologies": n_tech,
                "count_langages": n_lang,
                "count_frameworks": n_fw,
                "count_unique_docs_max": max(n_tech, n_lang, n_fw),
            }
    return {"type": "unknown"}


def format_special_stats_block(data: Dict[str, Any]) -> str:
    t = data.get("type")
    if t in ("top_frameworks", "top_technologies", "top_languages"):
        rows = data.get("rows") or []
        if not rows:
            return "Aucune donnée agrégée disponible pour ce champ."
        lines = [f"📊 **Agrégation `{data.get('field')}`** (nombre de CVs par valeur) :\n"]
        for name, cnt in rows[:15]:
            lines.append(f"  • {name} : {cnt} CV(s)")
        return "\n".join(lines)
    if t == "count_python":
        return f"📊 **Développeurs / profils avec Python (champ langages)** : **{data.get('count', 0)}** CV(s)."
    if t in ("count_ai_projects", "count_cloud_projects"):
        return f"📊 **{data.get('label')}** : **{data.get('count', 0)}** CV(s) (recherche texte projets + extrait CV)."
    if t == "top_universities":
        rows = data.get("rows") or []
        if not rows:
            return "Aucun diplôme agrégé."
        lines = ["📊 **Formations / diplômes les plus fréquents** (regroupement approximatif) :\n"]
        for name, cnt in rows[:10]:
            lines.append(f"  • ({cnt}×) {name}")
        return "\n".join(lines)
    if t == "count_category_topic":
        return f"📊 **{data.get('label')}** : **{data.get('count', 0)}** CV(s)."
    if t == "count_named_skill":
        skill = data.get("skill", "?")
        return (
            f"📊 **Occurrences de {skill}** (comptage par champ keyword) :\n"
            f"  • technologies : {data.get('count_technologies', 0)} CV(s)\n"
            f"  • langages : {data.get('count_langages', 0)} CV(s)\n"
            f"  • frameworks : {data.get('count_frameworks', 0)} CV(s)"
        )
    return ""
