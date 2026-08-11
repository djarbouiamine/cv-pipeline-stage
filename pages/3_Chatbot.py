# pages/3_Chatbot.py
"""Chatbot dual-mode – Conversation générale + RAG sur les CVs.

Modes :
- **general** : questions libres (Python, explications, conversation…) sans base CV
- **cv**      : recherche / comparaison / stats sur les CVs indexés

Ce module implémente un routage intelligent des questions CV :
- Questions sémantiques → recherche kNN vectorielle (mode par défaut,
  fonctionne pour N'IMPORTE QUELLE question, même sans mot-clé reconnu)
- Questions factuelles/agrégatives → requêtes ES triées/filtrées/agrégées
- Comparaisons nommées → recherche par noms de candidats

Le LLM reçoit un prompt strict anti-hallucination pour ne jamais inventer
de valeurs numériques ni déduire d'informations non explicitement fournies.

CORRECTIFS APPLIQUÉS (par rapport à la version précédente) :
  [FIX-A] retrieve_by_names() comparait le nom COMPLET du CV à un set de
          noms PARTIELS avec une égalité stricte -> ne matchait jamais ->
          liste de CVs vide -> le LLM halluciner une réponse plausible.
          Corrigé en test de sous-chaîne bidirectionnel.
  [FIX-B] Garde-fou : si docs est vide (hors mode stats), on n'appelle
          JAMAIS le LLM. On répond directement qu'aucun CV ne correspond.
  [FIX-C] En mode "comparison", les noms renvoyés par le classifieur LLM
          (souvent partiels, ex: "Rim") sont maintenant systématiquement
          re-résolus vers les noms complets réels de la base via
          find_candidates_in_question(), qui est déterministe (recherche
          ES + regex), au lieu d'être utilisés tels quels.
  [FIX-D] retrieve_top_n() accepte maintenant un paramètre `category` pour
          combiner "top N" + "catégorie" (ex: "Top 5 en cybersécurité"),
          ce qui ne fonctionnait pas avant (les deux filtres étaient
          mutuellement exclusifs).
  [FIX-E] Le mode "semantic" reste TOUJOURS le filet de sécurité universel :
          toute question qui ne matche aucun mode structuré (ranking,
          stats, filter, classement, comparison) retombe automatiquement
          en recherche vectorielle kNN, donc aucune question ne reste
          sans réponse exploitable.
  [FIX-F] Validation légère post-génération : si la réponse du LLM contient
          un score décimal qui n'existe dans AUCUN des CVs du contexte,
          on affiche un avertissement visible (au lieu de laisser passer
          silencieusement une hallucination).
"""

import os
import sys
import re
import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any

# Ajouter le répertoire parent au PATH pour pouvoir importer les modules du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import du client Elasticsearch et du modèle d'embedding partagé
from es_client import get_es_client
from cv_extractor import embedding_model, call_llm_structured  # modèle partagé (ou dummy)
from chatbot.intent import classify_intent
from chatbot.system_prompt import CV_RAG_SYSTEM_PROMPT
from chatbot.llm_client import (
    get_llm,
    build_chat_messages,
    GENERAL_SYSTEM_PROMPT,
)
from chatbot.response_format import (
    build_semantic_answer_instructions,
    COMPARISON_RESPONSE_INSTRUCTIONS,
    STATS_RESPONSE_INSTRUCTIONS,
    detect_unverifiable_criteria,
)
from chatbot.category_resolver import (
    detect_topic_in_question,
    resolve_category,
    get_known_categories,
    match_category_in_index,
    topic_fallback_keywords,
    build_topic_keyword_query,
)
from chatbot.mandatory_criteria import (
    detect_mandatory_requirements,
    split_docs_by_requirements,
    partial_match_score,
    requirements_summary,
)
from chatbot.es_aggregations import detect_aggregation_intent, run_special_aggregation, format_special_stats_block
from chatbot.recommendation_ranking import is_recommendation_question
from chatbot.final_ranking import (
    rerank_docs_by_job_relevance,
    split_exact_skill_matches,
    extract_query_skills,
)
from chatbot.recruiter_helpers import (
    apply_recruiter_filters,
    parse_recruiter_filters,
    build_evidence_lines,
    recruiter_dimension_bars,
    detect_cv_red_flags,
    suggest_interview_questions,
    compute_answer_confidence,
    build_search_process_lines,
    follow_up_suggestions,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

FIELDS = [
    "nom", "email", "telephone", "localisation", "categorie_principale",
    "score_qualite_globale", "score_qualite_globale_sur_10",
    "scores_categories", "scores_categories_ponderes", "scores_categories_ponderes_sur_10",
    "technologies", "langages", "frameworks", "outils_devops",
    "annees_experience", "diplomes", "certifications",
    "projets", "description_projets", "alertes_parcours",
    "text"
]

# ---------------------------------------------------------------------------
# Mots-clés pour détecter les questions agrégatives / factuelles (FR + EN)
# Ces mots-clés ne sont plus qu'un FALLBACK — la classification LLM passe
# en premier et couvre les formulations non prévues ici.
# ---------------------------------------------------------------------------

MOTS_CLES_RANKING = [
    "max", "maximum", "meilleur", "meilleure",
    "plus élevé", "plus elevé", "plus élevée", "plus haute", "plus haut",
    "top", "premier", "première",
    "best", "highest", "score le plus",
    "le plus d'", "le plus de",
    "le plus expérimenté", "le plus experimente",
    "most experienced", "most",
]

MOTS_CLES_RANKING_ASC = [
    "min", "minimum", "moins bon", "moins bonne", "moins",
    "pire", "dernier", "dernière",
    "plus bas", "plus basse", "plus faible",
    "lowest", "worst",
    "le moins d'", "le moins de",
    "le moins expérimenté", "le moins experimente",
    "least experienced", "least",
]

MOTS_CLES_STATS = [
    "combien", "nombre de", "moyenne", "total", "statistique",
    "stats", "how many", "average", "count",
]

MOTS_CLES_FILTRAGE = [
    "catégorie", "categorie", "catégories", "categories",
    "tous les cv", "tous les cvs", "liste des", "lister",
    "all cv", "all cvs",
]

MOTS_CLES_CLASSEMENT = [
    "classement", "classer", "trier", "rang", "ranking", "rank",
    "top 3", "top 5", "top 10", "top3", "top5", "top10",
]

MOTS_CLES_COMPARAISON = [
    "compare", "comparer", "comparaison", "versus", "vs",
    "différence entre", "difference entre",
    "entre",
]

SORT_FIELD_MAPPING = {
    "expérience":        ("annees_experience",        "années d'expérience"),
    "experience":        ("annees_experience",        "années d'expérience"),
    "expérimenté":       ("annees_experience",        "années d'expérience"),
    "experimente":       ("annees_experience",        "années d'expérience"),
    "senior":            ("annees_experience",        "années d'expérience"),
    "junior":            ("annees_experience",        "années d'expérience"),
    "années":            ("annees_experience",        "années d'expérience"),
    "annees":            ("annees_experience",        "années d'expérience"),
    "ans d'expérience":  ("annees_experience",        "années d'expérience"),
    "ans d'experience":  ("annees_experience",        "années d'expérience"),
    "most experienced":  ("annees_experience",        "années d'expérience"),
    "sur 10":            ("score_qualite_globale_sur_10", "score qualité /10"),
    "/10":               ("score_qualite_globale_sur_10", "score qualité /10"),
    "note sur 10":       ("score_qualite_globale_sur_10", "score qualité /10"),
    "score":             ("score_qualite_globale",   "score qualité globale"),
    "qualité":           ("score_qualite_globale",   "score qualité globale"),
    "qualite":           ("score_qualite_globale",   "score qualité globale"),
    "note":              ("score_qualite_globale",   "score qualité globale"),
}

# Alias historique (la résolution passe par chatbot.category_resolver)
CATEGORY_TERM_MAP: dict = {}

SYSTEM_PROMPT = CV_RAG_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helper : recherche de mots-clés par mot entier (word boundary)
# ---------------------------------------------------------------------------

def contains_keyword(text: str, keywords: list) -> bool:
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text):
            return True
    return False


# ---------------------------------------------------------------------------
# Retrieval : recherche vectorielle (kNN sémantique)
# ---------------------------------------------------------------------------

def embed_query(text: str) -> np.ndarray:
    vec = embedding_model.encode(text, normalize_embeddings=True)
    return np.array(vec)


def get_total_cv_count(es) -> int:
    try:
        return es.count(index="cvs")["count"]
    except Exception:
        return 0


def compute_dynamic_k(total: int, min_k: int = 8, max_k: int = 25) -> int:
    if total <= 0:
        return min_k
    if total <= max_k:
        return max(total, 1)
    return max(min_k, max_k)


def retrieve_top_k(es, query_vec: np.ndarray, k: int = 8) -> List[Dict]:
    body = {
        "size": k,
        "knn": {
            "field": "embedding_cv",
            "query_vector": query_vec.tolist(),
            "k": k,
            "num_candidates": max(50, k * 4),
        },
        "_source": FIELDS
    }
    res = es.search(index="cvs", body=body)
    hits = res["hits"]["hits"]
    docs = []
    for h in hits:
        d = dict(h["_source"])
        d["_retrieval_score"] = h.get("_score")
        docs.append(d)
    return docs


# ---------------------------------------------------------------------------
# Retrieval : requêtes ES par agrégation native et tri
# ---------------------------------------------------------------------------

def _attach_none_score(docs: List[Dict]) -> List[Dict]:
    for d in docs:
        d.setdefault("_retrieval_score", None)
    return docs


def retrieve_all_sorted(es, sort_field: str = "score_qualite_globale",
                        order: str = "desc") -> List[Dict]:
    total = es.count(index="cvs")["count"]
    res = es.search(
        index="cvs",
        size=total,
        sort=[{sort_field: {"order": order}}],
        _source=FIELDS,
    )
    docs = [dict(h["_source"]) for h in res["hits"]["hits"]]
    return _attach_none_score(docs)


def retrieve_extremum(es, sort_field: str = "score_qualite_globale",
                      order: str = "desc") -> Tuple[Optional[float], List[Dict], List[Dict]]:
    agg_type = "max" if order == "desc" else "min"

    agg_res = es.search(
        index="cvs",
        size=0,
        aggs={
            "extremum": {agg_type: {"field": sort_field}}
        }
    )
    value = agg_res["aggregations"]["extremum"]["value"]
    if value is None:
        return None, [], []

    docs_res = es.search(
        index="cvs",
        size=50,
        query={"term": {sort_field: value}},
        _source=FIELDS,
    )
    docs_extremum = _attach_none_score([dict(h["_source"]) for h in docs_res["hits"]["hits"]])

    all_docs = retrieve_all_sorted(es, sort_field=sort_field, order=order)

    return value, docs_extremum, all_docs


def retrieve_top_n(es, sort_field: str = "score_qualite_globale",
                   order: str = "desc", n: int = 5,
                   category: Optional[str] = None) -> List[Dict]:
    """[FIX-D] Accepte désormais un filtre `category` optionnel."""
    es_cat = None
    if category:
        es_cat, _ = resolve_category(category, es, hint=category)
        if not es_cat:
            es_cat = match_category_in_index(category, get_known_categories(es))
    query = {"term": {"categorie_principale": es_cat}} if es_cat else {"match_all": {}}
    if category and not es_cat:
        kw = topic_fallback_keywords(category)
        if kw:
            return retrieve_by_topic_keywords(es, kw)[:n]
    res = es.search(
        index="cvs",
        size=n,
        query=query,
        sort=[{sort_field: {"order": order}}],
        _source=FIELDS,
    )
    docs = [dict(h["_source"]) for h in res["hits"]["hits"]]
    return _attach_none_score(docs)


def retrieve_by_category(es, category: str) -> List[Dict]:
    known = get_known_categories(es)
    es_cat = match_category_in_index(category, known)
    topic = category
    if not es_cat:
        _, topic = resolve_category(category, es, hint=category)
        es_cat = match_category_in_index(topic or category, known)
    if not es_cat:
        keywords = topic_fallback_keywords(topic or category)
        if keywords:
            return retrieve_by_topic_keywords(es, keywords)
        return []

    total = es.count(index="cvs")["count"]
    res = es.search(
        index="cvs",
        size=total,
        query={"term": {"categorie_principale": es_cat}},
        sort=[{"score_qualite_globale": {"order": "desc"}}],
        _source=FIELDS,
    )
    docs = [dict(h["_source"]) for h in res["hits"]["hits"]]
    return _attach_none_score(docs)


def retrieve_by_topic_keywords(es, keywords: List[str]) -> List[Dict]:
    if not keywords:
        return []
    total = es.count(index="cvs")["count"]
    res = es.search(
        index="cvs",
        size=total,
        query=build_topic_keyword_query(keywords),
        sort=[{"score_qualite_globale": {"order": "desc"}}],
        _source=FIELDS,
    )
    docs = [dict(h["_source"]) for h in res["hits"]["hits"]]
    return _attach_none_score(docs)


def retrieve_stats(es) -> Dict[str, Any]:
    total = es.count(index="cvs")["count"]

    res = es.search(
        index="cvs",
        size=0,
        aggs={
            "by_categ": {
                "terms": {"field": "categorie_principale", "size": 20},
                "aggs": {
                    "avg_score": {"avg": {"field": "score_qualite_globale"}},
                    "max_score": {"max": {"field": "score_qualite_globale"}},
                    "min_score": {"min": {"field": "score_qualite_globale"}},
                }
            },
            "global_avg": {"avg": {"field": "score_qualite_globale"}},
            "global_max": {"max": {"field": "score_qualite_globale"}},
            "global_min": {"min": {"field": "score_qualite_globale"}},
            "avg_experience": {"avg": {"field": "annees_experience"}},
            "exp_histogram": {
                "histogram": {"field": "annees_experience", "interval": 1, "min_doc_count": 0}
            },
            "top_technologies": {"terms": {"field": "technologies", "size": 20}},
            "top_langages": {"terms": {"field": "langages", "size": 15}},
        }
    )
    aggs = res["aggregations"]
    aggs["total_cvs"] = total
    return aggs


def retrieve_by_names(es, names: List[str]) -> List[Dict]:
    """[FIX-A] Le bug original comparait le NOM COMPLET du CV (ex:
    "rim zayani") à un set de noms potentiellement PARTIELS (ex: {"rim"})
    avec une égalité stricte -> ne matchait jamais -> liste vide.
    Corrigé : on teste maintenant si un des noms cherchés est une
    sous-chaîne du nom complet (ou l'inverse), dans les deux sens."""
    should_clauses = [{"match": {"nom": name}} for name in names]
    total = es.count(index="cvs")["count"]
    res = es.search(
        index="cvs",
        size=total,
        query={
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        },
        _source=FIELDS,
    )
    hits = [dict(h["_source"]) for h in res["hits"]["hits"]]

    names_lower = [n.strip().lower() for n in names if n.strip()]
    filtered = []
    for h in hits:
        nom_complet = h.get("nom", "").strip().lower()
        if not nom_complet:
            continue
        if any(n in nom_complet or nom_complet in n for n in names_lower):
            filtered.append(h)

    return _attach_none_score(filtered)


def get_all_candidate_names(es) -> List[str]:
    total = es.count(index="cvs")["count"]
    res = es.search(
        index="cvs",
        size=total,
        _source=["nom"],
    )
    names = []
    for h in res["hits"]["hits"]:
        nom = h["_source"].get("nom", "").strip()
        if nom:
            names.append(nom)
    return names


def find_candidates_in_question(question: str, es) -> List[str]:
    q_lower = question.lower()
    all_names = get_all_candidate_names(es)
    matched_names = []

    mots_stop = {
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "en",
        "qui", "que", "sur", "par", "pour", "dans", "avec", "est", "son", "sa",
        "ses", "leur", "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta",
        "tes", "nos", "vos", "vs", "the", "and", "or", "for", "not", "are",
        "compare", "comparer", "comparaison", "score", "scores", "entre",
        "meilleur", "meilleure", "plus", "moins",
    }

    for full_name in all_names:
        full_name_lower = full_name.lower()

        if full_name_lower in q_lower:
            if full_name not in matched_names:
                matched_names.append(full_name)
            continue

        parts = full_name_lower.split()
        for part in parts:
            if len(part) < 3 or part in mots_stop:
                continue
            if re.search(r'\b' + re.escape(part) + r'\b', q_lower):
                if full_name not in matched_names:
                    matched_names.append(full_name)
                break

    return matched_names


# ---------------------------------------------------------------------------
# Filtre thématique déterministe post-kNN
# ---------------------------------------------------------------------------

def filter_docs_by_topic(question: str, docs: List[Dict]) -> Tuple[List[Dict], Optional[str]]:
    topic = detect_topic_in_question(question)
    if not topic:
        return docs, None

    filtres = []
    for doc in docs:
        cat = doc.get("categorie_principale") or ""
        if cat and match_category_in_index(topic, [cat]):
            filtres.append(doc)
            continue

        morceaux = []
        projets = doc.get("projets") or []
        if isinstance(projets, list):
            morceaux.extend(projets)
        desc_projets = doc.get("description_projets") or []
        if isinstance(desc_projets, list):
            morceaux.extend(desc_projets)
        morceaux.extend(doc.get("technologies") or [])
        morceaux.append((doc.get("text") or "")[:3000])
        haystack = " ".join(str(m) for m in morceaux).lower()

        keys = topic_fallback_keywords(topic) + [topic.lower()]
        if any(
            re.search(r"\b" + re.escape(t) + r"\b", haystack)
            for t in keys
            if len(t) > 2
        ):
            filtres.append(doc)

    if not filtres:
        return docs, None

    return filtres, topic


# ---------------------------------------------------------------------------
# Classifieur LLM — détermine uniquement le MODE de la question.
# [FIX-C] Les noms/catégories qu'il renvoie ne sont plus utilisés
# directement : ils servent seulement de HINT, la résolution finale passe
# toujours par find_candidates_in_question() / detect_category_in_question()
# qui sont déterministes et fiables.
# ---------------------------------------------------------------------------

CLASSIFIER_PROMPT_VERSION = "v2"


def _get_known_categories(es) -> List[str]:
    return get_known_categories(es)


def classify_question(
    question: str,
    known_categories: List[str],
    known_names: List[str],
) -> Dict:
    _VALID_MODES = {
        "semantic", "ranking_desc", "ranking_asc",
        "stats", "filter", "classement", "comparison",
    }
    _VALID_SORT_FIELDS = {
        "score_qualite_globale",
        "score_qualite_globale_sur_10",
        "annees_experience",
    }

    categories_str = ", ".join(known_categories) if known_categories else "aucune catégorie connue"
    names_str = ", ".join(known_names) if known_names else "aucun candidat connu"

    prompt = f"""Tu es un classifieur d'intention pour un moteur de recherche de CVs.

Catégories de candidats présentes dans la base : {categories_str}

Noms des candidats indexés : {names_str}

Ta mission : analyser la question et retourner UNIQUEMENT un JSON avec ces champs.

Valeurs possibles pour "mode" :
- "semantic"      : recherche sémantique générale — MODE PAR DÉFAUT si la
                     question est ambiguë, ouverte, ou ne rentre dans aucun
                     autre mode. Ce mode gère TOUTE question libre (profil,
                     compétence, projet, expérience décrite en langage
                     naturel, etc.), donc en cas de doute choisis toujours
                     "semantic" plutôt que d'inventer un autre mode.
- "ranking_desc"  : candidat(s) avec la valeur MAXIMALE d'un champ numérique
- "ranking_asc"   : candidat(s) avec la valeur MINIMALE d'un champ numérique
- "stats"         : statistiques, comptages, moyennes sur la base
- "filter"        : lister tous les CVs d'une catégorie ou de la base entière
- "classement"    : afficher un top N candidats triés (peut être combiné à
                     une catégorie, ex: "Top 5 en cybersécurité")
- "comparison"    : comparer nommément plusieurs candidats

Valeurs possibles pour "sort_field" :
- "score_qualite_globale"
- "score_qualite_globale_sur_10"
- "annees_experience"
- null

Exemples few-shot (retourner uniquement le JSON, jamais de texte autour) :

Q: "Trouve-moi un profil avec expérience en LLM et MLOps"
→ {{"mode": "semantic", "sort_field": null, "category": "Intelligence Artificielle", "candidate_names": [], "top_n": null, "confidence": 0.9}}

Q: "quelqu'un qui maîtrise Kubernetes"
→ {{"mode": "semantic", "sort_field": null, "category": "Cloud & DevOps", "candidate_names": [], "top_n": null, "confidence": 0.75}}

Q: "Quel candidat a le meilleur score ?"
→ {{"mode": "ranking_desc", "sort_field": "score_qualite_globale", "category": null, "candidate_names": [], "top_n": null, "confidence": 0.95}}

Q: "Le CV avec le score le plus bas"
→ {{"mode": "ranking_asc", "sort_field": "score_qualite_globale", "category": null, "candidate_names": [], "top_n": null, "confidence": 0.95}}

Q: "Quel candidat a le plus d'expérience ?"
→ {{"mode": "ranking_desc", "sort_field": "annees_experience", "category": null, "candidate_names": [], "top_n": null, "confidence": 0.9}}

Q: "Combien de CVs dans la catégorie IA ?"
→ {{"mode": "stats", "sort_field": null, "category": "Intelligence Artificielle", "candidate_names": [], "top_n": null, "confidence": 0.9}}

Q: "Top 3 des candidats"
→ {{"mode": "classement", "sort_field": "score_qualite_globale", "category": null, "candidate_names": [], "top_n": 3, "confidence": 0.95}}

Q: "Top 5 en cybersécurité"
→ {{"mode": "classement", "sort_field": "score_qualite_globale", "category": "Cybersécurité", "candidate_names": [], "top_n": 5, "confidence": 0.9}}

Q: "Compare Rim et Mohamed"
→ {{"mode": "comparison", "sort_field": null, "category": null, "candidate_names": ["Rim", "Mohamed"], "top_n": null, "confidence": 0.95}}

Q: "Liste tous les CVs en cybersécurité"
→ {{"mode": "filter", "sort_field": null, "category": "Cybersécurité", "candidate_names": [], "top_n": null, "confidence": 0.9}}

RÈGLES IMPORTANTES :
- Si la question est ambiguë ou ne correspond clairement à aucun mode structuré,
  réponds mode=semantic avec confidence <= 0.5. Ne force JAMAIS un mode
  structuré si tu n'es pas sûr — le mode semantic gère bien toute question
  libre.
- "candidate_names" et "category" sont des indices, pas la vérité finale :
  ils seront re-vérifiés par le code. Fais de ton mieux mais ne t'inquiète
  pas d'être imprécis sur ces deux champs.
- Retourne UNIQUEMENT le JSON, sans texte autour, sans balises ```.

Question à classifier : {question}"""

    raw = call_llm_structured(prompt, provider_order=["groq", "openrouter", "gemini"])

    mode = raw.get("mode", "semantic")
    if mode not in _VALID_MODES:
        mode = "semantic"

    sort_field = raw.get("sort_field") or None
    if sort_field and sort_field not in _VALID_SORT_FIELDS:
        sort_field = "score_qualite_globale"

    try:
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    if confidence < 0.5:
        mode = "semantic"

    try:
        top_n_raw = raw.get("top_n")
        top_n = int(top_n_raw) if top_n_raw is not None else None
        if top_n is not None:
            top_n = max(1, min(top_n, 50))
    except (TypeError, ValueError):
        top_n = None

    return {
        "mode": mode,
        "sort_field": sort_field,
        "category": raw.get("category") or None,
        "candidate_names": raw.get("candidate_names") or [],
        "top_n": top_n,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Routeur intelligent : détermine le type de recherche à effectuer
# ---------------------------------------------------------------------------

def detect_category_in_question(question: str) -> Optional[str]:
    """Thème canonique détecté (résolu vers l'index dans retrieve_by_category)."""
    return detect_topic_in_question(question)


def extract_top_n(question: str) -> int:
    match = re.search(r"top\s*(\d+)", question.lower())
    if match:
        return min(int(match.group(1)), 50)
    return 5


def enrich_semantic_retrieval(
    question: str,
    docs: List[Dict],
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Critères obligatoires, filtres recruteur, classement adéquation poste."""
    meta: Dict[str, Any] = {
        "no_exact_match": False,
        "mandatory_label": "",
        "unverifiable_criteria": [],
        "is_recommendation": False,
        "rejected": [],
        "query_skills": extract_query_skills(question),
        "returned": len(docs),
        "confidence_label": "",
        "confidence_pct": 0.0,
        "confidence_reasons": [],
        "search_process": [],
        "follow_ups": [],
    }
    filt = parse_recruiter_filters(question)
    docs = apply_recruiter_filters(docs, filt)

    unverifiable = detect_unverifiable_criteria(question)
    if unverifiable:
        meta["unverifiable_criteria"] = unverifiable
        meta["no_exact_match"] = True
        if not meta["mandatory_label"]:
            meta["mandatory_label"] = ", ".join(unverifiable)
        for d in docs:
            for key in (
                "_match_score", "_match_components", "_match_strengths",
                "_match_weaknesses", "_selection_reasons",
            ):
                d.pop(key, None)

    reqs = detect_mandatory_requirements(question)
    if reqs:
        matching, others = split_docs_by_requirements(docs, reqs)
        if not matching:
            meta["no_exact_match"] = True
            meta["mandatory_label"] = requirements_summary(reqs)
            for o in others[:12]:
                meta["rejected"].append({
                    "nom": o.get("nom", "?"),
                    "reason": f"Critère obligatoire non satisfait ({meta['mandatory_label']})",
                })
            others.sort(
                key=lambda d: (
                    -partial_match_score(d, reqs),
                    -(d.get("_retrieval_score") or 0),
                )
            )
            docs = others[:8]
        else:
            for o in others:
                meta["rejected"].append({
                    "nom": o.get("nom", "?"),
                    "reason": f"Critère obligatoire non satisfait ({requirements_summary(reqs)})",
                })
            docs = matching

    exact, others_skill, strict_skills = split_exact_skill_matches(question, docs)
    if strict_skills and meta["query_skills"]:
        if not exact:
            meta["no_exact_match"] = True
            if not meta["mandatory_label"]:
                meta["mandatory_label"] = "compétences : " + ", ".join(meta["query_skills"])
            for o in others_skill:
                meta["rejected"].append({
                    "nom": o.get("nom", "?"),
                    "reason": o.get("_reject_reason", "compétences incomplètes"),
                })
            docs = others_skill[:8]
        else:
            for o in others_skill:
                meta["rejected"].append({
                    "nom": o.get("nom", "?"),
                    "reason": o.get("_reject_reason", "compétences incomplètes"),
                })
            docs = exact

    docs = rerank_docs_by_job_relevance(question, docs)
    meta["returned"] = len(docs)

    if is_recommendation_question(question):
        meta["is_recommendation"] = True

    q_lower = question.lower()
    if any(k in q_lower for k in ("entretien", "interview", "question")) and len(docs) == 1:
        meta["interview_questions"] = suggest_interview_questions(docs[0])

    meta["follow_ups"] = follow_up_suggestions(question, docs, "semantic")
    return docs, meta


def detect_sort_field(question: str) -> Tuple[str, str]:
    q = question.lower()
    for keyword in sorted(SORT_FIELD_MAPPING.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(keyword) + r'\b', q):
            return SORT_FIELD_MAPPING[keyword]
    return ("score_qualite_globale", "score qualité globale")


def extract_names_from_question(question: str, es=None) -> List[str]:
    if es is not None:
        return find_candidates_in_question(question, es)

    q = question.strip()
    patterns = [
        r"compar\w*\s+(.+?)\s+(?:et|and|vs|versus)\s+(.+?)(?:\s*\?|\s*$|\s*,)",
        r"entre\s+(.+?)\s+(?:et|and)\s+(.+?)(?:\s*\?|\s*$|\s*,)",
        r"(.+?)\s+(?:vs|versus)\s+(.+?)(?:\s*\?|\s*$|\s*,)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            names = [match.group(1).strip(), match.group(2).strip()]
            cleaned = []
            for name in names:
                name = re.sub(r"^(le|la|les|un|une|des|du|de|l')\s+", "", name, flags=re.IGNORECASE)
                name = re.sub(r"\s+(cv|candidat|profil)s?\s*$", "", name, flags=re.IGNORECASE)
                if len(name) >= 2:
                    cleaned.append(name)
            if len(cleaned) >= 2:
                return cleaned
    return []


def is_comparison_question(question: str, es=None) -> bool:
    q = question.lower()
    comparison_keywords = [
        "compare", "comparer", "comparaison",
        "versus", "vs",
        "différence entre", "difference entre",
    ]
    has_keyword = contains_keyword(q, comparison_keywords)

    if es is not None:
        matched = find_candidates_in_question(question, es)
        if has_keyword:
            return len(matched) >= 2
        full_matches = [n for n in matched if n.lower() in q.lower()]
        return len(full_matches) >= 2

    if has_keyword:
        names = extract_names_from_question(question)
        return len(names) >= 2
    return False


def route_question(question: str, es) -> Tuple[Optional[List[Dict]], Optional[Dict], str, str]:
    """
    Routage intelligent de la question.

    Principe [FIX-E] : le mode "semantic" est le filet de sécurité
    universel. Toute question qui ne correspond clairement à aucun mode
    structuré (ranking / stats / filter / classement / comparison) — que
    ce soit via le classifieur LLM ou via les mots-clés de fallback —
    retombe automatiquement en recherche vectorielle kNN. Aucune question
    ne reste donc sans réponse exploitable, même si elle ne contient aucun
    mot-clé prévu.

    Retourne :
        - docs : liste de CVs (ou None si mode stats pur)
        - stats : dict de statistiques (ou None si pas mode stats)
        - mode : 'comparison' | 'ranking_desc' | 'ranking_asc' | 'stats' | 'filter' | 'classement' | 'semantic'
        - description : description humaine du mode de retrieval utilisé
    """
    q = question.lower()

    agg_intent = detect_aggregation_intent(question)
    if agg_intent:
        special = run_special_aggregation(es, agg_intent, question)
        stats = retrieve_stats(es)
        stats["special_aggregation"] = special
        docs = retrieve_all_sorted(es)
        block = format_special_stats_block(special)
        desc = f"📊 Agrégation ES ({agg_intent})"
        if block:
            stats["special_aggregation_text"] = block
        return docs, stats, "stats", desc

    # ── Classifieur LLM (détermine le mode + donne des hints) ───────────
    try:
        known_categories = _get_known_categories(es)
        known_names = get_all_candidate_names(es)
        clf = classify_question(question, known_categories, known_names)

        mode_llm       = clf["mode"]
        sort_field_llm = clf["sort_field"] or "score_qualite_globale"
        category_hint  = clf["category"]
        names_hint     = clf["candidate_names"]
        top_n_llm      = clf["top_n"] or 5
        conf           = clf["confidence"]

        # [FIX-C] Le hint de catégorie du LLM est revalidé contre le
        # détecteur déterministe. S'il ne matche pas une catégorie connue
        # au sens de CATEGORY_TERM_MAP, on retombe sur la détection texte.
        category_llm = detect_category_in_question(question) or category_hint
        es_cat, topic_label = resolve_category(question, es, hint=category_llm)

        _field_labels = {
            "annees_experience":             "années d'expérience",
            "score_qualite_globale_sur_10":  "score qualité /10",
            "score_qualite_globale":         "score qualité globale",
        }
        field_label_llm = _field_labels.get(sort_field_llm, "score qualité globale")
        conf_badge = f" (🧠 LLM {conf:.0%})"

        if mode_llm == "comparison":
            # [FIX-C] Ne JAMAIS faire confiance aux noms partiels du LLM
            # directement. On les utilise seulement comme point de départ
            # pour une recherche déterministe des noms complets réels.
            names = find_candidates_in_question(question, es)
            if not names and names_hint:
                # Le classifieur a proposé des noms partiels (ex: "Rim") :
                # on les recherche dans la base pour les résoudre en noms
                # complets, plutôt que de les utiliser tels quels.
                names = find_candidates_in_question(" ".join(names_hint), es)
            if len(names) >= 2:
                docs = retrieve_by_names(es, names)
                names_str = " et ".join(names)
                return docs, None, "comparison", f"🔄 Comparaison directe : {names_str}{conf_badge}"
            # Pas assez de noms résolus de façon fiable → on ne force pas
            # une comparaison bancale, on retombe en semantic ci-dessous.
            raise ValueError("comparison : noms non résolus de façon fiable")

        elif mode_llm == "classement":
            n = top_n_llm
            docs = retrieve_top_n(es, sort_field=sort_field_llm, order="desc", n=n, category=category_llm)
            label = f"🏆 Classement top {n} par {field_label_llm}"
            if category_llm:
                label += f" en {category_llm}"
            return docs, None, "classement", f"{label}{conf_badge}"

        elif mode_llm == "ranking_asc":
            value, _, all_docs = retrieve_extremum(es, sort_field=sort_field_llm, order="asc")
            desc = f"📊 Agrégation min sur {field_label_llm} → {value} ({len(all_docs)} CVs){conf_badge}"
            return all_docs, None, "ranking_asc", desc

        elif mode_llm == "ranking_desc":
            value, _, all_docs = retrieve_extremum(es, sort_field=sort_field_llm, order="desc")
            desc = f"📊 Agrégation max sur {field_label_llm} → {value} ({len(all_docs)} CVs){conf_badge}"
            return all_docs, None, "ranking_desc", desc

        elif mode_llm == "stats":
            stats = retrieve_stats(es)
            if category_llm:
                docs = retrieve_by_category(es, category_llm)
            else:
                docs = retrieve_all_sorted(es, sort_field=sort_field_llm)
            return docs, stats, "stats", f"📊 Statistiques agrégées depuis Elasticsearch{conf_badge}"

        elif mode_llm == "filter":
            if category_llm or topic_label:
                label = es_cat or topic_label or category_llm
                docs = retrieve_by_category(es, label or category_llm)
                return docs, None, "filter", f"🏷️ CVs filtrés : {label}{conf_badge}"
            docs = retrieve_all_sorted(es, sort_field=sort_field_llm)
            return docs, None, "filter", f"📋 Liste de tous les CVs{conf_badge}"

        elif mode_llm == "semantic":
            total = get_total_cv_count(es)
            k = compute_dynamic_k(total)
            q_vec = embed_query(question)
            docs = retrieve_top_k(es, q_vec, k=k)

            if es_cat:
                docs_filtered = [d for d in docs if d.get("categorie_principale") == es_cat]
                # CORRECTIF : filtre catégorie strict -> filtre souple.
                # Avant, dès qu'AU MOINS 1 CV matchait la catégorie
                # détectée, tout le reste du pool sémantique (souvent 13
                # CVs) était jeté -> un candidat unique et parfois peu
                # pertinent se retrouvait seul en lice par élimination,
                # pas par mérite (ex: "data analyst" -> 1 seul CV analysé
                # sur 13, alors que catégorie_principale n'est qu'UN des
                # 2-3 domaines identifiés par le LLM pour chaque CV -- un
                # bon candidat peut très bien avoir ce domaine en 2e ou 3e
                # position et se faire exclure à tort).
                # Seuil minimal avant d'appliquer le filtre strict : sinon
                # on garde le pool complet et on laisse le reclassement
                # (skills + sémantique générique) départager correctement.
                if len(docs_filtered) >= 3:
                    docs = docs_filtered
                    topic_filtre = es_cat
                else:
                    for d in docs:
                        d["_category_match"] = d.get("categorie_principale") == es_cat
                    topic_filtre = None
            elif topic_label and not es_cat:
                kw_docs = retrieve_by_topic_keywords(es, topic_fallback_keywords(topic_label))
                if kw_docs:
                    docs = kw_docs[: max(k, len(kw_docs))]
                    topic_filtre = topic_label
                else:
                    docs, topic_filtre = filter_docs_by_topic(question, docs)
            else:
                docs, topic_filtre = filter_docs_by_topic(question, docs)

            coverage = f" ({len(docs)}/{total} CVs analysés)" if total else ""
            if topic_filtre:
                desc = f"🔍 Recherche sémantique filtrée sur « {topic_filtre} »{coverage}{conf_badge}"
            else:
                desc = f"🔍 Recherche sémantique (kNN vectoriel){coverage}{conf_badge}"
            return docs, None, "semantic", desc

    except Exception:
        # LLM indisponible, JSON invalide, comparaison non résolue, etc.
        # → on retombe silencieusement sur le routage par mots-clés.
        pass

    # ── Routage par mots-clés (fallback si le LLM classifieur échoue) ───

    if is_comparison_question(question, es):
        names = extract_names_from_question(question, es)
        if len(names) >= 2:
            docs = retrieve_by_names(es, names)
            names_str = " et ".join(names)
            return docs, None, "comparison", f"🔄 Comparaison directe : {names_str}"

    if contains_keyword(q, MOTS_CLES_CLASSEMENT):
        n = extract_top_n(q)
        sort_field, field_label = detect_sort_field(question)
        category = detect_category_in_question(question)
        docs = retrieve_top_n(es, sort_field=sort_field, order="desc", n=n, category=category)
        label = f"🏆 Classement des top {n} CVs par {field_label}"
        if category:
            label += f" en {category}"
        return docs, None, "classement", label

    # Note ASC vérifié avant DESC : "le score le plus bas" contient
    # littéralement "score le plus", qui matche aussi MOTS_CLES_RANKING.
    if contains_keyword(q, MOTS_CLES_RANKING_ASC):
        sort_field, field_label = detect_sort_field(question)
        value, docs_extremum, all_docs = retrieve_extremum(es, sort_field=sort_field, order="asc")
        desc = f"📊 Agrégation min sur {field_label} → {value} ({len(all_docs)} CVs en contexte)"
        return all_docs, None, "ranking_asc", desc

    if contains_keyword(q, MOTS_CLES_RANKING):
        sort_field, field_label = detect_sort_field(question)
        value, docs_extremum, all_docs = retrieve_extremum(es, sort_field=sort_field, order="desc")
        desc = f"📊 Agrégation max sur {field_label} → {value} ({len(all_docs)} CVs en contexte)"
        return all_docs, None, "ranking_desc", desc

    if contains_keyword(q, MOTS_CLES_STATS):
        agg_intent = detect_aggregation_intent(question)
        stats = retrieve_stats(es)
        if agg_intent:
            special = run_special_aggregation(es, agg_intent, question)
            stats["special_aggregation"] = special
            stats["special_aggregation_text"] = format_special_stats_block(special)
        category = detect_category_in_question(question)
        if category:
            docs = retrieve_by_category(es, category)
        else:
            sort_field, _ = detect_sort_field(question)
            docs = retrieve_all_sorted(es, sort_field=sort_field)
        return docs, stats, "stats", "📊 Statistiques agrégées depuis Elasticsearch"

    if contains_keyword(q, MOTS_CLES_FILTRAGE):
        category = detect_category_in_question(question)
        if category:
            docs = retrieve_by_category(es, category)
            return docs, None, "filter", f"🏷️ CVs filtrés par catégorie : {category}"
        sort_field, _ = detect_sort_field(question)
        docs = retrieve_all_sorted(es, sort_field=sort_field)
        return docs, None, "filter", "📋 Liste de tous les CVs"

    # [FIX-E] Filet de sécurité universel : TOUTE question qui n'a matché
    # aucun mode structuré ci-dessus atterrit ici, en recherche sémantique.
    # C'est ce qui garantit qu'aucune question "hors mots-clés" ne reste
    # sans réponse exploitable.
    total = get_total_cv_count(es)
    k = compute_dynamic_k(total)
    q_vec = embed_query(question)
    docs = retrieve_top_k(es, q_vec, k=k)
    docs, topic_filtre = filter_docs_by_topic(question, docs)

    coverage = f" ({len(docs)}/{total} CVs analysés)" if total else ""
    if topic_filtre:
        desc = f"🔍 Recherche sémantique (kNN) filtrée sur le thème « {topic_filtre} »{coverage}"
    else:
        desc = f"🔍 Recherche sémantique (kNN vectoriel){coverage}"
    return docs, None, "semantic", desc


# ---------------------------------------------------------------------------
# Construction du prompt contextuel selon le mode de retrieval
# ---------------------------------------------------------------------------

def format_cv_entry(
    i: int,
    doc: Dict,
    include_text: bool = False,
    query_skills: Optional[List[str]] = None,
) -> str:
    nom = doc.get("nom", "Inconnu")
    score = doc.get("score_qualite_globale", "N/A")
    score_10 = doc.get("score_qualite_globale_sur_10", "N/A")
    categ = doc.get("categorie_principale", "N/A")
    exp = doc.get("annees_experience", "N/A")
    localisation = doc.get("localisation", "") or "information non disponible"
    techs = ", ".join(doc.get("technologies", [])[:8]) or "N/A"
    langs = ", ".join(doc.get("langages", [])[:5]) or "N/A"
    fworks = ", ".join(doc.get("frameworks", [])[:5]) or "N/A"
    diplomes = doc.get("diplomes", "N/A")
    if isinstance(diplomes, list):
        diplomes = " | ".join(diplomes[:3])

    projets = doc.get("projets") or []
    if isinstance(projets, list) and projets:
        projets_str = " | ".join(projets[:5])
    else:
        projets_str = "aucun projet renseigné"

    desc_projets = doc.get("description_projets") or doc.get("evaluation_projets") or []
    if isinstance(desc_projets, list) and desc_projets:
        desc_parts = []
        for item in desc_projets[:4]:
            if isinstance(item, dict):
                nom_p = item.get("nom", "")
                desc_p = item.get("description", "")
                desc_parts.append(f"{nom_p}: {desc_p}".strip(": "))
            else:
                desc_parts.append(str(item))
        desc_projets_str = " | ".join(desc_parts) if desc_parts else "non renseigné"
    else:
        desc_projets_str = "non renseigné"

    certifications = doc.get("certifications") or []
    if isinstance(certifications, list) and certifications:
        cert_str = ", ".join(str(c) for c in certifications[:4])
    else:
        cert_str = "aucune"

    match_line = ""
    if doc.get("_match_score") is not None:
        plus = ", ".join(doc.get("_match_strengths") or []) or "—"
        minus = ", ".join(doc.get("_match_weaknesses") or doc.get("_missing_skills") or []) or "—"
        comp = doc.get("_match_components") or {}
        comp_str = ", ".join(f"{k}={v}%" for k, v in comp.items()) if comp else "—"
        reasons = ", ".join(doc.get("_selection_reasons") or []) or "—"
        match_line = (
            f"  • Job Fit Score : {doc['_match_score']}% "
            f"(computed for this query — NOT stored in database)\n"
            f"  • Job Fit breakdown : {comp_str}\n"
            f"  • Why selected : {reasons}\n"
            f"  • Strengths : {plus}\n"
            f"  • Gaps / missing : {minus}\n"
        )

    evidence_lines = build_evidence_lines(doc, query_skills)
    evidence_block = ""
    if evidence_lines:
        evidence_block = "  • **Preuves (extrait structuré)** :\n    " + "\n    ".join(evidence_lines) + "\n"

    dim_bars = recruiter_dimension_bars(doc)
    dim_block = ""
    if dim_bars:
        dim_block = "  • **Scores recruteur par domaine** :\n" + "\n".join(dim_bars) + "\n"

    flags = detect_cv_red_flags(doc)
    flags_block = ""
    if flags:
        flags_block = "  • **Signaux d'attention CV** : " + "; ".join(flags) + "\n"

    entry = (
        f"**CV {i} – {nom}**\n"
        f"  • CV Quality Score : {score}/100 ({score_10}/10) "
        f"(stored in database — field score_qualite_globale)\n"
        f"{match_line}"
        f"{dim_block}"
        f"{evidence_block}"
        f"{flags_block}"
        f"  • Catégorie principale : {categ}\n"
        f"  • Localisation : {localisation}\n"
        f"  • Années d'expérience : {exp}\n"
        f"  • Technologies : {techs}\n"
        f"  • Langages : {langs}\n"
        f"  • Frameworks : {fworks}\n"
        f"  • Projets : {projets_str}\n"
        f"  • Descriptions projets : {desc_projets_str}\n"
        f"  • Certifications : {cert_str}\n"
        f"  • Diplômes : {diplomes}\n"
    )

    if include_text:
        text_preview = doc.get("text", "")[:700]
        if text_preview:
            entry += f"  • Extrait du CV : {text_preview}...\n"

    return entry


def build_prompt(question: str, docs: Optional[List[Dict]], stats: Optional[Dict],
                 mode: str, field_label: str = "score qualité globale",
                 total_cvs: Optional[int] = None,
                 semantic_meta: Optional[Dict[str, Any]] = None) -> str:
    semantic_meta = semantic_meta or {}

    if mode in ("ranking_desc", "ranking_asc", "classement"):
        order_label = "du plus haut au plus bas" if mode != "ranking_asc" else "du plus bas au plus haut"
        sources = "\n---\n".join(
            [format_cv_entry(i + 1, doc) for i, doc in enumerate(docs)]
        )
        return (
            f"**Question** : {question}\n\n"
            f"Voici TOUS les CVs triés par **{field_label}** ({order_label}). "
            f"Les valeurs sont des données EXACTES provenant de la base de données :\n\n"
            f"{sources}\n\n"
            f"Répondez à la question en vous basant UNIQUEMENT sur ces données triées. "
            f"Le critère de tri est **{field_label}** — utilisez ce champ pour identifier "
            f"le meilleur ou le pire candidat selon la question posée. "
            f"Les valeurs indiquées sont les valeurs réelles — ne les modifiez pas."
        )

    elif mode == "comparison":
        sources = "\n---\n".join(
            [format_cv_entry(i + 1, doc, include_text=True) for i, doc in enumerate(docs)]
        )
        return (
            f"**Question** : {question}\n\n"
            f"Voici les CVs des candidats demandés pour la comparaison :\n\n"
            f"{sources}\n\n"
            f"{COMPARISON_RESPONSE_INSTRUCTIONS.strip()}\n\n"
            f"Comparez ces candidats en vous basant UNIQUEMENT sur les données ci-dessus. "
            f"Les valeurs indiquées sont exactes — ne les inventez pas. "
            f"Si une information n'est pas présente (ex: localisation), dites-le "
            f"explicitement au lieu de la deviner."
        )

    elif mode == "stats":
        stats_text = ""
        if stats:
            total = stats.get("total_cvs", "?")
            avg = stats.get("global_avg", {}).get("value")
            mx = stats.get("global_max", {}).get("value")
            mn = stats.get("global_min", {}).get("value")
            avg_exp = stats.get("avg_experience", {}).get("value")

            stats_text = f"📊 **Statistiques globales** :\n"
            stats_text += f"  • Total de CVs indexés : {total}\n"
            if avg is not None:
                stats_text += f"  • Score qualité moyen : {avg:.1f}/100\n"
            if mx is not None:
                stats_text += f"  • Score qualité maximum : {mx:.1f}/100\n"
            if mn is not None:
                stats_text += f"  • Score qualité minimum : {mn:.1f}/100\n"
            if avg_exp is not None:
                stats_text += f"  • Années d'expérience moyennes : {avg_exp:.1f}\n"

            special_txt = stats.get("special_aggregation_text")
            if special_txt:
                stats_text += f"\n{special_txt}\n"

            buckets = stats.get("by_categ", {}).get("buckets", [])
            if buckets:
                stats_text += "\n📂 **Par catégorie** (nombre de CVs par categorie_principale) :\n"
                for b in buckets:
                    cat_name = b["key"]
                    cat_count = b["doc_count"]
                    cat_avg = b.get("avg_score", {}).get("value")
                    cat_max = b.get("max_score", {}).get("value")
                    avg_str = f", score moyen: {cat_avg:.1f}" if cat_avg else ""
                    max_str = f", score max: {cat_max:.1f}" if cat_max else ""
                    stats_text += f"  • {cat_name} : {cat_count} CV(s){avg_str}{max_str}\n"

            hist = stats.get("exp_histogram", {}).get("buckets", [])
            if hist:
                stats_text += "\n📈 **Histogramme expérience (années)** — comptage par tranche :\n"
                for b in hist[:12]:
                    stats_text += f"  • {b.get('key', '?')} ans : {b.get('doc_count', 0)} CV(s)\n"

            for agg_key, label in (("top_technologies", "Technologies"), ("top_langages", "Langages")):
                buckets_t = stats.get(agg_key, {}).get("buckets", [])
                if buckets_t:
                    stats_text += f"\n🔧 **Fréquence {label}** (champ keyword ES) :\n"
                    for b in buckets_t[:12]:
                        stats_text += f"  • {b['key']} : {b['doc_count']} CV(s)\n"

        cv_text = ""
        if docs:
            cv_text = "\n\n📋 **Détails des CVs** :\n" + "\n---\n".join(
                [format_cv_entry(i + 1, doc) for i, doc in enumerate(docs[:10])]
            )

        return (
            f"**Question** : {question}\n\n"
            f"{stats_text}{cv_text}\n\n"
            f"{STATS_RESPONSE_INSTRUCTIONS.strip()}\n\n"
            f"Répondez à la question en utilisant UNIQUEMENT les statistiques "
            f"et données ci-dessus. Tous les chiffres sont des valeurs EXACTES."
        )

    elif mode == "filter":
        sources = "\n---\n".join(
            [format_cv_entry(i + 1, doc) for i, doc in enumerate(docs)]
        )
        return (
            f"**Question** : {question}\n\n"
            f"Voici les CVs correspondant au filtre appliqué :\n\n"
            f"{sources}\n\n"
            f"Répondez à la question en vous basant UNIQUEMENT sur ces CVs. "
            f"Les scores indiqués sont des valeurs exactes de la base de données."
        )

    else:  # semantic
        q_skills = semantic_meta.get("query_skills") or extract_query_skills(question)
        answer_instructions = build_semantic_answer_instructions(
            question,
            len(docs),
            total_cvs=total_cvs,
            no_exact_match=semantic_meta.get("no_exact_match", False),
            mandatory_label=semantic_meta.get("mandatory_label") or None,
            unverifiable_criteria=semantic_meta.get("unverifiable_criteria") or None,
            is_recommendation=semantic_meta.get("is_recommendation", False),
            confidence_label=semantic_meta.get("confidence_label"),
            confidence_pct=semantic_meta.get("confidence_pct"),
            confidence_reasons=semantic_meta.get("confidence_reasons"),
            follow_ups=semantic_meta.get("follow_ups"),
            rejected=semantic_meta.get("rejected"),
            search_process=semantic_meta.get("search_process"),
            interview_questions=semantic_meta.get("interview_questions"),
        )
        if semantic_meta.get("unverifiable_criteria"):
            uv = ", ".join(semantic_meta["unverifiable_criteria"])
            prefix = (
                f"**IMPORTANT** : Le critère demandé ({uv}) n'est pas renseigné de façon "
                f"fiabilisable dans les CVs indexés (pas de champ employeur actuel / salaire). "
                f"Ne pas afficher de Job Fit % sur ce critère. "
                f"CVs ci-dessous = similarité générale uniquement.\n\n"
            )
        elif semantic_meta.get("no_exact_match"):
            prefix = (
                "**IMPORTANT** : AUCUNE correspondance exacte pour le critère obligatoire. "
                "CVs ci-dessous = profils les plus proches uniquement.\n\n"
            )
        else:
            prefix = ""
        if semantic_meta.get("no_exact_match") or semantic_meta.get("unverifiable_criteria"):
            sources = "\n---\n".join(
                [
                    format_cv_entry(i + 1, doc, include_text=True, query_skills=q_skills)
                    for i, doc in enumerate(docs[:5])
                ]
            )
        else:
            sources = "\n---\n".join(
                [
                    format_cv_entry(i + 1, doc, include_text=True, query_skills=q_skills)
                    for i, doc in enumerate(docs)
                ]
            )
        rejected_block = ""
        rej = semantic_meta.get("rejected") or []
        if rej:
            rejected_block = "\n\n**Profils écartés (hors recommandation exacte)** :\n" + "\n".join(
                f"- {r.get('nom', '?')} : {r.get('reason', '')}" for r in rej[:10]
            )
        coverage_note = ""
        if total_cvs is not None and docs is not None and len(docs) < total_cvs:
            coverage_note = (
                f"\n\nNote : {len(docs)} CV(s) sur {total_cvs} indexés transmis "
                f"(les plus pertinents pour cette recherche)."
            )
        return (
            f"**Question** : {question}\n\n"
            f"{prefix}"
            f"Voici les {len(docs)} CVs les plus pertinents pour cette question "
            f"(déjà filtrés par thème si applicable) :\n\n"
            f"{sources}"
            f"{rejected_block}\n\n"
            f"{answer_instructions}\n"
            f"Les scores indiqués sont des valeurs exactes — ne les inventez pas. "
            f"Si une information n'est pas dans les données, dites-le explicitement."
            f"{coverage_note}"
        )


# ---------------------------------------------------------------------------
# [FIX-F] Validation légère post-génération : détecte les scores décimaux
# mentionnés dans la réponse qui n'existent dans AUCUN CV du contexte.
# Ce n'est pas une preuve formelle d'hallucination (le LLM peut légitimement
# citer un pourcentage ou une moyenne calculée), mais c'est un signal utile
# à afficher pour la traçabilité.
# ---------------------------------------------------------------------------

def find_suspicious_numbers(answer: str, docs: List[Dict]) -> List[float]:
    """Detect decimal values in the answer that don't match any known CV field.

    Stored DB scores (quality, experience) and computed job-fit scores
    (_match_score, _match_components) are whitelisted so the checker
    does not flag legitimate Job Fit percentages.
    """
    valid_values = set()
    for d in docs:
        for field in ("score_qualite_globale", "score_qualite_globale_sur_10", "annees_experience"):
            v = d.get(field)
            if isinstance(v, (int, float)):
                valid_values.add(round(float(v), 1))
        match = d.get("_match_score")
        if isinstance(match, (int, float)):
            valid_values.add(round(float(match), 1))
        for comp_val in (d.get("_match_components") or {}).values():
            if isinstance(comp_val, (int, float)):
                valid_values.add(round(float(comp_val), 1))
        raw_sem = d.get("_retrieval_score")
        if isinstance(raw_sem, (int, float)):
            valid_values.add(round(float(raw_sem), 1))
            valid_values.add(round(float(raw_sem), 3))

    mentioned = {float(x) for x in re.findall(r'\b\d+\.\d+\b', answer)}
    # Tolère les échelles standard (ex: /5, /10, /100).
    tolerated = {5.0, 10.0, 100.0}
    suspicious = sorted(mentioned - valid_values - tolerated)
    return suspicious


# ---------------------------------------------------------------------------
# Traçabilité : construction du tableau de sources
# ---------------------------------------------------------------------------

def build_sources_dataframe(docs: List[Dict], mode: str) -> pd.DataFrame:
    rows = []
    for doc in docs:
        raw_score = doc.get("_retrieval_score")
        cv_quality = doc.get("score_qualite_globale")
        cv_quality_str = f"{cv_quality}/100" if isinstance(cv_quality, (int, float)) else "N/A"
        if doc.get("_match_score") is not None:
            job_fit_str = f"{doc['_match_score']}% (computed)"
        elif mode == "semantic" and raw_score is not None:
            job_fit_str = f"semantic {raw_score:.3f}"
        else:
            job_fit_str = "N/A"
        rows.append({
            "Nom": doc.get("nom", "?"),
            "Job Fit (this query)": job_fit_str,
            "CV Quality (database)": cv_quality_str,
            "Catégorie": doc.get("categorie_principale", "-"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(page_title="💬 Assistant", layout="wide")
st.title("💬 Assistant – Conversation & Recherche CV")
st.caption("Mode automatique : questions générales OU recherche dans vos CVs indexés")

provider = st.sidebar.selectbox(
    "Provider LLM",
    options=["groq", "openrouter", "mistral", "gemini"],
    index=0,
)

mode_override = st.sidebar.radio(
    "Mode de réponse",
    options=["auto", "general", "cv"],
    format_func=lambda x: {
        "auto": "🔄 Automatique (recommandé)",
        "general": "💬 Conversation générale",
        "cv": "📋 Base de CVs uniquement",
    }[x],
    index=0,
)

st.sidebar.divider()
if st.sidebar.button("🗑️ Effacer la conversation"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### 💡 Exemples")
st.sidebar.markdown("""
**Conversation générale** :
- *Qu'est-ce que Python ?*
- *Explique le machine learning*
- *Bonjour, comment ça va ?*

**Base de CVs (RAG)** :
- *Trouve un profil orienté IA*
- *Quel CV a le meilleur score ?*
- *Top 5 en cybersécurité*
- *Compare Rim et Mohamed*
""")

es = get_es_client()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("mode_badge"):
            st.caption(msg["mode_badge"])
        if msg.get("warning"):
            st.warning(msg["warning"])
        if msg.get("sources_df") is not None and not msg["sources_df"].empty:
            st.markdown(msg.get("trace_badge", ""))
            st.dataframe(msg["sources_df"], use_container_width=True, hide_index=True)
        if msg.get("stats"):
            with st.expander("📊 Voir les statistiques brutes", expanded=False):
                st.json(msg["stats"])

# Question pré-remplie depuis le Dashboard (action rapide)
_prefill = st.session_state.pop("chatbot_prefill", None)
question = st.chat_input("Posez votre question sur les CVs…")
if not question and _prefill:
    question = _prefill

if question and question.strip():
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "avatar": "👤",
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    llm = get_llm(provider)
    warning_msg = None
    stats = None
    sources_df = pd.DataFrame()
    trace_badge = ""

    # --- Détection d'intention (general vs cv) ---
    known_names = get_all_candidate_names(es)
    if mode_override == "general":
        user_intent = {"intent": "general", "confidence": 1.0, "source": "manual"}
    elif mode_override == "cv":
        user_intent = {"intent": "cv", "confidence": 1.0, "source": "manual"}
    else:
        with st.spinner("🧠 Détection du type de question…"):
            user_intent = classify_intent(question, known_names, call_llm_structured)

    intent_label = (
        f"{'💬 Conversation générale' if user_intent['intent'] == 'general' else '📋 Recherche CV'} "
        f"({user_intent['source']}, confiance {user_intent['confidence']:.0%})"
    )

    if user_intent["intent"] == "general":
        # --- Mode conversation générale (sans RAG) ---
        mode_description = intent_label
        with st.spinner("✍️ Génération de la réponse…"):
            messages = build_chat_messages(
                st.session_state.chat_history,
                GENERAL_SYSTEM_PROMPT,
            )
            try:
                answer = llm.generate_chat(messages)
            except Exception as e:
                answer = f"❌ Erreur lors de la génération : {e}"
        mode = "general"
    else:
        # --- Mode CV : routage RAG + génération ---
        with st.spinner("🔎 Analyse de la question et récupération des CVs…"):
            docs, stats, mode, mode_description = route_question(question, es)
        mode_description = f"{intent_label} → {mode_description}"

        if (not docs) and mode != "stats":
            q_vec = embed_query(question)
            docs = retrieve_top_k(es, q_vec, k=min(5, max(3, compute_dynamic_k(get_total_cv_count(es)))))
            mode = "semantic"
            mode_description = f"{intent_label} → 🔍 Aucun résultat structuré — profils les plus proches (sémantique)"
            semantic_meta = {"no_exact_match": True, "mandatory_label": "critères de la recherche"}
        else:
            semantic_meta = {}

        if mode != "stats" and docs:
            if mode == "semantic":
                docs, semantic_meta = enrich_semantic_retrieval(question, docs)
            conf_label, conf_pct, conf_reasons = compute_answer_confidence(
                semantic_meta, docs, question,
            )
            semantic_meta["confidence_label"] = conf_label
            semantic_meta["confidence_pct"] = conf_pct
            semantic_meta["confidence_reasons"] = conf_reasons
            semantic_meta["search_process"] = build_search_process_lines(
                question, semantic_meta, mode,
            )

        if (not docs) and mode != "stats":
            answer = (
                "Aucune correspondance exacte trouvée dans la base indexée pour cette recherche. "
                "Essayez de reformuler (autre catégorie, compétence, ou nom de candidat)."
            )
        else:
            _, field_label = detect_sort_field(question)
            total_cvs = get_total_cv_count(es) if mode == "semantic" else None
            prompt = build_prompt(
                question, docs, stats, mode,
                field_label=field_label,
                total_cvs=total_cvs,
                semantic_meta=semantic_meta,
            )
            with st.spinner("✍️ Génération de la réponse…"):
                try:
                    answer = llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
                except Exception as e:
                    answer = f"❌ Erreur lors de la génération : {e}"

            sources_df = build_sources_dataframe(docs, mode) if docs else pd.DataFrame()
            if docs:
                suspicious = find_suspicious_numbers(answer, docs)
                if suspicious:
                    warning_msg = (
                        f"⚠️ La réponse mentionne des valeurs ({', '.join(str(s) for s in suspicious)}) "
                        f"qui ne correspondent à aucun score connu (CV Quality en base, "
                        f"Job Fit calculé, ou expérience). Vérifiez cette réponse avant de vous y fier."
                    )

        n_analyses = len(docs) if docs else 0
        trace_badge = (
            f"🛡️ **Anti-hallucination** : {n_analyses} CV(s) analysé(s). "
            f"CV Quality = stored in database · Job Fit = computed for this query."
        ) if n_analyses else ""

    # --- Affichage ---
    with st.chat_message("assistant", avatar="🤖"):
        st.caption(mode_description)
        st.markdown(answer)
        if warning_msg:
            st.warning(warning_msg)
        if not sources_df.empty:
            st.markdown(trace_badge)
            st.dataframe(sources_df, use_container_width=True, hide_index=True)
        if stats:
            with st.expander("📊 Voir les statistiques brutes", expanded=False):
                st.json(stats)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "avatar": "🤖",
        "mode_badge": mode_description,
        "warning": warning_msg,
        "sources_df": sources_df,
        "trace_badge": trace_badge,
        "stats": stats,
    })