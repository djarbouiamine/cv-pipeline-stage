# pages/3_Chatbot.py
"""Chatbot RAG – Recherche intelligente dans les CVs.

Ce module implémente un routage intelligent des questions :
- Questions sémantiques → recherche kNN vectorielle
- Questions factuelles/agrégatives → requêtes ES triées/filtrées/agrégées
- Comparaisons nommées → recherche par noms de candidats

Le LLM reçoit un prompt strict anti-hallucination pour ne jamais inventer
de valeurs numériques ni déduire d'informations non explicitement fournies.
"""

import os
import sys
import re
import streamlit as st
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

# Ajouter le répertoire parent au PATH pour pouvoir importer les modules du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import du client Elasticsearch et du modèle d'embedding partagé
from es_client import get_es_client
from cv_extractor import embedding_model  # modèle partagé (ou dummy)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Champs à récupérer depuis Elasticsearch pour chaque CV
FIELDS = [
    "nom", "email", "localisation", "categorie_principale",
    "score_qualite_globale", "score_qualite_globale_sur_10",
    "technologies", "langages", "frameworks",
    "annees_experience", "diplomes", "certifications",
    "text"
]

# ---------------------------------------------------------------------------
# Mots-clés pour détecter les questions agrégatives / factuelles (FR + EN)
# ---------------------------------------------------------------------------

# Ranking descendant : questions demandant le meilleur / le plus élevé
MOTS_CLES_RANKING = [
    "max", "maximum", "meilleur", "meilleure",
    "plus élevé", "plus elevé", "plus élevée", "plus haute", "plus haut",
    "top", "premier", "première",
    "best", "highest", "score le plus",
    "le plus d'", "le plus de",
    "le plus expérimenté", "le plus experimente",
    "most experienced", "most",
]

# Ranking ascendant : questions demandant le pire / le plus bas
MOTS_CLES_RANKING_ASC = [
    "min", "minimum", "moins bon", "moins bonne", "moins",
    "pire", "dernier", "dernière",
    "plus bas", "plus basse", "plus faible",
    "lowest", "worst",
    "le moins d'", "le moins de",
    "le moins expérimenté", "le moins experimente",
    "least experienced", "least",
]

# Statistiques / comptage
MOTS_CLES_STATS = [
    "combien", "nombre de", "moyenne", "total", "statistique",
    "stats", "how many", "average", "count",
]

# Filtrage par catégorie
MOTS_CLES_FILTRAGE = [
    "catégorie", "categorie", "catégories", "categories",
    "tous les cv", "tous les cvs", "liste des", "lister",
    "all cv", "all cvs",
]

# Classement / Top N
MOTS_CLES_CLASSEMENT = [
    "classement", "classer", "trier", "rang", "ranking", "rank",
    "top 3", "top 5", "top 10", "top3", "top5", "top10",
]

# Comparaison nommée ("compare X et Y")
MOTS_CLES_COMPARAISON = [
    "compare", "comparer", "comparaison", "versus", "vs",
    "différence entre", "difference entre",
    "entre", "et",
]

# Mapping mots-clés → champ ES pour le tri dynamique (2e étape de détection)
# Clé = mot-clé détecté dans la question, Valeur = (champ_es, label_humain)
# Note : trié du plus long au plus court pour prioriser les correspondances précises
SORT_FIELD_MAPPING = {
    # Expérience
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
    # Score qualité (sur 10)
    "sur 10":            ("score_qualite_globale_sur_10", "score qualité /10"),
    "/10":               ("score_qualite_globale_sur_10", "score qualité /10"),
    "note sur 10":       ("score_qualite_globale_sur_10", "score qualité /10"),
    # Score qualité (défaut) — doit rester en dernier car c'est le fallback
    "score":             ("score_qualite_globale",   "score qualité globale"),
    "qualité":           ("score_qualite_globale",   "score qualité globale"),
    "qualite":           ("score_qualite_globale",   "score qualité globale"),
    "note":              ("score_qualite_globale",   "score qualité globale"),
}

# Mapping de catégories connues pour la détection automatique
CATEGORIES_CONNUES = [
    "intelligence artificielle", "ia", "ai", "artificial intelligence",
    "cybersécurité", "cybersecurite", "cyber", "sécurité informatique",
    "data", "data science", "data engineering", "big data",
    "réseau", "reseaux", "réseau", "network", "networking",
    "développement", "developpement", "dev", "software",
    "cloud", "devops",
    "iot", "embarqué", "embarque", "embedded",
    "technologies de l'information", "ict",
]

# Prompt système anti-hallucination — renforcé (inclut la règle anti-déduction externe)
SYSTEM_PROMPT = """Vous êtes un assistant expert en analyse de CVs d'ingénieurs.

RÈGLES STRICTES :
1. Répondez UNIQUEMENT à partir des données fournies ci-dessous.
2. Si vous comparez des scores ou classez des candidats, basez-vous EXACTEMENT
   sur les chiffres fournis. Ne déduisez JAMAIS un classement par intuition ou
   à partir du contenu textuel.
3. Citez toujours les valeurs numériques exactes (scores, années d'expérience)
   entre parenthèses.
4. Si l'information demandée n'est pas dans le contexte fourni, dites-le
   clairement : "Cette information n'est pas disponible dans les données fournies."
5. Structurez votre réponse de manière claire avec des puces ou un tableau si
   plusieurs candidats sont comparés.
6. Ne devinez JAMAIS. Ne supposez JAMAIS. Utilisez uniquement les données
   explicitement fournies.
7. Ne déduisez JAMAIS une information non présente explicitement dans les données
   fournies. Par exemple, ne déduisez pas une ville à partir d'un nom de domaine
   d'adresse email. Si le champ localisation est vide ou absent, dites explicitement
   que cette information n'est pas disponible pour ce candidat, sans deviner à
   partir d'autres champs comme l'email.
8. Pour les questions de type "quel candidat a le meilleur X" ou "le plus élevé",
   basez-vous EXCLUSIVEMENT sur le champ numérique indiqué dans le contexte
   (score qualité globale, années d'expérience, etc.), pas sur votre interprétation
   du contenu textuel du CV.

RÈGLE ANTI-DÉDUCTION EXTERNE (critique) : Ne répondez JAMAIS à une question sur
la localisation, l'employeur, ou tout attribut personnel d'un candidat en vous
basant sur une déduction externe à partir du nom d'une école, d'une entreprise,
ou d'un domaine d'email mentionné dans le CV (par exemple : ne déduisez jamais
une ville à partir du nom d'une école, même si vous connaissez sa localisation
réelle par ailleurs). Utilisez UNIQUEMENT la valeur du champ correspondant tel
qu'extrait directement du CV (ex: le champ 'localisation'). Si ce champ est vide,
absent, ou null dans les données fournies, répondez explicitement 'information
non disponible pour ce candidat' — même si vous pensez connaître la réponse grâce
à des connaissances générales externes au CV. Cette règle s'applique à TOUTE
information non explicitement écrite dans les champs du CV fourni."""


# ---------------------------------------------------------------------------
# Helper : récupération d'un provider LLM et wrapper minimal
# ---------------------------------------------------------------------------

def get_llm(provider: str):
    """Retourne un objet avec une méthode `generate(prompt)`.
    Pour chaque provider, on charge la clé API depuis le .env et on utilise
    le SDK approprié. Si la clé est manquante, on renvoie un dummy qui
    renvoie simplement le prompt en guise de réponse (utile pour le dev).
    """
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not api_key:
        st.warning(f"⚠️ Clé API manquante pour le provider '{provider}'. Le chatbot utilisera une réponse factice.")
        class DummyLLM:
            def generate(self, prompt: str) -> str:
                return f"[Réponse factice] {prompt[:200]}..."
        return DummyLLM()

    # Import du SDK seulement si la clé est disponible – on évite les imports inutiles
    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # modèle Groq recommandé
    elif provider == "openrouter":
        from openrouter import OpenRouter
        client = OpenRouter(api_key=api_key)
        model_name = "openrouter/mistralai/mixtral-8x7b-instruct"
    elif provider == "mistral":
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        model_name = "mistral-large-latest"
    elif provider == "gemini":
        from google.generativeai import configure as gemini_configure
        gemini_configure(api_key=api_key)
        import google.generativeai as genai
        model_name = "gemini-1.5-pro"
    else:
        raise ValueError(f"Provider inconnu : {provider}")

    class RealLLM:
        def __init__(self, client, model_name: str):
            self.client = client
            self.model_name = model_name

        def generate(self, prompt: str) -> str:
            if provider == "groq":
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                return resp.choices[0].message.content
            elif provider == "openrouter":
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp.choices[0].message.content
            elif provider == "mistral":
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp.choices[0].message.content
            elif provider == "gemini":
                model = genai.GenerativeModel(self.model_name)
                full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
                resp = model.generate_content(full_prompt)
                return resp.text
            else:
                return "[Impossible de générer]"

    return RealLLM(client, model_name)


# ---------------------------------------------------------------------------
# Retrieval : recherche vectorielle (kNN sémantique)
# ---------------------------------------------------------------------------

def embed_query(text: str) -> np.ndarray:
    """Encode la requête avec le même modèle que les CVs.
    Retourne un vecteur numpy normalisé (norme L2 = 1) pour la comparaison.
    """
    vec = embedding_model.encode(text, normalize_embeddings=True)
    return np.array(vec)


def retrieve_top_k(es, query_vec: np.ndarray, k: int = 8) -> List[Dict]:
    """
    Retourne les *k* CVs les plus similaires à `query_vec`.
    Utilise la requête native `knn` d'Elasticsearch.
    k=8 pour couvrir l'intégralité d'un petit index, num_candidates=50.
    """
    body = {
        "size": k,
        "knn": {
            "field": "embedding_cv",
            "query_vector": query_vec.tolist(),
            "k": k,
            "num_candidates": 50,
        },
        "_source": FIELDS
    }
    res = es.search(index="cvs", body=body)
    hits = res["hits"]["hits"]
    return [h["_source"] for h in hits]


# ---------------------------------------------------------------------------
# Retrieval : requêtes ES par agrégation native et tri
# ---------------------------------------------------------------------------

def retrieve_all_sorted(es, sort_field: str = "score_qualite_globale",
                        order: str = "desc") -> List[Dict]:
    """Récupère TOUS les CVs triés par le champ spécifié.
    Utilise es.count() pour obtenir le size dynamique → ne rate jamais un doc.
    """
    total = es.count(index="cvs")["count"]
    res = es.search(
        index="cvs",
        size=total,   # ← size = nombre total de docs dans l'index
        sort=[{sort_field: {"order": order}}],
        _source=FIELDS,
    )
    return [h["_source"] for h in res["hits"]["hits"]]


def retrieve_extremum(es, sort_field: str = "score_qualite_globale",
                      order: str = "desc") -> Tuple[Optional[float], List[Dict], List[Dict]]:
    """Trouve la valeur extrême (max ou min) sur un champ donné, puis récupère :
    1. Le(s) document(s) ayant exactement cette valeur extrême
    2. TOUS les documents triés par ce champ (pour que le LLM ait le contexte complet)

    Retourne : (valeur_extremum, docs_extremum, tous_les_docs_triés)

    Utilise une agrégation ES native (max/min) → indépendant du size.
    Puis récupère TOUS les docs triés via retrieve_all_sorted() pour que le LLM
    puisse vérifier/comparer l'ensemble des candidats.
    """
    agg_type = "max" if order == "desc" else "min"

    # Étape 1 : agrégation native pour trouver la valeur extrême exacte
    agg_res = es.search(
        index="cvs",
        size=0,   # ← pas de docs, juste l'agrégation
        aggs={
            "extremum": {agg_type: {"field": sort_field}}
        }
    )
    value = agg_res["aggregations"]["extremum"]["value"]
    if value is None:
        return None, [], []

    # Étape 2 : récupérer le(s) doc(s) qui ont exactement la valeur extrême
    docs_res = es.search(
        index="cvs",
        size=50,   # ← suffisant pour les docs avec la valeur exacte
        query={"term": {sort_field: value}},
        _source=FIELDS,
    )
    docs_extremum = [h["_source"] for h in docs_res["hits"]["hits"]]

    # Étape 3 : récupérer TOUS les docs triés par ce champ (contexte complet)
    # Cela permet au LLM de voir tous les candidats et de confirmer le classement
    all_docs = retrieve_all_sorted(es, sort_field=sort_field, order=order)

    return value, docs_extremum, all_docs


def retrieve_top_n(es, sort_field: str = "score_qualite_globale",
                   order: str = "desc", n: int = 5) -> List[Dict]:
    """Récupère les N meilleurs/pires CVs par tri ES classique.
    À utiliser UNIQUEMENT pour les requêtes 'top N' explicites où la
    troncature est voulue par l'utilisateur (ex: "top 3 candidats").
    size=N est ici voulu, ce n'est pas un bug.
    """
    res = es.search(
        index="cvs",
        size=n,   # ← size = N demandé par l'utilisateur
        sort=[{sort_field: {"order": order}}],
        _source=FIELDS,
    )
    return [h["_source"] for h in res["hits"]["hits"]]


def retrieve_by_category(es, category: str) -> List[Dict]:
    """Récupère TOUS les CVs filtrés par categorie_principale.
    Utilise es.count() pour un size dynamique → ne rate jamais un doc.
    """
    total = es.count(index="cvs")["count"]
    res = es.search(
        index="cvs",
        size=total,   # ← size = nombre total de docs
        query={"term": {"categorie_principale": category}},
        sort=[{"score_qualite_globale": {"order": "desc"}}],
        _source=FIELDS,
    )
    return [h["_source"] for h in res["hits"]["hits"]]


def retrieve_stats(es) -> Dict[str, Any]:
    """Récupère les statistiques agrégées via agrégations ES natives :
    - total, score moyen/max/min, expérience moyenne
    - ventilation par catégorie (terms sur categorie_principale)

    size=0 car on ne veut que les agrégations, pas les documents.
    """
    total = es.count(index="cvs")["count"]

    res = es.search(
        index="cvs",
        size=0,   # ← pas de docs, uniquement les agrégations
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
        }
    )
    aggs = res["aggregations"]
    aggs["total_cvs"] = total
    return aggs


def retrieve_by_names(es, names: List[str]) -> List[Dict]:
    """Récupère les CVs correspondant à une liste de noms de candidats.
    Utilise une query bool/should avec match sur le champ 'nom' pour chaque
    nom demandé (opérateur OR entre les noms) — tolérant aux fautes de frappe.
    """
    should_clauses = [{"match": {"nom": name}} for name in names]
    total = es.count(index="cvs")["count"]
    res = es.search(
        index="cvs",
        size=total,   # ← on veut tous les résultats matchant
        query={
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        },
        _source=FIELDS,
    )
    hits = [h["_source"] for h in res["hits"]["hits"]]

    # ── FIX (homonymie) : la requête `match` fait un OR entre les MOTS du nom
    # (utile pour tolérer les fautes de frappe côté utilisateur), mais ça peut
    # ramener un candidat "en trop" qui partage juste un prénom ou un nom de
    # famille avec un des candidats demandés (ex: "Rim ZAYANI" matche aussi
    # "Yesmine ZAYANI" à cause du mot "zayani" commun). `names` contient déjà
    # les noms EXACTS tels qu'indexés dans ES (résolus en amont par
    # find_candidates_in_question), donc on filtre ici pour ne garder QUE les
    # candidats explicitement demandés — pas de faux positif par homonymie.
    names_lower = {n.strip().lower() for n in names}
    hits = [h for h in hits if h.get("nom", "").strip().lower() in names_lower]

    return hits


def get_all_candidate_names(es) -> List[str]:
    """Récupère la liste de TOUS les noms de candidats indexés dans ES.
    Utilisé par la détection de comparaison pour vérifier si des noms
    connus apparaissent dans la question de l'utilisateur.
    """
    total = es.count(index="cvs")["count"]
    res = es.search(
        index="cvs",
        size=total,
        _source=["nom"],   # ← on ne récupère que le champ 'nom'
    )
    names = []
    for h in res["hits"]["hits"]:
        nom = h["_source"].get("nom", "").strip()
        if nom:
            names.append(nom)
    return names


def find_candidates_in_question(question: str, es) -> List[str]:
    """Détecte quels noms de candidats connus (indexés dans ES) apparaissent
    dans le texte de la question. Recherche insensible à la casse et tolérante :
    - Correspondance exacte du nom complet (ex: "Mohamed Amine")
    - Correspondance partielle par partie du nom (prénom ou nom de famille)
      si la partie fait au moins 3 caractères (pour éviter les faux positifs
      avec des mots courts comme "Ali" qui pourraient être des mots courants,
      mais garder les prénoms courts significatifs comme "Rim")

    Retourne la liste des noms complets (tels qu'indexés dans ES) détectés.
    """
    q_lower = question.lower()
    all_names = get_all_candidate_names(es)
    matched_names = []

    # Mots très courants en français/anglais à ne pas considérer comme des noms
    # (pour éviter les faux positifs avec des parties de noms très courtes)
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

        # 1) Correspondance exacte du nom complet dans la question
        if full_name_lower in q_lower:
            if full_name not in matched_names:
                matched_names.append(full_name)
            continue

        # 2) Correspondance par partie du nom (prénom ou nom de famille)
        #    On vérifie chaque partie individuellement
        parts = full_name_lower.split()
        for part in parts:
            # Ignorer les parties trop courtes (< 3 chars) ou les mots-stop
            if len(part) < 3 or part in mots_stop:
                continue
            # Vérifier que la partie apparaît comme un mot entier dans la question
            # (pas comme sous-chaîne d'un autre mot) via boundary regex
            if re.search(r'\b' + re.escape(part) + r'\b', q_lower):
                if full_name not in matched_names:
                    matched_names.append(full_name)
                break  # une partie suffit pour matcher ce candidat

    return matched_names


# ---------------------------------------------------------------------------
# Routeur intelligent : détermine le type de recherche à effectuer
# ---------------------------------------------------------------------------

def detect_category_in_question(question: str) -> Optional[str]:
    """Essaie de détecter une catégorie ES dans la question de l'utilisateur.
    Retourne la catégorie normalisée ou None.
    """
    q = question.lower()

    # Mapping de termes courants vers les valeurs ES réelles
    category_mapping = {
        "ia": "Intelligence Artificielle",
        "ai": "Intelligence Artificielle",
        "intelligence artificielle": "Intelligence Artificielle",
        "artificial intelligence": "Intelligence Artificielle",
        "cybersécurité": "Cybersécurité",
        "cybersecurite": "Cybersécurité",
        "cyber": "Cybersécurité",
        "sécurité informatique": "Cybersécurité",
        "data": "Data Science",
        "data science": "Data Science",
        "data engineering": "Data Science",
        "big data": "Data Science",
        "réseau": "Réseaux",
        "reseaux": "Réseaux",
        "réseaux": "Réseaux",
        "network": "Réseaux",
        "networking": "Réseaux",
        "développement": "Développement Logiciel",
        "developpement": "Développement Logiciel",
        "dev": "Développement Logiciel",
        "software": "Développement Logiciel",
        "cloud": "Cloud & DevOps",
        "devops": "Cloud & DevOps",
        "iot": "IoT & Embarqué",
        "embarqué": "IoT & Embarqué",
        "embarque": "IoT & Embarqué",
        "embedded": "IoT & Embarqué",
        "ict": "Technologies de l'Information",
        "technologies de l'information": "Technologies de l'Information",
    }

    # On cherche du plus long au plus court pour prioriser les correspondances précises
    for term in sorted(category_mapping.keys(), key=len, reverse=True):
        if term in q:
            return category_mapping[term]

    return None


def extract_top_n(question: str) -> int:
    """Extrait le nombre N d'un 'top N' dans la question. Par défaut 5."""
    match = re.search(r"top\s*(\d+)", question.lower())
    if match:
        return min(int(match.group(1)), 50)  # max 50 pour éviter les abus
    return 5


def detect_sort_field(question: str) -> Tuple[str, str]:
    """2e étape de détection : après avoir détecté un besoin de tri,
    inspecte les mots-clés de la question pour choisir le bon champ ES.

    Retourne :
        - sort_field : nom du champ ES (ex. 'annees_experience')
        - field_label : label humain (ex. "années d'expérience")

    La recherche se fait du plus long au plus court pour prioriser les
    correspondances précises (ex: "ans d'expérience" avant "années").
    """
    q = question.lower()

    # Chercher du plus long au plus court pour prioriser les correspondances précises
    for keyword in sorted(SORT_FIELD_MAPPING.keys(), key=len, reverse=True):
        if keyword in q:
            return SORT_FIELD_MAPPING[keyword]

    # Fallback : score qualité globale
    return ("score_qualite_globale", "score qualité globale")


def extract_names_from_question(question: str, es=None) -> List[str]:
    """Détecte les noms de candidats connus (indexés dans ES) mentionnés dans
    la question. Remplace l'ancienne extraction par regex qui échouait pour
    les formulations non standard.

    Si `es` est fourni, utilise find_candidates_in_question() pour une
    détection robuste basée sur les noms réels de l'index ES.
    Sinon, utilise l'ancien fallback regex (gardé pour compatibilité).

    Retourne une liste de noms complets détectés (peut être vide).
    """
    # --- Méthode principale : détection basée sur les noms ES réels ---
    if es is not None:
        return find_candidates_in_question(question, es)

    # --- Fallback regex (gardé pour compatibilité si es n'est pas dispo) ---
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
    """Détecte si la question est une comparaison entre candidats nommés.

    Nouvelle logique (BUG A corrigé) :
    - Récupère tous les noms de candidats connus dans l'index ES
    - Vérifie si AU MOINS 2 de ces noms apparaissent dans la question
    - Fonctionne quelle que soit la formulation ("compare X et Y",
      "X vs Y", "Compare les scores de X et Y", etc.)
    """
    if es is not None:
        matched = find_candidates_in_question(question, es)
        return len(matched) >= 2

    # Fallback sans ES (ancien comportement)
    q = question.lower()
    has_keyword = any(m in q for m in ["compare", "comparer", "comparaison",
                                        "versus", " vs ", "différence entre",
                                        "difference entre"])
    if has_keyword:
        names = extract_names_from_question(question)
        return len(names) >= 2
    return False


def route_question(question: str, es) -> Tuple[Optional[List[Dict]], Optional[Dict], str, str]:
    """
    Routage intelligent de la question.

    Étape 1 : détecter le TYPE de question (comparaison, classement, tri, stats, filtre, sémantique)
    Étape 2 : si c'est un tri, détecter le CHAMP à trier via detect_sort_field()
              (expérience, score, etc.) — JAMAIS un champ fixe

    Le champ de tri est TOUJOURS déterminé dynamiquement par detect_sort_field()
    en analysant les mots-clés de la question. Par exemple :
    - "score le plus élevé" → sort_field = score_qualite_globale
    - "le plus expérimenté" → sort_field = annees_experience

    Retourne :
        - docs : liste de CVs (ou None si mode stats pur)
        - stats : dict de statistiques (ou None si pas mode stats)
        - mode : 'comparison' | 'ranking_desc' | 'ranking_asc' | 'stats' | 'filter' | 'classement' | 'semantic'
        - description : description humaine du mode de retrieval utilisé
    """
    q = question.lower()

    # --- Mode : Comparaison nommée ("compare X et Y", "X vs Y", etc.) ---
    # Priorité maximale : si l'utilisateur nomme ≥2 candidats connus, on les
    # cherche directement dans ES, quelle que soit la formulation de la question.
    # BUG A corrigé : détection basée sur les noms réels indexés dans ES,
    # et non plus sur des patterns regex fragiles.
    if is_comparison_question(question, es):
        names = extract_names_from_question(question, es)
        docs = retrieve_by_names(es, names)
        names_str = " et ".join(names)
        return docs, None, "comparison", f"🔄 Comparaison directe : {names_str}"

    # --- Mode : Classement complet / Top N (tri classique, troncature voulue) ---
    # detect_sort_field() détermine dynamiquement le champ de tri
    if any(m in q for m in MOTS_CLES_CLASSEMENT):
        n = extract_top_n(q)
        sort_field, field_label = detect_sort_field(question)  # ← champ dynamique
        docs = retrieve_top_n(es, sort_field=sort_field, order="desc", n=n)
        return docs, None, "classement", f"🏆 Classement des top {n} CVs par {field_label}"

    # --- Mode : Ranking descendant (meilleur / max / plus élevé) ---
    # Utilise une agrégation ES native max puis récupère TOUS les docs triés
    # detect_sort_field() choisit le bon champ (score vs expérience vs ...)
    if any(m in q for m in MOTS_CLES_RANKING):
        sort_field, field_label = detect_sort_field(question)  # ← champ dynamique
        value, docs_extremum, all_docs = retrieve_extremum(es, sort_field=sort_field, order="desc")
        desc = f"📊 Agrégation max sur {field_label} → {value} ({len(all_docs)} CVs en contexte)"
        # On envoie TOUS les docs triés au LLM pour qu'il ait le contexte complet
        return all_docs, None, "ranking_desc", desc

    # --- Mode : Ranking ascendant (pire / min / plus bas) ---
    # Même logique que le ranking desc mais en ordre ascendant
    if any(m in q for m in MOTS_CLES_RANKING_ASC):
        sort_field, field_label = detect_sort_field(question)  # ← champ dynamique
        value, docs_extremum, all_docs = retrieve_extremum(es, sort_field=sort_field, order="asc")
        desc = f"📊 Agrégation min sur {field_label} → {value} ({len(all_docs)} CVs en contexte)"
        # On envoie TOUS les docs triés au LLM pour qu'il ait le contexte complet
        return all_docs, None, "ranking_asc", desc

    # --- Mode : Statistiques / Comptage (agrégations ES natives) ---
    # Pour "combien de candidats en IA" → agrégation terms sur categorie_principale
    # Pour "moyenne des scores" → agrégation avg
    if any(m in q for m in MOTS_CLES_STATS):
        stats = retrieve_stats(es)
        # On récupère aussi les CVs pour donner du contexte au LLM
        category = detect_category_in_question(question)
        if category:
            # Si une catégorie est détectée, filtrer par cette catégorie
            docs = retrieve_by_category(es, category)
        else:
            # Sinon, récupérer tous les CVs triés par le champ pertinent
            sort_field, _ = detect_sort_field(question)
            docs = retrieve_all_sorted(es, sort_field=sort_field)
        return docs, stats, "stats", "📊 Statistiques agrégées depuis Elasticsearch"

    # --- Mode : Filtrage par catégorie ---
    if any(m in q for m in MOTS_CLES_FILTRAGE):
        category = detect_category_in_question(question)
        if category:
            docs = retrieve_by_category(es, category)
            return docs, None, "filter", f"🏷️ CVs filtrés par catégorie : {category}"
        # Pas de catégorie détectée → liste complète
        sort_field, _ = detect_sort_field(question)
        docs = retrieve_all_sorted(es, sort_field=sort_field)
        return docs, None, "filter", "📋 Liste de tous les CVs"

    # --- Mode : Recherche sémantique (kNN) — par défaut ---
    # k=8 et num_candidates=50 pour couvrir tout l'index
    q_vec = embed_query(question)
    docs = retrieve_top_k(es, q_vec, k=8)
    return docs, None, "semantic", "🔍 Recherche sémantique (kNN vectoriel)"


# ---------------------------------------------------------------------------
# Construction du prompt contextuel selon le mode de retrieval
# ---------------------------------------------------------------------------

def format_cv_entry(i: int, doc: Dict, include_text: bool = False) -> str:
    """Formate un CV en texte structuré pour le prompt LLM.
    Inclut le champ 'localisation' pour éviter que le LLM ne le déduise.
    """
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

    entry = (
        f"**CV {i} – {nom}**\n"
        f"  • Score qualité globale : {score}/100 ({score_10}/10)\n"
        f"  • Catégorie principale : {categ}\n"
        f"  • Localisation : {localisation}\n"
        f"  • Années d'expérience : {exp}\n"
        f"  • Technologies : {techs}\n"
        f"  • Langages : {langs}\n"
        f"  • Frameworks : {fworks}\n"
        f"  • Diplômes : {diplomes}\n"
    )

    if include_text:
        text_preview = doc.get("text", "")[:400]
        if text_preview:
            entry += f"  • Extrait du CV : {text_preview}...\n"

    return entry


def build_prompt(question: str, docs: Optional[List[Dict]], stats: Optional[Dict],
                 mode: str, field_label: str = "score qualité globale") -> str:
    """Construit le prompt utilisateur adapté au mode de retrieval.

    Le paramètre field_label permet d'indiquer au LLM le champ exact utilisé
    pour le tri/agrégation (ex: "années d'expérience", "score qualité globale").
    """

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
            f"Comparez ces candidats en vous basant UNIQUEMENT sur les données ci-dessus. "
            f"Présentez la comparaison sous forme de tableau si possible. "
            f"Les valeurs indiquées sont exactes — ne les inventez pas. "
            f"Si une information n'est pas présente (ex: localisation), dites-le "
            f"explicitement au lieu de la deviner."
        )

    elif mode == "stats":
        # Construire le texte des statistiques
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

            buckets = stats.get("by_categ", {}).get("buckets", [])
            if buckets:
                stats_text += "\n📂 **Par catégorie** :\n"
                for b in buckets:
                    cat_name = b["key"]
                    cat_count = b["doc_count"]
                    cat_avg = b.get("avg_score", {}).get("value")
                    cat_max = b.get("max_score", {}).get("value")
                    avg_str = f", score moyen: {cat_avg:.1f}" if cat_avg else ""
                    max_str = f", score max: {cat_max:.1f}" if cat_max else ""
                    stats_text += f"  • {cat_name} : {cat_count} CV(s){avg_str}{max_str}\n"

        # Ajouter aussi les CVs si disponibles
        cv_text = ""
        if docs:
            cv_text = "\n\n📋 **Détails des CVs** :\n" + "\n---\n".join(
                [format_cv_entry(i + 1, doc) for i, doc in enumerate(docs[:10])]
            )

        return (
            f"**Question** : {question}\n\n"
            f"{stats_text}{cv_text}\n\n"
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
        sources = "\n---\n".join(
            [format_cv_entry(i + 1, doc, include_text=True) for i, doc in enumerate(docs)]
        )
        return (
            f"**Question** : {question}\n\n"
            f"Voici les {len(docs)} CVs les plus pertinents sémantiquement pour cette question :\n\n"
            f"{sources}\n\n"
            f"Répondez de façon concise en vous basant UNIQUEMENT sur les informations "
            f"ci-dessus. Les scores indiqués sont des valeurs exactes — ne les inventez pas. "
            f"Si une information n'est pas disponible dans les données (ex: localisation), "
            f"dites-le explicitement au lieu de la deviner à partir d'autres champs."
        )


# ---------------------------------------------------------------------------
# UI Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(page_title="💬 Chatbot RAG", layout="wide")
st.title("💬 Chatbot RAG – Recherche intelligente dans les CVs")
st.caption("Recherche sémantique (kNN) + requêtes factuelles (tri, agrégation, filtrage)")

# Sidebar – choisir le provider LLM
provider = st.sidebar.selectbox(
    "Provider LLM",
    options=["groq", "openrouter", "mistral", "gemini"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("### 💡 Exemples de questions")
st.sidebar.markdown("""
**Sémantique (kNN)** :
- *Trouve-moi un profil orienté IA avec de l'embarqué*
- *Candidat avec expérience en machine learning*

**Factuelle (tri/agrégation)** :
- *Quel CV a le meilleur score ?*
- *Top 3 des candidats*
- *Combien de CVs en catégorie IA ?*
- *Le CV avec le score le plus bas*
- *Quel candidat a le plus d'expérience ?*

**Comparaison** :
- *Compare Rim et Mohamed*
""")

# Initialiser le client ES (fallback est géré dans es_client)
es = get_es_client()

# Historique de la conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Afficher l'historique des messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("mode_badge"):
            st.caption(msg["mode_badge"])

# Input de l'utilisateur
question = st.chat_input("Posez votre question sur les CVs…")

if question and question.strip():
    # Afficher le message utilisateur
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "avatar": "👤",
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # --- Phase 1 : Routage intelligent ---
    with st.spinner("🧠 Analyse de la question et récupération des données…"):
        docs, stats, mode, mode_description = route_question(question, es)

    # --- Phase 2 : Construction du prompt contextuel ---
    # Passer le label du champ de tri au prompt pour que le LLM sache
    # quel champ a été utilisé pour le tri (score, expérience, etc.)
    _, field_label = detect_sort_field(question)
    prompt = build_prompt(question, docs, stats, mode, field_label=field_label)

    # --- Phase 3 : Appel au LLM ---
    llm = get_llm(provider)
    with st.spinner("✍️ Génération de la réponse…"):
        answer = llm.generate(prompt)

    # --- Phase 4 : Affichage de la réponse ---
    with st.chat_message("assistant", avatar="🤖"):
        st.caption(mode_description)
        st.markdown(answer)

    # Sauvegarder dans l'historique
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "avatar": "🤖",
        "mode_badge": mode_description,
    })

    # --- Phase 5 : Sources détaillées ---
    if docs:
        with st.expander("📎 Voir les sources (CVs) utilisés", expanded=False):
            for i, doc in enumerate(docs):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                with col1:
                    st.markdown(f"**{doc.get('nom', '?')}**")
                with col2:
                    score = doc.get("score_qualite_globale", "-")
                    st.metric("Score", f"{score}/100" if score != "-" else "-")
                with col3:
                    st.markdown(f"📂 {doc.get('categorie_principale', '-')}")
                with col4:
                    exp = doc.get("annees_experience", "-")
                    st.markdown(f"📅 {exp} ans" if exp != "-" else "📅 -")

                tech = ", ".join(doc.get("technologies", [])[:6])
                if tech:
                    st.markdown(f"  🛠️ *{tech}*")
                st.divider()

    if stats:
        with st.expander("📊 Voir les statistiques brutes", expanded=False):
            st.json(stats)