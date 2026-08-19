# pages/2_Recherche.py
"""Recherche avancée – Filtrage multi-critères des CVs indexés dans Elasticsearch.

Fonctionnalités :
- Recherche textuelle libre (multi_match sur nom, localisation, projets, diplomes, certifications)
- Filtres keyword multi-select (catégorie, technologies, langages, frameworks, BDD, DevOps, langues)
- Filtres nested (scores_categories : domaine + score min ; experiences_pro : domaine + poste)
- Filtres range (score qualité min, années d'expérience min)
- Toggle alertes de parcours
- Tri dynamique (pertinence, score qualité, expérience, score domaine nested)
- Pagination from/size avec navigation
- Agrégations cachées avec bouton de rafraîchissement manuel
"""

import os
import sys
import streamlit as st
from typing import List, Dict, Optional, Tuple, Any

# Ajouter le répertoire parent au PATH pour importer es_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from es_client import get_es_client
from cv_cache import CVCache
from cv_removal import remove_cv

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

INDEX_NAME = "cvs"

# Nombre de résultats par page (options proposées à l'utilisateur)
OPTIONS_PAR_PAGE = [5, 10, 20, 50]
DEFAUT_PAR_PAGE = 10

# Champs _source à récupérer (tous les champs utiles pour l'affichage)
SOURCE_FIELDS = [
    "nom", "email", "telephone", "linkedin", "localisation",
    "categorie_principale",
    "domaine_1", "score_1", "domaine_2", "score_2", "domaine_3", "score_3",
    "scores_categories", "scores_categories_ponderes", "scores_categories_ponderes_sur_10",
    "technologies", "langages", "frameworks", "bases_de_donnees", "outils_devops",
    "langues",
    "projets", "diplomes", "certifications",
    "score_qualite_globale", "score_qualite_globale_sur_10",
    "annees_experience",
    "alertes_parcours",
    "experiences_pro",
]

# Champs keyword pour lesquels on propose un multi-select (label UI → champ ES)
CHAMPS_KEYWORD_MULTISELECT = {
    "Technologies": "technologies",
    "Langages": "langages",
    "Frameworks": "frameworks",
    "Bases de données": "bases_de_donnees",
    "Outils DevOps": "outils_devops",
    "Langues": "langues",
}


# ---------------------------------------------------------------------------
# Fonctions utilitaires : vérification de l'index
# ---------------------------------------------------------------------------

def index_existe(es) -> bool:
    """Vérifie si l'index ES existe, sans planter si ES est un DummyES."""
    try:
        return es.indices.exists(index=INDEX_NAME)
    except Exception:
        return False


def compter_documents(es) -> int:
    """Retourne le nombre de documents dans l'index, 0 si erreur."""
    try:
        return es.count(index=INDEX_NAME)["count"]
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Agrégations ES — cachées 5 min avec @st.cache_data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_distinct_keyword_values(_es_id: str, es_host: str, field: str) -> List[str]:
    """Agrégation terms sur un champ keyword (technologies, langages, etc.).

    _es_id et es_host sont des paramètres fictifs pour que le cache Streamlit
    puisse hasher l'appel (l'objet ES n'est pas hashable directement).
    Retourne une liste triée de valeurs distinctes, ou [] si erreur.
    """
    es = get_es_client()
    try:
        res = es.search(
            index=INDEX_NAME,
            size=0,
            aggs={
                "valeurs": {"terms": {"field": field, "size": 500}}
            },
        )
        buckets = res.get("aggregations", {}).get("valeurs", {}).get("buckets", [])
        return sorted([b["key"] for b in buckets])
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def get_distinct_domaines_nested(_es_id: str) -> List[str]:
    """Agrégation nested terms sur scores_categories.domaine.

    Retourne la liste triée des domaines distincts présents dans les CVs.
    """
    es = get_es_client()
    try:
        res = es.search(
            index=INDEX_NAME,
            size=0,
            aggs={
                "nested_scores": {
                    "nested": {"path": "scores_categories"},
                    "aggs": {
                        "domaines": {
                            "terms": {"field": "scores_categories.domaine", "size": 100}
                        }
                    }
                }
            },
        )
        buckets = (
            res.get("aggregations", {})
            .get("nested_scores", {})
            .get("domaines", {})
            .get("buckets", [])
        )
        return sorted([b["key"] for b in buckets])
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def get_distinct_domaines_experience(_es_id: str) -> List[str]:
    """Agrégation nested terms sur experiences_pro.domaine."""
    es = get_es_client()
    try:
        res = es.search(
            index=INDEX_NAME,
            size=0,
            aggs={
                "nested_exp": {
                    "nested": {"path": "experiences_pro"},
                    "aggs": {
                        "domaines": {
                            "terms": {"field": "experiences_pro.domaine", "size": 100}
                        }
                    }
                }
            },
        )
        buckets = (
            res.get("aggregations", {})
            .get("nested_exp", {})
            .get("domaines", {})
            .get("buckets", [])
        )
        return sorted([b["key"] for b in buckets])
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def get_score_range(_es_id: str) -> Tuple[float, float]:
    """Retourne (min, max) du champ score_qualite_globale."""
    es = get_es_client()
    try:
        res = es.search(
            index=INDEX_NAME,
            size=0,
            aggs={
                "min_score": {"min": {"field": "score_qualite_globale"}},
                "max_score": {"max": {"field": "score_qualite_globale"}},
            },
        )
        aggs = res.get("aggregations", {})
        mn = aggs.get("min_score", {}).get("value")
        mx = aggs.get("max_score", {}).get("value")
        if mn is None or mx is None:
            return (0.0, 100.0)
        return (float(mn), float(mx))
    except Exception:
        return (0.0, 100.0)


@st.cache_data(ttl=300, show_spinner=False)
def get_experience_range(_es_id: str) -> Tuple[float, float]:
    """Retourne (min, max) du champ annees_experience."""
    es = get_es_client()
    try:
        res = es.search(
            index=INDEX_NAME,
            size=0,
            aggs={
                "min_exp": {"min": {"field": "annees_experience"}},
                "max_exp": {"max": {"field": "annees_experience"}},
            },
        )
        aggs = res.get("aggregations", {})
        mn = aggs.get("min_exp", {}).get("value")
        mx = aggs.get("max_exp", {}).get("value")
        if mn is None or mx is None:
            return (0.0, 30.0)
        return (float(mn), float(mx))
    except Exception:
        return (0.0, 30.0)


# ---------------------------------------------------------------------------
# Construction de la requête ES
# ---------------------------------------------------------------------------

def build_search_query(
    texte_libre: str,
    categories: List[str],
    domaine_nested: str,
    score_domaine_min: float,
    filtres_keyword: Dict[str, List[str]],
    exp_pro_domaine: str,
    exp_pro_poste: str,
    score_qualite_min: float,
    score_qualite_min_global: float,
    annees_exp_min: float,
    annees_exp_min_global: float,
    avec_alertes: bool,
) -> Dict:
    """Construit la requête bool/must avec toutes les clauses actives.

    Chaque filtre actif ajoute une clause à `must`. Les filtres vides/non
    sélectionnés sont ignorés (pas de clause ajoutée = match_all implicite).
    """
    must_clauses = []

    # 1. Recherche textuelle libre (multi_match cross_fields)
    if texte_libre.strip():
        must_clauses.append({
            "multi_match": {
                "query": texte_libre.strip(),
                "fields": ["nom^3", "localisation", "projets", "diplomes", "certifications"],
                "type": "cross_fields",
                "operator": "and",
            }
        })

    # 2. Filtre catégorie principale (terms)
    if categories:
        must_clauses.append({
            "terms": {"categorie_principale": categories}
        })

    # 3. Filtre domaine nested + score min (scores_categories)
    if domaine_nested:
        nested_must = [{"term": {"scores_categories.domaine": domaine_nested}}]
        if score_domaine_min > 0:
            nested_must.append({
                "range": {"scores_categories.score": {"gte": score_domaine_min}}
            })
        must_clauses.append({
            "nested": {
                "path": "scores_categories",
                "query": {"bool": {"must": nested_must}},
            }
        })

    # 4. Filtres keyword multi-select (technologies, langages, etc.)
    for label, field in CHAMPS_KEYWORD_MULTISELECT.items():
        valeurs = filtres_keyword.get(field, [])
        if valeurs:
            must_clauses.append({
                "terms": {field: valeurs}
            })

    # 5. Filtre expérience pro nested (domaine + poste)
    if exp_pro_domaine or exp_pro_poste.strip():
        nested_exp_must = []
        if exp_pro_domaine:
            nested_exp_must.append({
                "term": {"experiences_pro.domaine": exp_pro_domaine}
            })
        if exp_pro_poste.strip():
            nested_exp_must.append({
                "match": {"experiences_pro.poste": exp_pro_poste.strip()}
            })
        must_clauses.append({
            "nested": {
                "path": "experiences_pro",
                "query": {"bool": {"must": nested_exp_must}},
            }
        })

    # 6. Filtre score qualité minimum (range) — seulement si > min global
    if score_qualite_min > score_qualite_min_global:
        must_clauses.append({
            "range": {"score_qualite_globale": {"gte": score_qualite_min}}
        })

    # 7. Filtre années d'expérience minimum (range) — seulement si > min global
    if annees_exp_min > annees_exp_min_global:
        must_clauses.append({
            "range": {"annees_experience": {"gte": annees_exp_min}}
        })

    # 8. Toggle alertes de parcours (exists)
    if avec_alertes:
        must_clauses.append({
            "exists": {"field": "alertes_parcours"}
        })

    # Assemblage final
    if must_clauses:
        return {"bool": {"must": must_clauses}}
    else:
        return {"match_all": {}}


def build_sort_clause(
    tri: str,
    domaine_nested: str,
) -> List[Dict]:
    """Construit la clause de tri ES en fonction du choix utilisateur.

    Respecte les contraintes du mapping : pas de tri natif sur les champs
    text (nom, diplomes, certifications).
    """
    if tri == "Score qualité ↓":
        return [{"score_qualite_globale": {"order": "desc"}}]
    elif tri == "Score qualité ↑":
        return [{"score_qualite_globale": {"order": "asc"}}]
    elif tri == "Expérience ↓":
        return [{"annees_experience": {"order": "desc"}}]
    elif tri == "Expérience ↑":
        return [{"annees_experience": {"order": "asc"}}]
    elif tri == "Score domaine ↓" and domaine_nested:
        # Tri nested sur le score du domaine sélectionné
        return [{
            "scores_categories.score": {
                "order": "desc",
                "nested": {
                    "path": "scores_categories",
                    "filter": {
                        "term": {"scores_categories.domaine": domaine_nested}
                    }
                }
            }
        }]
    else:
        # Pertinence (_score) — défaut
        return [{"_score": {"order": "desc"}}]


# ---------------------------------------------------------------------------
# Exécution de la recherche
# ---------------------------------------------------------------------------

def executer_recherche(
    es,
    query: Dict,
    sort: List[Dict],
    page: int,
    par_page: int,
) -> Tuple[List[Dict], int]:
    """Exécute la recherche ES avec pagination from/size.

    Retourne :
        - hits : liste de dicts (contenu _source de chaque CV)
        - total : nombre total de résultats (pour calculer le nb de pages)
    """
    from_offset = (page - 1) * par_page

    try:
        res = es.search(
            index=INDEX_NAME,
            query=query,
            sort=sort,
            from_=from_offset,
            size=par_page,
            _source=SOURCE_FIELDS,
            track_total_hits=True,
        )
    except Exception as e:
        st.error(f"❌ Erreur lors de la recherche ES : {e}")
        return [], 0

    hits = [{"_id": h["_id"], **h["_source"]} for h in res["hits"]["hits"]]
    total_hits = res["hits"]["total"]
    # total peut être un dict {"value": N, "relation": "eq"} ou un int
    if isinstance(total_hits, dict):
        total = total_hits.get("value", 0)
    else:
        total = int(total_hits)

    return hits, total


# ---------------------------------------------------------------------------
def _score_color(score):
    if score is None: return "#666"
    if score >= 70: return "#43e97b"
    if score >= 50: return "#f6d365"
    return "#fa709a"


def afficher_cv(doc: Dict, rang: int, es=None, cache=None):
    """Affiche un CV dans une carte stylise avec toggle expand/collapse."""
    doc_id = doc.get("_id", "")
    nom = doc.get("nom", "Inconnu")
    categ = doc.get("categorie_principale", "—")
    score = doc.get("score_qualite_globale")
    exp = doc.get("annees_experience")

    score_str = f"{score:.1f}/100" if score is not None else "—"
    exp_str   = f"{exp:.1f} ans" if exp is not None else "—"
    color     = _score_color(score)
    key       = f"cv_open_{rang}_{nom}"

    if key not in st.session_state:
        st.session_state[key] = False

    # ── Header card (toujours visible) ─────────────────────────────────────
    st.markdown(f"""
<div style="
  background:linear-gradient(145deg,#1a1f2e,#1e2535);
  border:1px solid rgba(255,255,255,0.07);
  border-left:4px solid {color};
  border-radius:12px;
  padding:0.9rem 1.2rem;
  margin-bottom:0.4rem;
  display:flex;
  align-items:center;
  gap:1rem;
">
  <span style="
    background:rgba(255,255,255,0.06);
    border-radius:50%;
    width:32px;height:32px;
    display:flex;align-items:center;justify-content:center;
    font-weight:800;color:#aaa;font-size:0.8rem;flex-shrink:0;
  ">{rang}</span>
  <div style="flex:1">
    <span style="color:#fff;font-weight:700;font-size:1rem">{nom}</span>
    <span style="color:rgba(255,255,255,0.45);font-size:0.82rem;margin-left:0.8rem">{categ}</span>
  </div>
  <span style="color:{color};font-weight:700;font-size:0.9rem">{score_str}</span>
  <span style="color:rgba(255,255,255,0.4);font-size:0.82rem;margin-left:0.8rem">Exp: {exp_str}</span>
</div>
""", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 8])
    with col_btn:
        label = "▼ Détails" if not st.session_state[key] else "▲ Fermer"
        if st.button(label, key=f"btn_{key}", width='content'):
            st.session_state[key] = not st.session_state[key]

    if not st.session_state[key]:
        return

    # ── Détails (expandés) ─────────────────────────────────────────────────
    with st.container():
        st.markdown('<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem">', unsafe_allow_html=True)

        # Contact
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"📧 **Email** : {doc.get('email','—')}")
        with col2:
            st.markdown(f"📞 **Tél** : {doc.get('telephone','—') or '—'}")
        with col3:
            st.markdown(f"📍 **Lieu** : {doc.get('localisation','—') or '—'}")

        linkedin = doc.get("linkedin")
        if linkedin:
            st.markdown(f"🔗 **LinkedIn** : {linkedin}")

        st.divider()

        # Scores KPI
        score_10 = doc.get("score_qualite_globale_sur_10")
        c1, c2, c3 = st.columns(3)
        with c1:
            if score is not None:
                st.metric("Score qualité", f"{score:.1f}/100")
        with c2:
            if score_10 is not None:
                st.metric("Score /10", f"{score_10:.1f}/10")
        with c3:
            if exp is not None:
                st.metric("Expérience", f"{exp:.1f} ans")

        # Top domaines
        domaines_top = []
        for i in range(1, 4):
            d = doc.get(f"domaine_{i}")
            s = doc.get(f"score_{i}")
            if d and s is not None:
                domaines_top.append(f"{d} ({s:.1f})")
        if domaines_top:
            st.markdown(f"🏷️ **Top domaines** : {' · '.join(domaines_top)}")

        st.divider()

        # Compétences
        for label, field in CHAMPS_KEYWORD_MULTISELECT.items():
            valeurs = doc.get(field, [])
            if isinstance(valeurs, list) and valeurs:
                tags = " ".join(f'<span style="background:rgba(79,172,254,0.12);color:#4facfe;border:1px solid rgba(79,172,254,0.25);border-radius:5px;padding:0.1rem 0.5rem;font-size:0.75rem;margin:0.1rem;display:inline-block">{v}</span>' for v in valeurs)
                st.markdown(f"**🛠️ {label}** :", unsafe_allow_html=False)
                st.markdown(tags, unsafe_allow_html=True)

        # Diplômes
        diplomes = doc.get("diplomes")
        if diplomes:
            if isinstance(diplomes, list): diplomes = " | ".join(diplomes)
            st.markdown(f"🎓 **Diplômes** : {diplomes}")

        certifs = doc.get("certifications")
        if certifs:
            if isinstance(certifs, list): certifs = " | ".join(certifs)
            st.markdown(f"📜 **Certifications** : {certifs}")

        # Projets
        projets = doc.get("projets")
        if projets:
            if isinstance(projets, list): projets = " | ".join(projets)
            st.markdown(f"📁 **Projets** : {projets}")

        # Expériences
        experiences = doc.get("experiences_pro", [])
        if isinstance(experiences, list) and experiences:
            st.divider()
            st.markdown("💼 **Expériences professionnelles** :")
            for xp in experiences:
                poste  = xp.get("poste", "—")
                dom_xp = xp.get("domaine", "")
                debut  = xp.get("date_debut", "")
                fin    = xp.get("date_fin", "")
                poids  = xp.get("poids_pertinence")
                periode   = f"{debut} → {fin}" if debut or fin else ""
                poids_str = f" (pertinence : {poids:.2f})" if poids is not None else ""
                dom_str   = f" [{dom_xp}]" if dom_xp else ""
                st.markdown(f"  • **{poste}**{dom_str} — {periode}{poids_str}")

        # Alertes
        alertes = doc.get("alertes_parcours")
        if alertes:
            if isinstance(alertes, list): alertes = " | ".join(alertes)
            st.warning(f"⚠️ **Alertes** : {alertes}")

        # Scores par catégorie
        scores_cats = doc.get("scores_categories_ponderes_sur_10", [])
        if isinstance(scores_cats, list) and scores_cats:
            st.divider()
            st.markdown("📊 **Scores par catégorie (/10)** :")
            for sc in sorted(scores_cats, key=lambda x: x.get("score", 0), reverse=True):
                dom = sc.get("domaine", "—")
                val = sc.get("score")
                if val is not None:
                    st.markdown(f"  • {dom} : **{val:.1f}**/10")

        # ── Supprimer ce CV ────────────────────────────────────────────────
        if es is not None and cache is not None:
            st.divider()
            _del_key = f"confirm_del_{rang}_{nom}"
            _confirm = st.checkbox(
                f"🗑️ Supprimer **{nom}** complètement du système",
                key=_del_key,
            )
            if _confirm:
                st.warning(
                    "Cette action efface le CV de **Elasticsearch**, "
                    "**cv_cache.json**, **cvs_data.json** et le **fichier PDF**."
                )
                if st.button(
                    "🗑️ Confirmer la suppression",
                    type="primary",
                    key=f"btn_del_{rang}_{nom}",
                ):
                    with st.spinner("Suppression en cours..."):
                        _src = {
                            "nom":       doc.get("nom", ""),
                            "email":     doc.get("email", ""),
                            "telephone": doc.get("telephone", ""),
                            "filename":  doc.get("filename", ""),
                        }
                        _rpt = remove_cv(es, cache, doc_id, _src)
                    if _rpt.get("success"):
                        _parts = []
                        if _rpt.get("elasticsearch"): _parts.append("Elasticsearch")
                        if _rpt.get("cache"):         _parts.append("cache")
                        if _rpt.get("output"):        _parts.append("JSON/Excel")
                        if _rpt.get("pdf"):           _parts.append("fichier PDF")
                        st.success(f"✅ **{nom}** supprimé de : {' · '.join(_parts)}")
                        st.rerun()
                    else:
                        st.error("❌ Suppression échouée. Vérifiez qu'Elasticsearch est démarré.")

        st.markdown('</div>', unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# UI Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(page_title="🔎 Recherche", layout="wide", page_icon="🔎")

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from theme import inject_theme, hero, sec
inject_theme()
hero("🔎", "Recherche Avancée de CVs", "Filtrage multi-critères • Tri • Pagination • Agrégations Elasticsearch", badge="📁 Index cvs")


es = get_es_client()
cache = CVCache()

# --- Vérification de l'index ---
if not index_existe(es):
    st.info("ℹ️ L'index Elasticsearch n'existe pas encore. Veuillez d'abord indexer des CVs.")
    st.stop()

total_docs = compter_documents(es)
if total_docs == 0:
    st.info("ℹ️ L'index est vide — aucun CV indexé. Veuillez d'abord ajouter des CVs.")
    st.stop()

# Identifiant pour le cache (non-hashable ES client workaround)
_es_cache_id = "es_default"

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR : Filtres
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.header("🔧 Filtres de recherche")

# Bouton de rafraîchissement des filtres (vide le cache des agrégations)
if st.sidebar.button("🔄 Rafraîchir les filtres"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

# ── 1. Recherche textuelle libre ──
texte_libre = st.sidebar.text_input(
    "🔍 Recherche libre",
    value="",
    placeholder="Nom, localisation, projet, diplôme…",
    help="Recherche full-text sur nom, localisation, projets, diplômes, certifications.",
)

st.sidebar.divider()

# ── 2. Catégorie principale ──
categories_dispo = get_distinct_keyword_values(_es_cache_id, "", "categorie_principale")
categories_choisies = st.sidebar.multiselect(
    "📂 Catégorie principale",
    options=categories_dispo,
    default=[],
    help="Filtrer par catégorie(s) principale(s).",
)

st.sidebar.divider()

# ── 3. Domaine + score min (nested : scores_categories) ──
st.sidebar.subheader("🏷️ Domaine & score")
domaines_dispo = get_distinct_domaines_nested(_es_cache_id)
domaine_choisi = st.sidebar.selectbox(
    "Domaine",
    options=[""] + domaines_dispo,
    index=0,
    format_func=lambda x: "— Tous —" if x == "" else x,
    help="Filtrer par domaine (scores_categories).",
)
score_domaine_min = 0.0
if domaine_choisi:
    score_domaine_min = st.sidebar.slider(
        "Score minimum dans ce domaine",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
    )

st.sidebar.divider()

# ── 4. Filtres keyword multi-select ──
st.sidebar.subheader("🛠️ Compétences techniques")
filtres_keyword_valeurs: Dict[str, List[str]] = {}

for label, field in CHAMPS_KEYWORD_MULTISELECT.items():
    valeurs_dispo = get_distinct_keyword_values(_es_cache_id, "", field)
    selection = st.sidebar.multiselect(
        label,
        options=valeurs_dispo,
        default=[],
    )
    if selection:
        filtres_keyword_valeurs[field] = selection

st.sidebar.divider()

# ── 5. Expérience professionnelle (nested) ──
st.sidebar.subheader("💼 Expérience professionnelle")
domaines_exp_dispo = get_distinct_domaines_experience(_es_cache_id)
exp_pro_domaine = st.sidebar.selectbox(
    "Domaine d'expérience",
    options=[""] + domaines_exp_dispo,
    index=0,
    format_func=lambda x: "— Tous —" if x == "" else x,
    help="Filtrer par domaine dans les expériences professionnelles.",
)
exp_pro_poste = st.sidebar.text_input(
    "Intitulé de poste",
    value="",
    placeholder="Ex: Data Scientist, Ingénieur…",
    help="Recherche full-text dans les intitulés de postes.",
)

st.sidebar.divider()

# ── 6. Score qualité minimum (range) ──
score_min_global, score_max_global = get_score_range(_es_cache_id)
# Garde-fou : pas de slider si min == max
if score_min_global < score_max_global:
    score_qualite_min = st.sidebar.slider(
        "⭐ Score qualité minimum",
        min_value=score_min_global,
        max_value=score_max_global,
        value=score_min_global,
        step=1.0,
        help="Afficher uniquement les CVs avec un score ≥ cette valeur.",
    )
else:
    score_qualite_min = score_min_global
    st.sidebar.caption(f"Score qualité : {score_min_global:.0f} (valeur unique)")

# ── 7. Années d'expérience minimum (range) ──
exp_min_global, exp_max_global = get_experience_range(_es_cache_id)
if exp_min_global < exp_max_global:
    annees_exp_min = st.sidebar.slider(
        "📅 Années d'expérience minimum",
        min_value=exp_min_global,
        max_value=exp_max_global,
        value=exp_min_global,
        step=0.5,
        format="%.1f",
        help="Afficher uniquement les CVs avec ≥ ce nombre d'années d'expérience.",
    )
else:
    annees_exp_min = exp_min_global
    st.sidebar.caption(f"Expérience : {exp_min_global:.1f} ans (valeur unique)")

st.sidebar.divider()

# ── 8. Toggle alertes de parcours ──
avec_alertes = st.sidebar.checkbox(
    "⚠️ Avec alertes de parcours uniquement",
    value=False,
    help="Ne montrer que les CVs ayant des alertes de parcours.",
)

# ═══════════════════════════════════════════════════════════════════════════
# ZONE PRINCIPALE : Options de tri / pagination + résultats
# ═══════════════════════════════════════════════════════════════════════════

# --- Tri et pagination (barre d'options en haut) ---
col_tri, col_par_page, col_espace = st.columns([2, 1, 3])

with col_tri:
    # Options de tri dynamiques
    options_tri = [
        "Pertinence",
        "Score qualité ↓",
        "Score qualité ↑",
        "Expérience ↓",
        "Expérience ↑",
    ]
    # Ajouter l'option tri par score domaine si un domaine nested est sélectionné
    if domaine_choisi:
        options_tri.append("Score domaine ↓")

    tri_choisi = st.selectbox("Trier par", options=options_tri, index=0)

with col_par_page:
    par_page = st.selectbox(
        "Résultats / page",
        options=OPTIONS_PAR_PAGE,
        index=OPTIONS_PAR_PAGE.index(DEFAUT_PAR_PAGE),
    )

# --- Construction de la requête ---
query = build_search_query(
    texte_libre=texte_libre,
    categories=categories_choisies,
    domaine_nested=domaine_choisi,
    score_domaine_min=score_domaine_min,
    filtres_keyword=filtres_keyword_valeurs,
    exp_pro_domaine=exp_pro_domaine,
    exp_pro_poste=exp_pro_poste,
    score_qualite_min=score_qualite_min,
    score_qualite_min_global=score_min_global,
    annees_exp_min=annees_exp_min,
    annees_exp_min_global=exp_min_global,
    avec_alertes=avec_alertes,
)

sort = build_sort_clause(tri_choisi, domaine_choisi)

# --- Gestion de la pagination via session_state ---
if "page_recherche" not in st.session_state:
    st.session_state.page_recherche = 1

page_courante = st.session_state.page_recherche

# --- Exécution de la recherche ---
hits, total_resultats = executer_recherche(es, query, sort, page_courante, par_page)

# Calcul du nombre total de pages
nb_pages = max(1, (total_resultats + par_page - 1) // par_page)

# Ramener la page courante dans les bornes (cas où les filtres ont changé)
if page_courante > nb_pages:
    st.session_state.page_recherche = 1
    page_courante = 1
    # Relancer la recherche avec la bonne page
    hits, total_resultats = executer_recherche(es, query, sort, page_courante, par_page)

st.divider()

# --- Résumé des résultats ---
debut = (page_courante - 1) * par_page + 1
fin = min(page_courante * par_page, total_resultats)

if total_resultats == 0:
    st.warning("⚠️ Aucun CV ne correspond à ces critères. Essayez d'élargir vos filtres.")
    st.stop()

st.markdown(
    f"**{total_resultats}** CV(s) trouvé(s) — "
    f"affichage de **{debut}** à **{fin}** "
    f"(page **{page_courante}** / **{nb_pages}**)"
)

# --- Affichage des résultats ---
for i, doc in enumerate(hits):
    rang = debut + i
    afficher_cv(doc, rang, es=es, cache=cache)

# --- Navigation pagination ---
st.divider()
col_prev, col_info, col_next = st.columns([1, 3, 1])

with col_prev:
    if page_courante > 1:
        if st.button("⬅️ Précédent"):
            st.session_state.page_recherche = page_courante - 1
            st.rerun()

with col_info:
    st.markdown(
        f"<div style='text-align: center;'>Page {page_courante} / {nb_pages} "
        f"({total_resultats} résultats)</div>",
        unsafe_allow_html=True,
    )

with col_next:
    if page_courante < nb_pages:
        if st.button("Suivant ➡️"):
            st.session_state.page_recherche = page_courante + 1
            st.rerun()
