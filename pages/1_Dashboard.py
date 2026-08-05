# pages/1_Dashboard.py
"""Dashboard analytique – KPIs, graphiques et tableau filtrable.

Récupère TOUS les CVs indexés dans Elasticsearch (via es.count() pour
le total dynamique), construit un DataFrame pandas, applique des filtres
interactifs (sidebar), puis affiche les KPIs et graphiques sur le
DataFrame filtré.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

# Ajouter le répertoire parent au path pour permettre l'import de es_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from es_client import get_es_client

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("📊 Dashboard analytique")

es = get_es_client()

# ---------------------------------------------------------------------------
# 1. Récupérer dynamiquement TOUS les CVs (pas un size fixe en dur)
# ---------------------------------------------------------------------------
total = es.count(index="cvs")["count"]

if total == 0:
    st.info("ℹ️ Aucun CV indexé dans Elasticsearch. Veuillez d'abord ajouter des CVs.")
    st.stop()

# Requête ES avec size = total pour tout récupérer
all_cvs = es.search(
    index="cvs",
    size=total,
    _source=[
        "nom", "email", "categorie_principale",
        "score_qualite_globale", "score_qualite_globale_sur_10",
        "annees_experience", "technologies",
        "langages", "frameworks",
    ]
)

# Construire le DataFrame à partir de tous les documents
rows = []
for hit in all_cvs["hits"]["hits"]:
    src = hit["_source"]
    rows.append({
        "Nom": src.get("nom", "—"),
        "Email": src.get("email", "—"),
        "Catégorie": src.get("categorie_principale", "—"),
        "Score qualité": src.get("score_qualite_globale"),
        "Score /10": src.get("score_qualite_globale_sur_10"),
        "Années exp.": src.get("annees_experience"),
        "Technologies": ", ".join(src.get("technologies", [])[:6]),
        "Langages": ", ".join(src.get("langages", [])[:5]),
    })

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 2. Conversion numérique robuste (éviter les strings/None qui cassent les moyennes)
# ---------------------------------------------------------------------------
df["Score qualité"] = pd.to_numeric(df["Score qualité"], errors="coerce")
df["Score /10"] = pd.to_numeric(df["Score /10"], errors="coerce")
df["Années exp."] = pd.to_numeric(df["Années exp."], errors="coerce")

# ---------------------------------------------------------------------------
# 3. Filtres interactifs dans la sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("🔧 Filtres")

# Multiselect sur la catégorie principale
categories_disponibles = sorted(df["Catégorie"].dropna().unique().tolist())
categories_choisies = st.sidebar.multiselect(
    "Catégorie principale",
    options=categories_disponibles,
    default=categories_disponibles,
    help="Sélectionnez une ou plusieurs catégories à afficher.",
)

# Slider score qualité minimum
# Garde-fou : si min == max (un seul score distinct), on ne peut pas afficher
# de slider (Streamlit exige min_value < max_value strictement).
score_min_global = float(df["Score qualité"].min()) if df["Score qualité"].notna().any() else 0.0
score_max_global = float(df["Score qualité"].max()) if df["Score qualité"].notna().any() else 100.0
if score_min_global < score_max_global:
    score_min = st.sidebar.slider(
        "Score qualité minimum",
        min_value=score_min_global,
        max_value=score_max_global,
        value=score_min_global,
        step=1.0,
        help="Afficher uniquement les CVs avec un score ≥ cette valeur.",
    )
else:
    score_min = score_min_global
    st.sidebar.caption(f"Score qualité : {score_min_global:.0f} (valeur unique, pas de filtre possible)")

# Slider années d'expérience minimum
# Même garde-fou que pour le score : pas de slider si une seule valeur distincte
exp_min_global = float(df["Années exp."].min()) if df["Années exp."].notna().any() else 0.0
exp_max_global = float(df["Années exp."].max()) if df["Années exp."].notna().any() else 20.0
if exp_min_global < exp_max_global:
    exp_min = st.sidebar.slider(
        "Années d'expérience minimum",
        min_value=exp_min_global,
        max_value=exp_max_global,
        value=exp_min_global,
        step=0.1,
        format="%.1f",
        help="Afficher uniquement les CVs avec ≥ ce nombre d'années d'expérience.",
    )
else:
    exp_min = exp_min_global
    st.sidebar.caption(f"Expérience : {exp_min_global:.1f} ans (valeur unique, pas de filtre possible)")

# Appliquer les filtres
df_filtre = df[
    (df["Catégorie"].isin(categories_choisies))
    & (df["Score qualité"].fillna(0) >= score_min)
    & (df["Années exp."].fillna(0) >= exp_min)
]

# ---------------------------------------------------------------------------
# 5. Gérer le cas DataFrame vide après filtrage
# ---------------------------------------------------------------------------
if df_filtre.empty:
    st.warning("⚠️ Aucun CV ne correspond à ces filtres. Modifiez les filtres dans la barre latérale.")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs en haut de page (calculés sur le DataFrame FILTRÉ)
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

nb_filtre = len(df_filtre)
avg_score = df_filtre["Score qualité"].mean()
avg_exp = df_filtre["Années exp."].mean()
nb_categories = df_filtre["Catégorie"].nunique()

col1.metric(
    label="📄 CVs affichés",
    value=f"{nb_filtre} / {total}",
)
col2.metric(
    label="⭐ Score qualité moyen",
    value=f"{avg_score:.1f} / 100" if pd.notna(avg_score) else "N/A",
)
col3.metric(
    label="📅 Exp. moyenne",
    value=f"{avg_exp:.2f} ans" if pd.notna(avg_exp) else "N/A",
)
col4.metric(
    label="📂 Catégories",
    value=nb_categories,
)

st.divider()

# ---------------------------------------------------------------------------
# Graphique donut : Répartition par catégorie (données filtrées)
# ---------------------------------------------------------------------------
st.subheader("📂 Répartition par catégorie principale")

cat_counts = df_filtre["Catégorie"].value_counts()

if not cat_counts.empty:
    fig_pie = px.pie(
        values=cat_counts.values,
        names=cat_counts.index,
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.warning("Aucune catégorie trouvée dans les données filtrées.")

st.divider()

# ---------------------------------------------------------------------------
# Graphique barres : Score qualité moyen par catégorie (données filtrées)
# ---------------------------------------------------------------------------
st.subheader("📊 Score qualité moyen par catégorie")

score_par_cat = (
    df_filtre.groupby("Catégorie")["Score qualité"]
    .mean()
    .dropna()
    .sort_values(ascending=False)
)

if not score_par_cat.empty:
    fig_bar = px.bar(
        x=score_par_cat.index,
        y=score_par_cat.values.round(1),
        labels={"x": "Catégorie", "y": "Score qualité moyen"},
        color=score_par_cat.values,
        color_continuous_scale="Viridis",
    )
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("Pas assez de données pour afficher le graphique des scores par catégorie.")

st.divider()

# ---------------------------------------------------------------------------
# Graphique barres : Expérience moyenne par catégorie (données filtrées)
# ---------------------------------------------------------------------------
st.subheader("📅 Années d'expérience moyennes par catégorie")

exp_par_cat = (
    df_filtre.groupby("Catégorie")["Années exp."]
    .mean()
    .dropna()
    .sort_values(ascending=False)
)

if not exp_par_cat.empty:
    fig_exp = px.bar(
        x=exp_par_cat.index,
        y=exp_par_cat.values.round(2),
        labels={"x": "Catégorie", "y": "Années d'expérience moyennes"},
        color=exp_par_cat.values,
        color_continuous_scale="Plasma",
    )
    fig_exp.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_exp, use_container_width=True)
else:
    st.info("Pas assez de données pour afficher le graphique d'expérience par catégorie.")

st.divider()

# ---------------------------------------------------------------------------
# 4. Tableau récapitulatif avec slider de nombre de lignes
# ---------------------------------------------------------------------------
st.subheader("📋 Tableau des CVs")

# Slider pour limiter l'affichage (pas de re-requête ES)
# Garde-fou : si 0 ou 1 seul CV filtré, pas de slider (min_value doit être < max_value)
max_affichage = len(df_filtre)
if max_affichage > 1:
    nombre_a_afficher = st.slider(
        "Nombre de CVs à afficher",
        min_value=1,
        max_value=max_affichage,
        value=min(10, max_affichage),
    )
else:
    nombre_a_afficher = max_affichage
    st.caption(f"1 seul CV correspond aux filtres — affiché ci-dessous.")

# Colonnes à afficher dans le tableau
colonnes_tableau = ["Nom", "Email", "Catégorie", "Score qualité", "Score /10",
                     "Années exp.", "Technologies", "Langages"]

st.dataframe(
    df_filtre[colonnes_tableau].head(nombre_a_afficher),
    use_container_width=True,
    hide_index=True,
)

# Info résumé en bas de page
st.caption(
    f"Affichage de {nombre_a_afficher} CV(s) sur {len(df_filtre)} filtrés "
    f"({total} au total dans l'index)."
)
