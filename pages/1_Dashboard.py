# pages/1_Dashboard.py
"""Dashboard analytique – KPIs, graphiques et tableau filtrable.

Récupère TOUS les CVs indexés dans Elasticsearch (cache 60 s), construit un
DataFrame pandas, applique des filtres interactifs (sidebar), puis affiche
les KPIs et graphiques sur le DataFrame filtré, organisés en onglets.
"""

import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from es_client import get_es_client
from cv_cache import CVCache
from cv_removal import remove_cv

st.set_page_config(page_title="Dashboard", layout="wide")

SCORE_BUCKETS = [
    ("0–50", 0, 50),
    ("50–70", 50, 70),
    ("70–85", 70, 85),
    ("85–100", 85, 101),
]

QUALITY_TIERS = [
    ("🟢 Excellent (85+)", 85, 101),
    ("🟡 Bon (70–84)", 70, 85),
    ("🟠 Moyen (50–69)", 50, 70),
    ("🔴 Faible (<50)", 0, 50),
]


@st.cache_data(ttl=60)
def load_cvs_data() -> tuple[int, pd.DataFrame]:
    """Charge tous les CVs depuis Elasticsearch (mis en cache 60 s)."""
    es = get_es_client()
    total = es.count(index="cvs")["count"]
    if total == 0:
        return 0, pd.DataFrame()

    all_cvs = es.search(
        index="cvs",
        size=total,
        _source=[
            "nom", "email", "telephone", "categorie_principale",
            "score_qualite_globale", "score_qualite_globale_sur_10",
            "annees_experience", "technologies",
            "langages", "frameworks", "bases_de_donnees", "outils_devops",
            "alertes_parcours", "indexed_at", "filename",
        ],
    )

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
            "_doc_id": hit["_id"],
            "_filename": src.get("filename", ""),
            "_source_raw": src,
            "_competences_raw": (
                (src.get("technologies") or [])
                + (src.get("langages") or [])
                + (src.get("frameworks") or [])
                + (src.get("bases_de_donnees") or [])
                + (src.get("outils_devops") or [])
            ),
            "_alertes_raw": src.get("alertes_parcours") or [],
            "_indexed_at_raw": src.get("indexed_at"),
        })

    df = pd.DataFrame(rows)
    df["Score qualité"] = pd.to_numeric(df["Score qualité"], errors="coerce")
    df["Score /10"] = pd.to_numeric(df["Score /10"], errors="coerce")
    df["Années exp."] = pd.to_numeric(df["Années exp."], errors="coerce")
    return total, df


def apply_plotly_theme(fig):
    """Thème transparent compatible clair/sombre (sans font_color forcé)."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20),
    )
    return fig


def format_delta(current: float, base: float, suffix: str = "", decimals: int = 1):
    """Formate un delta KPI par rapport à la base entière."""
    if pd.isna(current) or pd.isna(base):
        return None
    diff = current - base
    if abs(diff) < 0.05:
        return None
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.{decimals}f}{suffix} vs base"


def cv_has_skills(competences: list, skills: list) -> bool:
    """True si le CV possède toutes les compétences demandées (insensible à la casse)."""
    if not skills:
        return True
    comp_lower = [c.lower() for c in competences]
    for skill in skills:
        s = skill.lower()
        if not any(s in c or c in s for c in comp_lower):
            return False
    return True


def count_in_buckets(scores: pd.Series, buckets: list[tuple]) -> pd.DataFrame:
    """Compte les scores dans des tranches définies."""
    counts = []
    for label, low, high in buckets:
        n = ((scores >= low) & (scores < high)).sum()
        counts.append({"Tranche": label, "CVs": int(n)})
    return pd.DataFrame(counts)


def open_chatbot_with(question: str):
    """Envoie une question pré-remplie vers la page Chatbot."""
    st.session_state.chatbot_prefill = question
    st.switch_page("pages/3_Chatbot.py")


def alert_type(alerte: str) -> str:
    if alerte.startswith("Trou"):
        return "gap"
    if alerte.startswith("Chevauchement"):
        return "overlap"
    return "other"


st.title("📊 Dashboard analytique")

total, df = load_cvs_data()

if total == 0 or df.empty:
    st.info("ℹ️ Aucun CV indexé dans Elasticsearch. Veuillez d'abord ajouter des CVs.")
    st.stop()

# ---------------------------------------------------------------------------
# Filtres sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("🔧 Filtres")

categories_disponibles = sorted(df["Catégorie"].dropna().unique().tolist())
categories_choisies = st.sidebar.multiselect(
    "Catégorie principale",
    options=categories_disponibles,
    default=categories_disponibles,
    help="Sélectionnez une ou plusieurs catégories à afficher.",
)

all_skills = sorted({
    comp for comps in df["_competences_raw"] for comp in comps if comp
})
skills_choisies = st.sidebar.multiselect(
    "Compétences requises",
    options=all_skills,
    default=[],
    help="Afficher uniquement les CVs possédant TOUTES ces compétences.",
)

score_min_global = float(df["Score qualité"].min()) if df["Score qualité"].notna().any() else 0.0
score_max_global = float(df["Score qualité"].max()) if df["Score qualité"].notna().any() else 100.0
if score_min_global < score_max_global:
    score_min = st.sidebar.slider(
        "Score qualité minimum",
        min_value=score_min_global,
        max_value=score_max_global,
        value=score_min_global,
        step=1.0,
    )
else:
    score_min = score_min_global
    st.sidebar.caption(f"Score qualité : {score_min_global:.0f} (valeur unique)")

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
    )
else:
    exp_min = exp_min_global
    st.sidebar.caption(f"Expérience : {exp_min_global:.1f} ans (valeur unique)")

st.sidebar.divider()
st.sidebar.subheader("🗑️ Supprimer un CV")

delete_doc_ids = [""] + df["_doc_id"].astype(str).tolist()


def _format_delete_option(doc_id: str) -> str:
    if not doc_id:
        return "— Choisir —"
    match = df[df["_doc_id"].astype(str) == doc_id]
    if match.empty:
        return doc_id
    row = match.iloc[0]
    return f"{row['Nom']} ({row['Email']})"


cv_to_delete_id = st.sidebar.selectbox(
    "CV à supprimer",
    options=delete_doc_ids,
    format_func=_format_delete_option,
    key="delete_cv_select",
)
confirm_delete = st.sidebar.checkbox(
    "Je confirme la suppression définitive",
    key="confirm_delete_cv",
)

if st.sidebar.button(
    "🗑️ Supprimer",
    type="primary",
    disabled=not cv_to_delete_id or not confirm_delete,
    use_container_width=True,
):
    matches = df[df["_doc_id"].astype(str) == cv_to_delete_id]
    if matches.empty:
        st.sidebar.error("CV introuvable — actualisez la page et réessayez.")
    else:
        row = matches.iloc[0]
        cache = CVCache()
        es = get_es_client()
        report = remove_cv(
            es=es,
            cache=cache,
            doc_id=str(row["_doc_id"]),
            source=row["_source_raw"],
        )
        load_cvs_data.clear()
        if report.get("success"):
            details = []
            if report.get("elasticsearch"):
                details.append("dashboard")
            if report.get("cache"):
                details.append(f"cache ({report.get('cache_count', 1)})")
            if report.get("pdf"):
                details.append("PDF cvs/uploads")
            if report.get("output"):
                details.append("JSON/Excel")
            extra = f" ({', '.join(details)})" if details else ""
            st.toast(f"CV « {row['Nom']} » supprimé{extra}.", icon="✅")
            st.session_state.pop("delete_cv_select", None)
            st.session_state.pop("confirm_delete_cv", None)
            st.rerun()
        else:
            st.sidebar.error("Échec : impossible de supprimer ce CV. Vérifiez qu'Elasticsearch est démarré.")

df_filtre = df[
    df["Catégorie"].isin(categories_choisies)
    & (df["Score qualité"].fillna(0) >= score_min)
    & (df["Années exp."].fillna(0) >= exp_min)
    & df["_competences_raw"].apply(lambda comps: cv_has_skills(comps, skills_choisies))
]

if df_filtre.empty:
    st.warning("⚠️ Aucun CV ne correspond à ces filtres. Modifiez les filtres dans la barre latérale.")
    st.stop()

# KPIs de référence (base entière)
base_avg_score = df["Score qualité"].mean()
base_avg_exp = df["Années exp."].mean()
nb_filtre = len(df_filtre)
avg_score = df_filtre["Score qualité"].mean()
avg_exp = df_filtre["Années exp."].mean()
nb_alertes_filtre = sum(len(a) for a in df_filtre["_alertes_raw"])
nb_alertes_base = sum(len(a) for a in df["_alertes_raw"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("📄 CVs affichés", f"{nb_filtre} / {total}")
col2.metric(
    "⭐ Score qualité moyen",
    f"{avg_score:.1f} / 100" if pd.notna(avg_score) else "N/A",
    delta=format_delta(avg_score, base_avg_score),
    delta_color="inverse",
)
col3.metric(
    "📅 Exp. moyenne",
    f"{avg_exp:.2f} ans" if pd.notna(avg_exp) else "N/A",
    delta=format_delta(avg_exp, base_avg_exp, suffix=" ans", decimals=2),
    delta_color="inverse",
)
col4.metric(
    "⚠️ Alertes",
    nb_alertes_filtre,
    delta=format_delta(nb_alertes_filtre, nb_alertes_base, suffix="", decimals=0),
    delta_color="inverse",
)

if nb_filtre < 4:
    st.info("ℹ️ Peu de CVs dans cette sélection — ajoutez-en ou élargissez les filtres pour des comparaisons fiables.")

st.divider()

tab_overview, tab_skills, tab_quality, tab_pipeline, tab_alerts = st.tabs(
    ["Vue d'ensemble", "Compétences", "Qualité", "Pipeline", "Alertes"]
)

# ---------------------------------------------------------------------------
# Onglet Vue d'ensemble
# ---------------------------------------------------------------------------
with tab_overview:
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("📂 Répartition par catégorie")
        cat_counts = df_filtre["Catégorie"].value_counts()
        if not cat_counts.empty:
            fig_pie = apply_plotly_theme(px.pie(
                values=cat_counts.values,
                names=cat_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            ))
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

            if len(cat_counts) == 1:
                cat = cat_counts.index[0]
                if st.button(f"💬 Recommander en {cat}", key=f"chat_cat_{cat}"):
                    open_chatbot_with(f"Recommande les meilleurs candidats en {cat}")
        else:
            st.warning("Aucune catégorie trouvée.")

    with c_right:
        st.subheader("📊 Distribution des scores")
        scores_valid = df_filtre["Score qualité"].dropna()
        if not scores_valid.empty:
            hist_df = count_in_buckets(scores_valid, SCORE_BUCKETS)
            fig_hist = apply_plotly_theme(px.bar(
                hist_df,
                x="Tranche",
                y="CVs",
                color="CVs",
                color_continuous_scale="Teal",
                text="CVs",
            ))
            fig_hist.update_traces(textposition="outside")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Pas de scores disponibles.")

    st.subheader("🎯 Expérience vs score qualité")
    scatter_df = df_filtre.dropna(subset=["Années exp.", "Score qualité"])
    if len(scatter_df) >= 2:
        fig_scatter = apply_plotly_theme(px.scatter(
            scatter_df,
            x="Années exp.",
            y="Score qualité",
            color="Catégorie",
            hover_name="Nom",
            labels={"Années exp.": "Années d'expérience", "Score qualité": "Score qualité"},
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Ajoutez plus de CVs pour afficher le nuage de points expérience / score.")

    st.divider()
    st.subheader("📋 Tableau des CVs")

    export_cols = ["Nom", "Email", "Catégorie", "Score qualité", "Score /10",
                   "Années exp.", "Technologies", "Langages"]
    btn_col, dl_col = st.columns([2, 1])
    with dl_col:
        csv_data = df_filtre[export_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Exporter CSV",
            data=csv_data,
            file_name="cvs_filtres.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with btn_col:
        if st.button("💬 Demander au Chatbot sur cette sélection", use_container_width=True):
            parts = []
            if len(categories_choisies) < len(categories_disponibles):
                parts.append(f"catégorie {', '.join(categories_choisies)}")
            if skills_choisies:
                parts.append(f"compétences {', '.join(skills_choisies)}")
            if score_min > score_min_global:
                parts.append(f"score ≥ {score_min:.0f}")
            if exp_min > exp_min_global:
                parts.append(f"expérience ≥ {exp_min:.1f} ans")
            ctx = " avec " + ", ".join(parts) if parts else ""
            open_chatbot_with(f"Recommande les meilleurs candidats{ctx}")

    max_affichage = len(df_filtre)
    if max_affichage > 1:
        nombre_a_afficher = st.slider(
            "Nombre de CVs à afficher",
            min_value=1,
            max_value=max_affichage,
            value=min(10, max_affichage),
            key="table_rows",
        )
    else:
        nombre_a_afficher = max_affichage

    table_view = df_filtre[export_cols].head(nombre_a_afficher)
    st.dataframe(table_view, use_container_width=True, hide_index=True)

    noms_filtre = df_filtre["Nom"].tolist()
    compare_col1, compare_col2 = st.columns([3, 1])
    with compare_col1:
        candidat_compare = st.selectbox(
            "Comparer un candidat via le Chatbot",
            options=[""] + noms_filtre,
            format_func=lambda x: "— Choisir —" if x == "" else x,
            key="compare_select",
        )
    with compare_col2:
        st.write("")
        st.write("")
        if st.button("💬 Comparer", disabled=not candidat_compare):
            cat = df_filtre.loc[df_filtre["Nom"] == candidat_compare, "Catégorie"].iloc[0]
            open_chatbot_with(
                f"Compare {candidat_compare} avec les 2 meilleurs profils en {cat}"
            )

    st.caption(
        f"Affichage de {nombre_a_afficher} CV(s) sur {len(df_filtre)} filtrés "
        f"({total} au total dans l'index)."
    )

# ---------------------------------------------------------------------------
# Onglet Compétences
# ---------------------------------------------------------------------------
with tab_skills:
    st.subheader("🛠️ Top Technologies")
    competences_counter = Counter()
    for liste_competences in df_filtre["_competences_raw"]:
        competences_counter.update(liste_competences)

    if competences_counter:
        top20 = competences_counter.most_common(20)
        top20_df = pd.DataFrame(top20, columns=["Compétence", "Occurrences"]).sort_values(
            "Occurrences", ascending=True
        )
        fig_tech = apply_plotly_theme(px.bar(
            top20_df,
            x="Occurrences",
            y="Compétence",
            orientation="h",
            color="Occurrences",
            color_continuous_scale="Teal",
        ))
        fig_tech.update_layout(height=max(400, 20 * len(top20)))
        st.plotly_chart(fig_tech, use_container_width=True)

        st.markdown("**Actions rapides**")
        skill_cols = st.columns(min(4, len(top20)))
        for i, (skill, _) in enumerate(top20[:4]):
            with skill_cols[i]:
                if st.button(f"💬 {skill}", key=f"chat_skill_{skill}"):
                    open_chatbot_with(f"Trouve des candidats avec {skill}")

        st.subheader("🔍 Couverture des compétences")
        common = [s for s, n in competences_counter.items() if n >= 3]
        rare = [s for s, n in competences_counter.items() if n <= 2]
        cov1, cov2 = st.columns(2)
        with cov1:
            st.markdown("**Les plus fréquentes**")
            st.write(", ".join(sorted(common, key=lambda s: -competences_counter[s])[:15]) or "—")
        with cov2:
            st.markdown("**Présentes mais rares (≤ 2 CVs)**")
            st.write(", ".join(sorted(rare)[:20]) or "—")
    else:
        st.info("Aucune compétence trouvée dans les CVs filtrés.")

# ---------------------------------------------------------------------------
# Onglet Qualité
# ---------------------------------------------------------------------------
with tab_quality:
    q_left, q_right = st.columns(2)

    with q_left:
        st.subheader("📊 Score moyen par catégorie")
        score_par_cat = (
            df_filtre.groupby("Catégorie")["Score qualité"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )
        if not score_par_cat.empty:
            fig_bar = apply_plotly_theme(px.bar(
                x=score_par_cat.index,
                y=score_par_cat.values.round(1),
                labels={"x": "Catégorie", "y": "Score qualité moyen"},
                color=score_par_cat.values,
                color_continuous_scale="Viridis",
            ))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Pas assez de données.")

    with q_right:
        st.subheader("📅 Expérience moyenne par catégorie")
        exp_par_cat = (
            df_filtre.groupby("Catégorie")["Années exp."]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )
        if not exp_par_cat.empty:
            fig_exp = apply_plotly_theme(px.bar(
                x=exp_par_cat.index,
                y=exp_par_cat.values.round(2),
                labels={"x": "Catégorie", "y": "Années d'expérience moyennes"},
                color=exp_par_cat.values,
                color_continuous_scale="Plasma",
            ))
            st.plotly_chart(fig_exp, use_container_width=True)
        else:
            st.info("Pas assez de données.")

    st.subheader("🏷️ Niveaux de qualité")
    tier_rows = []
    for label, low, high in QUALITY_TIERS:
        n = ((df_filtre["Score qualité"] >= low) & (df_filtre["Score qualité"] < high)).sum()
        tier_rows.append({"Niveau": label, "CVs": int(n)})
    tier_df = pd.DataFrame(tier_rows)
    if tier_df["CVs"].sum() > 0:
        fig_tiers = apply_plotly_theme(px.bar(
            tier_df,
            x="Niveau",
            y="CVs",
            color="Niveau",
            text="CVs",
        ))
        fig_tiers.update_traces(textposition="outside")
        fig_tiers.update_layout(showlegend=False)
        st.plotly_chart(fig_tiers, use_container_width=True)

# ---------------------------------------------------------------------------
# Onglet Pipeline
# ---------------------------------------------------------------------------
with tab_pipeline:
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    dates_idx = pd.to_datetime(df_filtre["_indexed_at_raw"], errors="coerce", utc=True)
    this_week = (dates_idx >= week_ago).sum()
    this_month = (dates_idx >= month_ago).sum()
    last_indexed = dates_idx.max()

    p1, p2, p3 = st.columns(3)
    p1.metric("📥 CVs cette semaine", int(this_week))
    p2.metric("📥 CVs ce mois", int(this_month))
    p3.metric(
        "🕐 Dernière indexation",
        last_indexed.strftime("%d/%m/%Y") if pd.notna(last_indexed) else "N/A",
    )

    st.subheader("📅 Timeline d'ajout des CVs")
    nb_sans_date = dates_idx.isna().sum()
    if nb_sans_date > 0:
        st.info(
            f"ℹ️ {nb_sans_date} CV(s) sans champ `indexed_at` — exclus de la courbe ci-dessous."
        )

    timeline_df = pd.DataFrame({"indexed_at": dates_idx}).dropna()
    if not timeline_df.empty:
        timeline_df = timeline_df.sort_values("indexed_at")
        timeline_df["Cumul CVs indexés"] = range(1, len(timeline_df) + 1)
        fig_timeline = apply_plotly_theme(px.area(
            timeline_df,
            x="indexed_at",
            y="Cumul CVs indexés",
            labels={"indexed_at": "Date d'indexation"},
        ))
        fig_timeline.update_traces(line_color="#2ca9a9")
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("Aucune date d'indexation exploitable.")

# ---------------------------------------------------------------------------
# Onglet Alertes
# ---------------------------------------------------------------------------
with tab_alerts:
    df_avec_alertes = df_filtre[df_filtre["_alertes_raw"].apply(len) > 0]
    nb_avec_alertes = len(df_avec_alertes)
    nb_gaps = nb_overlaps = nb_other = 0
    for alertes in df_filtre["_alertes_raw"]:
        for a in alertes:
            t = alert_type(a)
            if t == "gap":
                nb_gaps += 1
            elif t == "overlap":
                nb_overlaps += 1
            else:
                nb_other += 1

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("📄 CVs avec alerte", f"{nb_avec_alertes} / {len(df_filtre)}")
    a2.metric("🕳️ Trous", nb_gaps)
    a3.metric("🔀 Chevauchements", nb_overlaps)
    a4.metric("🔢 Total alertes", nb_gaps + nb_overlaps + nb_other)

    if nb_avec_alertes > 0:
        lignes_alertes = []
        for _, row in df_avec_alertes.iterrows():
            for alerte in row["_alertes_raw"]:
                if alerte.startswith("Trou"):
                    icone = "🕳️"
                elif alerte.startswith("Chevauchement"):
                    icone = "🔀"
                else:
                    icone = "⚠️"
                lignes_alertes.append({"Candidat": row["Nom"], "Alerte": f"{icone} {alerte}"})
        st.dataframe(pd.DataFrame(lignes_alertes), use_container_width=True, hide_index=True)
    else:
        st.success("✅ Aucune alerte de parcours détectée sur les CVs filtrés.")
