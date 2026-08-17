import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import inject_theme, hero

st.set_page_config(
    page_title="CV Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

# Style the page_link buttons to look like cards
st.markdown("""
<style>
/* ── Nav card override for page_link ── */
[data-testid="stPageLink-NavLink"] {
    background: linear-gradient(145deg,#1a1f2e,#242938) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 1.5rem 1rem !important;
    text-align: center !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 0.3rem !important;
    transition: all 0.2s ease !important;
    text-decoration: none !important;
}
[data-testid="stPageLink-NavLink"]:hover {
    border-color: rgba(79,172,254,0.4) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

hero(
    "🧠",
    "CV Pipeline Intelligence",
    "Extraction • Classification • Scoring • Recherche sémantique • RAG conversationnel",
    badge="⚡ Powered by Elasticsearch + LLM",
)

# ── Navigation cards ──────────────────────────────────────────────────────────
PAGES = [
    ("📊", "Dashboard",   "KPIs & Visualisations",        "#4facfe", "pages/1_Dashboard.py"),
    ("🔎", "Recherche",   "Filtrage avancé",              "#43e97b", "pages/2_Recherche.py"),
    ("💬", "Chatbot RAG", "Assistant IA conversationnel", "#fa709a", "pages/3_Chatbot.py"),
    ("📤", "Ajouter CV",  "Upload & Indexation",          "#f6d365", "pages/4_Ajouter_CV.py"),
]

cols = st.columns(4, gap="medium")
for col, (icon, name, desc, color, path) in zip(cols, PAGES):
    with col:
        # Card shell (visual only)
        st.markdown(f"""
<div style="
  background:linear-gradient(145deg,#1a1f2e,#242938);
  border:1px solid rgba(255,255,255,0.08);
  border-top:4px solid {color};
  border-radius:14px;
  padding:1.5rem 1rem 0.6rem;
  text-align:center;
  margin-bottom:0.3rem;
">
  <div style="font-size:2.4rem;margin-bottom:0.4rem">{icon}</div>
  <div style="color:#fff;font-weight:700;font-size:1.05rem;margin-bottom:0.15rem">{name}</div>
  <div style="color:rgba(255,255,255,0.38);font-size:0.77rem;margin-bottom:0.5rem">{desc}</div>
</div>""", unsafe_allow_html=True)
        # Functional navigation button
        st.page_link(path, label=f"Ouvrir {name}", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Info bar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.07);
  border-left:4px solid #4facfe;
  border-radius:12px;
  padding:1rem 1.5rem;
  color:rgba(255,255,255,0.6);
  font-size:0.84rem;
  line-height:1.8;
">
  <strong style="color:rgba(255,255,255,0.9)">🛡️ Principe anti-hallucination</strong> &mdash;
  Tous les calculs finaux sont du <strong style="color:#4facfe">Python déterministe</strong>.
  Le LLM évalue uniquement la qualité par unité (1 projet, 1 certification à la fois).
  Les contraintes utilisateur (compétences, score min) sont appliquées comme
  <strong style="color:#43e97b">hard filters Elasticsearch</strong> avant tout appel LLM.
</div>
""", unsafe_allow_html=True)
