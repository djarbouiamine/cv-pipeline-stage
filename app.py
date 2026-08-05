import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Explorateur CV",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🗂️ Explorateur de CVs – Pipeline cv-pipeline-stage")
st.markdown(
    """
Cette application Streamlit possède **quatre pages** :
- 📊 **Dashboard** – KPI et visualisations globales.
- 🔎 **Recherche** – Filtrage avancé des CVs.
- 💬 **Chatbot RAG** – Assistant conversationnel.
- 📝 **Ajouter CV** – Upload, extraction et indexation.

Utilisez le menu de navigation à gauche pour accéder à chaque fonctionnalité.
"""
)

# Sidebar navigation (Streamlit auto‑loads scripts in `pages/`)
st.sidebar.header("Navigation")
pages = {
    "📊 Dashboard": "pages/1_Dashboard.py",
    "🔎 Recherche": "pages/2_Recherche.py",
    "💬 Chatbot RAG": "pages/3_Chatbot.py",
    "📝 Ajouter CV": "pages/4_Ajouter_CV.py",
}
for name, path in pages.items():
    st.sidebar.caption(f"{name} → {Path(path).name}")

# Aucun code supplémentaire nécessaire : chaque fichier `pages/*.py`
# contient sa propre logique (st.title, visualisations, etc.).
