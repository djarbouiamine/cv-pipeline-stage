"""
theme.py — Design system partagé pour toutes les pages Streamlit cv-pipeline.
Importer et appeler inject_theme() en debut de chaque page, apres set_page_config().
"""
import streamlit as st

# Palette de couleurs
BLUE   = "#4facfe"
CYAN   = "#00f2fe"
GREEN  = "#43e97b"
PINK   = "#fa709a"
YELLOW = "#f6d365"
PURPLE = "#a18cd1"
DARK1  = "#0f2027"
DARK2  = "#203a43"
DARK3  = "#2c5364"

SHARED_CSS = """
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.main .block-container { padding-top: 0.8rem !important; max-width: 1400px; }

/* ── Hero banner ─────────────────────────────────────────────────────── */
.cv-hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 18px;
    padding: 1.8rem 2.4rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.07);
    position: relative;
    overflow: hidden;
}
.cv-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(79,172,254,0.15), transparent 70%);
    border-radius: 50%;
}
.cv-hero-icon { font-size: 2.8rem; line-height: 1; z-index: 1; }
.cv-hero-body { z-index: 1; }
.cv-hero-title {
    color: #fff;
    font-size: 1.65rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.6px;
    line-height: 1.2;
}
.cv-hero-sub {
    color: rgba(255,255,255,0.55);
    font-size: 0.88rem;
    margin: 0.25rem 0 0;
}
.cv-hero-badge {
    margin-left: auto;
    background: rgba(79,172,254,0.15);
    border: 1px solid rgba(79,172,254,0.4);
    border-radius: 50px;
    padding: 0.35rem 1rem;
    color: #7ee8fa;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
    z-index: 1;
}

/* ── KPI metric cards ────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1a1f2e, #242938) !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.3rem !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4) !important;
}
[data-testid="stMetricLabel"] p {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #fff !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] > div:first-child {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px 6px;
    border: 1px solid rgba(255,255,255,0.07);
    gap: 4px;
    margin-bottom: 0.8rem;
}
button[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.2s ease !important;
    color: rgba(255,255,255,0.55) !important;
    border: none !important;
    background: transparent !important;
}
button[data-baseweb="tab"]:hover {
    color: rgba(255,255,255,0.85) !important;
    background: rgba(255,255,255,0.06) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #4facfe, #00f2fe) !important;
    color: #000 !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(79,172,254,0.45) !important;
}

/* ── Section title style ─────────────────────────────────────────────── */
.sec-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: rgba(255,255,255,0.8);
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0.3rem 0 0.7rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027 0%, #1a2a35 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.82) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSlider { padding: 0 0.3rem; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    transition: all 0.2s ease !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px) !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4facfe, #00f2fe) !important;
    color: #000 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(79,172,254,0.35) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.85) !important;
}

/* ── Download button ─────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.35) !important;
}

/* ── Inputs / Text inputs ────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 9px !important;
    color: #fff !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #4facfe !important;
    box-shadow: 0 0 0 2px rgba(79,172,254,0.2) !important;
}

/* ── Selectbox / Multiselect ─────────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 9px !important;
    color: #fff !important;
}

/* ── Dataframe ───────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] iframe {
    border-radius: 12px !important;
}
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    overflow: hidden;
}

/* ── Info / Warning / Success / Error boxes ─────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── File uploader ───────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(79,172,254,0.06) !important;
    border: 2px dashed rgba(79,172,254,0.4) !important;
    border-radius: 14px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(79,172,254,0.7) !important;
    background: rgba(79,172,254,0.1) !important;
}

/* ── Chat message bubbles ────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    padding: 0.9rem 1.1rem !important;
    margin-bottom: 0.5rem !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: rgba(79,172,254,0.1) !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background: rgba(255,255,255,0.04) !important;
}

/* ── Chat input ─────────────────────────────────────────────────────── */
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(79,172,254,0.4) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    color: #fff !important;
}

/* ── Expander ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Spinner ─────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] { color: #4facfe !important; }

/* ── Result card (CV search results) ────────────────────────────────── */
.cv-card {
    background: linear-gradient(145deg, #1a1f2e, #1e2535);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
}
.cv-card:hover {
    border-color: rgba(79,172,254,0.4);
    box-shadow: 0 6px 25px rgba(79,172,254,0.1);
}
.cv-card-rank {
    position: absolute;
    top: 1rem; right: 1.2rem;
    background: rgba(79,172,254,0.15);
    border: 1px solid rgba(79,172,254,0.3);
    border-radius: 50px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 700;
    color: #4facfe;
}
.cv-card-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.2rem;
}
.cv-card-meta {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.5);
    margin-bottom: 0.7rem;
}
.cv-score-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}
.score-high  { background: rgba(67,233,123,0.15); color: #43e97b; border: 1px solid rgba(67,233,123,0.3); }
.score-mid   { background: rgba(246,211,101,0.15); color: #f6d365; border: 1px solid rgba(246,211,101,0.3); }
.score-low   { background: rgba(250,112,154,0.15); color: #fa709a; border: 1px solid rgba(250,112,154,0.3); }
.cv-skill-tag {
    display: inline-block;
    background: rgba(79,172,254,0.12);
    color: #4facfe;
    border: 1px solid rgba(79,172,254,0.25);
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.15rem;
}
</style>
"""


def inject_theme():
    """Injecte le CSS premium dans la page Streamlit courante."""
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


def hero(icon: str, title: str, subtitle: str, badge: str = "") -> None:
    """Affiche le hero banner premium en haut de page."""
    badge_html = f'<div class="cv-hero-badge">{badge}</div>' if badge else ""
    st.markdown(f"""
<div class="cv-hero">
  <div class="cv-hero-icon">{icon}</div>
  <div class="cv-hero-body">
    <p class="cv-hero-title">{title}</p>
    <p class="cv-hero-sub">{subtitle}</p>
  </div>
  {badge_html}
</div>
""", unsafe_allow_html=True)


def sec(title: str) -> None:
    """Affiche un titre de section stylise."""
    st.markdown(f'<p class="sec-title">{title}</p>', unsafe_allow_html=True)


def score_badge(score: float) -> str:
    """Retourne le HTML d'un badge de score colore selon la valeur."""
    if score >= 70:
        cls = "score-high"
    elif score >= 50:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="cv-score-badge {cls}">{score:.1f}/100</span>'


def skill_tags(skills: list, max_n: int = 8) -> str:
    """Retourne le HTML de tags de competences."""
    tags = "".join(f'<span class="cv-skill-tag">{s}</span>' for s in skills[:max_n])
    if len(skills) > max_n:
        tags += f'<span class="cv-skill-tag">+{len(skills)-max_n}</span>'
    return tags
