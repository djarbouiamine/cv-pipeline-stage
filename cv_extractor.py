import json
import time
import os
import re
import requests
import numpy as np
from datetime import date
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    # Fallback if the heavy dependency cannot be imported (e.g., missing torch)
    # Provide a dummy class that mimics the encode method returning zero vectors.
    class _DummySentenceTransformer:
        def encode(self, text, normalize_embeddings=False):
            # Return a list of 384 zeros (dimension of the original model)
            return [0.0] * 384
    SentenceTransformer = _DummySentenceTransformer
    print(f"⚠️ SentenceTransformer import failed ({e}); using dummy embedding model.")

# Charger le fichier .env s'il existe
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()
                except Exception:
                    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ── Clients (créés seulement si la clé existe) ──────────────────
client_groq = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        client_groq = Groq(api_key=GROQ_API_KEY)
    except ImportError:
        print("⚠️ SDK groq non installé (pip install groq).")

client_gemini = None
if GEMINI_API_KEY:
    try:
        from google import genai
        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
    except ImportError:
        print("⚠️ SDK google-genai non installé.")

AVAILABLE_KEYS = {
    "groq": bool(GROQ_API_KEY) and client_groq is not None,
    "mistral": bool(MISTRAL_API_KEY),
    "openrouter": bool(OPENROUTER_API_KEY),
    "gemini": bool(GEMINI_API_KEY) and client_gemini is not None,
}

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-20b",
    "mistral": "mistral-small-latest",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "gemini": "gemini-2.5-flash",
}


def _get_env_float(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"⚠️ Valeur invalide pour {name} dans .env, utilisation du défaut ({default})")
        return default


# ── Poids du score de qualité (configurables via .env) ──────────
QUALITY_WEIGHTS = {
    "diplome": _get_env_float("QUALITY_WEIGHT_DIPLOME", 25),
    "certifications": _get_env_float("QUALITY_WEIGHT_CERTIFICATIONS", 20),
    "diversite_technique": _get_env_float("QUALITY_WEIGHT_TECH", 20),
    "projets": _get_env_float("QUALITY_WEIGHT_PROJETS", 25),
    "langues": _get_env_float("QUALITY_WEIGHT_LANGUES", 10),
}

# Normalisation automatique : peu importe ce que l'utilisateur met dans le .env
# (même si ça ne fait pas 100 au total), le ratio entre les poids est conservé
# et le total est toujours ramené à 100%.
_total_w = sum(QUALITY_WEIGHTS.values())
if _total_w <= 0:
    raise ValueError("La somme des QUALITY_WEIGHT_* dans .env doit être > 0")
QUALITY_WEIGHTS = {k: (v / _total_w) * 100 for k, v in QUALITY_WEIGHTS.items()}


# ── Bascule automatique entre providers (configurable via .env) ──
FALLBACK_ENABLED = os.environ.get("AUTO_FALLBACK", "false").strip().lower() in ("1", "true", "yes")
FALLBACK_ORDER = [p.strip() for p in os.environ.get(
    "FALLBACK_ORDER", "groq,openrouter,mistral,gemini"
).split(",") if p.strip()]
MAX_AUTO_WAIT_S = _get_env_float("GROQ_MAX_AUTO_WAIT_S", 1200)


class QuotaTimeoutError(Exception):
    """Levée quand le délai d'attente du quota dépasse le plafond configuré."""
    pass


# ---------------------------------------------------------------------------
# Normalisation des noms de domaines par embeddings sémantiques.
# But : éviter que le LLM invente un nom de domaine légèrement différent à
# chaque CV ("Cybersécurité" vs "Cybersecurity" vs "Sécurité informatique"),
# ce qui provoquerait une explosion du nombre de champs dans Elasticsearch
# (chaque clé de scores_categories devient un champ mappé).
# ---------------------------------------------------------------------------

# Initialise le modèle d'embedding, ou un dummy en cas d'erreur précédente
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
SIMILARITE_SEUIL = _get_env_float("DOMAINE_SIMILARITE_SEUIL", 0.75)

# Seuil dédié pour comparer le domaine d'une EXPÉRIENCE (libellé souvent
# court et très spécifique, ex: "Automated Water Meter Testing") aux
# domaines du candidat. Volontairement plus bas que SIMILARITE_SEUIL
# (normalisation des noms de domaines entre CVs) car ici on compare des
# textes de nature différente — un stage précis vs un domaine large — donc
# la similarité brute est structurellement plus faible même quand le lien
# est réel.
# Plancher de similarité : en dessous, le poids est forcé à 0 plutôt que de
# garder un résidu de bruit non nul (ex: similarité 0.05 entre deux domaines
# vraiment sans rapport ne doit pas contribuer, même marginalement).
# Volontairement bas — ce n'est plus un seuil de décision inclus/exclus,
# juste un filtre anti-bruit résiduel.
SEUIL_PLANCHER_EXPERIENCE = _get_env_float("EXPERIENCE_SEUIL_PLANCHER", 0.15)

DOMAINES_REF_PATH = "output/.domaines_reference.json"
DOMAINES_REFERENCE = []      # noms de domaines validés (str)
DOMAINES_EMBEDDINGS = []     # vecteurs numpy correspondants, même index


def _charger_domaines_reference():
    """Recharge la liste de référence depuis le disque si elle existe."""
    global DOMAINES_REFERENCE, DOMAINES_EMBEDDINGS
    if os.path.exists(DOMAINES_REF_PATH):
        try:
            with open(DOMAINES_REF_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            DOMAINES_REFERENCE = data.get("noms", [])
            DOMAINES_EMBEDDINGS = [np.array(e) for e in data.get("embeddings", [])]
            print(f"📂 {len(DOMAINES_REFERENCE)} domaines de référence chargés depuis {DOMAINES_REF_PATH}")
        except Exception as e:
            print(f"⚠️ Impossible de charger {DOMAINES_REF_PATH} : {e}")


def _sauver_domaines_reference():
    """Sauvegarde la liste de référence sur disque (noms + embeddings)."""
    os.makedirs("output", exist_ok=True)
    try:
        with open(DOMAINES_REF_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "noms": DOMAINES_REFERENCE,
                "embeddings": [e.tolist() for e in DOMAINES_EMBEDDINGS],
            }, f, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Impossible de sauvegarder {DOMAINES_REF_PATH} : {e}")


_charger_domaines_reference()


def normaliser_domaine(nom_domaine):
    """
    Compare un nom de domaine aux domaines déjà connus (similarité cosinus
    sur les embeddings). Renvoie le nom existant si assez proche, sinon
    ajoute ce nouveau nom à la référence et le renvoie tel quel.
    """
    if not nom_domaine:
        return nom_domaine

    if not DOMAINES_REFERENCE:
        emb = embedding_model.encode(nom_domaine)
        DOMAINES_REFERENCE.append(nom_domaine)
        DOMAINES_EMBEDDINGS.append(emb)
        return nom_domaine

    nouveau_emb = embedding_model.encode(nom_domaine)
    matrice = np.array(DOMAINES_EMBEDDINGS)
    similarites = np.dot(matrice, nouveau_emb) / (
        np.linalg.norm(matrice, axis=1) * np.linalg.norm(nouveau_emb) + 1e-10
    )
    meilleur_idx = int(np.argmax(similarites))

    if similarites[meilleur_idx] >= SIMILARITE_SEUIL:
        return DOMAINES_REFERENCE[meilleur_idx]

    DOMAINES_REFERENCE.append(nom_domaine)
    DOMAINES_EMBEDDINGS.append(nouveau_emb)
    return nom_domaine


def normaliser_categories(data):
    """
    Applique normaliser_domaine() à chaque clé de scores_categories.
    Si deux clés du même CV se retrouvent fusionnées sur le même nom
    normalisé, on garde le score le plus élevé des deux.
    """
    if "scores_categories" not in data or not data["scores_categories"]:
        return data

    nouveau_dict = {}
    for domaine, score in data["scores_categories"].items():
        domaine_normalise = normaliser_domaine(domaine)
        nouveau_dict[domaine_normalise] = max(score, nouveau_dict.get(domaine_normalise, 0))

    data["scores_categories"] = nouveau_dict
    if nouveau_dict:
        data["categorie_principale"] = max(nouveau_dict, key=nouveau_dict.get)

    return data


# ---------------------------------------------------------------------------
# Calcul des années d'expérience — dates extraites brutes par le LLM,
# parsing + calcul de durée faits en Python (plus fiable que de demander
# au LLM de compter lui-même).
# ---------------------------------------------------------------------------

MOIS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
    "jan": 1, "fév": 2, "fev": 2, "avr": 4, "juil": 7, "sept": 9,
    "oct": 10, "nov": 11, "déc": 12, "dec": 12,
}
MOIS_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

MOTS_PRESENT = ["present", "présent", "en cours", "actuel", "aujourd'hui", "now", "current", "ongoing"]


def parse_date_cv(date_brute):
    """
    Convertit une date brute (texte du CV) en objet date() Python.
    Retourne None si la date est vide ou illisible (fallback silencieux,
    ne fait jamais planter le pipeline pour une date mal formatée).
    Gère "present"/"en cours" en renvoyant la date du jour.
    """
    if not date_brute:
        return None
    texte = str(date_brute).strip().lower()
    if not texte:
        return None

    if any(mot in texte for mot in MOTS_PRESENT):
        return date.today()

    # Format "MM/YYYY" ou "MM-YYYY"
    m = re.match(r"^(\d{1,2})[/\-](\d{4})$", texte)
    if m:
        mois, annee = int(m.group(1)), int(m.group(2))
        if 1 <= mois <= 12:
            return date(annee, mois, 1)

    # Format "Mois AAAA" (français ou anglais, ex: "janvier 2022", "Jan 2021")
    m = re.match(r"^([a-zéû]+)\.?\s+(\d{4})$", texte)
    if m:
        mot_mois, annee = m.group(1), int(m.group(2))
        mois = MOIS_FR.get(mot_mois) or MOIS_EN.get(mot_mois)
        if mois:
            return date(annee, mois, 1)

    # Format "AAAA" seul
    m = re.match(r"^(\d{4})$", texte)
    if m:
        return date(int(m.group(1)), 1, 1)

    # Format déjà ISO "AAAA-MM-JJ" ou "AAAA-MM"
    m = re.match(r"^(\d{4})-(\d{1,2})", texte)
    if m:
        annee, mois = int(m.group(1)), int(m.group(2))
        if 1 <= mois <= 12:
            return date(annee, mois, 1)

    return None  # non reconnu — on ne devine pas, on laisse l'appelant gérer


def _similarite_domaine(domaine_experience, domaine_principal):
    """
    Similarité cosinus brute (0-1) entre le domaine d'une expérience et le
    domaine principal du candidat. Ne tranche PAS inclus/exclus — renvoie
    la mesure continue, utilisée ensuite comme poids de pertinence.
    """
    if not domaine_experience or not domaine_principal:
        return 0.0
    emb_exp = embedding_model.encode(domaine_experience)
    emb_principal = embedding_model.encode(domaine_principal)
    similarite = np.dot(emb_exp, emb_principal) / (
        np.linalg.norm(emb_exp) * np.linalg.norm(emb_principal) + 1e-10
    )
    return float(max(0.0, similarite))  # jamais négatif, plus simple à interpréter comme poids


def _date_vers_index_mois(d):
    """Convertit une date en index mois absolu (ex: juillet 2026 -> 2026*12+7)."""
    return d.year * 12 + d.month


def calculate_experience_annees(data):
    """
    Calcule le nombre d'années d'expérience PONDÉRÉE par pertinence au
    domaine principal du candidat. Chaque expérience contribue à hauteur de
    sa similarité sémantique au domaine principal (0 à 1), pas de façon
    binaire — un stage à moitié pertinent compte pour moitié de sa durée.

    Pour les mois couverts par plusieurs expériences en parallèle, on garde
    le POIDS MAXIMUM du mois (pas la somme) pour ne pas compter deux fois la
    même période.

    Renvoie (annees_experience, details) où details est la liste des
    expériences enrichies d'un champ "poids_pertinence" (0-1, arrondi),
    utile pour comprendre a posteriori d'où vient le chiffre final.
    """
    experiences = data.get("experiences_pro") or []
    domaine_principal = data.get("categorie_principale") or ""

    poids_par_mois = {}  # index_mois -> poids max observé pour ce mois
    details = []

    for exp in experiences:
        if not isinstance(exp, dict):
            continue

        domaine_exp = exp.get("domaine") or ""
        similarite = _similarite_domaine(domaine_exp, domaine_principal)
        poids = similarite if similarite >= SEUIL_PLANCHER_EXPERIENCE else 0.0

        detail = dict(exp)
        detail["poids_pertinence"] = round(poids, 2)

        d_debut = parse_date_cv(exp.get("date_debut"))
        d_fin = parse_date_cv(exp.get("date_fin"))
        if d_debut is None or d_fin is None or d_fin < d_debut or poids == 0.0:
            # Date illisible/incohérente, ou poids nul -> pas de contribution,
            # mais l'expérience reste visible dans "details" avec son poids.
            details.append(detail)
            continue

        m_debut, m_fin = _date_vers_index_mois(d_debut), _date_vers_index_mois(d_fin)
        for m in range(m_debut, m_fin + 1):  # bornes incluses
            poids_par_mois[m] = max(poids_par_mois.get(m, 0.0), poids)

        details.append(detail)

    mois_total_pondere = sum(poids_par_mois.values())
    annees = round(mois_total_pondere / 12, 1)

    return annees, details


# ---------------------------------------------------------------------------
# Détection d'alertes de parcours (trous chronologiques, chevauchements).
# Purement informatif : on signale, on ne juge pas — un chevauchement peut
# être parfaitement légitime (stage + projet freelance en parallèle).
# ---------------------------------------------------------------------------

SEUIL_TROU_MOIS = int(_get_env_float("ALERTE_TROU_MOIS", 6))
SEUIL_CHEVAUCHEMENT_MOIS = int(_get_env_float("ALERTE_CHEVAUCHEMENT_MOIS", 2))


def _label_experience(exp_tuple):
    """Construit un libellé lisible 'Poste (Domaine)' pour les messages d'alerte."""
    _, _, poste, domaine = exp_tuple
    if domaine:
        return f"{poste} ({domaine})"
    return poste


def calculate_alertes_parcours(data, seuil_trou_mois=None, seuil_chevauchement_mois=None):
    """
    Analyse le parcours du candidat et signale :
    - un trou chronologique >= seuil_trou_mois dans la timeline globale
      (calculé sur l'UNION des périodes couvertes, pas paire par paire —
      sinon un trou "comblé" par une 3e expérience serait signalé à tort)
    - un chevauchement entre deux expériences dont la durée réelle
      d'INTERSECTION (pas juste l'écart de dates de début) >= seuil_chevauchement_mois
      Comparaison de TOUTES les paires d'expériences (pas seulement les
      paires consécutives), car une expérience courte peut être entièrement
      imbriquée dans une plus longue sans lui être adjacente dans le tri
      par date de début (ex: un stage de 3 mois au milieu d'un poste
      associatif d'1 an) — comparer seulement les paires consécutives
      donnerait alors une durée de chevauchement fausse (calculée sur
      l'écart de dates plutôt que sur la vraie durée commune).
    Ne considère que les expériences dont les deux dates sont parsables.
    Renvoie une liste de messages texte, prête à afficher/stocker.
    """
    seuil_trou_mois = seuil_trou_mois if seuil_trou_mois is not None else SEUIL_TROU_MOIS
    seuil_chevauchement_mois = (
        seuil_chevauchement_mois if seuil_chevauchement_mois is not None else SEUIL_CHEVAUCHEMENT_MOIS
    )
    experiences = data.get("experiences_pro") or []

    items = []
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        d_debut = parse_date_cv(exp.get("date_debut"))
        d_fin = parse_date_cv(exp.get("date_fin"))
        if d_debut is not None and d_fin is not None and d_fin >= d_debut:
            items.append((
                d_debut, d_fin,
                exp.get("poste") or "expérience non nommée",
                exp.get("domaine") or "",
            ))

    alertes = []

    # --- Chevauchements : intersection réelle, sur TOUTES les paires ---
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            d1, f1, poste1, domaine1 = items[i]
            d2, f2, poste2, domaine2 = items[j]

            inter_debut = max(d1, d2)
            inter_fin = min(f1, f2)
            if inter_debut > inter_fin:
                continue  # pas de recouvrement du tout entre ces deux-là

            duree_chevauchement = _date_vers_index_mois(inter_fin) - _date_vers_index_mois(inter_debut) + 1
            if duree_chevauchement >= seuil_chevauchement_mois:
                label1 = _label_experience((d1, f1, poste1, domaine1))
                label2 = _label_experience((d2, f2, poste2, domaine2))
                alertes.append(
                    f"Chevauchement de {duree_chevauchement} mois entre \"{label1}\" et \"{label2}\""
                )

    # --- Trous : sur l'UNION des périodes couvertes (timeline fusionnée) ---
    items_tries = sorted(items, key=lambda it: _date_vers_index_mois(it[0]))

    blocs = []  # chaque bloc : {"debut_idx", "fin_idx", "label_debut", "label_fin"}
    for d, f, poste, domaine in items_tries:
        d_idx, f_idx = _date_vers_index_mois(d), _date_vers_index_mois(f)
        if blocs and d_idx <= blocs[-1]["fin_idx"] + 1:
            if f_idx > blocs[-1]["fin_idx"]:
                blocs[-1]["fin_idx"] = f_idx
                blocs[-1]["label_fin"] = (poste, domaine)
            # sinon : cette expérience est entièrement contenue dans le bloc
            # existant, elle ne change pas sa borne de fin
        else:
            blocs.append({
                "debut_idx": d_idx, "fin_idx": f_idx,
                "label_debut": (poste, domaine), "label_fin": (poste, domaine),
            })

    for i in range(len(blocs) - 1):
        gap = blocs[i + 1]["debut_idx"] - blocs[i]["fin_idx"] - 1
        if gap >= seuil_trou_mois:
            poste_fin, domaine_fin = blocs[i]["label_fin"]
            poste_debut, domaine_debut = blocs[i + 1]["label_debut"]
            label_actuelle = _label_experience((None, None, poste_fin, domaine_fin))
            label_suivante = _label_experience((None, None, poste_debut, domaine_debut))
            alertes.append(
                f"Trou de {gap} mois entre \"{label_actuelle}\" et \"{label_suivante}\""
            )

    return alertes


# ---------------------------------------------------------------------------
# Score de qualité du CV
# ---------------------------------------------------------------------------

def calculate_diploma_score(diplomes):
    """Score 0-100 basé sur le diplôme le plus élevé mentionné (mots-clés FR/EN)."""
    if not diplomes:
        return 20
    text = " ".join(diplomes).lower()

    doctorat_kw = ["doctorat", "phd", "docteur", "thèse", "these"]
    ingenieur_master_kw = [
        "ingénieur", "ingenieur", "ingénierie", "ingenierie",
        "engineering", "engineer", "master", "mastère", "mastere",
        "bac+5", "msc", "m.sc", "master's degree", "masters degree",
    ]
    licence_kw = [
        "licence", "bac+3", "bachelor", "bachelor's degree",
        "computer science", "informatique",
    ]
    bts_dut_kw = ["bts", "dut", "bac+2"]

    if any(k in text for k in doctorat_kw):
        return 100
    if any(k in text for k in ingenieur_master_kw):
        return 85
    if any(k in text for k in licence_kw):
        return 60
    if any(k in text for k in bts_dut_kw):
        return 40
    return 30


def calculate_certif_score(data):
    """
    Score 0-100 agrégé à partir des évaluations qualitatives du LLM
    (evaluation_certifications -> score_qualite par certification).
    Agrégation : 50% moyenne générale + 50% meilleure certification.
    """
    evaluations = data.get("evaluation_certifications") or []
    scores = []
    for ev in evaluations:
        if not isinstance(ev, dict):
            continue
        try:
            scores.append(float(ev.get("score_qualite", 0)))
        except (TypeError, ValueError):
            continue

    if scores:
        moyenne = sum(scores) / len(scores)
        meilleure = max(scores)
        return round(min(100, moyenne * 0.5 + meilleure * 0.5), 1)

    certifications = data.get("certifications") or []
    if not certifications:
        return 0
    return min(100, len(certifications) * 20)


def calculate_tech_score(data):
    """Score 0-100 basé sur la diversité technique (langages, frameworks, etc.)."""
    total = 0
    for field in ["langages", "frameworks", "bases_de_donnees", "outils_devops", "technologies"]:
        values = data.get(field) or []
        if isinstance(values, list):
            total += len(values)
    return min(100, total * 4)


def calculate_projet_score(data):
    """
    Score 0-100 agrégé à partir des évaluations qualitatives du LLM
    (evaluation_projets -> score_importance par projet).
    Agrégation : 50% moyenne générale + 50% meilleur projet.
    """
    evaluations = data.get("evaluation_projets") or []
    scores = []
    for ev in evaluations:
        if not isinstance(ev, dict):
            continue
        try:
            scores.append(float(ev.get("score_importance", 0)))
        except (TypeError, ValueError):
            continue

    if scores:
        moyenne = sum(scores) / len(scores)
        meilleur = max(scores)
        return round(min(100, moyenne * 0.5 + meilleur * 0.5), 1)

    projets = data.get("projets") or []
    nb = len(projets) if isinstance(projets, list) else 0
    return min(100, nb * 15)


def map_niveau_langue(niveau_brut):
    """Convertit le niveau de langue brut (texte du CV) en score 0-100."""
    if not niveau_brut:
        return 50
    n = str(niveau_brut).lower()

    natif = ["natif", "maternelle", "bilingue", "native", "mother tongue"]
    courant = ["courant", "avancé", "avance", "advanced", "fluent", "c1", "c2"]
    intermediaire = ["intermédiaire", "intermediaire", "intermediate", "professionnel", "working", "b1", "b2"]
    debutant = ["notions", "scolaire", "débutant", "debutant", "basique", "basic", "elementary", "beginner", "a1", "a2"]

    if any(k in n for k in natif):
        return 100
    if any(k in n for k in courant):
        return 85
    if any(k in n for k in intermediaire):
        return 60
    if any(k in n for k in debutant):
        return 30
    return 50


def calculate_langue_score(data):
    """Score 0-100 basé sur les niveaux de langue extraits par le LLM."""
    evaluations = data.get("evaluation_langues") or []
    scores = []
    for ev in evaluations:
        if isinstance(ev, dict):
            scores.append(map_niveau_langue(ev.get("niveau_brut", "")))

    if scores:
        return round(sum(scores) / len(scores), 1)

    langues = data.get("langues") or []
    if not langues:
        return 0
    return min(100, len(langues) * 25)


def calculate_quality_score(data):
    """
    Calcule le score de qualité global (0-100) du CV.
    Pondération (configurable via .env, voir QUALITY_WEIGHTS) :
    Diplôme | Certifications | Diversité technique | Projets | Langues
    """
    score_diplome = calculate_diploma_score(data.get("diplomes") or [])
    score_certif = calculate_certif_score(data)
    score_tech = calculate_tech_score(data)
    score_projet = calculate_projet_score(data)
    score_langue = calculate_langue_score(data)

    score_global = (
        score_diplome * QUALITY_WEIGHTS["diplome"] / 100
        + score_certif * QUALITY_WEIGHTS["certifications"] / 100
        + score_tech * QUALITY_WEIGHTS["diversite_technique"] / 100
        + score_projet * QUALITY_WEIGHTS["projets"] / 100
        + score_langue * QUALITY_WEIGHTS["langues"] / 100
    )

    return {
        "score_qualite_globale": round(score_global, 1),
        "score_qualite_globale_sur_10": round(score_global / 10, 1),
        "details_score": {
            "diplome": score_diplome,
            "certifications": score_certif,
            "diversite_technique": score_tech,
            "projets": score_projet,
            "langues": score_langue,
        },
    }


def calculate_domain_scores_ponderes(data, score_qualite_globale):
    """Score final par domaine = 60% pertinence (LLM) + 40% qualité globale du profil."""
    scores_categories = data.get("scores_categories") or {}
    result_100 = {}
    result_10 = {}
    for domaine, pertinence in scores_categories.items():
        try:
            pertinence = float(pertinence)
        except (TypeError, ValueError):
            pertinence = 0.0
        score_final = pertinence * 0.6 + score_qualite_globale * 0.4
        result_100[domaine] = round(score_final, 1)
        result_10[domaine] = round(score_final / 10, 1)
    return result_100, result_10


# ---------------------------------------------------------------------------
# Conversion objet -> liste nested, pour correspondre au mapping Elasticsearch
# "nested" ({"domaine": ..., "score": ...} par entrée). Doit toujours être
# appelée en TOUTE DERNIÈRE étape : normaliser_categories() et
# calculate_domain_scores_ponderes() ont besoin du format objet (dict) pour
# fonctionner, donc cette conversion arrive après elles.
# ---------------------------------------------------------------------------
def convertir_en_liste_nested(valeur):
    """
    Convertit un dict {domaine: score} en liste [{"domaine": ..., "score": ...}].
    Ne fait rien si c'est déjà une liste (ou vide/None) — sécurité en cas de
    double appel ou de format déjà correct.
    """
    if isinstance(valeur, dict):
        return [{"domaine": d, "score": s} for d, s in valeur.items()]
    return valeur or []


def get_retry_delay(err, default=10.0):
    """Extrait le délai d'attente recommandé depuis les headers HTTP si possible."""
    try:
        response = getattr(err, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                return float(retry_after)
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# Prompt partagé — utilisé par cv_extractor ET cv_comparator, pour que la
# comparaison entre providers porte sur le MÊME prompt / format de sortie.
# ---------------------------------------------------------------------------
def build_prompt(text, domaines_existants=None):
    """
    domaines_existants : liste de noms de domaines déjà utilisés pour
    d'autres CVs (par défaut, la référence globale DOMAINES_REFERENCE,
    mise à jour au fil de l'extraction). Injectée dans le prompt pour que
    le LLM réutilise le même vocabulaire au lieu d'inventer une variante.
    """
    if domaines_existants is None:
        domaines_existants = DOMAINES_REFERENCE

    bloc_domaines = ""
    if domaines_existants:
        bloc_domaines = f"""
Domaines déjà utilisés pour d'autres candidats (RÉUTILISE EXACTEMENT la
même écriture si un domaine correspond au profil de ce candidat ; ne crée
un nouveau nom que si aucun de ceux-ci ne convient vraiment) :
{", ".join(domaines_existants)}
"""

    return f"""
Tu es un expert RH. Analyse ce CV et extrais les informations dans un JSON.
Retourne UNIQUEMENT le JSON, rien d'autre.

IMPORTANT : Le texte peut être mal formaté à cause de l'OCR.
Corrige les erreurs évidentes.

RÈGLE ANTI-INVENTION (très importante) :
N'invente JAMAIS d'information absente du CV. Si une donnée n'existe pas,
laisse le champ vide ([], "" ou null). Les descriptions doivent être basées
UNIQUEMENT sur ce qui est écrit dans le CV, jamais complétées ou enrichies.

Pour la classification :
- Identifie TOI-MÊME les 2 ou 3 domaines professionnels les plus pertinents
  pour ce candidat (ne choisis pas dans une liste fixe, trouve les domaines
  qui correspondent vraiment à son profil)
{bloc_domaines}
- Donne un score de pertinence de 0 à 100 pour chaque domaine identifié
- Indique le domaine principal (le plus haut score)

Pour les projets :
- Pour CHAQUE projet, écris une description en UNE SEULE LIGNE, basée
  uniquement sur ce qui est dans le CV (pas d'invention de détails)
- Pour CHAQUE projet, évalue individuellement son importance/complexité
  (score_importance, 0 à 100) selon :
  • la complexité technique et les technologies utilisées
  • l'impact ou l'envergure (nombre d'utilisateurs, taille d'équipe, contexte
    professionnel vs académique/personnel)
  • le rôle du candidat (lead, contributeur principal, simple participant)

Pour les certifications :
- Pour CHAQUE certification, évalue sa qualité/reconnaissance
  (score_qualite, 0 à 100) selon :
  • la réputation de l'organisme émetteur (AWS/Google/Microsoft/Cisco/PMI
    etc. = élevé ; MOOC générique non précisé = plus faible)
  • la pertinence par rapport au profil du candidat
  • le niveau indiqué (fondamental vs avancé/expert), UNIQUEMENT si écrit
    dans le CV — ne devine pas un niveau non mentionné

Pour les langues :
- Pour CHAQUE langue, extrais le niveau EXACTEMENT comme écrit dans le CV
  (ex: "courant", "B2", "langue maternelle", "notions") dans "niveau_brut".
  N'évalue pas de score toi-même pour les langues, contente-toi d'extraire
  le texte tel quel.

Pour les expériences professionnelles (stages, emplois, alternances) :
- Liste CHAQUE expérience séparément dans "experiences_pro"
- N'inclus QUE du vrai travail professionnel : stage en entreprise, emploi,
  alternance, consulting. N'inclus JAMAIS un rôle associatif, bénévole, un
  bureau/poste dans un club étudiant ou une organisation (ex: "Community
  manager" dans un chapitre IEEE, "Membre" ou "Responsable" d'une
  association, d'un club robotique, d'un bureau des étudiants...) — même
  si ce rôle a un titre qui ressemble à un poste et des dates précises.
  Ces activités ne sont PAS de l'expérience professionnelle.
- "date_debut" et "date_fin" : recopie le texte de date EXACTEMENT comme
  écrit dans le CV (ex: "2021", "Jan 2021", "01/2021", "janvier 2022").
  Si le poste est en cours, mets "present" pour "date_fin".
  Ne calcule AUCUNE durée toi-même, ne convertis aucune date.
- "domaine" : le domaine professionnel de CETTE expérience précise
  (ex: "développement web", "réseaux", "marketing") — peut être différent
  du domaine principal du candidat si un stage était hors-sujet.

RAPPEL : pour les projets, certifications et langues, donne uniquement des
évaluations INDIVIDUELLES. Ne calcule AUCUN score global, AUCUNE moyenne
toi-même — l'agrégation finale est faite ensuite par le code Python.

Format attendu :
{{
    "nom": "...",
    "email": "...",
    "telephone": "...",
    "linkedin": "...",
    "localisation": "...",
    "scores_categories": {{
        "Domaine trouvé par l'IA 1": 0,
        "Domaine trouvé par l'IA 2": 0
    }},
    "categorie_principale": "le domaine avec le score le plus élevé",
    "technologies": ["...", "..."],
    "langages": ["...", "..."],
    "frameworks": ["...", "..."],
    "bases_de_donnees": ["...", "..."],
    "outils_devops": ["...", "..."],
    "projets": ["nom du projet 1", "nom du projet 2"],
    "evaluation_projets": [
        {{"nom": "nom du projet 1", "description": "une seule ligne", "score_importance": 0}}
    ],
    "diplomes": ["...", "..."],
    "certifications": ["...", "..."],
    "evaluation_certifications": [
        {{"nom": "nom de la certification", "score_qualite": 0}}
    ],
    "langues": ["...", "..."],
    "evaluation_langues": [
        {{"langue": "...", "niveau_brut": "tel qu'écrit dans le CV"}}
    ],
    "experiences_pro": [
        {{"poste": "...", "domaine": "...", "date_debut": "tel qu'écrit", "date_fin": "tel qu'écrit ou present"}}
    ]
}}

Voici le CV :
{text}
"""


# ---------------------------------------------------------------------------
# Appels bruts par provider — retournent un dict JSON déjà parsé.
# ---------------------------------------------------------------------------
def _call_groq(prompt, model):
    if not client_groq:
        raise RuntimeError("Client Groq non configuré (GROQ_API_KEY manquante)")
    completion = client_groq.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = completion.choices[0].message.content.strip()
    print("=== RAW GROQ CONTENT ===")
    print(repr(content[:1000]))
    return json.loads(content)


def _call_mistral(prompt, model):
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY manquante")
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post("https://api.mistral.ai/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} : {resp.text[:200]}")
    resp.encoding = "utf-8"
    content = resp.json()["choices"][0]["message"]["content"]
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def _call_openrouter(prompt, model):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY manquante")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "cv-pipeline-stage",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} : {resp.text[:200]}")
    resp.encoding = "utf-8"
    content = resp.json()["choices"][0]["message"]["content"]
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def _call_gemini(prompt, model):
    if not client_gemini:
        raise RuntimeError("Client Gemini non configuré (GEMINI_API_KEY manquante)")
    from google.genai import types
    config = types.GenerateContentConfig(response_mime_type="application/json")
    response = client_gemini.models.generate_content(model=model, contents=prompt, config=config)
    result = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(result)


PROVIDER_FUNCTIONS = {
    "groq": _call_groq,
    "mistral": _call_mistral,
    "openrouter": _call_openrouter,
    "gemini": _call_gemini,
}


def call_llm_structured(prompt: str, provider_order: list = None) -> dict:
    """
    Appelle le premier provider disponible dans *provider_order* et retourne
    un dict JSON parsé.

    Conçu pour des appels courts (classification, routage) — PAS pour
    l'extraction complète de CV (utiliser extract_cv_data() pour ça).

    Comportement par provider :
    - Quota / rate-limit (429)   → passe directement au provider suivant.
    - Erreur transitoire (5xx)   → 2 retries avec backoff exponentiel.
    - Autre erreur               → passe au provider suivant.
    - Tous providers épuisés     → lève RuntimeError.

    provider_order : liste ordonnée de providers à essayer.
                     Défaut : ["groq", "openrouter", "mistral", "gemini"]
    """
    if provider_order is None:
        provider_order = ["groq", "openrouter", "mistral", "gemini"]

    last_error = None
    for provider in provider_order:
        if not AVAILABLE_KEYS.get(provider):
            continue
        call_fn = PROVIDER_FUNCTIONS[provider]
        model = DEFAULT_MODELS[provider]
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return call_fn(prompt, model)
            except Exception as e:
                last_error = e
                if _is_quota_error(e):
                    # Quota épuisé → inutile de réessayer, on passe au suivant
                    break
                elif _is_transient_server_error(e):
                    if attempt < max_retries - 1:
                        time.sleep((2 ** attempt) * 2.0)
                        continue
                    break
                else:
                    # Erreur non récupérable pour ce provider → suivant
                    break

    raise RuntimeError(
        f"call_llm_structured : tous les providers ont échoué. "
        f"Dernière erreur : {last_error}"
    )


def _is_quota_error(e):
    err_str = str(e).lower()
    if hasattr(e, "status_code") and getattr(e, "status_code") == 429:
        return True
    return any(k in err_str for k in ["429", "quota", "rate limit", "resource_exhausted", "too many requests"])


def _is_transient_server_error(e):
    err_str = str(e).lower()
    return any(k in err_str for k in ["500", "502", "503", "504", "unavailable"])


# ---------------------------------------------------------------------------
# Extraction principale — provider au choix de l'utilisateur.
# ---------------------------------------------------------------------------
def extract_cv_data(text, provider="groq", model=None):
    """
    Prend le texte brut d'un CV, extrait via le provider demandé,
    normalise les noms de domaines (embeddings) et enrichit avec le score
    de qualité calculé par Python.

    provider : "groq" (défaut), "mistral", "openrouter" ou "gemini"

    Lève QuotaTimeoutError si le quota est dépassé au-delà du plafond
    MAX_AUTO_WAIT_S (au lieu d'attendre indéfiniment) — voir extract_cv_data_auto
    pour la bascule automatique vers un autre provider dans ce cas.
    """
    if provider not in PROVIDER_FUNCTIONS:
        raise ValueError(f"Provider inconnu : '{provider}'. Choix possibles : {list(PROVIDER_FUNCTIONS)}")
    if not AVAILABLE_KEYS.get(provider):
        raise RuntimeError(f"Clé API manquante ou client non initialisé pour '{provider}' (vérifie ton .env)")

    model = model or DEFAULT_MODELS[provider]
    prompt = build_prompt(text)
    call_fn = PROVIDER_FUNCTIONS[provider]

    max_retries = 5
    last_error = None

    for attempt in range(max_retries):
        try:
            parsed = call_fn(prompt, model)
            parsed = normaliser_categories(parsed)

            quality = calculate_quality_score(parsed)
            parsed["score_qualite_globale"] = quality["score_qualite_globale"]
            parsed["score_qualite_globale_sur_10"] = quality["score_qualite_globale_sur_10"]
            parsed["details_score_qualite"] = quality["details_score"]

            annees_exp, details_experiences = calculate_experience_annees(parsed)
            parsed["annees_experience"] = annees_exp
            parsed["experiences_pro"] = details_experiences  # enrichi de poids_pertinence par expérience

            parsed["alertes_parcours"] = calculate_alertes_parcours(parsed)

            ponderes_100, ponderes_10 = calculate_domain_scores_ponderes(
                parsed, quality["score_qualite_globale"]
            )
            ponderes_100, ponderes_10 = calculate_domain_scores_ponderes(
                parsed, quality["score_qualite_globale"]
            )
            parsed["scores_categories_ponderes"] = ponderes_100
            parsed["scores_categories_ponderes_sur_10"] = ponderes_10

            # Flatten : domaine_1/score_1, domaine_2/score_2, domaine_3/score_3
            # (pour affichage simple dans Kibana, en plus du format nested)
            domaines_tries = sorted(parsed["scores_categories"].items(), key=lambda x: x[1], reverse=True)
            for i, (domaine, score) in enumerate(domaines_tries[:3], start=1):
                parsed[f"domaine_{i}"] = domaine
                parsed[f"score_{i}"] = score

            # Conversion finale objet -> liste, pour correspondre au mapping
            # Elasticsearch "nested". TOUTE DERNIÈRE étape (voir commentaire
            # au-dessus de la définition de convertir_en_liste_nested).
            parsed["scores_categories"] = convertir_en_liste_nested(parsed["scores_categories"])
            parsed["scores_categories_ponderes"] = convertir_en_liste_nested(parsed["scores_categories_ponderes"])
            parsed["scores_categories_ponderes_sur_10"] = convertir_en_liste_nested(parsed["scores_categories_ponderes_sur_10"])


            # Construire un texte enrichi pour l'embedding : texte brut + champs structurés
            # Cela garantit que TOUTES les technologies/langages/frameworks extraits par le LLM
            # sont inclus dans le vecteur, même si le texte brut ne les mentionne pas tous
            # explicitement (ex: profil full-stack avec React dans frameworks et Node.js
            # dans technologies).
            champs_enrichissement = []
            for champ in ["technologies", "langages", "frameworks", "bases_de_donnees",
                          "outils_devops", "certifications"]:
                valeurs = parsed.get(champ, [])
                if isinstance(valeurs, list) and valeurs:
                    champs_enrichissement.append(", ".join(valeurs))
            categ = parsed.get("categorie_principale", "")
            if categ:
                champs_enrichissement.append(categ)
            texte_enrichi = text + "\n" + " | ".join(champs_enrichissement)

            embedding = embedding_model.encode(texte_enrichi, normalize_embeddings=True)
            parsed["embedding_cv"] = embedding.tolist()
            return parsed

        except json.JSONDecodeError as je:
            print(f"⚠️ JSON invalide reçu de {provider} (essai {attempt + 1}/{max_retries}) : {je}")
            last_error = je
            time.sleep(2.0)
            continue

        except Exception as e:
            last_error = e
            if _is_quota_error(e):
                wait_time = get_retry_delay(e, default=10.0) + 1.0
                if wait_time > MAX_AUTO_WAIT_S:
                    raise QuotaTimeoutError(
                        f"Quota {provider} dépasse le plafond configuré "
                        f"({wait_time:.0f}s > {MAX_AUTO_WAIT_S:.0f}s)"
                    ) from e
                print(f"⚠️ Quota/rate limit {provider} (429). Attente de {wait_time:.2f}s "
                      f"(essai {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            elif _is_transient_server_error(e):
                wait_time = (2 ** attempt) * 5.0
                print(f"⚠️ Erreur serveur {provider} temporaire. Attente de {wait_time:.2f}s "
                      f"(essai {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                # Erreur non récupérable (clé invalide, modèle inexistant, etc.)
                raise e

    raise RuntimeError(
        f"❌ Extraction impossible avec '{provider}' après {max_retries} tentatives. "
        f"Dernière erreur : {last_error}"
    )


def extract_cv_data_auto(text, provider=None, model=None):
    """
    Comme extract_cv_data(), mais si AUTO_FALLBACK=true dans le .env,
    bascule automatiquement sur le provider suivant de FALLBACK_ORDER
    quand le quota du provider actuel dépasse GROQ_MAX_AUTO_WAIT_S.

    Si AUTO_FALLBACK=false (défaut), le comportement est identique à
    extract_cv_data() : l'erreur remonte telle quelle, sans bascule.
    """
    order = [provider] if provider else []
    order += [p for p in FALLBACK_ORDER if p not in order]

    last_error = None
    for p in order:
        if not AVAILABLE_KEYS.get(p):
            continue
        try:
            return extract_cv_data(text, provider=p, model=model if p == provider else None)
        except QuotaTimeoutError as e:
            last_error = e
            if not FALLBACK_ENABLED:
                raise
            print(f"↪️  Bascule automatique : {p} → provider suivant ({e})")
            continue

    raise last_error or RuntimeError("Aucun provider disponible pour l'extraction.")


def extract_all_cvs(cvs, provider="groq", model=None):
    """
    Prend la liste des CVs du Reader, extrait tous avec le provider choisi.
    Un CV en échec est loggé et ignoré, il ne bloque plus tout le lot.
    """
    extracted = []

    for i, cv in enumerate(cvs):
        if i > 0:
            print("⏳ Pause de 2 secondes avant l'extraction suivante...")
            time.sleep(2.0)

        print(f"🤖 Extraction ({provider}) : {cv['filename']}")
        try:
            data = extract_cv_data_auto(cv["text"], provider=provider, model=model)
            extracted.append({"filename": cv["filename"], "text": cv["text"], "data": data, "provider": provider})
            print(f"✅ Extrait : {cv['filename']}")
        except Exception as e:
            print(f"❌ Échec sur {cv['filename']} : {e}")
            extracted.append({"filename": cv["filename"], "data": None, "error": str(e), "provider": provider})

    _sauver_domaines_reference()
    return extracted


# TEST / CLI
if __name__ == "__main__":
    import argparse
    from cv_reader import read_all_cvs

    parser = argparse.ArgumentParser(description="Extraction + classification de CVs.")
    parser.add_argument("--provider", default="groq", choices=list(PROVIDER_FUNCTIONS.keys()),
                         help="Provider LLM à utiliser (défaut: groq)")
    parser.add_argument("--model", default=None, help="Nom exact du modèle (sinon défaut du provider)")
    args = parser.parse_args()

    if not AVAILABLE_KEYS.get(args.provider):
        raise SystemExit(f"❌ Clé API manquante pour '{args.provider}' dans .env")

    cvs = read_all_cvs("cvs/")
    results = extract_all_cvs(cvs, provider=args.provider, model=args.model)

    for result in results:
        print(f"\n{'='*50}")
        print(f"📄 {result['filename']}")
        print(f"{'='*50}")
        if result.get("data"):
            print(json.dumps(result["data"], indent=2, ensure_ascii=False))
        else:
            print(f"❌ Erreur : {result.get('error')}")