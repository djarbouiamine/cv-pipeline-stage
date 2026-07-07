import json
import time
import os
import requests

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
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-small-latest",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "gemini": "gemini-2.5-flash",
}


# ---------------------------------------------------------------------------
# Score de qualité du CV
# Principe : le LLM évalue qualitativement (projets, certifications) ou
# extrait du texte brut (niveau de langue) ; Python fait TOUT le calcul
# et l'agrégation. Aucun calcul de score global n'est fait par le LLM.
# Ça marche pareil quel que soit le provider utilisé pour l'extraction.
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
    Fallback sur un comptage simple si le LLM n'a pas fourni d'évaluation.
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
    Pondération : Diplôme 25% | Certifications 20% | Diversité technique 20%
                | Projets 25% | Langues 10%
    Aucun appel LLM supplémentaire : reproductible, sans coût de quota additionnel.
    """
    score_diplome = calculate_diploma_score(data.get("diplomes") or [])
    score_certif = calculate_certif_score(data)
    score_tech = calculate_tech_score(data)
    score_projet = calculate_projet_score(data)
    score_langue = calculate_langue_score(data)

    score_global = (
        score_diplome * 0.25
        + score_certif * 0.20
        + score_tech * 0.20
        + score_projet * 0.25
        + score_langue * 0.10
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
def build_prompt(text):
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
    et enrichit avec le score de qualité calculé par Python.

    provider : "groq" (défaut), "mistral", "openrouter" ou "gemini"
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

            quality = calculate_quality_score(parsed)
            parsed["score_qualite_globale"] = quality["score_qualite_globale"]
            parsed["score_qualite_globale_sur_10"] = quality["score_qualite_globale_sur_10"]
            parsed["details_score_qualite"] = quality["details_score"]

            ponderes_100, ponderes_10 = calculate_domain_scores_ponderes(
                parsed, quality["score_qualite_globale"]
            )
            parsed["scores_categories_ponderes"] = ponderes_100
            parsed["scores_categories_ponderes_sur_10"] = ponderes_10

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
            data = extract_cv_data(cv["text"], provider=provider, model=model)
            extracted.append({"filename": cv["filename"], "data": data, "provider": provider})
            print(f"✅ Extrait : {cv['filename']}")
        except Exception as e:
            print(f"❌ Échec sur {cv['filename']} : {e}")
            extracted.append({"filename": cv["filename"], "data": None, "error": str(e), "provider": provider})

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