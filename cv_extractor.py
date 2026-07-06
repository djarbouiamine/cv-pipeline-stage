import json
import time
import re
import os
import hashlib
import requests

# ---------------------------------------------------------------------------
# Chargement du .env
# ---------------------------------------------------------------------------
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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

CACHE_PATH = "output/.cv_cache.json"

# À incrémenter à chaque fois qu'on change le prompt ou la logique de scoring,
# pour que le cache s'invalide automatiquement au lieu de renvoyer d'anciens
# résultats qui n'ont pas les nouveaux champs/comportements.
# v3-qualite-stricte : justifications forcées à citer un élément concret du CV
PROMPT_VERSION = "v3-qualite-stricte"


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------
def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print("⚠️ Cache corrompu ou vide, on repart d'un cache neuf.")
            return {}
    return {}


def _save_cache(cache):
    os.makedirs("output", exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _hash_text(text):
    combined = f"{PROMPT_VERSION}::{text}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# PROMPT — extraction + évaluation qualité en un seul appel
# ---------------------------------------------------------------------------
def build_prompt(text):
    return f"""
Tu es un expert RH. Analyse ce CV et extrais les informations dans un JSON.
Retourne UNIQUEMENT le JSON, rien d'autre, pas de ```markdown```.

IMPORTANT : Le texte peut être mal formaté à cause de l'OCR.
Corrige les erreurs évidentes.

Pour la classification :
- Identifie TOI-MÊME les 2 ou 3 domaines professionnels les plus pertinents
  pour ce candidat (ne choisis pas dans une liste fixe, trouve les domaines
  qui correspondent vraiment à son profil)
- Donne un score de pertinence de 0 à 100 pour chaque domaine identifié
- Indique le domaine principal (le plus haut score)

Pour l'évaluation qualité (evaluation_qualite), IMPORTANT : évalue la QUALITÉ,
pas la quantité. Un candidat avec 1 seul projet exceptionnel doit avoir une
meilleure note qu'un candidat avec 5 projets basiques. Si une liste est vide,
mets la note à 0.

- qualite_projets (0-10) : base-toi sur la complexité technique réelle
  (architecture, stack avancé), la présence de résultats mesurables
  (métriques, déploiement réel), et l'autonomie/originalité du projet
  (recherche/perso vs exercice scolaire encadré).
- qualite_diplomes (0-10) : base-toi sur le niveau de l'établissement et la
  cohérence avec le profil visé.
- qualite_certifications (0-10) : base-toi sur la reconnaissance réelle de la
  certification (ex: Cisco, DataCamp, AWS) vs simple certificat de
  participation à un événement.

Donne une justification courte (1 phrase) pour chaque note. La justification
DOIT citer un élément concret et précis du CV (nom exact d'un projet, d'une
techno, d'un diplôme...). Interdiction de phrases génériques du type
"démontre une bonne maîtrise des compétences techniques" sans rien de
spécifique — si tu ne peux pas citer un élément précis, c'est que la note
doit probablement être plus basse.

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
    "projets": ["...", "..."],
    "description_projets": {{
        "nom du projet 1": "description courte"
    }},
    "diplomes": ["...", "..."],
    "certifications": ["...", "..."],
    "langues": ["...", "..."],
    "evaluation_qualite": {{
        "qualite_projets": 0,
        "justification_projets": "...",
        "qualite_diplomes": 0,
        "justification_diplomes": "...",
        "qualite_certifications": 0,
        "justification_certifications": "..."
    }}
}}

Voici le CV :
{text}
"""


class RateLimitError(Exception):
    """Erreur de rate limit temporaire (ex: tokens/minute). On peut réessayer après un délai."""
    def __init__(self, message, retry_after=5.0):
        super().__init__(message)
        self.retry_after = retry_after


def _parse_retry_after(text):
    """Cherche un délai d'attente suggéré dans le message d'erreur (ex: 'try again in 3.2s')."""
    match = re.search(r'try again in ([\d\.]+)s', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'retryDelay["\']?\s*:\s*["\']?([\d\.]+)s', text)
    if match:
        return float(match.group(1))
    return 5.0


def _clean_json(raw_text):
    result = raw_text.strip()
    result = result.replace("```json", "").replace("```", "").strip()
    return json.loads(result)


# ---------------------------------------------------------------------------
# APPELS PROVIDERS
# ---------------------------------------------------------------------------
def call_groq(prompt, model="llama-3.3-70b-versatile"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 4000
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code == 429:
        raise RateLimitError(resp.text[:300], retry_after=_parse_retry_after(resp.text))
    if resp.status_code != 200:
        raise RuntimeError(f"Groq {resp.status_code}: {resp.text[:300]}")
    content = resp.json()["choices"][0]["message"]["content"]
    return _clean_json(content)


def call_gemini(prompt, model="gemini-2.5-flash"):
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "max_output_tokens": 8192
        }
    )
    return _clean_json(response.text)


def call_openrouter(prompt, model):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 4000
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code == 429:
        raise RateLimitError(resp.text[:300], retry_after=_parse_retry_after(resp.text))
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter {model} {resp.status_code}: {resp.text[:300]}")
    content = resp.json()["choices"][0]["message"]["content"]
    return _clean_json(content)


# ---------------------------------------------------------------------------
# Récupération dynamique des modèles gratuits OpenRouter
# ---------------------------------------------------------------------------
# Les slugs des modèles gratuits changent souvent (ex: deepseek-r1:free et
# qwen3-235b-a22b:free sont passés payants sans prévenir). Plutôt que de
# coder des noms en dur qui se périment, on interroge l'API OpenRouter pour
# avoir la liste réelle des modèles gratuits au moment de l'exécution.
_OPENROUTER_FREE_MODELS_CACHE = None

def get_free_openrouter_models(max_models=3):
    global _OPENROUTER_FREE_MODELS_CACHE
    if _OPENROUTER_FREE_MODELS_CACHE is not None:
        return _OPENROUTER_FREE_MODELS_CACHE

    fallback = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "openai/gpt-oss-20b:free",
    ]

    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        free_models = []
        for m in models:
            pricing = m.get("pricing", {})
            try:
                prompt_price = float(pricing.get("prompt", 1))
                completion_price = float(pricing.get("completion", 1))
            except (TypeError, ValueError):
                continue
            if prompt_price == 0 and completion_price == 0:
                free_models.append(m["id"])

        if free_models:
            _OPENROUTER_FREE_MODELS_CACHE = free_models[:max_models]
            print(f"🔎 Modèles gratuits OpenRouter détectés : {_OPENROUTER_FREE_MODELS_CACHE}")
            return _OPENROUTER_FREE_MODELS_CACHE
    except Exception as e:
        print(f"⚠️ Impossible de récupérer la liste des modèles gratuits OpenRouter ({e}), utilisation du fallback.")

    _OPENROUTER_FREE_MODELS_CACHE = fallback
    return fallback


# Liste des candidats (providers/modèles) essayés dans l'ordre, avec fallback.
# Groq et Gemini sont fixes, les modèles OpenRouter sont récupérés dynamiquement
# au moment de l'exécution pour éviter les noms périmés.
def build_candidates():
    candidates = [
        {"provider": "groq", "model": "llama-3.3-70b-versatile", "fn": call_groq},
        {"provider": "gemini", "model": "gemini-2.5-flash", "fn": call_gemini},
    ]
    for model_name in get_free_openrouter_models():
        candidates.append({"provider": "openrouter", "model": model_name, "fn": call_openrouter})
    return candidates


CANDIDATES = build_candidates()


def _try_candidate(candidate, prompt):
    """Appelle un provider. Retourne (data, model_used) ou lève une exception."""
    data = candidate["fn"](prompt, model=candidate["model"])
    return data, f"{candidate['provider']}:{candidate['model']}"


# ---------------------------------------------------------------------------
# SCORE GLOBAL /10 — basé sur la QUALITÉ, pas la quantité
# ---------------------------------------------------------------------------
# - Qualité des projets           -> 4 pts
# - Qualité des diplômes          -> 2 pts
# - Qualité des certifications    -> 2 pts
# - Langues (nombre)              -> 1.5 pts
# - Complétude profil basique     -> 0.5 pt
def compute_score_global(data):
    evaluation = data.get("evaluation_qualite") or {}

    qualite_projets = evaluation.get("qualite_projets", 0) or 0
    qualite_diplomes = evaluation.get("qualite_diplomes", 0) or 0
    qualite_certifications = evaluation.get("qualite_certifications", 0) or 0

    score = 0.0
    score += (qualite_projets / 10) * 4
    score += (qualite_diplomes / 10) * 2
    score += (qualite_certifications / 10) * 2

    # Langues : le nombre reste pertinent (parler plusieurs langues = atout réel)
    langues = data.get("langues") or []
    if len(langues) >= 3:
        score += 1.5
    elif len(langues) == 2:
        score += 1.0
    elif len(langues) == 1:
        score += 0.5

    # Complétude profil basique (0.5 pt)
    contact_fields = ["email", "telephone", "linkedin", "localisation"]
    filled = sum(1 for f in contact_fields if data.get(f))
    score += (filled / len(contact_fields)) * 0.5

    return round(min(score, 10.0), 1)


# ---------------------------------------------------------------------------
# EXTRACTION AVEC CACHE + FALLBACK MULTI-PROVIDER
# ---------------------------------------------------------------------------
def extract_cv_data(text, force_refresh=False):
    """
    Prend le texte brut d'un CV.
    Essaie chaque provider dans l'ordre (Groq -> Gemini -> OpenRouter...).
    Utilise le cache si le texte a déjà été traité, sauf si force_refresh=True
    (utile pour retraiter un CV précis sans devoir vider tout le cache).
    Retourne un dictionnaire structuré avec score_global calculé.
    """
    cache = _load_cache()
    text_hash = _hash_text(text)

    if not force_refresh and text_hash in cache:
        print("💾 Résultat trouvé dans le cache, appel API évité.")
        return cache[text_hash]["data"]

    prompt = build_prompt(text)

    last_error = None
    for candidate in CANDIDATES:
        max_retries_same_provider = 2
        for attempt in range(max_retries_same_provider):
            try:
                print(f"🤖 Tentative avec {candidate['provider']} ({candidate['model']})...")
                data, model_used = _try_candidate(candidate, prompt)
                data["score_global"] = compute_score_global(data)
                data["modele_utilise"] = model_used

                cache[text_hash] = {
                    "data": data,
                    "model_used": model_used
                }
                _save_cache(cache)

                print(f"✅ Succès avec {model_used}")
                return data

            except RateLimitError as e:
                last_error = e
                wait_time = min(e.retry_after + 1.0, 65.0)
                print(f"⏳ Rate limit temporaire sur {candidate['provider']} ({candidate['model']}). "
                      f"Attente de {wait_time:.1f}s avant retry (essai {attempt + 1}/{max_retries_same_provider})...")
                time.sleep(wait_time)
                continue

            except Exception as e:
                last_error = e
                print(f"⚠️ Échec avec {candidate['provider']} ({candidate['model']}): {str(e)[:200]}")
                break  # on passe directement au provider suivant (erreur non temporaire)

    raise RuntimeError(f"Tous les providers ont échoué. Dernière erreur : {last_error}")


def re_extract_cvs(cvs, filenames_to_refresh):
    """
    Retraite uniquement les CVs dont le nom de fichier est dans filenames_to_refresh,
    en forçant un nouvel appel API et en ignorant le cache existant pour ces
    fichiers-là uniquement (utile pour ne pas tout relancer).
    """
    refreshed = []
    for cv in cvs:
        if cv['filename'] in filenames_to_refresh:
            print(f"🔄 Retraitement forcé : {cv['filename']}")
            data = extract_cv_data(cv['text'], force_refresh=True)
            refreshed.append({"filename": cv['filename'], "data": data})
    return refreshed


def extract_all_cvs(cvs):
    """
    Prend la liste des CVs du Reader.
    Retourne la liste avec les données extraites.
    """
    extracted = []

    for i, cv in enumerate(cvs):
        if i > 0:
            print("⏳ Pause de 2 secondes avant l'extraction suivante...")
            time.sleep(2.0)

        print(f"📄 Extraction : {cv['filename']}")
        try:
            data = extract_cv_data(cv['text'])
            extracted.append({
                "filename": cv['filename'],
                "data": data
            })
            print(f"✅ Extrait : {cv['filename']} (score_global: {data.get('score_global')}/10)")
        except Exception as e:
            print(f"❌ Échec définitif pour {cv['filename']} (tous providers épuisés) : {str(e)[:200]}")
            extracted.append({
                "filename": cv['filename'],
                "data": None,
                "erreur": str(e)
            })
            continue

    return extracted


# TEST
if __name__ == "__main__":
    from cv_reader import read_all_cvs

    cvs = read_all_cvs("cvs/")
    results = extract_all_cvs(cvs)

    for result in results:
        print(f"\n{'='*50}")
        print(f"📄 {result['filename']}")
        print(f"{'='*50}")
        print(json.dumps(result['data'], indent=2, ensure_ascii=False))