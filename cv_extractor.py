import json
import time
import re
import os
import hashlib
import argparse

from google import genai
from google.genai.errors import APIError
import requests

# ── Charger le fichier .env s'il existe ───────────────────────
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


def get_retry_delay(err):
    """Tente d'extraire le délai d'attente (retry delay) à partir de l'exception."""
    try:
        if hasattr(err, 'details') and isinstance(err.details, dict):
            error_dict = err.details.get('error', {})
            details_list = error_dict.get('details', [])
            for detail in details_list:
                if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                    delay_str = detail.get('retryDelay', '')
                    if delay_str.endswith('s'):
                        return float(delay_str[:-1])
                    return float(delay_str)
    except Exception:
        pass
    try:
        match = re.search(r'Please retry in ([\d\.]+)s', err.message or '')
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return 10.0


CACHE_PATH = "output/.cv_cache.json"


def get_text_hash(text):
    return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()


def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Impossible de charger le cache : {e}")
    return {}


def save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Impossible de sauvegarder le cache : {e}")


# ── Clients API (tous gratuits) ────────────────────────────────
gemini_key = os.environ.get("GEMINI_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")
openrouter_key = os.environ.get("OPENROUTER_API_KEY")
mistral_key = os.environ.get("MISTRAL_API_KEY")

client_gemini = genai.Client(api_key=gemini_key) if gemini_key else None

client_groq = None
if groq_key:
    try:
        from groq import Groq
        client_groq = Groq(api_key=groq_key)
    except ImportError:
        print("⚠️ SDK Groq non installé. Impossible d'utiliser Groq.")

# Modèle par défaut utilisé quand un provider est forcé sans préciser de modèle
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
    "mistral": "mistral-small-latest",
    "openrouter": "openrouter/free",
}

AVAILABLE_KEYS = {
    "groq": bool(groq_key),
    "gemini": bool(gemini_key),
    "mistral": bool(mistral_key),
    "openrouter": bool(openrouter_key),
}


def _call_openrouter(prompt, model):
    """Appelle un modèle gratuit sur OpenRouter et retourne le JSON parsé."""
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY manquante dans .env")

    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "cv-pipeline-stage",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter {resp.status_code} ({model}) : {resp.text}")

    content = resp.json()["choices"][0]["message"]["content"]
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def _call_mistral(prompt, model):
    """Appelle un modèle gratuit sur Mistral et retourne le JSON parsé."""
    if not mistral_key:
        raise RuntimeError("MISTRAL_API_KEY manquante dans .env")

    headers = {
        "Authorization": f"Bearer {mistral_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Mistral {resp.status_code} ({model}) : {resp.text[:200]}")

    content = resp.json()["choices"][0]["message"]["content"]
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def _call_gemini(prompt, model):
    if not client_gemini:
        raise RuntimeError("GEMINI_API_KEY manquante dans .env")
    from google.genai import types
    config = types.GenerateContentConfig(response_mime_type="application/json")
    response = client_gemini.models.generate_content(model=model, contents=prompt, config=config)
    result = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(result)


def _call_groq(prompt, model):
    if not client_groq:
        raise RuntimeError("GROQ_API_KEY manquante dans .env")
    completion = client_groq.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content.strip())


def build_prompt(text):
    return f"""
Tu es un expert RH. Analyse ce CV et extrais les informations dans un JSON.
Retourne UNIQUEMENT le JSON, rien d'autre.

IMPORTANT : Le texte peut être mal formaté à cause de l'OCR.
Corrige les erreurs évidentes.

Pour la classification :
- Identifie TOI-MÊME les 2 ou 3 domaines professionnels les plus pertinents
  pour ce candidat (ne choisis pas dans une liste fixe, trouve les domaines
  qui correspondent vraiment à son profil)
- Donne un score de pertinence de 0 à 100 pour chaque domaine identifié
- Indique le domaine principal (le plus haut score)

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
    "langues": ["...", "..."]
}}

Voici le CV :
{text}
"""


def _dispatch_call(provider, prompt, model):
    if provider == "gemini":
        return _call_gemini(prompt, model)
    elif provider == "groq":
        return _call_groq(prompt, model)
    elif provider == "openrouter":
        return _call_openrouter(prompt, model)
    elif provider == "mistral":
        return _call_mistral(prompt, model)
    else:
        raise ValueError(f"Provider inconnu : {provider}")


def extract_cv_data(text, provider=None, model=None):
    """
    Prend le texte brut d'un CV.

    - Si `provider` est précisé : utilise UNIQUEMENT ce fournisseur/modèle,
      SANS fallback automatique. Si ça échoue, l'erreur remonte telle quelle.
    - Si `provider` est None (comportement par défaut, celui du pipeline de
      production) : essaie Groq en premier, puis bascule sur OpenRouter en
      cas d'échec, comme avant.

    Retourne (dictionnaire structuré, nom du modèle utilisé).
    """
    prompt = build_prompt(text)

    if provider is not None:
        if provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Provider '{provider}' inconnu. Choix possibles : {list(DEFAULT_MODELS.keys())}"
            )
        if not AVAILABLE_KEYS.get(provider):
            raise RuntimeError(
                f"Aucune clé API configurée pour '{provider}' dans .env. "
                f"Providers disponibles : {[p for p, ok in AVAILABLE_KEYS.items() if ok]}"
            )
        chosen_model = model or DEFAULT_MODELS[provider]
        print(f"   🤖 Extraction forcée avec {provider} ({chosen_model})...")
        data = _dispatch_call(provider, prompt, chosen_model)
        return data, chosen_model

    # ── Comportement par défaut : Groq en premier, OpenRouter en secours ──
    candidates = [
        {"provider": "groq", "model": model or DEFAULT_MODELS["groq"]},
        {"provider": "openrouter", "model": DEFAULT_MODELS["openrouter"]},
    ]

    last_exception = None

    for candidate in candidates:
        cprovider = candidate["provider"]
        cmodel = candidate["model"]

        if not AVAILABLE_KEYS.get(cprovider):
            continue

        print(f"   🤖 Tentative avec {cprovider} ({cmodel})...")

        if cprovider == "groq":
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = _call_groq(prompt, cmodel)
                    return data, cmodel
                except Exception as e:
                    if hasattr(e, 'status_code') and e.status_code in [429, 500, 502, 503, 504]:
                        wait_time = (2 ** attempt) * 5.0
                        print(f"   ⚠️ Erreur Groq ({e.status_code}). Attente {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        last_exception = e
                    else:
                        last_exception = e
                        break
            continue

        elif cprovider == "openrouter":
            try:
                data = _call_openrouter(prompt, cmodel)
                return data, cmodel
            except Exception as e:
                print(f"   ⚠️ Échec OpenRouter ({cmodel}) : {e}")
                last_exception = e
                continue

    raise last_exception or Exception("Tous les modèles et fournisseurs d'extraction ont échoué.")


def extract_all_cvs(cvs, provider=None, model=None):
    """
    Prend la liste des CVs du Reader.
    Retourne la liste avec les données extraites + le modèle utilisé.
    Un CV en échec n'arrête pas le traitement des suivants.

    provider / model : voir extract_cv_data(). None = comportement par défaut
    (Groq puis OpenRouter en fallback).
    """
    extracted = []
    cache = load_cache()

    for i, cv in enumerate(cvs):
        filename = cv['filename']
        text = cv['text']
        text_hash = get_text_hash(text)
        cache_key = f"{text_hash}:{provider or 'default'}:{model or 'default'}"

        if cache_key in cache:
            print(f"📦 Récupération depuis le cache : {filename}")
            extracted.append({
                "filename": filename,
                "model_used": cache[cache_key].get("model_used", "cache"),
                "data": cache[cache_key]["data"]
            })
            continue

        if len(extracted) > 0:
            print("⏳ Pause de 2 secondes avant l'extraction suivante...")
            time.sleep(2.0)

        print(f"🤖 Extraction : {filename}")
        try:
            data, model_used = extract_cv_data(text, provider=provider, model=model)
            extracted.append({
                "filename": filename,
                "model_used": model_used,
                "data": data
            })
            cache[cache_key] = {
                "filename": filename,
                "model_used": model_used,
                "data": data
            }
            save_cache(cache)
            print(f"✅ Extrait avec {model_used} : {filename}")
        except Exception as e:
            print(f"❌ Échec de l'extraction pour {filename} : {e}")
            extracted.append({
                "filename": filename,
                "model_used": None,
                "data": None,
                "error": str(e)
            })
            continue

    return extracted


# TEST / ligne de commande
if __name__ == "__main__":
    from cv_reader import read_all_cvs

    parser = argparse.ArgumentParser(description="Extraction de CVs avec choix optionnel du provider IA.")
    parser.add_argument(
        "--provider",
        choices=list(DEFAULT_MODELS.keys()),
        default=None,
        help="Force un fournisseur précis (gemini, groq, mistral, openrouter). "
             "Par défaut : Groq avec fallback OpenRouter automatique.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Nom exact du modèle à utiliser (optionnel, sinon modèle par défaut du provider).",
    )
    args = parser.parse_args()

    cvs = read_all_cvs("cvs/")
    results = extract_all_cvs(cvs, provider=args.provider, model=args.model)

    for result in results:
        print(f"\n{'='*50}")
        print(f"📄 {result['filename']} (modèle : {result['model_used']})")
        print(f"{'='*50}")
        if result.get("data"):
            print(json.dumps(result['data'], indent=2, ensure_ascii=False))
        else:
            print(f"⚠️ Échec : {result.get('error')}")