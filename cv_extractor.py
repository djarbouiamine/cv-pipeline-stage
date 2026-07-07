import os
import sys
import json
import time
import hashlib
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from quota_tracker import quota

# Force stdout/stderr to use UTF-8 to prevent encoding issues on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

# Initialisation du client Gemini
client_gemini = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        import google.genai as genai
        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
    except ImportError:
        print("⚠️ SDK google-genai non installé.")

# ── Config providers ──────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def check_key(name, value):
    if not value:
        print(f"❌ {name} est VIDE ou introuvable dans .env !")
    else:
        print(f"✅ {name} chargée ({value[:6]}...{value[-4:]})")


check_key("GROQ_API_KEY", GROQ_API_KEY)
check_key("MISTRAL_API_KEY", MISTRAL_API_KEY)
check_key("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
check_key("GEMINI_API_KEY", GEMINI_API_KEY)

CACHE_PATH = "output/.cv_cache.json"
FAILED_LOG_PATH = "output/.failed_cvs.json"


# ── Limiteur de débit (par minute, par provider) ──────────────
class RateLimiter:
    def __init__(self, max_calls, period_seconds):
        self.max_calls = max_calls
        self.period = period_seconds
        self.lock = threading.Lock()
        self.calls = []

    def wait_for_slot(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                wait_time = self.period - (now - self.calls[0]) + 0.1
            else:
                wait_time = 0
            if wait_time > 0:
                print(f"⏳ Rate limiter : pause {wait_time:.1f}s...")
        if wait_time > 0:
            time.sleep(wait_time)
        with self.lock:
            self.calls.append(time.time())


# Chaque provider a son propre limiteur, partagé entre tous les agents qui l'utilisent
groq_limiter = RateLimiter(max_calls=20, period_seconds=60)
mistral_limiter = RateLimiter(max_calls=5, period_seconds=1)
openrouter_limiter = RateLimiter(max_calls=15, period_seconds=60)
gemini_limiter = RateLimiter(max_calls=15, period_seconds=60)

# Un seul appel Groq à la fois, peu importe combien de CVs/agents en parallèle
groq_concurrency_lock = threading.Semaphore(1)


class ProviderExhausted(Exception):
    """Levée quand un provider a un Retry-After trop long (quota probablement journalier)."""
    pass


class ProviderAuthError(Exception):
    """Levée sur 401 — clé invalide."""
    pass


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs("output", exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def log_failed_cv(filename, reason):
    os.makedirs("output", exist_ok=True)
    failed = []
    if os.path.exists(FAILED_LOG_PATH):
        with open(FAILED_LOG_PATH, "r", encoding="utf-8") as f:
            failed = json.load(f)
    failed.append({"filename": filename, "reason": reason, "timestamp": time.time()})
    with open(FAILED_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)


cache_lock = threading.Lock()


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_json(raw):
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── Retry HTTP : attend si court, lève ProviderExhausted si trop long ─
def _http_post_with_retry(limiter, url, headers, payload, max_retries=2,
                           base_wait=10, max_acceptable_wait=60):
    host = url.split('/')[2]
    for attempt in range(max_retries):
        limiter.wait_for_slot()
        r = requests.post(url, headers=headers, json=payload, timeout=60)

        if r.status_code == 401:
            raise ProviderAuthError(f"401 Unauthorized sur {host} — clé API invalide ou vide.")

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", base_wait * (2 ** attempt)))

            if retry_after > max_acceptable_wait:
                raise ProviderExhausted(
                    f"Quota probablement épuisé sur {host} (Retry-After={retry_after}s)"
                )

            print(f"⏳ 429 sur {host} — attente {retry_after}s "
                  f"(tentative {attempt + 1}/{max_retries})...")
            time.sleep(retry_after)
            continue

        r.raise_for_status()
        return r

    raise ProviderExhausted(f"Échec après {max_retries} tentatives sur {host}")


# ── Appels bruts par provider ──────────────────────────────────
def call_groq(prompt, json_mode=True):
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    with groq_concurrency_lock:
        r = _http_post_with_retry(
            groq_limiter,
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            payload=payload,
        )
    return r.json()["choices"][0]["message"]["content"]


def call_mistral(prompt, json_mode=True):
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = _http_post_with_retry(
        mistral_limiter,
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
        payload=payload,
    )
    return r.json()["choices"][0]["message"]["content"]


def call_openrouter(prompt, json_mode=True):
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = _http_post_with_retry(
        openrouter_limiter,
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        payload=payload,
    )
    return r.json()["choices"][0]["message"]["content"]


def call_gemini(prompt, json_mode=True):
    if not client_gemini:
        raise ProviderAuthError("GEMINI_API_KEY non configurée ou client Gemini non initialisé.")

    gemini_limiter.wait_for_slot()

    from google.genai import types
    config = None
    if json_mode:
        config = types.GenerateContentConfig(response_mime_type="application/json")

    try:
        response = client_gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
            raise ProviderExhausted(f"Quota Gemini épuisé ou limite atteinte : {e}")
        raise


PROVIDER_FUNCTIONS = {
    "groq": call_groq,
    "mistral": call_mistral,
    "openrouter": call_openrouter,
    "gemini": call_gemini,
}


# ── Appel générique avec cascade de fallback ───────────────────
def call_with_fallback(prompt, provider_order, agent_label=""):
    """
    Essaie chaque provider de provider_order dans l'ordre.

    AVANT chaque appel : vérifie le quota tracker. Si le provider est
    déjà marqué épuisé aujourd'hui, on saute directement au suivant —
    zéro requête HTTP, zéro attente.

    Bascule automatiquement au suivant si le provider actuel est
    à sec (quota épuisé), en erreur d'auth, en erreur réseau,
    ou s'il renvoie un format inattendu. Un ProviderExhausted marque
    aussi le provider comme épuisé pour le reste de la journée, pour
    que les prochains CV du batch ne le retentent même pas.
    """
    last_error = None
    for provider_name in provider_order:

        if not quota.can_call(provider_name):
            print(f"   ⏭️ {agent_label} : {provider_name} déjà épuisé aujourd'hui — skip direct")
            continue

        fn = PROVIDER_FUNCTIONS[provider_name]
        try:
            raw = fn(prompt)
            quota.record_call(provider_name)
            return clean_json(raw), provider_name
        except ProviderExhausted as e:
            print(f"   ⚠️ {agent_label} : {provider_name} indisponible ({e}) — bascule...")
            quota.mark_exhausted(provider_name)
            last_error = e
            continue
        except ProviderAuthError as e:
            print(f"   ⚠️ {agent_label} : {provider_name} erreur de clé ({e}) — bascule...")
            last_error = e
            continue
        except json.JSONDecodeError as e:
            print(f"   ⚠️ {agent_label} : {provider_name} a renvoyé un JSON invalide — bascule...")
            quota.record_call(provider_name)
            last_error = e
            continue
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ {agent_label} : {provider_name} erreur réseau/HTTP ({e}) — bascule...")
            last_error = e
            continue
        except (KeyError, IndexError) as e:
            print(f"   ⚠️ {agent_label} : {provider_name} réponse malformée ({e}) — bascule...")
            quota.record_call(provider_name)
            last_error = e
            continue
        except Exception as e:
            print(f"   ⚠️ {agent_label} : {provider_name} a échoué ({type(e).__name__}: {e}) — bascule...")
            last_error = e
            continue

    raise RuntimeError(
        f"{agent_label} : tous les providers ont échoué ({provider_order}). "
        f"Dernière erreur : {last_error}"
    )


# ── Agent 1 : Extraction pure ──────────────────────────────────
# Ordre de préférence : Groq d'abord, puis Gemini, puis OpenRouter, puis Mistral
AGENT1_PROVIDER_ORDER = ["groq", "gemini", "openrouter", "mistral"]


def agent1_extract(cv_text):
    prompt = f"""Extrais uniquement les faits de ce CV en JSON.
Pas de scoring, pas de jugement. Juste ce qui est écrit.

Format :
{{"nom":"","email":"","telephone":"","linkedin":"","localisation":"",
"technologies":[],"langages":[],"frameworks":[],"bases_de_donnees":[],
"outils_devops":[],"projets":[],"description_projets":{{}},
"diplomes":[],"certifications":[],"langues":[]}}

CV :
{cv_text}
"""
    data, provider_used = call_with_fallback(prompt, AGENT1_PROVIDER_ORDER, "Agent1-Extraction")
    return data, provider_used


# ── Agent 2 : Scoring qualitatif ────────────────────────────────
# Ordre de préférence : Mistral d'abord, puis Gemini, puis OpenRouter, puis Groq
AGENT2_PROVIDER_ORDER = ["mistral", "gemini", "openrouter", "groq"]


def agent2_score(extracted_data):
    prompt = f"""Voici les données extraites d'un CV (JSON). Juge la QUALITÉ,
pas la quantité. Une justification courte (1 phrase max) par critère.

Données :
{json.dumps(extracted_data, ensure_ascii=False)}

Réponds en JSON :
{{"qualite_projets":{{"score":0,"justification":""}},
"qualite_diplomes":{{"score":0,"justification":""}},
"qualite_certifications":{{"score":0,"justification":""}}}}
"""
    data, provider_used = call_with_fallback(prompt, AGENT2_PROVIDER_ORDER, "Agent2-Scoring")
    return data, provider_used


# ── Agent 3 : Anti-hallucination ────────────────────────────────
# Ordre de préférence : OpenRouter d'abord, puis Gemini, puis Mistral, puis Groq
AGENT3_PROVIDER_ORDER = ["openrouter", "gemini", "mistral", "groq"]


def agent3_verify(extracted_data, scoring_data):
    prompt = f"""Tu vérifies si les justifications ci-dessous inventent des
détails absents des données sources. Si un chiffre/date/rang précis n'existe
pas dans les données sources, retire-le de la justification et baisse le
score de 1 point.

Données sources :
{json.dumps(extracted_data, ensure_ascii=False)}

Scoring à vérifier :
{json.dumps(scoring_data, ensure_ascii=False)}

Réponds avec le JSON corrigé, même format que le scoring.
"""
    data, provider_used = call_with_fallback(prompt, AGENT3_PROVIDER_ORDER, "Agent3-Verification")
    return data, provider_used


# ── Pipeline pour un seul CV ───────────────────────────────────
def process_one_cv(cv):
    text_hash = hash_text(cv["text"])

    with cache_lock:
        cache = load_cache()
        if text_hash in cache:
            return {"filename": cv["filename"], "data": cache[text_hash], "cached": True}

    try:
        extracted, provider1 = agent1_extract(cv["text"])
        scoring, provider2 = agent2_score(extracted)
        verified_scoring, provider3 = agent3_verify(extracted, scoring)
    except RuntimeError as e:
        log_failed_cv(cv["filename"], str(e))
        raise

    final_data = {
        **extracted,
        "scoring_qualitatif": verified_scoring,
        "_providers_used": {
            "agent1_extraction": provider1,
            "agent2_scoring": provider2,
            "agent3_verification": provider3,
        },
    }

    with cache_lock:
        cache = load_cache()
        cache[text_hash] = final_data
        save_cache(cache)

    return {"filename": cv["filename"], "data": final_data, "cached": False}


# ── Orchestrateur (traite un lot en parallèle) ─────────────────
def process_batch(cvs_batch, max_workers=2):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_cv, cv): cv for cv in cvs_batch}
        for future in as_completed(futures):
            cv = futures[future]
            try:
                result = future.result()
                results.append(result)
                tag = "📦 (cache)" if result.get("cached") else "✅"
                providers = result["data"].get("_providers_used", {})
                print(f"{tag} Terminé : {cv['filename']} — {providers}")
            except Exception as e:
                print(f"❌ Erreur définitive sur {cv['filename']} : {e}")
    return results


def process_all_cvs(cvs, max_workers=2, batch_size=3, pause_between_batches=15):
    """
    Découpe la liste de CVs en lots de `batch_size`.
    Chaque lot est traité en parallèle (max_workers), puis on attend
    `pause_between_batches` secondes avant d'attaquer le lot suivant.
    Ça laisse le temps aux quotas minute/TPM de se reposer entre les lots,
    sans avoir à attendre un quota journalier complet.
    """
    all_results = []
    nb_batches = (len(cvs) + batch_size - 1) // batch_size

    for i in range(0, len(cvs), batch_size):
        batch = cvs[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"\n🚀 Lot {batch_num}/{nb_batches} — {len(batch)} CV(s) : "
              f"{[cv['filename'] for cv in batch]}")
        print(f"📊 Quotas restants : {quota.status()}")

        batch_results = process_batch(batch, max_workers=max_workers)
        all_results.extend(batch_results)

        # Pas de pause après le dernier lot
        is_last_batch = (i + batch_size) >= len(cvs)
        if not is_last_batch and pause_between_batches > 0:
            print(f"⏸️ Pause de {pause_between_batches}s avant le lot suivant "
                  f"(laisse les quotas minute se reposer)...")
            time.sleep(pause_between_batches)

    return all_results


# ── Compatibilité avec cv_agent.py ─────────────────────────────
def build_prompt(cv_text):
    return f"""Extrais uniquement les faits de ce CV en JSON.
Pas de scoring, pas de jugement. Juste ce qui est écrit.

Format :
{{"nom":"","email":"","telephone":"","linkedin":"","localisation":"",
"technologies":[],"langages":[],"frameworks":[],"bases_de_donnees":[],
"outils_devops":[],"projets":[],"description_projets":{{}},
"diplomes":[],"certifications":[],"langues":[]}}

CV :
{cv_text}
"""

def _call_groq(prompt, model=None):
    raw = call_groq(prompt)
    return clean_json(raw)

def _call_gemini(prompt, model=None):
    raw = call_gemini(prompt)
    return clean_json(raw)

def _call_mistral(prompt, model=None):
    raw = call_mistral(prompt)
    return clean_json(raw)

def _call_openrouter(prompt, model=None):
    raw = call_openrouter(prompt)
    return clean_json(raw)

AVAILABLE_KEYS = {
    "groq": bool(GROQ_API_KEY),
    "mistral": bool(MISTRAL_API_KEY),
    "openrouter": bool(OPENROUTER_API_KEY),
    "gemini": bool(GEMINI_API_KEY),
}

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-small-latest",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "gemini": "gemini-2.5-flash",
}


if __name__ == "__main__":
    from cv_reader import read_all_cvs
    from cv_saver import save_to_json, save_to_excel

    # On repart d'un fichier .failed_cvs.json propre pour ce run précis,
    # sinon un ancien fichier d'un run précédent fausse le message final.
    if os.path.exists(FAILED_LOG_PATH):
        os.remove(FAILED_LOG_PATH)

    cvs = read_all_cvs("cvs/")
    results = process_all_cvs(cvs, max_workers=2)

    print(f"\n🎉 {len(results)}/{len(cvs)} CVs traités avec succès.")
    print(f"📊 Quotas utilisés aujourd'hui : {quota.status()}")

    # Le nombre de CVs vraiment échoués = différence entre CVs lus et résultats obtenus
    nb_echecs = len(cvs) - len(results)
    if nb_echecs > 0:
        print(f"⚠️ {nb_echecs} CV(s) ont échoué définitivement (tous providers épuisés) — "
              f"voir {FAILED_LOG_PATH}.")

    # ── Affichage détaillé : extraction + scoring pour chaque CV ──
    for r in results:
        print(f"\n{'='*60}")
        print(f"📄 {r['filename']}" + (" (depuis le cache)" if r.get("cached") else ""))
        print(f"{'='*60}")
        print(json.dumps(r["data"], indent=2, ensure_ascii=False))

    # ── Export JSON + Excel ─────────────────────────────────────
    save_to_json(results)
    save_to_excel(results)

    print("\n🎉 Terminé ! Fichiers dans le dossier output/ "
          "(cvs_data.json, cvs_data.xlsx, .cv_cache.json)")