import json
import time
import os
import re
import csv
from datetime import datetime

from google import genai
from google.genai.errors import APIError
import requests

# ── Charger .env ──────────────────────────────────────────────
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
        print("⚠️ SDK Groq non installé.")

# ── Champs attendus pour mesurer la complétude ────────────────
EXPECTED_FIELDS = [
    "nom", "email", "telephone", "linkedin", "localisation",
    "scores_categories", "categorie_principale", "technologies",
    "langages", "frameworks", "bases_de_donnees", "outils_devops",
    "projets", "description_projets", "diplomes", "certifications", "langues"
]

# ── Modèles à comparer (doublon Llama-OpenRouter retiré) ──────
CANDIDATES = [
    {"name": "Llama-3.3-70B (Groq)",     "provider": "groq",       "model": "llama-3.3-70b-versatile"},
    {"name": "GPT-OSS-20B (OpenRouter)", "provider": "openrouter", "model": "openai/gpt-oss-20b:free"},
    {"name": "OpenRouter Free (auto)",   "provider": "openrouter", "model": "openrouter/free"},
    {"name": "Gemini 2.5 Flash",         "provider": "gemini",     "model": "gemini-2.5-flash"},
    {"name": "Gemini 2.5 Flash-Lite",    "provider": "gemini",     "model": "gemini-2.5-flash-lite"},
    {"name": "Mistral Small (Mistral)",  "provider": "mistral",    "model": "mistral-small-latest"},
]


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


def _call_openrouter(prompt, model):
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY manquante")

    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "cv-pipeline-comparaison",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers, json=payload, timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} : {resp.text[:200]}")

    content = resp.json()["choices"][0]["message"]["content"]
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def _call_gemini(prompt, model):
    if not client_gemini:
        raise RuntimeError("Client Gemini non configuré")
    from google.genai import types
    config = types.GenerateContentConfig(response_mime_type="application/json")
    response = client_gemini.models.generate_content(model=model, contents=prompt, config=config)
    result = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(result)


def _call_groq(prompt, model):
    if not client_groq:
        raise RuntimeError("Client Groq non configuré")
    completion = client_groq.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content.strip())


def _call_mistral(prompt, model):
    if not mistral_key:
        raise RuntimeError("MISTRAL_API_KEY manquante")

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
        headers=headers, json=payload, timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} : {resp.text[:200]}")

    content = resp.json()["choices"][0]["message"]["content"]
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


# ── Évaluation de la Justesse (Vérité terrain) ────────────────
GROUND_TRUTH = {
    "CV_Amine_Jarboui.pdf": {
        "nom": "Amine Jarboui",
        "email": "amine.jarboui@supcom.tn",
        "telephone": "29400323",
        "categories_valides": ["full-stack", "backend", "devops", "logiciel", "web", "software", "supcom"]
    },
    "CV_Hayder_Khouildi_EN.pdf": {
        "nom": "Hayder Khouildi",
        "email": "khouildihayder7@gmail.com",
        "telephone": "53847400",
        "categories_valides": ["embedded", "embarqué", "iot", "systems", "système", "internet of things"]
    },
    "cv_hazem.pdf": {
        "nom": "Hazem Bellili",
        "email": "hazem.bellili@supcom.tn",
        "telephone": "51877031",
        "categories_valides": ["logiciel", "software", "devops", "cloud", "télécom", "telecom"]
    },
    "cv_yesmine_zayani.pdf": {
        "nom": "Yesmine ZAYANI",
        "email": "yessminezayeni5@gmail.com",
        "telephone": "54384270",
        "categories_valides": ["intelligence artificielle", "machine learning", "ia", "ai", "imagerie", "medical", "médicale"]
    }
}


def evaluate_extraction_accuracy(filename, data):
    if not isinstance(data, dict) or filename not in GROUND_TRUTH:
        return None, None

    gt = GROUND_TRUTH[filename]
    
    # 1. Évaluation de l'extraction (Nom, Email, Téléphone)
    nom_ext = str(data.get("nom") or "").strip().lower()
    email_ext = str(data.get("email") or "").strip().lower()
    tel_ext = str(data.get("telephone") or "").strip().lower()

    nom_correct = 1.0 if gt["nom"].lower() in nom_ext or nom_ext in gt["nom"].lower() else 0.0
    email_correct = 1.0 if gt["email"].lower() in email_ext or email_ext in gt["email"].lower() else 0.0
    
    tel_digits_gt = "".join(c for c in gt["telephone"] if c.isdigit())
    tel_digits_ext = "".join(c for c in tel_ext if c.isdigit())
    
    if tel_digits_gt and tel_digits_ext:
        tel_correct = 1.0 if tel_digits_gt in tel_digits_ext or tel_digits_ext in tel_digits_gt else 0.0
    else:
        tel_correct = 1.0 if not tel_digits_gt and not tel_digits_ext else 0.0

    justesse_pct = round((nom_correct + email_correct + tel_correct) / 3.0 * 100, 1)

    # 2. Évaluation de la classification
    cat_principale = str(data.get("categorie_principale") or "").strip().lower()
    scores = data.get("scores_categories") or {}
    
    classification_pertinente = False
    if any(keyword in cat_principale for keyword in gt["categories_valides"]):
        classification_pertinente = True
    else:
        for cat, score in scores.items():
            if isinstance(score, (int, float)) and score >= 70:
                if any(keyword in cat.lower() for keyword in gt["categories_valides"]):
                    classification_pertinente = True
                    break

    return justesse_pct, classification_pertinente


def completeness_score(data):
    if not isinstance(data, dict):
        return 0.0
    filled = 0
    for field in EXPECTED_FIELDS:
        val = data.get(field)
        if val not in (None, "", [], {}):
            filled += 1
    return round(filled / len(EXPECTED_FIELDS) * 100, 1)


def get_error_type(e):
    err_str = str(e).lower()
    if isinstance(e, json.JSONDecodeError):
        return "parsing"
    
    is_quota = False
    if hasattr(e, 'code') and e.code == 429:
        is_quota = True
    elif hasattr(e, 'status_code') and e.status_code == 429:
        is_quota = True
    elif any(keyword in err_str for keyword in ["429", "resource_exhausted", "quota", "rate limit", "exhausted", "too many requests"]):
        is_quota = True
        
    if is_quota:
        return "quota"
        
    if any(keyword in err_str for keyword in ["timeout", "timed out"]):
        return "timeout"
        
    if any(keyword in err_str for keyword in ["json", "decode", "parsing", "format"]):
        return "parsing"
        
    return "other"


def run_one_model(candidate, prompt, max_retries_503=2):
    """Exécute un modèle sur un prompt. Retry léger uniquement sur les 503 (transitoires)."""
    provider = candidate["provider"]
    model = candidate["model"]
    result = {
        "name": candidate["name"],
        "provider": provider,
        "model": model,
        "success": False,
        "latency_s": None,
        "completeness_pct": 0.0,
        "error": None,
        "error_type": None,
        "justesse_extraction_pct": None,
        "classification_pertinente": None,
    }

    start = time.time()
    attempt = 0
    while attempt <= max_retries_503:
        try:
            if provider == "gemini":
                data = _call_gemini(prompt, model)
            elif provider == "groq":
                data = _call_groq(prompt, model)
            elif provider == "openrouter":
                data = _call_openrouter(prompt, model)
            elif provider == "mistral":
                data = _call_mistral(prompt, model)
            else:
                raise ValueError(f"Provider inconnu : {provider}")

            result["latency_s"] = round(time.time() - start, 2)
            result["success"] = True
            result["completeness_pct"] = completeness_score(data)
            result["data"] = data
            return result

        except Exception as e:
            err_str = str(e)
            is_503 = "503" in err_str or "UNAVAILABLE" in err_str
            if is_503 and attempt < max_retries_503:
                wait = 5 * (attempt + 1)
                print(f"(503, retry dans {wait}s) ", end="")
                time.sleep(wait)
                attempt += 1
                continue
            result["latency_s"] = round(time.time() - start, 2)
            result["error"] = err_str[:200]
            result["error_type"] = get_error_type(e)
            return result

    return result


PROVIDER_DELAYS = {
    "gemini": 7.0,
    "groq": 2.0,
    "openrouter": 2.0,
    "mistral": 2.0
}


def compare_models_on_cvs(cvs, candidates_to_test=None, delay_between_calls=2.0):
    if candidates_to_test is None:
        candidates_to_test = CANDIDATES
    all_results = []

    for cv in cvs:
        filename = cv["filename"]
        text = cv["text"]
        prompt = build_prompt(text)

        print(f"\n{'='*60}")
        print(f"📄 CV : {filename}")
        print(f"{'='*60}")

        for candidate in candidates_to_test:
            if candidate["provider"] == "gemini" and not client_gemini:
                continue
            if candidate["provider"] == "groq" and not client_groq:
                continue
            if candidate["provider"] == "openrouter" and not openrouter_key:
                continue
            if candidate["provider"] == "mistral" and not mistral_key:
                continue

            print(f"   🤖 Test : {candidate['name']}... ", end="")
            res = run_one_model(candidate, prompt)
            res["filename"] = filename

            if res["success"]:
                acc, rel = evaluate_extraction_accuracy(filename, res["data"])
                res["justesse_extraction_pct"] = acc
                res["classification_pertinente"] = rel
                
                justesse_str = f" | justesse {acc}%" if acc is not None else ""
                class_str = f" | classif pert: {'oui' if rel else 'non'}" if rel is not None else ""
                print(f"✅ {res['latency_s']}s | complétude {res['completeness_pct']}%{justesse_str}{class_str}")
            else:
                print(f"❌ {res['error']} (Type: {res['error_type']})")

            all_results.append(res)
            
            # Pause dynamique par provider pour éviter les 429 (quota)
            delay = PROVIDER_DELAYS.get(candidate["provider"], delay_between_calls)
            time.sleep(delay)

    return all_results


def summarize_results(all_results):
    summary = {}
    for r in all_results:
        name = r["name"]
        if name not in summary:
            summary[name] = {
                "total": 0,
                "success": 0,
                "latencies": [],
                "completeness": [],
                "err_quota": 0,
                "err_timeout": 0,
                "err_parsing": 0,
                "err_other": 0,
                "justesse_scores": [],
                "classification_pertinentes": []
            }
        s = summary[name]
        s["total"] += 1
        if r["success"]:
            s["success"] += 1
            s["latencies"].append(r["latency_s"])
            s["completeness"].append(r["completeness_pct"])
            if r.get("justesse_extraction_pct") is not None:
                s["justesse_scores"].append(r["justesse_extraction_pct"])
            if r.get("classification_pertinente") is not None:
                s["classification_pertinentes"].append(r["classification_pertinente"])
        else:
            err_type = r.get("error_type")
            if err_type == "quota":
                s["err_quota"] += 1
            elif err_type == "timeout":
                s["err_timeout"] += 1
            elif err_type == "parsing":
                s["err_parsing"] += 1
            else:
                s["err_other"] += 1

    rows = []
    for name, s in summary.items():
        success_rate = round(s["success"] / s["total"] * 100, 1) if s["total"] else 0
        avg_latency = round(sum(s["latencies"]) / len(s["latencies"]), 2) if s["latencies"] else None
        avg_completeness = round(sum(s["completeness"]) / len(s["completeness"]), 1) if s["completeness"] else None
        
        avg_justesse = round(sum(s["justesse_scores"]) / len(s["justesse_scores"]), 1) if s["justesse_scores"] else None
        
        class_evals = s["classification_pertinentes"]
        avg_pertinence = round(sum(1.0 for v in class_evals if v) / len(class_evals) * 100, 1) if class_evals else None

        rows.append({
            "modele": name,
            "taux_succes_pct": success_rate,
            "latence_moyenne_s": avg_latency,
            "completude_moyenne_pct": avg_completeness,
            "erreur_quota": s["err_quota"],
            "erreur_timeout": s["err_timeout"],
            "erreur_parsing": s["err_parsing"],
            "erreur_autre": s["err_other"],
            "justesse_moyenne_pct": avg_justesse,
            "classification_pertinente_pct": avg_pertinence,
            "nb_tests": s["total"],
        })

    rows.sort(key=lambda x: (-x["taux_succes_pct"], -(x["justesse_moyenne_pct"] or 0), -(x["completude_moyenne_pct"] or 0)))
    return rows


def print_summary_table(rows):
    print(f"\n{'='*125}")
    print("📊 TABLEAU COMPARATIF AMÉLIORÉ")
    print(f"{'='*125}")
    print(f"{'Modèle':<28} {'Succès %':>9} {'Latence (s)':>12} {'Complétude %':>13} {'Justesse %':>11} {'Classif %':>10} {'Quota':>6} {'Timeout':>8} {'Parse':>6} {'Autre':>6} {'Tests':>6}")
    print("-" * 125)
    for row in rows:
        lat = row["latence_moyenne_s"] if row["latence_moyenne_s"] is not None else "-"
        comp = row["completude_moyenne_pct"] if row["completude_moyenne_pct"] is not None else "-"
        just = row["justesse_moyenne_pct"] if row["justesse_moyenne_pct"] is not None else "-"
        classif = row["classification_pertinente_pct"] if row["classification_pertinente_pct"] is not None else "-"
        print(f"{row['modele']:<28} {row['taux_succes_pct']:>8}% {lat!s:>12} {comp!s:>12}% {just!s:>10}% {classif!s:>9}% {row['erreur_quota']:>6} {row['erreur_timeout']:>8} {row['erreur_parsing']:>6} {row['erreur_autre']:>6} {row['nb_tests']:>6}")


def save_reports(all_results, summary_rows, suffix="", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_path = os.path.join(output_dir, f"comparaison_detail_{suffix}{timestamp}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(output_dir, f"comparaison_resume_{suffix}{timestamp}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "modele", "taux_succes_pct", "latence_moyenne_s", "completude_moyenne_pct",
            "erreur_quota", "erreur_timeout", "erreur_parsing", "erreur_autre",
            "justesse_moyenne_pct", "classification_pertinente_pct", "nb_tests"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n💾 Rapport détaillé : {detail_path}")
    print(f"💾 Résumé CSV       : {summary_path}")


if __name__ == "__main__":
    import argparse
    import sys
    from cv_reader import read_all_cvs

    parser = argparse.ArgumentParser(description="Étude comparative des LLMs sur l'extraction de CVs.")
    parser.add_argument("--all", action="store_true", help="Tester tous les modèles candidats.")
    parser.add_argument("--provider", action="append", help="Filtrer par fournisseur (groq, gemini, openrouter, mistral). Peut être répété.")
    parser.add_argument("--model", action="append", help="Filtrer par nom exact du modèle. Peut être répété.")
    
    args = parser.parse_args()

    # Dictionnaire des clés pour valider leur existence
    API_KEYS = {
        "gemini": gemini_key,
        "groq": groq_key,
        "openrouter": openrouter_key,
        "mistral": mistral_key
    }
    
    known_providers = set(c["provider"] for c in CANDIDATES)
    known_models = set(c["model"] for c in CANDIDATES)

    # 1. Validation de l'existence des filtres ciblés
    if args.provider:
        for p in args.provider:
            if p not in known_providers:
                print(f"❌ Fournisseur '{p}' inconnu.")
                print(f"Fournisseurs existants dans CANDIDATES : {', '.join(sorted(known_providers))}")
                sys.exit(1)

    if args.model:
        for m in args.model:
            if m not in known_models:
                print(f"❌ Modèle '{m}' inconnu.")
                print(f"Modèles existants dans CANDIDATES : {', '.join(sorted(known_models))}")
                sys.exit(1)

    # 2. Validation de la configuration des clés API pour les filtres ciblés
    available_providers_with_keys = [p for p, k in API_KEYS.items() if k]
    
    if args.provider:
        for p in args.provider:
            if not API_KEYS.get(p):
                print(f"❌ Clé API manquante dans .env pour le fournisseur '{p}'.")
                print(f"Fournisseurs configurés disponibles : {', '.join(available_providers_with_keys)}")
                sys.exit(1)

    if args.model:
        for m in args.model:
            cand = next((c for c in CANDIDATES if c["model"] == m), None)
            if cand:
                p = cand["provider"]
                if not API_KEYS.get(p):
                    print(f"❌ Clé API manquante dans .env pour le fournisseur '{p}' (requis pour le modèle '{m}').")
                    print(f"Fournisseurs configurés disponibles : {', '.join(available_providers_with_keys)}")
                    sys.exit(1)

    # 3. Filtrage des candidats
    if args.all or not (args.provider or args.model):
        candidates_to_test = CANDIDATES.copy()
    else:
        candidates_to_test = []
        for candidate in CANDIDATES:
            match_provider = (args.provider and candidate["provider"] in args.provider)
            match_model = (args.model and candidate["model"] in args.model)
            if match_provider or match_model:
                candidates_to_test.append(candidate)

    if not candidates_to_test:
        print("⚠️ Aucun candidat ne correspond aux filtres spécifiés.")
        sys.exit(0)

    # 4. Calcul du suffixe pour le nom de fichier
    providers_tested = set(c["provider"] for c in candidates_to_test)
    suffix = ""
    if len(providers_tested) == 1:
        suffix = f"{list(providers_tested)[0]}_"

    cvs = read_all_cvs("cvs/")
    # cvs = cvs[:3]  # décommente pour un test rapide

    all_results = compare_models_on_cvs(cvs, candidates_to_test=candidates_to_test, delay_between_calls=2.0)
    summary_rows = summarize_results(all_results)
    print_summary_table(summary_rows)
    save_reports(all_results, summary_rows, suffix=suffix)