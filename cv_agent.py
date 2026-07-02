"""
cv_agent.py — Étape "agent" : pipeline à 2 étapes (extraction + vérification)

Contrairement à cv_extractor.py qui fait UN seul appel LLM (extraction +
classification en une fois), ce module ajoute une 2ème étape : un agent de
vérification qui relit le CV et le JSON produit, et corrige la classification
si elle ne correspond pas vraiment au profil du candidat.

Objectif : combiner la vitesse d'un modèle rapide (ex: Groq) avec une
meilleure fiabilité de classification (point faible identifié dans
cv_comparator.py : Groq seul = 75% de pertinence de classification).
"""

import json
import time

from cv_extractor import (
    build_prompt,
    _call_groq,
    _call_gemini,
    _call_mistral,
    _call_openrouter,
    AVAILABLE_KEYS,
    DEFAULT_MODELS,
)


def build_verification_prompt(cv_text, extracted_data):
    """
    Construit le prompt du 2ème agent : il relit le CV et le JSON déjà
    produit, et vérifie/corrige UNIQUEMENT la classification (le reste du
    JSON n'est pas retouché pour ne pas introduire de nouvelles erreurs).
    """
    categorie_actuelle = extracted_data.get("categorie_principale", "")
    scores_actuels = extracted_data.get("scores_categories", {})

    return f"""
Tu es un expert RH senior chargé de VÉRIFIER une classification déjà faite
par un premier système, pas de refaire toute l'extraction.

Voici le CV original :
{cv_text}

Voici la classification proposée par le premier système :
- Domaine principal : {categorie_actuelle}
- Scores par domaine : {json.dumps(scores_actuels, ensure_ascii=False)}

Ta tâche :
1. Relis le CV attentivement.
2. Vérifie si le domaine principal proposé correspond vraiment au profil
   du candidat.
3. Si c'est correct, renvoie exactement la même classification.
4. Si ce n'est PAS correct ou incomplet, corrige-la avec les domaines qui
   correspondent réellement au profil.

Retourne UNIQUEMENT un JSON avec ce format, rien d'autre :
{{
    "scores_categories": {{
        "Domaine 1": 0,
        "Domaine 2": 0
    }},
    "categorie_principale": "le domaine avec le score le plus élevé",
    "correction_appliquee": true ou false,
    "justification": "courte explication de la vérification (1 phrase)"
}}
"""


def _dispatch(provider, prompt, model):
    if provider == "groq":
        return _call_groq(prompt, model)
    elif provider == "gemini":
        return _call_gemini(prompt, model)
    elif provider == "mistral":
        return _call_mistral(prompt, model)
    elif provider == "openrouter":
        return _call_openrouter(prompt, model)
    else:
        raise ValueError(f"Provider inconnu : {provider}")


def extract_with_agent(
    cv_text,
    extraction_provider="groq",
    extraction_model=None,
    verification_provider="gemini",
    verification_model=None,
):
    """
    Pipeline à 2 étapes :
      1. Extraction rapide (provider/model au choix, ex: Groq)
      2. Vérification/correction de la classification (provider/model au
         choix, idéalement un modèle plus précis, ex: Gemini)

    Retourne un dict avec :
      - toutes les données extraites à l'étape 1
      - la classification éventuellement corrigée à l'étape 2
      - des métadonnées sur ce que l'agent a fait (correction_appliquee,
        justification, latence de chaque étape)
    """
    extraction_model = extraction_model or DEFAULT_MODELS[extraction_provider]
    verification_model = verification_model or DEFAULT_MODELS[verification_provider]

    if not AVAILABLE_KEYS.get(extraction_provider):
        raise RuntimeError(f"Clé API manquante pour '{extraction_provider}' (extraction)")
    if not AVAILABLE_KEYS.get(verification_provider):
        raise RuntimeError(f"Clé API manquante pour '{verification_provider}' (vérification)")

    # ── Étape 1 : extraction rapide ──────────────────────────
    t0 = time.time()
    prompt_extraction = build_prompt(cv_text)
    extracted_data = _dispatch(extraction_provider, prompt_extraction, extraction_model)
    t1 = time.time()

    # ── Étape 2 : agent de vérification de la classification ─
    prompt_verif = build_verification_prompt(cv_text, extracted_data)
    verif_result = _dispatch(verification_provider, prompt_verif, verification_model)
    t2 = time.time()

    # Fusion : on garde toutes les données de l'étape 1, on remplace
    # uniquement la classification par la version vérifiée
    final_data = dict(extracted_data)
    final_data["scores_categories"] = verif_result.get("scores_categories", extracted_data.get("scores_categories"))
    final_data["categorie_principale"] = verif_result.get("categorie_principale", extracted_data.get("categorie_principale"))

    return {
        "data": final_data,
        "agent_meta": {
            "extraction_provider": extraction_provider,
            "extraction_model": extraction_model,
            "verification_provider": verification_provider,
            "verification_model": verification_model,
            "correction_appliquee": verif_result.get("correction_appliquee", False),
            "justification": verif_result.get("justification", ""),
            "latence_extraction_s": round(t1 - t0, 2),
            "latence_verification_s": round(t2 - t1, 2),
            "latence_totale_s": round(t2 - t0, 2),
        }
    }


# TEST
if __name__ == "__main__":
    from cv_reader import read_all_cvs

    cvs = read_all_cvs("cvs/")

    for cv in cvs:
        print(f"\n{'='*60}")
        print(f"📄 {cv['filename']}")
        print(f"{'='*60}")
        try:
            result = extract_with_agent(
                cv["text"],
                extraction_provider="groq",
                verification_provider="mistral",   # au lieu de gemini (quota 20/jour trop bas)
            )
            meta = result["agent_meta"]
            print(f"   Étape 1 ({meta['extraction_provider']}/{meta['extraction_model']}) : {meta['latence_extraction_s']}s")
            print(f"   Étape 2 ({meta['verification_provider']}/{meta['verification_model']}) : {meta['latence_verification_s']}s")
            print(f"   Correction appliquée : {meta['correction_appliquee']}")
            if meta["correction_appliquee"]:
                print(f"   Justification : {meta['justification']}")
            print(f"   Domaine final : {result['data'].get('categorie_principale')}")
        except Exception as e:
            print(f"   ❌ Échec : {e}")