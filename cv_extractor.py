from google import genai
from google.genai.errors import APIError
import json
import time
import re
import os

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

def get_retry_delay(err):
    """
    Tente d'extraire le délai d'attente (retry delay) à partir de l'exception.
    """
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
        
    return 10.0  # Valeur par défaut si non trouvé

# Nouvelle bibliothèque google-genai
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def extract_cv_data(text):
    """
    Prend le texte brut d'un CV
    Retourne un dictionnaire structuré
    """
    
    prompt = f"""
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

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )
            result = response.text.strip()
            result = result.replace("```json", "").replace("```", "").strip()
            return json.loads(result)
        except APIError as e:
            if e.code in [429, 500, 502, 503, 504]:
                if e.code == 429:
                    delay = get_retry_delay(e)
                    wait_time = delay + 1.0
                    error_msg = "Quota dépassé (429 RESOURCE_EXHAUSTED)"
                else:
                    wait_time = (2 ** attempt) * 5.0
                    error_msg = f"Erreur serveur temporaire ({e.code})"
                print(f"⚠️ {error_msg}. Attente de {wait_time:.2f}s avant de réessayer (essai {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
    
    # Dernier essai hors du bloc try/except pour propager l'erreur finale si tout échoue
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    result = response.text.strip()
    result = result.replace("```json", "").replace("```", "").strip()
    return json.loads(result)


def extract_all_cvs(cvs):
    """
    Prend la liste des CVs du Reader
    Retourne la liste avec les données extraites
    """
    extracted = []
    
    for i, cv in enumerate(cvs):
        if i > 0:
            # Petit délai de politesse entre chaque CV pour éviter de saturer les quotas gratuits
            print("⏳ Pause de 2 secondes avant l'extraction suivante...")
            time.sleep(2.0)
            
        print(f"🤖 Extraction : {cv['filename']}")
        data = extract_cv_data(cv['text'])
        extracted.append({
            "filename": cv['filename'],
            "data": data
        })
        print(f"✅ Extrait : {cv['filename']}")
    
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