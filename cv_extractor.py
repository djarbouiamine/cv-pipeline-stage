from groq import Groq, APIError, RateLimitError, APIConnectionError
import json
import time
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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "⚠️ GROQ_API_KEY manquant. Ajoute-le dans ton fichier .env "
        "(GROQ_API_KEY=ta_cle_ici) avant de lancer l'extraction."
    )

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Score de qualité du CV
# Principe : le LLM évalue qualitativement (projets, certifications) ou
# extrait du texte brut (niveau de langue) ; Python fait TOUT le calcul
# et l'agrégation. Aucun calcul de score global n'est fait par le LLM.
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

    # Fallback (ancien format en cache / réponse incomplète)
    certifications = data.get("certifications") or []
    if not certifications:
        return 0
    return min(100, len(certifications) * 20)


def calculate_tech_score(data):
    """
    Score 0-100 basé sur la diversité technique : langages, frameworks,
    bases de données, outils DevOps, technologies confondus.
    """
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
    Fallback sur un comptage simple si le LLM n'a pas fourni d'évaluation.
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

    # Fallback (ancien format en cache / réponse incomplète)
    projets = data.get("projets") or []
    nb = len(projets) if isinstance(projets, list) else 0
    return min(100, nb * 15)


def map_niveau_langue(niveau_brut):
    """
    Convertit le niveau de langue brut (extrait tel quel du CV par le LLM)
    en score 0-100. C'est un mapping Python transparent, pas un jugement LLM.
    """
    if not niveau_brut:
        return 50  # neutre : langue citée sans niveau précisé
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
    """
    Score 0-100 basé sur les niveaux de langue extraits par le LLM
    (evaluation_langues -> niveau_brut), mappés en score par Python.
    Fallback sur un comptage simple si absent.
    """
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
    Pondération :
      - Diplôme               : 25%  (Python, mots-clés)
      - Certifications        : 20%  (LLM évalue chaque certif -> Python agrège)
      - Diversité technique   : 20%  (Python, comptage — inclut les frameworks)
      - Projets                : 25%  (LLM évalue chaque projet -> Python agrège)
      - Langues                : 10%  (LLM extrait le niveau -> Python mappe)
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
    """
    Score final par domaine = 60% pertinence domaine (donnée par le LLM)
    + 40% qualité globale du profil. Fourni en /100 et en /10.
    """
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
    """
    Tente d'extraire le délai d'attente recommandé à partir des headers
    de la réponse Groq (header 'retry-after'). Si absent, retourne
    une valeur par défaut.
    """
    try:
        response = getattr(err, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                return float(retry_after)
    except Exception:
        pass
    return default


def extract_cv_data(text):
    """
    Prend le texte brut d'un CV.
    Retourne un dictionnaire structuré (via Groq / llama-3.3-70b-versatile)
    enrichi des scores de qualité calculés par Python.
    """

    prompt = f"""
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
        {{
            "nom": "nom du projet 1",
            "description": "une seule ligne, basée sur le CV uniquement",
            "score_importance": 0
        }}
    ],
    "diplomes": ["...", "..."],
    "certifications": ["...", "..."],
    "evaluation_certifications": [
        {{
            "nom": "nom de la certification",
            "score_qualite": 0
        }}
    ],
    "langues": ["...", "..."],
    "evaluation_langues": [
        {{
            "langue": "...",
            "niveau_brut": "tel qu'écrit dans le CV"
        }}
    ]
}}

Voici le CV :
{text}
"""

    max_retries = 5
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            result = response.choices[0].message.content.strip()
            result = result.replace("```json", "").replace("```", "").strip()

            try:
                parsed = json.loads(result)

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
                # Le modèle a renvoyé un JSON invalide/tronqué : on retente
                # au lieu de planter tout le pipeline.
                print(
                    f"⚠️ JSON invalide reçu de Groq (essai {attempt + 1}/{max_retries}) : {je}"
                )
                last_error = je
                time.sleep(2.0)
                continue

        except RateLimitError as e:
            delay = get_retry_delay(e, default=10.0)
            wait_time = delay + 1.0
            print(
                f"⚠️ Quota/rate limit Groq (429). Attente de {wait_time:.2f}s "
                f"avant de réessayer (essai {attempt + 1}/{max_retries})..."
            )
            time.sleep(wait_time)
            last_error = e

        except APIConnectionError as e:
            wait_time = (2 ** attempt) * 5.0
            print(
                f"⚠️ Erreur de connexion à Groq. Attente de {wait_time:.2f}s "
                f"avant de réessayer (essai {attempt + 1}/{max_retries})..."
            )
            time.sleep(wait_time)
            last_error = e

        except APIError as e:
            status = getattr(e, "status_code", None)
            if status in [500, 502, 503, 504]:
                wait_time = (2 ** attempt) * 5.0
                print(
                    f"⚠️ Erreur serveur Groq ({status}). Attente de {wait_time:.2f}s "
                    f"avant de réessayer (essai {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
                last_error = e
            else:
                # Erreur non récupérable (ex: mauvaise clé API, modèle inexistant)
                raise e

    # Si tous les essais ont échoué, on lève une erreur explicite au lieu
    # de laisser un dernier appel "silencieux" planter le programme.
    raise RuntimeError(
        f"❌ Extraction impossible après {max_retries} tentatives. "
        f"Dernière erreur : {last_error}"
    )


def extract_all_cvs(cvs):
    """
    Prend la liste des CVs du Reader.
    Retourne la liste avec les données extraites.
    Un CV en échec est loggé et ignoré, il ne bloque plus tout le lot.
    """
    extracted = []

    for i, cv in enumerate(cvs):
        if i > 0:
            # Petit délai de politesse entre chaque CV pour éviter de saturer les quotas gratuits
            print("⏳ Pause de 2 secondes avant l'extraction suivante...")
            time.sleep(2.0)

        print(f"🤖 Extraction : {cv['filename']}")
        try:
            data = extract_cv_data(cv['text'])
            extracted.append({
                "filename": cv['filename'],
                "data": data
            })
            print(f"✅ Extrait : {cv['filename']}")
        except Exception as e:
            print(f"❌ Échec sur {cv['filename']} : {e}")
            extracted.append({
                "filename": cv['filename'],
                "data": None,
                "error": str(e)
            })

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
        if result.get("data"):
            print(json.dumps(result['data'], indent=2, ensure_ascii=False))
        else:
            print(f"❌ Erreur : {result.get('error')}")