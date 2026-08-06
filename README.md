# Pipeline de Traitement, Classification et Recherche de CVs

Pipeline complet d'**extraction**, de **classification**, de **scoring qualité**,
de **détection de doublons**, d'**indexation Elasticsearch** et de **recherche
(mots-clés + RAG conversationnel)** de CVs au format PDF. Utilise des LLMs
multi-fournisseurs (Groq, Gemini, Mistral, OpenRouter) pour l'extraction
qualitative, avec **tout le scoring, l'agrégation et les calculs faits en
Python déterministe** — le LLM ne fait jamais de calcul final, seulement de
l'évaluation qualitative par unité (voir section Anti-hallucination).

---

## 🛠️ Architecture du Projet

### Pipeline (scripts Python)

| Module | Rôle |
| :--- | :--- |
| **`cv_reader.py`** | Lit les PDF avec 3 stratégies en cascade : texte brut (PyMuPDF) → colonnes (pdfplumber) → OCR (Tesseract, pour les CVs scannés/images). |
| **`cv_extractor.py`** | Extraction LLM (4 fournisseurs) + tout le post-traitement déterministe : score de qualité pondéré, années d'expérience, alertes de parcours (trous/chevauchements), normalisation des noms de domaine par embeddings, scores de domaine pondérés (60 % pertinence LLM + 40 % qualité globale), calcul de l'embedding CV pour la recherche sémantique. |
| **`cv_saver.py`** | Exporte en **JSON** (`output/cvs_data.json`) et **Excel** (`output/cvs_data.xlsx`). |
| **`cv_comparator.py`** | Étude comparative des LLMs (succès, latence, complétude, justesse, pertinence classification) — partage le même prompt et les mêmes fonctions de scoring que `cv_extractor.py` pour une comparaison équitable. |
| **`cv_deduplication.py`** | Détection de doublons à 3 niveaux : (1) hash SHA-256 du texte brut = doublon exact, (2) email/téléphone identique = même candidat, (3) similarité sémantique (cosine sur embeddings) ≥ seuil = doublon probable. |
| **`cv_cache.py`** | Cache des extractions déjà faites, indexé par hash SHA-256 du fichier. Backend JSON par défaut, PostgreSQL si les variables `POSTGRES_*` sont renseignées. |
| **`es_client.py`** | Client Elasticsearch centralisé pour l'app Streamlit (mis en cache via `@st.cache_resource`), avec repli sur un client factice si ES est injoignable (évite de crasher le Dashboard). |
| **`create_index.py`** | Crée l'index Elasticsearch `cvs` avec le mapping complet (champs `nested` pour les scores par domaine et les expériences pro, `dense_vector` pour l'embedding). |
| **`cv_injector.py`** | Injecte en masse le JSON produit par `cv_saver.py` dans Elasticsearch. |

### Application Streamlit (`app.py` + `pages/`)

| Page | Rôle |
| :--- | :--- |
| **📊 Dashboard** (`pages/1_Dashboard.py`) | KPI et visualisations globales sur les CVs indexés. |
| **🔎 Recherche** (`pages/2_Recherche.py`) | Filtrage avancé multi-critères (domaine, compétences techniques, diplômes, langues, score qualité, années d'expérience, expérience pro) avec agrégations Elasticsearch mises en cache et bouton de rafraîchissement. |
| **💬 Chatbot RAG** (`pages/3_Chatbot.py`) | Assistant conversationnel : détecte l'intention de la question (comparaison, top N, statistiques, filtrage par catégorie...), récupère les CVs pertinents dans Elasticsearch (kNN sur l'embedding ou requêtes structurées), puis fait générer la réponse par un LLM **à partir des seuls documents récupérés**. |
| **📝 Ajouter CV** (`pages/ajouter_cv.py`) | Upload PDF → extraction → vérification de doublon (3 niveaux) → preview → indexation avec un ID Elasticsearch stable (hash du fichier). |

### Infrastructure

- **`docker-compose.yml`** : Elasticsearch 8.12 + Kibana 8.12, sécurité désactivée (`xpack.security.enabled=false`) — **pensé pour du développement local uniquement**, à ne jamais exposer tel quel sur un réseau public.

---

## 🧠 Principe architectural : anti-hallucination

Le LLM est utilisé **uniquement** pour de l'évaluation qualitative par unité
(un projet, une certification, un domaine à la fois). **Tous les calculs
finaux sont du Python déterministe** :

- Score de qualité global = moyenne pondérée de 5 composantes (diplôme,
  certifications, diversité technique, projets, langues), poids configurables
  via `.env` et automatiquement normalisés à 100 %.
- Score final par domaine = 60 % pertinence donnée par le LLM + 40 % score de
  qualité global — jamais un score inventé tel quel par le LLM.
- Années d'expérience et alertes de parcours (trous, chevauchements) :
  calculées à partir des dates extraites, pas estimées par le LLM.
- Détection de doublons : hash, égalité champ à champ, similarité cosinus —
  aucune décision de doublon n'est laissée au LLM.

---

## 📊 Résultats de l'Étude Comparative

Exécutée sur **10 CVs réels**, avec **5 configurations LLM** :

| Modèle / IA | Succès % | Latence Moy. | Complétude % | Justesse % | Classification % | Quota (429) | Tests |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Llama-3.3-70B (Groq)** | **100%** | **1.41s** | 90.0% | 100% | 100% | 0 | 10 |
| **GPT-OSS-20B (OpenRouter)** | 80% | 19.10s | 89.0% | 100% | 100% | 0 | 10 |
| **OpenRouter Free (auto)** | 80% | 46.48s | 88.2% | 100% | 100% | 1 | 10 |
| **Gemini 2.5 Flash** | 80% | 15.89s | 88.2% | 100% | 100% | 2 | 10 |
| **Gemini 2.5 Flash-Lite** | 40% | 21.82s | 89.7% | 100% | 100% | 6 | 10 |

> 💡 Les modèles Gemini atteignent 100 % de justesse et de pertinence de
> classification lors des appels réussis. Leurs taux de succès inférieurs
> viennent uniquement des quotas du tier gratuit (`429 RESOURCE_EXHAUSTED`),
> pas d'un défaut d'analyse.

---

## ⚙️ Installation et Configuration

### Prérequis
- Python 3.10+
- Tesseract OCR installé et accessible dans le PATH (pour l'OCR des CVs scannés)
- Poppler (requis par `pdf2image` pour la conversion PDF → image)
- Docker (pour Elasticsearch + Kibana)

### 1. Cloner le dépôt
```bash
git clone https://github.com/djarbouiamine/cv-pipeline-stage.git
cd cv-pipeline-stage
git checkout stage
```

### 2. Installer les dépendances Python
```bash
pip install -r requirements.txt
```

### 3. Lancer Elasticsearch + Kibana
```bash
docker compose up -d
```
Kibana est ensuite accessible sur http://localhost:5601, Elasticsearch sur
http://localhost:9200.

### 4. Créer l'index Elasticsearch
```bash
python create_index.py
```

### 5. Configurer les variables d'environnement
Créer un fichier `.env` à la racine :
```env
# Au moins une clé LLM est requise
GROQ_API_KEY=votre_cle_groq
GEMINI_API_KEY=votre_cle_gemini
OPENROUTER_API_KEY=votre_cle_openrouter
MISTRAL_API_KEY=votre_cle_mistral

# Elasticsearch (optionnel, défauts = local sans auth)
ELASTIC_HOST=http://localhost:9200
# ELASTIC_USERNAME=
# ELASTIC_PASSWORD=

# Bascule automatique entre providers en cas de quota dépassé (optionnel)
AUTO_FALLBACK=false
FALLBACK_ORDER=groq,openrouter,mistral,gemini
GROQ_MAX_AUTO_WAIT_S=1200

# Poids du score de qualité — normalisés automatiquement à 100% (optionnel)
QUALITY_WEIGHT_DIPLOME=25
QUALITY_WEIGHT_CERTIFICATIONS=20
QUALITY_WEIGHT_TECH=20
QUALITY_WEIGHT_PROJETS=25
QUALITY_WEIGHT_LANGUES=10

# Seuil de similarité pour la détection de doublon niveau 3 (optionnel)
DUPLICATE_SIMILARITY_THRESHOLD=0.90

# Cache PostgreSQL au lieu du JSON local (optionnel — sinon fallback JSON auto)
# POSTGRES_HOST=
# POSTGRES_PORT=
# POSTGRES_DB=
# POSTGRES_USER=
# POSTGRES_PASSWORD=
```
*(`.env` est ignoré par Git via `.gitignore`.)*

---

## 🚀 Utilisation

### Pipeline en ligne de commande

Extraire et classifier tous les CVs du dossier `cvs/` :
```bash
python cv_extractor.py
python cv_extractor.py --provider gemini
python cv_extractor.py --provider mistral --model mistral-medium-latest
```

Sauvegarder en JSON/Excel :
```bash
python cv_saver.py
```

Injecter en masse dans Elasticsearch :
```bash
python cv_injector.py
```

Étude comparative des LLMs :
```bash
python cv_comparator.py
python cv_comparator.py --provider groq --provider gemini
```

### Application Streamlit

```bash
streamlit run app.py
```
Donne accès aux 4 pages (Dashboard, Recherche, Chatbot RAG, Ajouter CV) via la
barre de navigation latérale.

---

## 📁 Structure du Projet

```
cv-pipeline-stage/
├── app.py                     # Point d'entrée Streamlit
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Recherche.py
│   ├── 3_Chatbot.py
│   └── ajouter_cv.py
├── cv_reader.py                # Lecture et OCR des PDF
├── cv_extractor.py             # Extraction LLM + scoring déterministe
├── cv_saver.py                 # Export JSON / Excel
├── cv_comparator.py            # Comparaison des LLMs
├── cv_deduplication.py         # Détection de doublons (3 niveaux)
├── cv_cache.py                 # Cache d'extraction (JSON ou PostgreSQL)
├── es_client.py                # Client Elasticsearch pour Streamlit
├── create_index.py             # Création du mapping ES
├── cv_injector.py               # Injection en masse dans ES
├── docker-compose.yml          # Elasticsearch + Kibana (local)
├── requirements.txt
├── cvs/                        # CVs PDF à analyser (non versionné)
├── output/                     # JSON / Excel / CSV générés (non versionné)
└── .env                        # Clés API et config (non versionné)
```

---

## 🚧 Points de vigilance identifiés (à traiter)

- `cvs_uploads/` (CVs uploadés via l'app) doit être ajouté à `.gitignore` —
  contient des documents personnels de candidats réels, à ne pas versionner.
- `requirements.txt` doit être complété (PyMuPDF, pytesseract, opencv-python,
  pdf2image, pdfplumber, groq, google-genai, requests, openpyxl, numpy,
  psycopg2-binary en optionnel).
- Harmoniser l'ID Elasticsearch entre `cv_injector.py` (injection en masse) et
  `pages/ajouter_cv.py` (ajout unitaire) — ce dernier utilise déjà un hash
  stable, à reporter dans `cv_injector.py` pour éviter les doublons d'index
  entre les deux chemins d'ajout.
