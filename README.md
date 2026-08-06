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
| **`cv_extractor.py`** | Extraction LLM (4 fournisseurs) + post-traitement déterministe : score de qualité pondéré, années d'expérience, alertes de parcours, normalisation des domaines par embeddings, scores de domaine pondérés (60 % pertinence LLM + 40 % qualité globale), embedding CV pour la recherche sémantique. |
| **`cv_saver.py`** | Exporte en **JSON** (`output/cvs_data.json`) et **Excel** (`output/cvs_data.xlsx`). |
| **`cv_comparator.py`** | Étude comparative des LLMs (succès, latence, complétude, justesse, pertinence classification). |
| **`cv_deduplication.py`** | Détection de doublons à 3 niveaux : (1) hash SHA-256 du texte brut, (2) email/téléphone identique, (3) similarité sémantique (cosine sur embeddings) ≥ seuil. |
| **`cv_cache.py`** | Cache des extractions déjà faites, indexé par hash SHA-256 du fichier. Backend JSON par défaut, PostgreSQL si les variables `POSTGRES_*` sont renseignées. |
| **`es_client.py`** | Client Elasticsearch centralisé pour l'app Streamlit (`@st.cache_resource`), avec repli sur un client factice si ES est injoignable. |
| **`create_index.py`** | Crée l'index Elasticsearch `cvs` avec le mapping complet (`nested`, `dense_vector` pour l'embedding). |
| **`cv_injector.py`** | Injecte en masse le JSON produit par `cv_saver.py` dans Elasticsearch. |

### Module Chatbot (`chatbot/`)

Logique métier extraite de la page Streamlit pour le routage, le ranking et les réponses recruteur :

| Module | Rôle |
| :--- | :--- |
| **`intent.py`** | Classification de l'intention (comparaison, top N, stats, filtrage, sémantique…). |
| **`category_resolver.py`** | Résolution des catégories/domaines mentionnés dans la question. |
| **`mandatory_criteria.py`** | Détection des critères obligatoires (diplôme, langue, compétence…) et filtrage des CVs. |
| **`recommendation_ranking.py`** | Classement des candidats pour une recommandation ciblée. |
| **`final_ranking.py`** | Ranking final multi-critères après récupération ES. |
| **`es_aggregations.py`** | Requêtes et agrégations Elasticsearch pour les questions statistiques. |
| **`recruiter_helpers.py`** | Helpers pour formater les réponses orientées recruteur. |
| **`llm_client.py`** | Client LLM dual-mode (conversation générale + RAG CV). |
| **`system_prompt.py`** | Prompts système anti-hallucination pour le RAG. |
| **`response_format.py`** | Instructions de format de réponse par type de question. |

### Application Streamlit (`app.py` + `pages/`)

| Page | Rôle |
| :--- | :--- |
| **📊 Dashboard** (`pages/1_Dashboard.py`) | KPI et visualisations globales sur les CVs indexés. |
| **🔎 Recherche** (`pages/2_Recherche.py`) | Filtrage avancé multi-critères avec agrégations Elasticsearch mises en cache. |
| **💬 Chatbot RAG** (`pages/3_Chatbot.py`) | Assistant dual-mode : conversation générale + RAG sur les CVs (kNN, requêtes structurées, comparaisons, stats, ranking recruteur). |
| **📝 Ajouter CV** (`pages/4_Ajouter_CV.py`) | Upload PDF → extraction → vérification de doublon (3 niveaux) → preview → indexation avec ID stable (hash du fichier). |

### Scripts utilitaires (`scripts/`)

| Script | Rôle |
| :--- | :--- |
| **`evaluate_chatbot.py`** | Suite d'évaluation du routage et de la récupération du chatbot (sans Streamlit). |

### Infrastructure

- **`docker-compose.yml`** : Elasticsearch 8.12 + Kibana 8.12, sécurité désactivée — **développement local uniquement**.

---

## 🧠 Principe architectural : anti-hallucination

Le LLM est utilisé **uniquement** pour de l'évaluation qualitative par unité
(un projet, une certification, un domaine à la fois). **Tous les calculs
finaux sont du Python déterministe** :

- Score de qualité global = moyenne pondérée de 5 composantes (diplôme,
  certifications, diversité technique, projets, langues).
- Score final par domaine = 60 % pertinence LLM + 40 % score de qualité global.
- Années d'expérience et alertes de parcours : calculées à partir des dates extraites.
- Détection de doublons : hash, égalité champ à champ, similarité cosinus.

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

> Les modèles Gemini atteignent 100 % de justesse lors des appels réussis.
> Les échecs viennent des quotas du tier gratuit (`429`), pas d'un défaut d'analyse.

---

## ⚙️ Installation et Configuration

### Prérequis

- Python 3.10+
- Tesseract OCR dans le PATH (CVs scannés)
- Poppler (requis par `pdf2image`)
- Docker (Elasticsearch + Kibana)

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

Kibana : http://localhost:5601 — Elasticsearch : http://localhost:9200

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

# Elasticsearch (optionnel)
ELASTIC_HOST=http://localhost:9200

# Bascule automatique entre providers (optionnel)
AUTO_FALLBACK=false
FALLBACK_ORDER=groq,openrouter,mistral,gemini
GROQ_MAX_AUTO_WAIT_S=1200

# Poids du score de qualité (optionnel)
QUALITY_WEIGHT_DIPLOME=25
QUALITY_WEIGHT_CERTIFICATIONS=20
QUALITY_WEIGHT_TECH=20
QUALITY_WEIGHT_PROJETS=25
QUALITY_WEIGHT_LANGUES=10

# Seuil de similarité doublon niveau 3 (optionnel)
DUPLICATE_SIMILARITY_THRESHOLD=0.90

# Cache PostgreSQL (optionnel — sinon JSON local)
# POSTGRES_HOST=
# POSTGRES_PORT=
# POSTGRES_DB=
# POSTGRES_USER=
# POSTGRES_PASSWORD=
```

*(`.env` et `cvs_uploads/` sont ignorés par Git.)*

---

## 🚀 Utilisation

### Pipeline en ligne de commande

```bash
# Extraire et classifier les CVs du dossier cvs/
python cv_extractor.py
python cv_extractor.py --provider gemini

# Sauvegarder en JSON/Excel
python cv_saver.py

# Injecter en masse dans Elasticsearch
python cv_injector.py

# Étude comparative des LLMs
python cv_comparator.py
python cv_comparator.py --provider groq --provider gemini

# Évaluer le chatbot (routage + récupération)
python scripts/evaluate_chatbot.py
```

### Application Streamlit

```bash
streamlit run app.py
```

Donne accès aux 4 pages via la barre de navigation latérale.

---

## 📁 Structure du Projet

```
cv-pipeline-stage/
├── app.py                      # Point d'entrée Streamlit
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Recherche.py
│   ├── 3_Chatbot.py
│   └── 4_Ajouter_CV.py
├── chatbot/                    # Logique RAG, intent, ranking recruteur
│   ├── intent.py
│   ├── category_resolver.py
│   ├── mandatory_criteria.py
│   ├── recommendation_ranking.py
│   ├── final_ranking.py
│   ├── es_aggregations.py
│   ├── recruiter_helpers.py
│   ├── llm_client.py
│   ├── system_prompt.py
│   └── response_format.py
├── scripts/
│   └── evaluate_chatbot.py
├── cv_reader.py
├── cv_extractor.py
├── cv_saver.py
├── cv_comparator.py
├── cv_deduplication.py
├── cv_cache.py
├── es_client.py
├── create_index.py
├── cv_injector.py
├── docker-compose.yml
├── requirements.txt
├── cvs/                        # CVs PDF pipeline CLI (non versionné)
├── cvs_uploads/                # CVs uploadés via l'app (non versionné)
├── output/                     # JSON / Excel générés (non versionné)
└── .env                        # Clés API (non versionné)
```

---

## 🚧 Points de vigilance

- **`cvs_uploads/`** contient des documents personnels de candidats — ne jamais les committer (désormais dans `.gitignore`).
- **`cv_injector.py`** utilise un ID numérique séquentiel (`0`, `1`, …) tandis que **`pages/4_Ajouter_CV.py`** utilise un hash stable du fichier. Harmoniser les deux chemins d'indexation pour éviter les doublons entre injection en masse et ajout unitaire.
- **`pages/3_Chatbot.py`** importe `from openrouter import OpenRouter` — vérifier que le package est installé ou remplacer par un appel `requests` comme dans `cv_extractor.py`.
