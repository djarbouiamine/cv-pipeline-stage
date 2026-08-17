# CV Pipeline — Intelligence de Recrutement

> **Extraction** • **Classification** • **Scoring** • **Recherche sémantique** • **Chatbot RAG**
>
> Pipeline complet pour analyser des CVs PDF avec LLM + Elasticsearch. Tous les calculs sont déterministes en Python — le LLM ne fait jamais de calcul final.

---

## ⚡ Démarrage rapide (5 étapes)

### Prérequis

| Outil | Version | Utilité |
|---|---|---|
| Python | 3.10+ | Runtime |
| Docker Desktop | Dernière | Elasticsearch + Kibana |
| Tesseract OCR | Dernière | CVs scannés/images |
| Poppler | Dernière | Conversion PDF |
| Clé API LLM | — | Groq (gratuit), Gemini, Mistral ou OpenRouter |

---

### Étape 1 — Cloner le projet

```bash
git clone https://github.com/djarbouiamine/cv-pipeline-stage.git
cd cv-pipeline-stage
git checkout stage
```

---

### Étape 2 — Installer les dépendances Python

```bash
pip install -r requirements.txt
```

---

### Étape 3 — Configurer les clés API

Créer un fichier `.env` à la racine du projet :

```env
# Minimum : une seule clé suffit (Groq recommandé — gratuit et rapide)
GROQ_API_KEY=votre_cle_groq
GEMINI_API_KEY=votre_cle_gemini
OPENROUTER_API_KEY=votre_cle_openrouter
MISTRAL_API_KEY=votre_cle_mistral

# Elasticsearch (optionnel — valeur par défaut : localhost:9200)
ELASTIC_HOST=http://localhost:9200
```

> **Obtenir une clé Groq gratuite** → https://console.groq.com

---

### Étape 4 — Lancer Elasticsearch + Kibana

```bash
docker compose up -d
```

Attendre ~30 secondes, puis vérifier :
- Elasticsearch → http://localhost:9200
- Kibana → http://localhost:5601

---

### Étape 5 — Installation automatique complète

```bash
python setup.py
```

Cette commande fait **tout automatiquement** :
1. Crée l'index Elasticsearch avec le bon mapping
2. Injecte les CVs depuis `output/cvs_data.json` (si présent)
3. Crée le dashboard Kibana complet (13 visualisations)

---

## 🚀 Lancer l'application

```bash
streamlit run app.py
```

→ Ouvre http://localhost:8501

---

## 📊 Pages de l'application

| Page | Accès | Description |
|---|---|---|
| **🧠 Accueil** | `/` | Vue d'ensemble + navigation |
| **📊 Dashboard** | `/1_Dashboard` | KPIs, scores, graphiques, alertes |
| **🔎 Recherche** | `/2_Recherche` | Filtrage avancé multi-critères |
| **💬 Chatbot RAG** | `/3_Chatbot` | Assistant IA conversationnel sur les CVs |
| **📤 Ajouter CV** | `/4_Ajouter_CV` | Upload PDF → extraction → indexation |

---

## 🗂️ Pipeline en ligne de commande

### Traiter des CVs depuis le dossier `cvs/`

```bash
# 1. Extraire et classifier les CVs (Groq par défaut)
python cv_extractor.py

# Avec un autre fournisseur
python cv_extractor.py --provider gemini

# 2. Sauvegarder en JSON + Excel
python cv_saver.py

# 3. Injecter dans Elasticsearch
python cv_injector.py
```

### Utilitaires

```bash
# Comparer les performances de plusieurs LLMs
python cv_comparator.py
python cv_comparator.py --provider groq --provider gemini

# Évaluer le routage du chatbot
python scripts/evaluate_chatbot.py
```

### Dashboard Kibana

```bash
# Créer/mettre à jour le dashboard (automatique avec setup.py)
python scripts/setup_kibana.py

# Le dashboard se met à jour automatiquement (refresh 30s)
# → http://localhost:5601/app/dashboards
```

---

## 📁 Structure du projet

```
cv-pipeline/
├── app.py                      ← Point d'entrée Streamlit
├── setup.py                    ← Installation en une commande
├── theme.py                    ← Design system partagé (CSS)
├── .env                        ← Clés API (à créer, non versionné)
│
├── pages/
│   ├── 1_Dashboard.py          ← KPIs et visualisations
│   ├── 2_Recherche.py          ← Recherche avancée
│   ├── 3_Chatbot.py            ← Chatbot RAG
│   └── 4_Ajouter_CV.py        ← Upload et indexation
│
├── chatbot/                    ← Logique RAG et ranking
│   ├── intent.py               ← Classification d'intention
│   ├── recruiter_helpers.py    ← Filtres hard + ranking recruteur
│   ├── llm_client.py           ← Client LLM dual-mode
│   └── ...
│
├── scripts/
│   ├── setup_kibana.py         ← Création automatique du dashboard Kibana
│   ├── export_kibana.py        ← Export NDJSON
│   └── evaluate_chatbot.py     ← Évaluation du routage
│
├── cv_extractor.py             ← Extraction LLM + scoring
├── cv_injector.py              ← Injection Elasticsearch
├── cv_deduplication.py         ← Détection doublons (3 niveaux)
├── cv_cache.py                 ← Cache SHA-256
├── create_index.py             ← Création de l'index ES
├── es_client.py                ← Client ES centralisé
├── docker-compose.yml          ← Elasticsearch 8.12 + Kibana 8.12
├── requirements.txt
│
├── kibana_dashboard.ndjson     ← Dashboard Kibana exporté (versionné)
├── cvs/                        ← CVs PDF pipeline CLI (non versionné)
├── cvs_uploads/                ← CVs uploadés via l'app (non versionné)
└── output/                     ← JSON / Excel générés (non versionné)
```

---

## 📊 Dashboard Kibana — Automatique

Le dashboard Kibana est créé automatiquement par `python setup.py`.
**Aucun clic manuel** — tout est généré par code via l'API REST Kibana.
Il se **rafraîchit toutes les 30 secondes** : chaque nouveau CV ajouté apparaît immédiatement.

### Visualisations incluses

| Graphique | Type | Description |
|---|---|---|
| Total CVs | Métrique | Nombre de CVs en temps réel |
| Score moyen | Métrique colorée | Rouge → Vert selon la qualité |
| Jauge qualité | Arc gauge | Score 0–100 avec seuils |
| Répartition catégories | Camembert | Distribution par domaine |
| Distribution scores | Histogramme | Tranches de 10 points |
| Top 10 candidats | Barres horizontales | Classés par score |
| Expérience par candidat | Barres | Années d'expérience |
| Top Technologies | Barres horizontales | 15 plus fréquentes |
| Top Frameworks | Barres horizontales | 15 plus fréquents |
| Langues | Camembert | Langues parlées |
| Score par domaine | Barres | Score moyen par catégorie |
| Nuage de compétences | Tag cloud | Toutes les technologies |
| **Tableau complet** | Tableau | Tous candidats + détails complets |

---

## 🧠 Principe anti-hallucination

| Composant | Rôle |
|---|---|
| **LLM** | Évaluation qualitative par unité (1 projet, 1 certification…) |
| **Python** | Tous les calculs finaux (scores, années exp., doublons) |
| **Elasticsearch** | Hard filters avant le LLM — jamais de filtre post-génération |

**Score qualité** = moyenne pondérée de 5 composantes :

| Composante | Poids par défaut |
|---|---|
| Diplôme | 25 % |
| Certifications | 20 % |
| Diversité technique | 20 % |
| Projets | 25 % |
| Langues | 10 % |

---

## 🔬 Résultats comparatifs LLM

Exécutés sur **10 CVs réels**, **5 configurations** :

| Modèle | Succès | Latence moy. | Complétude | Justesse | Quotas |
|---|---|---|---|---|---|
| **Llama-3.3-70B (Groq)** | **100 %** | **1.41 s** | 90 % | 100 % | 0 |
| GPT-OSS-20B (OpenRouter) | 80 % | 19.10 s | 89 % | 100 % | 0 |
| OpenRouter Free (auto) | 80 % | 46.48 s | 88 % | 100 % | 1 |
| Gemini 2.5 Flash | 80 % | 15.89 s | 88 % | 100 % | 2 |
| Gemini 2.5 Flash-Lite | 40 % | 21.82 s | 90 % | 100 % | 6 |

> Les échecs Gemini viennent des quotas du tier gratuit (429), pas d'un défaut d'analyse.

---

## ⚠️ Points importants

- **`cvs_uploads/`** et **`cvs/`** contiennent des données personnelles — ne jamais committer (dans `.gitignore`).
- **`cv_injector.py`** recrée l'index à chaque exécution. Pour ajouter des CVs sans tout supprimer, utiliser la page **"Ajouter CV"** de l'application Streamlit.
- Le champ `nom` doit être de type `text + keyword` pour que les graphiques Kibana "par candidat" fonctionnent — c'est configuré automatiquement dans `create_index.py`.
