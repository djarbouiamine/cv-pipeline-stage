# 🧠 CV Pipeline — Intelligence de Recrutement

> **Extraction** • **Classification** • **Scoring** • **Recherche sémantique** • **RAG Chatbot**
>
> Pipeline complet pour analyser des CVs PDF avec LLM + Elasticsearch.
> Tous les calculs finaux sont **Python déterministe** — le LLM ne fait jamais de calcul direct.

---

## ✅ Prérequis — Installez ceci avant tout

| Outil | Lien de téléchargement | Utilité |
|---|---|---|
| **Python 3.10+** | https://python.org/downloads | Runtime |
| **Docker Desktop** | https://docker.com/products/docker-desktop | Elasticsearch + Kibana |
| **Tesseract OCR** | https://github.com/UB-Mannheim/tesseract/wiki | Lire les CVs scannés |
| **Poppler** | https://github.com/oschwartz10612/poppler-windows/releases | Conversion PDF en image |
| **Clé API LLM** | https://console.groq.com *(gratuit)* | Extraction intelligente |

> 💡 **Groq est recommandé** : gratuit, rapide (< 2 s par CV), aucune carte bancaire requise.

---

## 🚀 Démarrage en 5 étapes

### Étape 1 — Cloner le projet

```bash
git clone https://github.com/djarbouiamine/cv-pipeline-stage.git
cd cv-pipeline-stage
```

---

### Étape 2 — Installer les dépendances Python

```bash
pip install -r requirements.txt
```

---

### Étape 3 — Configurer vos clés API

Créez un fichier **`.env`** à la racine du projet (même dossier que `app.py`) :

```env
# ── LLM — au moins UNE clé est obligatoire ──────────────────────
GROQ_API_KEY=votre_cle_groq          # Recommandé — gratuit sur console.groq.com
GEMINI_API_KEY=votre_cle_gemini      # Optionnel
OPENROUTER_API_KEY=votre_cle_or      # Optionnel
MISTRAL_API_KEY=votre_cle_mistral    # Optionnel

# ── Ordre de bascule automatique si quota atteint ───────────────
AUTO_FALLBACK=true
FALLBACK_ORDER=groq,openrouter,mistral,gemini

# ── Elasticsearch (valeur par défaut : localhost:9200) ───────────
ELASTIC_HOST=http://localhost:9200

# ── Pondération du score qualité (total = 100) ───────────────────
QUALITY_WEIGHT_DIPLOME=25
QUALITY_WEIGHT_CERTIFICATIONS=20
QUALITY_WEIGHT_TECH=20
QUALITY_WEIGHT_PROJETS=25
QUALITY_WEIGHT_LANGUES=10
```

---

### Étape 4 — Lancer Elasticsearch + Kibana

```bash
docker compose up -d
```

Attendez ~30 secondes, puis vérifiez que ces URLs répondent :
- **Elasticsearch** → http://localhost:9200
- **Kibana** → http://localhost:5601

---

### Étape 5 — Installation automatique complète

```bash
python setup.py
```

Cette **seule commande** fait tout dans l'ordre :

| Étape | Action |
|---|---|
| **0** | 🧹 Sync — supprime du cache/data/ES les CVs dont le PDF a été effacé |
| **1** | ✅ Vérifie qu'Elasticsearch et Kibana sont démarrés |
| **2** | 🗂️ Crée l'index Elasticsearch avec le bon mapping |
| **3a** | 📂 Charge les CVs déjà extraits depuis `output/cvs_data.json` |
| **3b** | 🔍 Scanne `cvs/` et `cvs_uploads/` pour détecter de nouveaux PDFs |
| **3c** | 🤖 Extrait les nouveaux PDFs via LLM avec **détection de doublons 3 niveaux** |
| **3d** | ⚡ Injecte tous les CVs dans Elasticsearch (ID stable — idempotent) |
| **4** | 📊 Crée le dashboard Kibana complet (13 visualisations) |

---

### Étape 6 — Lancer l'application Streamlit

```bash
streamlit run app.py
```

→ Ouvre **http://localhost:8501** dans votre navigateur.

---

## 📱 Pages de l'application

| Page | URL | Description |
|---|---|---|
| 🧠 **Accueil** | `/` | Vue d'ensemble et navigation |
| 📊 **Dashboard** | `/1_Dashboard` | KPIs, scores, graphiques en temps réel |
| 🔎 **Recherche** | `/2_Recherche` | Filtrage avancé multi-critères |
| 💬 **Chatbot RAG** | `/3_Chatbot` | Assistant IA conversationnel sur les CVs |
| 📤 **Ajouter CV** | `/4_Ajouter_CV` | Upload PDF → extraction LLM → indexation |

---

## 📂 Ajouter des CVs — 2 méthodes

### Méthode A : Via l'interface Streamlit (recommandée)

1. Ouvrez **http://localhost:8501**
2. Cliquez sur **"Ajouter CV"**
3. Déposez un fichier PDF
4. Cliquez **"Lancer l'extraction"** puis **"Valider et indexer"**

✅ Le CV est automatiquement ajouté à `cv_cache.json`, `cvs_data.json` **et** Elasticsearch.

### Méthode B : Via le dossier `cvs/`

1. Copiez vos fichiers PDF dans le dossier `cvs/`
2. Relancez simplement :

```bash
python setup.py
```

`setup.py` détecte automatiquement les nouveaux PDFs, applique la détection de doublons, extrait et injecte.

---

## 🔍 Détection de doublons — 3 niveaux

Appliquée identiquement dans `setup.py` ET dans la page Streamlit **"Ajouter CV"** :

| Niveau | Méthode | Cas détecté |
|---|---|---|
| **1 — Hash SHA-256** | Empreinte binaire du fichier | Même fichier PDF copié avec un autre nom |
| **2 — Email / Téléphone** | Correspondance exacte dans les données extraites | Même personne, CV différent (ex: `CV_Ahmed.pdf` et `CV_Ahmed_Copie.pdf`) |
| **3 — Similarité sémantique** | Cosinus entre embeddings multilingues | Même personne sans email/téléphone, ou CV anonymisé |

Un CV est **rejeté dès qu'un niveau est déclenché** — il n'est écrit ni dans `cv_cache.json`, ni dans `cvs_data.json`, ni dans Elasticsearch.

> Le niveau 3 est non-bloquant si le modèle d'embedding est indisponible — les niveaux 1 et 2 restent actifs.

---

## 🧹 Supprimer un CV / Synchroniser le cache

### Depuis l'interface Streamlit

Sur la page **"Ajouter CV"**, faites défiler jusqu'en bas :
- Section **"🔄 Synchroniser le cache"**
- **"🔍 Analyser"** → aperçu des entrées orphelines (dry-run)
- **"🧹 Nettoyer les orphelins"** → supprime de `cv_cache.json`, `cvs_data.json` et Elasticsearch

### Depuis le terminal

```bash
# Aperçu sans rien modifier
python cv_sync.py --dry-run

# Nettoyage réel
python cv_sync.py
```

### Via `python setup.py`

Le nettoyage est **automatique à l'étape 0**. Si vous avez supprimé des PDFs des dossiers `cvs/` ou `cvs_uploads/`, relancez simplement :

```bash
python setup.py
```

---

## 🗂️ Commandes utiles

```bash
# Lancer l'application Streamlit
streamlit run app.py

# Installation / réindexation complète
python setup.py

# Nettoyer le cache (orphelins)
python cv_sync.py

# Aperçu du nettoyage sans modifier
python cv_sync.py --dry-run

# Recréer seulement le dashboard Kibana
python scripts/setup_kibana.py

# Comparer les performances de plusieurs LLMs
python cv_comparator.py
python cv_comparator.py --provider groq --provider gemini
```

---

## 📁 Structure du projet

```
cv-pipeline/
├── app.py                    ← Point d'entrée Streamlit
├── setup.py                  ← Installation complète en UNE commande
├── cv_sync.py                ← Synchronisation / nettoyage des orphelins
├── theme.py                  ← Design system CSS partagé
├── .env                      ← Clés API (à créer, non versionné)
│
├── pages/
│   ├── 1_Dashboard.py        ← KPIs et visualisations
│   ├── 2_Recherche.py        ← Recherche avancée multi-critères
│   ├── 3_Chatbot.py          ← Chatbot RAG conversationnel
│   └── 4_Ajouter_CV.py      ← Upload PDF + sync orphelins
│
├── chatbot/                  ← Logique RAG et ranking
│   ├── intent.py             ← Classification d'intention
│   ├── recruiter_helpers.py  ← Filtres hard + ranking recruteur
│   └── llm_client.py         ← Client LLM multi-provider
│
├── scripts/
│   ├── setup_kibana.py       ← Création automatique du dashboard Kibana
│   └── evaluate_chatbot.py   ← Évaluation du routage chatbot
│
├── cv_extractor.py           ← Extraction LLM + scoring
├── cv_injector.py            ← Injection Elasticsearch (ID stable SHA-256)
├── cv_deduplication.py       ← Détection doublons 3 niveaux
├── cv_cache.py               ← Cache SHA-256 (JSON ou PostgreSQL)
├── cv_reader.py              ← Lecture PDF (texte / colonnes / OCR)
├── cv_saver.py               ← Sauvegarde JSON + Excel
├── cv_removal.py             ← Suppression ciblée d'un CV
├── cv_sync.py                ← Nettoyage des entrées orphelines
├── create_index.py           ← Création de l'index ES avec mapping
├── es_client.py              ← Client Elasticsearch centralisé
├── docker-compose.yml        ← Elasticsearch 8.x + Kibana 8.x
├── requirements.txt
│
├── cvs/                      ← PDFs pour pipeline CLI (non versionné)
├── cvs_uploads/              ← PDFs uploadés via Streamlit (non versionné)
└── output/
    ├── cv_cache.json         ← Cache des extractions (non versionné)
    ├── cvs_data.json         ← Données structurées — CVs uniques seulement
    └── cvs_data.xlsx         ← Export Excel
```

---

## 🔬 Résultats comparatifs LLM

Testés sur **10 CVs réels**, **5 configurations** :

| Modèle | Succès | Latence moy. | Complétude | Quotas |
|---|---|---|---|---|
| **Llama-3.3-70B (Groq)** | **100 %** | **1.41 s** | 90 % | 0 |
| GPT-OSS-20B (OpenRouter) | 80 % | 19.10 s | 89 % | 0 |
| OpenRouter Free (auto) | 80 % | 46.48 s | 88 % | 1 |
| Gemini 2.5 Flash | 80 % | 15.89 s | 88 % | 2 |
| Gemini 2.5 Flash-Lite | 40 % | 21.82 s | 90 % | 6 |

> Les échecs Gemini sont dus aux limites du tier gratuit (429), pas à un défaut d'analyse.

---

## 🧠 Principe anti-hallucination

| Composant | Rôle |
|---|---|
| **LLM** | Évalue la qualité par unité (1 projet, 1 certification à la fois) |
| **Python** | Effectue tous les calculs finaux (scores, années d'expérience, doublons) |
| **Elasticsearch** | Applique les filtres utilisateur **avant** tout appel LLM |

**Score qualité** = moyenne pondérée de 5 composantes (configurable dans `.env`) :

| Composante | Poids par défaut |
|---|---|
| Diplôme | 25 % |
| Projets | 25 % |
| Certifications | 20 % |
| Diversité technique | 20 % |
| Langues | 10 % |

---

## 🛠️ Résolution de problèmes fréquents

### ❌ Elasticsearch n'est pas disponible
```bash
docker compose up -d
# Attendre 30 secondes puis relancer
python setup.py
```

### ❌ Aucune clé API LLM configurée
Vérifiez votre fichier `.env` — il doit contenir au moins :
```env
GROQ_API_KEY=votre_cle
```
Obtenez une clé gratuite sur https://console.groq.com

### ❌ Le texte extrait du PDF est quasi vide
Le PDF est probablement une image scannée. Vérifiez que **Tesseract OCR** est installé :
```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

### ❌ Doublon détecté à l'upload dans Streamlit
Le CV est déjà présent (même email, téléphone ou contenu similaire).
- Si c'est une mise à jour : supprimez l'ancienne version depuis la page **"Ajouter CV"** → section **"Supprimer un CV"**
- Si c'est une erreur : vérifiez `output/cv_cache.json` pour voir quelle version est en cache

### ❌ Entrées fantômes dans le cache après suppression d'un PDF
```bash
python cv_sync.py
# ou via Streamlit : page "Ajouter CV" → "🔄 Synchroniser le cache"
```

### ❌ Les doublons apparaissent encore dans Kibana / Streamlit
Le fichier `cvs_data.json` contient peut-être des entrées dupliquées d'une ancienne version.
Relancez simplement :
```bash
python setup.py
```
`cv_injector.py` utilise désormais un **ID SHA-256 stable** par personne — même email = même document Elasticsearch, pas de doublon possible.

---

## ⚠️ Points importants

- **`cvs/`** et **`cvs_uploads/`** contiennent des **données personnelles** — dans `.gitignore`, ne jamais committer.
- **`cv_injector.py`** recrée l'index à chaque appel direct. Pour ajouter un CV sans tout supprimer, utilisez **la page Streamlit** ou **`python setup.py`**.
- **`cvs_data.json`** et **`cv_cache.json`** ne contiennent que des CVs **uniques** — la détection 3 niveaux garantit qu'aucune personne n'apparaît deux fois.
- Le dashboard Kibana se **rafraîchit automatiquement toutes les 30 secondes** — aucune action manuelle requise.
- Le champ `nom` est indexé `text + keyword` dans Elasticsearch — requis pour les graphiques "par candidat" dans Kibana.
