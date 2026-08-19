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

Copiez `.env.example` en `.env` et remplissez vos valeurs :

```bash
copy .env.example .env
```

Contenu du fichier **`.env`** :

```env
# ── LLM — au moins UNE clé est obligatoire ──────────────────────
GROQ_API_KEY=votre_cle_groq          # Recommandé — gratuit sur console.groq.com
GEMINI_API_KEY=votre_cle_gemini      # Optionnel
OPENROUTER_API_KEY=votre_cle_or      # Optionnel
MISTRAL_API_KEY=votre_cle_mistral    # Optionnel

# ── Fallback automatique si quota atteint ───────────────────────
AUTO_FALLBACK=true
FALLBACK_ORDER=groq,openrouter,mistral,gemini

# ── Elasticsearch ────────────────────────────────────────────────
ELASTIC_HOST=http://localhost:9200

# ── Score qualité — pondération (A+B+C+D+E = 100) ───────────────
# score_final = (diplome*A + certifications*B + tech*C + projets*D + langues*E) / 100
QUALITY_WEIGHT_DIPLOME=25        # A — niveau académique
QUALITY_WEIGHT_CERTIFICATIONS=20 # B — certifications (AWS, Azure, PMP…)
QUALITY_WEIGHT_TECH=20           # C — diversité technique
QUALITY_WEIGHT_PROJETS=25        # D — projets réalisés
QUALITY_WEIGHT_LANGUES=10        # E — langues maîtrisées

# ── Seuils de détection de doublons (Niveau 3 — similarité) ─────
# 0.90 = 2 CVs à 90% similaires → même personne (doublon rejeté)
DEDUP_SIMILARITY_THRESHOLD=0.90  # augmenter = moins strict
DEDUP_UPDATE_THRESHOLD=0.98      # seuil "mise à jour" (doit être > SIMILARITY)
```

> 📄 Consultez **`.env.example`** pour la liste complète des paramètres avec explications détaillées.

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
| 📊 **Dashboard** | `/1_Dashboard` | KPIs, scores, graphiques en temps réel depuis Elasticsearch |
| 🔎 **Recherche** | `/2_Recherche` | Filtrage avancé multi-critères + **suppression CV** |
| 💬 **Chatbot RAG** | `/3_Chatbot` | Assistant IA conversationnel sur les CVs |
| 📤 **Ajouter CV** | `/4_Ajouter_CV` | Upload PDF → extraction LLM → indexation |

---

## 📂 Ajouter des CVs — 2 méthodes

### Méthode A : Via l'interface Streamlit (recommandée)

1. Ouvrez **http://localhost:8501**
2. Cliquez sur **"Ajouter CV"**
3. Déposez un fichier PDF
4. Cliquez **"Lancer l'extraction"** puis **"Valider et indexer"**

✅ Le CV est automatiquement ajouté à `cv_cache.json`, `cvs_data.json` **et** Elasticsearch en une seule action.

### Méthode B : Via le dossier `cvs/`

1. Copiez vos fichiers PDF dans le dossier `cvs/`
2. Relancez simplement :

```bash
python setup.py
```

`setup.py` détecte automatiquement les nouveaux PDFs, applique la détection de doublons, extrait et injecte.

---

## 🗑️ Supprimer un CV — méthode correcte

> ⚠️ **N'utilisez jamais Kibana pour supprimer un CV.** Kibana ne supprime que le document Elasticsearch. Le fichier PDF, `cv_cache.json` et `cvs_data.json` ne seraient **pas** mis à jour — le CV réapparaîtrait au prochain `python setup.py`.

### ✅ Via la page Recherche Streamlit (recommandé)

1. Ouvrez la page **Recherche** → http://localhost:8501/2_Recherche
2. Trouvez le candidat → cliquez **▼ Détails**
3. Faites défiler jusqu'en bas → cochez **🗑️ Supprimer [Nom] complètement**
4. Cliquez **🗑️ Confirmer la suppression**

Un seul clic supprime **partout** :

| Elasticsearch | cv_cache.json | cvs_data.json + Excel | Fichier PDF |
|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ |

### ✅ Nettoyage automatique des orphelins (PDFs supprimés du disque)

Si vous avez supprimé des PDFs **manuellement** depuis l'explorateur, relancez :

```bash
python setup.py
# L'étape 0 détecte automatiquement les PDFs manquants et nettoie cache + data + ES

# ou directement :
python cv_sync.py          # nettoyage réel
python cv_sync.py --dry-run  # aperçu sans modifier
```

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

## 💾 Comprendre les fichiers de données

### `output/cv_cache.json` — Cache technique LLM

**Rôle :** Éviter de re-appeler le LLM pour un PDF déjà traité.

| Propriété | Détail |
|---|---|
| **Clé** | Hash SHA-256 du fichier PDF |
| **Contenu** | Extraction brute complète (toutes les infos du CV + métadonnées) |
| **Mise à jour** | À l'ajout, à la suppression, au nettoyage via `setup.py` ou `cv_sync.py` |

### `output/cvs_data.json` — Registre principal

**Rôle :** Source de vérité pour le dashboard Streamlit et les exports Excel.

| Propriété | Détail |
|---|---|
| **Clé** | Index numérique (liste JSON) |
| **Contenu** | Données structurées et nettoyées de chaque CV unique |
| **Mise à jour** | À l'ajout, à la suppression, au nettoyage via `setup.py` ou `cv_sync.py` |

### Différence clé

| | `cv_cache.json` | `cvs_data.json` |
|---|---|---|
| **Usage** | Éviter les appels LLM redondants | Dashboard + exports |
| **Clé** | Hash SHA-256 du PDF | Index numérique |
| **Contenu** | Extraction brute complète | Données structurées nettoyées |

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
├── theme.py                  ← Design system CSS partagé
├── .env                      ← Clés API (à créer, non versionné)
├── .env.example              ← Modèle .env avec tous les paramètres commentés
│
├── pages/
│   ├── 1_Dashboard.py        ← KPIs et visualisations temps réel
│   ├── 2_Recherche.py        ← Recherche avancée + suppression CV
│   ├── 3_Chatbot.py          ← Chatbot RAG conversationnel
│   └── 4_Ajouter_CV.py      ← Upload PDF → extraction LLM → indexation
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
├── cv_extractor.py           ← Extraction LLM + scoring qualité
├── cv_injector.py            ← Injection Elasticsearch (ID stable SHA-256)
├── cv_deduplication.py       ← Détection doublons 3 niveaux
├── cv_cache.py               ← Cache SHA-256 (JSON)
├── cv_reader.py              ← Lecture PDF (texte / colonnes / OCR)
├── cv_saver.py               ← Sauvegarde JSON + Excel
├── cv_removal.py             ← Suppression complète d'un CV (ES + cache + data + PDF)
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

| Composante | Variable `.env` | Poids par défaut |
|---|---|---|
| Diplôme | `QUALITY_WEIGHT_DIPLOME` | 25 % |
| Projets | `QUALITY_WEIGHT_PROJETS` | 25 % |
| Certifications | `QUALITY_WEIGHT_CERTIFICATIONS` | 20 % |
| Diversité technique | `QUALITY_WEIGHT_TECH` | 20 % |
| Langues | `QUALITY_WEIGHT_LANGUES` | 10 % |

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
- Pour mettre à jour : supprimez l'ancienne version depuis la page **Recherche** → **▼ Détails** → **🗑️ Supprimer**
- Puis uploadez la nouvelle version sur la page **Ajouter CV**

### ❌ Entrées fantômes dans le cache après suppression manuelle d'un PDF
```bash
python setup.py
# L'étape 0 nettoie automatiquement les orphelins

# ou directement :
python cv_sync.py
```

### ❌ Les doublons apparaissent encore dans Kibana / Streamlit
Relancez simplement :
```bash
python setup.py
```
`cv_injector.py` utilise un **ID SHA-256 stable** par personne — même email = même document ES, pas de doublon possible.

---

## ⚠️ Points importants

- **`cvs/`** et **`cvs_uploads/`** contiennent des **données personnelles** — dans `.gitignore`, ne jamais committer.
- **Ne jamais supprimer un CV depuis Kibana** — utilisez toujours la page **Recherche** de Streamlit pour une suppression complète (ES + cache + JSON + PDF).
- **`cvs_data.json`** et **`cv_cache.json`** ne contiennent que des CVs **uniques** — la détection 3 niveaux garantit qu'aucune personne n'apparaît deux fois.
- Le dashboard Kibana se **rafraîchit automatiquement toutes les 30 secondes** — aucune action manuelle requise.
- Le champ `nom` est indexé `text + keyword` dans Elasticsearch — requis pour les graphiques "par candidat" dans Kibana.
