# Pipeline de Traitement et de Classification de CVs (CV Pipeline Stage)

Ce projet est un pipeline complet d'extraction, de classification, de comparaison et d'indexation de CVs au format PDF. Il utilise des modèles de langage de grande taille (LLMs) gratuits pour transformer des CVs non structurés en données JSON structurées, les classifier thématiquement, et les injecter dans Elasticsearch.

---

## 🛠️ Architecture du Projet

Le projet s'articule autour des modules suivants :

1. **`cv_reader.py`** : Lit les fichiers PDF. Il applique trois stratégies successives :
   - Extraction de texte brut (PyMuPDF).
   - Extraction sur deux colonnes (pdfplumber) pour les CVs au design moderne.
   - Reconnaissance Optique de Caractères (OCR via Tesseract) pour les PDF numérisés ou sous forme d'images.
2. **`cv_extractor.py`** : Script principal d'extraction en production.
   - Par défaut, il tente l'extraction avec **Groq** (Llama 3.3 70B) pour sa rapidité et sa fiabilité, et bascule automatiquement sur **OpenRouter** en cas d'échec (429/503).
   - Un système de cache local évite de réanalyser un CV déjà traité.
   - Supporte des arguments CLI (`--provider` et `--model`) pour forcer l'utilisation d'une IA précise.
3. **`cv_saver.py`** : Exporte les données structurées extraites dans des fichiers JSON (`output/cvs_data.json`) et Excel (`output/cvs_data.xlsx`).
4. **`cv_injector.py`** : Charge les données extraites et les indexe dans une base **Elasticsearch** locale pour permettre la recherche plein texte et filtrée.
5. **`cv_comparator.py`** : Outil d'évaluation et d'étude comparative. Il exécute les différents LLMs gratuits sur les CVs pour évaluer leurs performances selon plusieurs métriques quantitatives et qualitatives.

---

## 📊 Résultats de l'Étude Comparative

Le script de comparaison a été exécuté sur un échantillon de 10 CVs réels en évaluant 5 configurations de LLMs. L'étude comparative mesure :
- **Le taux de succès** (requêtes ayant abouti à un JSON valide).
- **La latence moyenne** de réponse.
- **La complétude moyenne** (taux de remplissage des champs JSON attendus).
- **La justesse d'extraction** (nom, e-mail et téléphone validés manuellement par rapport à une vérité terrain).
- **La pertinence de la classification** thématique (pertinence du domaine principal identifié).
- **Le type d'erreurs** rencontrées (Quota 429, timeout, parsing JSON).

### Tableau de Synthèse Réel (10 CVs testés)

| Modèle / IA | Succès % | Latence Moyenne | Complétude % | Justesse Extraction | Classification Pertinente | Erreurs Quota (429) | Erreurs Parsing | Tests |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Llama-3.3-70B (Groq)** | **100.0%** | **1.41s** | 90.0% | 100.0% | 100.0% | 0 | 0 | 10 |
| **GPT-OSS-20B (OpenRouter)** | **80.0%** | 19.10s | 89.0% | 100.0% | 100.0% | 0 | 2 | 10 |
| **OpenRouter Free (auto)** | **80.0%** | 46.48s | 88.2% | 100.0% | 100.0% | 1 | 1 | 10 |
| **Gemini 2.5 Flash** | **80.0%** | 15.89s | 88.2% | 100.0% | 100.0% | 2 | 0 | 10 |
| **Gemini 2.5 Flash-Lite** | **40.0%** | 21.82s | 89.7% | 100.0% | 100.0% | 6 | 0 | 10 |

> 💡 **Observation Clé** : Les modèles Gemini ont un taux de justesse et de classification parfait (100 %) lors des appels réussis. Leurs taux de succès plus bas (80 % et 40 %) sont exclusivement causés par les limites de quotas de requêtes de leur formule gratuite (erreurs `429 RESOURCE_EXHAUSTED`), et non par un défaut d'analyse.

---

## ⚙️ Installation et Configuration

### Prérequis
- Python 3.10+
- Tesseract OCR (installé sur la machine et configuré dans le code)
- Elasticsearch (optionnel, requis uniquement pour `cv_injector.py`)

### 1. Cloner le dépôt
```bash
git clone https://github.com/djarbouiamine/cv-pipeline-stage.git
cd cv-pipeline-stage
```

### 2. Configurer les variables d'environnement
Créez un fichier `.env` à la racine du projet et renseignez vos clés d'API gratuites :
```env
GROQ_API_KEY=votre_cle_groq
GEMINI_API_KEY=votre_cle_gemini
OPENROUTER_API_KEY=votre_cle_openrouter
MISTRAL_API_KEY=votre_cle_mistral
```
*(Le fichier `.env` est automatiquement ignoré par Git via le `.gitignore` pour des raisons de sécurité).*

---

## 🚀 Utilisation

### 1. Exécuter le pipeline de production
Traite tous les CVs situés dans le dossier `cvs/` :
```bash
python cv_extractor.py
```
Pour forcer un fournisseur spécifique sans fallback :
```bash
python cv_extractor.py --provider gemini --model gemini-2.5-flash
```

### 2. Lancer l'étude comparative des modèles
Pour tester tous les modèles candidats :
```bash
python cv_comparator.py
```
Pour filtrer et comparer uniquement des fournisseurs précis :
```bash
python cv_comparator.py --provider groq --provider gemini
```
Pour tester un modèle unique :
```bash
python cv_comparator.py --model llama-3.3-70b-versatile
```

Les rapports détaillés en JSON et les résumés consolidés en CSV sont automatiquement générés dans le dossier `output/`.