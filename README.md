# Pipeline de Traitement et de Classification de CVs

Ce projet est un pipeline complet d'**extraction**, de **classification**, de **comparaison** et de **sauvegarde** de CVs au format PDF. Il utilise des modèles de langage (LLMs) multi-fournisseurs pour transformer des CVs non structurés en données JSON structurées et les classifier thématiquement.

---

## 🛠️ Architecture du Projet

Le projet s'articule autour de **4 modules principaux** :

| Module | Rôle |
| :--- | :--- |
| **`cv_reader.py`** | Lit les fichiers PDF avec 3 stratégies successives : extraction texte brut (PyMuPDF), extraction 2 colonnes (pdfplumber), OCR via Tesseract pour les PDF numérisés. |
| **`cv_extractor.py`** | Extraction et classification en production. Supporte 4 fournisseurs LLM (Groq, Gemini, Mistral, OpenRouter). Inclut le calcul du score de qualité, la gestion des quotas avec retry, et un cache local pour éviter les ré-extractions inutiles. |
| **`cv_saver.py`** | Exporte les données extraites en **JSON** (`output/cvs_data.json`) et **Excel** (`output/cvs_data.xlsx`). |
| **`cv_comparator.py`** | Outil d'évaluation comparative des LLMs. Mesure le taux de succès, la latence, la complétude, la justesse d'extraction, la pertinence de classification et la qualité globale sur un même corpus de CVs. |

> **Note** : Les modules `cv_extractor.py` et `cv_comparator.py` partagent exactement le même prompt, les mêmes fonctions d'appel aux fournisseurs et la même méthode de scoring — garantissant une comparaison équitable et reproductible.

---

## 📊 Résultats de l'Étude Comparative

Exécutée sur **10 CVs réels**, avec **5 configurations LLM** :

| Modèle / IA | Succès % | Latence Moy. | Complétude % | Justesse % | Classification % | Qualité Moy. | Quota (429) | Tests |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Llama-3.3-70B (Groq)** | **100%** | **1.41s** | 90.0% | 100% | 100% | — | 0 | 10 |
| **GPT-OSS-20B (OpenRouter)** | 80% | 19.10s | 89.0% | 100% | 100% | — | 0 | 10 |
| **OpenRouter Free (auto)** | 80% | 46.48s | 88.2% | 100% | 100% | — | 1 | 10 |
| **Gemini 2.5 Flash** | 80% | 15.89s | 88.2% | 100% | 100% | — | 2 | 10 |
| **Gemini 2.5 Flash-Lite** | 40% | 21.82s | 89.7% | 100% | 100% | — | 6 | 10 |

> 💡 **Observation clé** : Les modèles Gemini atteignent 100 % de justesse et de pertinence de classification lors des appels réussis. Leurs taux de succès inférieurs sont exclusivement causés par les limites de quota de leur tier gratuit (erreurs `429 RESOURCE_EXHAUSTED`), et non par un défaut d'analyse.

---

## ⚙️ Installation et Configuration

### Prérequis
- Python 3.10+
- Tesseract OCR (installé sur la machine et accessible dans le PATH)

### 1. Cloner le dépôt
```bash
git clone https://github.com/djarbouiamine/cv-pipeline-stage.git
cd cv-pipeline-stage
```

### 2. Installer les dépendances
```bash
pip install pymupdf pdfplumber pytesseract pillow groq google-genai requests openpyxl
```

### 3. Configurer les variables d'environnement
Créez un fichier `.env` à la racine du projet avec vos clés d'API :
```env
GROQ_API_KEY=votre_cle_groq
GEMINI_API_KEY=votre_cle_gemini
OPENROUTER_API_KEY=votre_cle_openrouter
MISTRAL_API_KEY=votre_cle_mistral
```
*(Au moins une clé est requise. Le fichier `.env` est ignoré par Git via `.gitignore`.)*

---

## 🚀 Utilisation

### 1. Extraire et classifier les CVs

Traite tous les CVs du dossier `cvs/` avec le fournisseur par défaut (Groq) :
```bash
python cv_extractor.py
```

Forcer un fournisseur et/ou un modèle spécifique :
```bash
python cv_extractor.py --provider gemini
python cv_extractor.py --provider mistral --model mistral-medium-latest
python cv_extractor.py --provider openrouter --model meta-llama/llama-3.3-70b-instruct:free
```

Fournisseurs disponibles : `groq`, `gemini`, `mistral`, `openrouter`.

### 2. Sauvegarder les résultats

```bash
python cv_saver.py
```
Génère `output/cvs_data.json` et `output/cvs_data.xlsx`.

### 3. Lancer l'étude comparative des modèles

Tester tous les modèles candidats :
```bash
python cv_comparator.py
```

Filtrer par fournisseur :
```bash
python cv_comparator.py --provider groq --provider gemini
```

Tester un modèle unique :
```bash
python cv_comparator.py --model llama-3.3-70b-versatile
```

Les rapports détaillés (JSON) et les résumés consolidés (CSV) sont automatiquement générés dans le dossier `output/`.

---

## 📁 Structure du Projet

```
cv-pipeline-stage/
├── cvs/                   # Dossier contenant les CVs PDF à analyser
├── output/                # Résultats générés (JSON, Excel, CSV)
│   ├── cvs_data.json
│   ├── cvs_data.xlsx
│   └── comparaison_*.csv
├── cv_reader.py           # Lecture et OCR des PDFs
├── cv_extractor.py        # Extraction multi-provider + scoring qualité
├── cv_saver.py            # Export JSON / Excel
├── cv_comparator.py       # Comparaison et évaluation des LLMs
├── .env                   # Clés API (non versionné)
└── README.md
```