import json
import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

def save_to_json(results, output_path="output/cvs_data.json"):
    """
    Sauvegarde tous les CVs en JSON
    """
    os.makedirs("output", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON sauvegardé : {output_path}")


def format_dict_scores(d):
    """
    Formate un dict {domaine: score} en une chaîne lisible pour Excel.
    Ex: {"Dev Web": 85, "IA": 70} -> "Dev Web: 85, IA: 70"
    """
    if not d or not isinstance(d, dict):
        return ""
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def save_to_excel(results, output_path="output/cvs_data.xlsx"):
    """
    Sauvegarde tous les CVs en Excel, avec les scores de qualité
    et de pertinence par domaine.
    """
    os.makedirs("output", exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "CVs"

    # Style entête
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(fill_type="solid", fgColor="2F75B6")

    # Colonnes
    headers = [
        "Fichier", "Nom", "Email", "Telephone", "LinkedIn", "Localisation",
        "Categorie principale",
        "Langages", "Frameworks", "Bases de données", "Outils DevOps",
        "Technologies", "Projets", "Diplômes", "Certifications", "Langues",
        "Score qualité (/100)", "Score qualité (/10)", "Détail score qualité",
        "Domaines & pertinence (LLM)", "Domaines pondérés (score final)",
    ]

    # Écrire les entêtes
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        # Colonnes de texte long un peu plus larges
        width = 40 if header in [
            "Détail score qualité", "Domaines & pertinence (LLM)",
            "Domaines pondérés (score final)", "Technologies", "Projets"
        ] else 25
        ws.column_dimensions[cell.column_letter].width = width

    # Écrire les données
    for row, result in enumerate(results, 2):
        data = result.get("data") or {}

        # Fonction helper pour éviter les erreurs null
        def safe_join(field):
            value = data.get(field) or []
            return ", ".join(value) if isinstance(value, list) else ""

        ws.cell(row=row, column=1, value=result["filename"])
        ws.cell(row=row, column=2, value=data.get("nom", ""))
        ws.cell(row=row, column=3, value=data.get("email", ""))
        ws.cell(row=row, column=4, value=data.get("telephone", ""))
        ws.cell(row=row, column=5, value=data.get("linkedin", ""))
        ws.cell(row=row, column=6, value=data.get("localisation", ""))
        ws.cell(row=row, column=7, value=data.get("categorie_principale", ""))
        ws.cell(row=row, column=8, value=safe_join("langages"))
        ws.cell(row=row, column=9, value=safe_join("frameworks"))
        ws.cell(row=row, column=10, value=safe_join("bases_de_donnees"))
        ws.cell(row=row, column=11, value=safe_join("outils_devops"))
        ws.cell(row=row, column=12, value=safe_join("technologies"))
        ws.cell(row=row, column=13, value=safe_join("projets"))
        ws.cell(row=row, column=14, value=safe_join("diplomes"))
        ws.cell(row=row, column=15, value=safe_join("certifications"))
        ws.cell(row=row, column=16, value=safe_join("langues"))

        # --- Nouvelles colonnes : scores de qualité ---
        ws.cell(row=row, column=17, value=data.get("score_qualite_globale", ""))
        ws.cell(row=row, column=18, value=data.get("score_qualite_globale_sur_10", ""))
        ws.cell(row=row, column=19, value=format_dict_scores(data.get("details_score_qualite")))
        ws.cell(row=row, column=20, value=format_dict_scores(data.get("scores_categories")))
        ws.cell(row=row, column=21, value=format_dict_scores(data.get("scores_categories_ponderes")))

        # Cas d'un CV en échec d'extraction (data = None) : on note l'erreur
        if not data and result.get("error"):
            ws.cell(row=row, column=2, value=f"❌ ÉCHEC : {result['error']}")

    wb.save(output_path)
    print(f"✅ Excel sauvegardé : {output_path}")


# TEST
if __name__ == "__main__":
    from cv_reader import read_all_cvs
    from cv_extractor import extract_all_cvs

    # Lire les CVs
    cvs = read_all_cvs("cvs/")

    # Extraire les données
    results = extract_all_cvs(cvs)

    # Sauvegarder
    save_to_json(results)
    save_to_excel(results)

    print("\n🎉 Terminé ! Fichiers dans le dossier output/")