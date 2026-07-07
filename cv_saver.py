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


def save_to_excel(results, output_path="output/cvs_data.xlsx"):
    """
    Sauvegarde tous les CVs en Excel
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
        "Fichier", "Nom", "Email", "Telephone", "LinkedIn",
        "Localisation", "Langages", "Frameworks", "Bases de données",
        "Outils DevOps", "Technologies", "Projets",
        "Diplômes", "Certifications", "Langues"
    ]
    
    # Écrire les entêtes
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[cell.column_letter].width = 25

    # Écrire les données
    # Écrire les données
    for row, result in enumerate(results, 2):
        data = result["data"]
        
        # Fonction helper pour éviter les erreurs null
        def safe_join(field):
            value = data.get(field) or []
            if not isinstance(value, list):
                return str(value) if value else ""
            
            str_items = []
            for item in value:
                if isinstance(item, dict):
                    name = item.get("nom") or item.get("title") or item.get("name")
                    if name:
                        str_items.append(str(name))
                    else:
                        str_items.append(json.dumps(item, ensure_ascii=False))
                else:
                    str_items.append(str(item))
            return ", ".join(str_items)
        
        ws.cell(row=row, column=1, value=result["filename"])
        ws.cell(row=row, column=2, value=data.get("nom", ""))
        ws.cell(row=row, column=3, value=data.get("email", ""))
        ws.cell(row=row, column=4, value=data.get("telephone", ""))
        ws.cell(row=row, column=5, value=data.get("linkedin", ""))
        ws.cell(row=row, column=6, value=data.get("localisation", ""))
        ws.cell(row=row, column=7, value=safe_join("langages"))
        ws.cell(row=row, column=8, value=safe_join("frameworks"))
        ws.cell(row=row, column=9, value=safe_join("bases_de_donnees"))
        ws.cell(row=row, column=10, value=safe_join("outils_devops"))
        ws.cell(row=row, column=11, value=safe_join("technologies"))
        ws.cell(row=row, column=12, value=safe_join("projets"))
        ws.cell(row=row, column=13, value=safe_join("diplomes"))
        ws.cell(row=row, column=14, value=safe_join("certifications"))
        ws.cell(row=row, column=15, value=safe_join("langues"))

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