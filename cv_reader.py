import fitz
import pytesseract
import os
import cv2
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\Library\bin"

def preprocess_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary)


def read_cv_columns(pdf_path):
    """
    Solution 2 — lit les PDFs avec colonnes (design graphique)
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            largeur = page.width
            
            # Colonne gauche
            gauche = page.crop((0, 0, largeur/2, page.height))
            text += gauche.extract_text() or ""
            text += "\n"
            
            # Colonne droite
            droite = page.crop((largeur/2, 0, largeur, page.height))
            text += droite.extract_text() or ""
            text += "\n"
    
    return text


def read_cv_text(pdf_path):
    """
    Essaie pymupdf → pdfplumber colonnes → OCR
    """
    text = ""

    # Essai 1 : pymupdf (PDF texte simple)
    doc = fitz.open(pdf_path)
    for page in doc:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        for block in blocks:
            if block[6] == 0:
                text += block[4] + "\n"
    doc.close()

    # Essai 2 : pdfplumber colonnes (PDF design graphique)
    if len(text.strip()) < 100:
        print(f"📐 Lecture colonnes pour : {os.path.basename(pdf_path)}")
        text = read_cv_columns(pdf_path)

    # Essai 3 : OCR (PDF image scannée)
    if len(text.strip()) < 100:
        print(f"🔍 OCR activé pour : {os.path.basename(pdf_path)}")
        images = convert_from_path(pdf_path,
                                   poppler_path=POPPLER_PATH,
                                   dpi=300)
        for image in images:
            clean_image = preprocess_image(image)
            text += pytesseract.image_to_string(
                clean_image,
                lang="fra+eng",
                config="--psm 6"
            )

    return text


def read_all_cvs(folder_path):
    all_cvs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            print(f"📖 Lecture de : {filename}")
            text = read_cv_text(full_path)
            all_cvs.append({
                "filename": filename,
                "text": text
            })
            print(f"✅ Terminé : {filename}")
    return all_cvs


# TEST
if __name__ == "__main__":
    cvs = read_all_cvs("cvs/")
    print(f"\n📄 Total CVs lus : {len(cvs)}")
    for cv in cvs:
        print(f"\n{'='*50}")
        print(f"📄 Fichier : {cv['filename']}")
        print(f"{'='*50}")
        print(cv['text'])