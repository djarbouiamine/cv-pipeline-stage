import json
import hashlib
from elasticsearch import Elasticsearch

from cv_extractor import embedding_model

es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "cvs"

MAPPING = {
    "mappings": {
        "properties": {
            "nom": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}
            },
            "email": {"type": "keyword"},
            "telephone": {"type": "keyword"},
            "linkedin": {"type": "keyword"},
            "localisation": {"type": "text"},
            "categorie_principale": {"type": "keyword"},
            "domaine_1": {"type": "keyword"}, "score_1": {"type": "float"},
            "domaine_2": {"type": "keyword"}, "score_2": {"type": "float"},
            "domaine_3": {"type": "keyword"}, "score_3": {"type": "float"},
            "scores_categories": {
                "type": "nested",
                "properties": {
                    "domaine": {"type": "keyword"},
                    "score": {"type": "float"}
                }
            },
            "scores_categories_ponderes": {
                "type": "nested",
                "properties": {
                    "domaine": {"type": "keyword"},
                    "score": {"type": "float"}
                }
            },
            "scores_categories_ponderes_sur_10": {
                "type": "nested",
                "properties": {
                    "domaine": {"type": "keyword"},
                    "score": {"type": "float"}
                }
            },
            "technologies": {"type": "keyword"},
            "langages": {"type": "keyword"},
            "frameworks": {"type": "keyword"},
            "bases_de_donnees": {"type": "keyword"},
            "outils_devops": {"type": "keyword"},
            "projets": {"type": "text"},
            "diplomes": {"type": "text"},
            "certifications": {"type": "text"},
            "langues": {"type": "keyword"},
            "score_qualite_globale": {"type": "float"},
            "score_qualite_globale_sur_10": {"type": "float"},
            "annees_experience": {"type": "float"},
            "alertes_parcours": {"type": "text"},
            "experiences_pro": {
                "type": "nested",
                "properties": {
                    "poste": {"type": "text"},
                    "domaine": {"type": "keyword"},
                    "date_debut": {"type": "keyword"},
                    "date_fin": {"type": "keyword"},
                    "poids_pertinence": {"type": "float"}
                }
            },
            "filename": {"type": "keyword"},

            # ── Champs RAG (ajoutés) ─────────────────────────
            "text": {"type": "text"},
            "embedding_cv": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}


def create_index(index_name=INDEX_NAME):
    if es.indices.exists(index=index_name):
        print(f"[WARN] L'index '{index_name}' existe deja. Supprime-le d'abord si tu veux le recreer.")
        return
    es.indices.create(index=index_name, body=MAPPING)
    print(f"[OK] Index '{index_name}' cree avec le mapping defini.")


def delete_index(index_name=INDEX_NAME):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"[OK] Index '{index_name}' supprime.")
    else:
        print(f"[INFO] L'index '{index_name}' n'existe pas, rien a supprimer.")


def inject_cvs(json_path="output/cvs_data.json", index_name=INDEX_NAME):
    """
    Lit le JSON produit par cv_extractor.py / cv_saver.py et injecte
    chaque CV dans Elasticsearch avec un ID stable (email+nom) pour eviter
    les doublons meme en cas de re-injection.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        print("[WARN] Aucun CV trouve dans le JSON -- rien a injecter.")
        return

    injected = 0
    skipped_dup = 0
    seen_emails: set = set()
    seen_phones: set = set()

    for result in results:
        data = result.get("data")
        if not data:
            print(f"[WARN] Ignore (pas de donnees) : {result.get('source_path')}")
            continue

        # ── Deduplication intra-fichier (email / telephone) ──────────────────
        entry_email = (data.get("email") or "").strip().lower()
        entry_phone = (data.get("telephone") or "").strip()
        entry_nom   = data.get("nom", result.get("filename", "?"))

        if entry_email and entry_email in seen_emails:
            print(f"[SKIP] Doublon (email) : {entry_nom} ({entry_email})")
            skipped_dup += 1
            continue
        if entry_phone and entry_phone in seen_phones:
            print(f"[SKIP] Doublon (telephone) : {entry_nom} ({entry_phone})")
            skipped_dup += 1
            continue

        if entry_email: seen_emails.add(entry_email)
        if entry_phone: seen_phones.add(entry_phone)

        doc = dict(data)  # copy to avoid mutating original
        doc["filename"] = result.get("filename") or result.get("source_path", "")
        doc["text"] = result.get("text", "")

        # ── ID stable : evite les doublons ES lors de re-injections ──────────
        # Priorite : hash du fichier > hash(email+nom) > hash(filename)
        raw_id = (
            result.get("hash")
            or (entry_email + entry_nom)
            or doc["filename"]
        )
        doc_id = hashlib.sha256(raw_id.encode("utf-8", errors="replace")).hexdigest()

        # ── Calcul de l'embedding ─────────────────────────────────────────────
        text_for_embedding = doc["text"].strip()
        if not text_for_embedding:
            print(f"[WARN] Pas de texte pour {doc['filename']} -- impossible de generer l'embedding")
            continue

        if embedding_model is None:
            print(f"[WARN] Modele d'embedding non disponible -- impossible d'injecter {doc['filename']}")
            continue

        champs_enrichissement = []
        for champ in ["technologies", "langages", "frameworks", "bases_de_donnees",
                      "outils_devops", "certifications"]:
            valeurs = doc.get(champ, [])
            if isinstance(valeurs, list) and valeurs:
                champs_enrichissement.append(", ".join(valeurs))
        categ = doc.get("categorie_principale", "")
        if categ:
            champs_enrichissement.append(categ)
        texte_enrichi = text_for_embedding + "\n" + " | ".join(champs_enrichissement)

        doc["embedding_cv"] = embedding_model.encode(
            texte_enrichi, normalize_embeddings=True
        ).tolist()

        es.index(index=index_name, id=doc_id, document=doc)
        injected += 1
        print(f"[OK] Injecte : {doc.get('nom', doc['filename'])}")

    print(f"\n[DONE] {injected}/{len(results)} CVs injectes dans l'index '{index_name}'")
    if skipped_dup:
        print(f"[INFO] {skipped_dup} doublon(s) ignore(s) (meme email/telephone).")



if __name__ == "__main__":
    delete_index()
    create_index()
    inject_cvs()