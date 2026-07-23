from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

MAPPING = {
    "mappings": {
        "properties": {
            "nom": {"type": "text"},
            "email": {"type": "keyword"},
            "telephone": {"type": "keyword"},
            "linkedin": {"type": "keyword"},
            "localisation": {"type": "text"},
            "categorie_principale": {"type": "keyword"},
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
            "filename": {"type": "keyword"},
        }
    }
}

def create_index(index_name="cvs"):
    if es.indices.exists(index=index_name):
        print(f"⚠️ L'index '{index_name}' existe déjà. Supprime-le d'abord si tu veux le recréer.")
        return
    es.indices.create(index=index_name, body=MAPPING)
    print(f"✅ Index '{index_name}' créé avec le mapping défini.")

def delete_index(index_name="cvs"):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"🗑️ Index '{index_name}' supprimé.")
    else:
        print(f"ℹ️ L'index '{index_name}' n'existe pas, rien à supprimer.")

if __name__ == "__main__":
    delete_index()   # ← supprime l'ancien index (avec mapping dynamique)
    create_index()   # ← recrée avec le nouveau mapping (nested)