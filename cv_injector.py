from elasticsearch import Elasticsearch
import json

# Connexion à Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Vérifier la connexion
try:
    info = es.info()
    print(f"✅ Connecté à Elasticsearch : {info['version']['number']}")
except Exception as e:
    print(f"❌ Erreur de connexion : {e}")
    exit()

INDEX_NAME = "cvs"


def inject_cvs(json_path="output/cvs_data.json"):
    """
    Lit le JSON et injecte chaque CV dans Elasticsearch
    """
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    for i, result in enumerate(results):
        data = result["data"]
        data["filename"] = result["filename"]  # on garde le nom du fichier

        # Injecte le document dans Elasticsearch
        es.index(index=INDEX_NAME, id=i, document=data)
        print(f"✅ Injecté : {data.get('nom', result['filename'])}")

    print(f"\n🎉 {len(results)} CVs injectés dans l'index '{INDEX_NAME}'")


# TEST
if __name__ == "__main__":
    inject_cvs()