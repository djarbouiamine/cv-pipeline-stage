from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# Récupère tous les CVs de l'index "cvs"
resultats = es.search(index="cvs", query={"match_all": {}}, size=50)

print(f"Total de CVs dans la base : {resultats['hits']['total']['value']}\n")
print("-" * 60)

for hit in resultats["hits"]["hits"]:
    data = hit["_source"]
    print(f"Nom          : {data.get('nom')}")
    print(f"Email        : {data.get('email')}")
    print(f"Catégorie    : {data.get('categorie_principale')}")
    print(f"Score global : {data.get('score_qualite_globale')}/100")
    print(f"Domaines     : {data.get('scores_categories')}")
    print(f"Projets      : {data.get('projets')}")
    print("-" * 60)