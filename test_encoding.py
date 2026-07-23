from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
es.index(index="test_encoding", id=1, document={"texte": "Cybersécurité et développement énergétique"})
print("Envoyé !")

# Relit directement via le client Python (pas PowerShell)
result = es.get(index="test_encoding", id=1)
print("Résultat relu :", result["_source"]["texte"])