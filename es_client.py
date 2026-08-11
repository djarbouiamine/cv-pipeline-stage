'''es_client.py
Connexion centralisée à Elasticsearch pour Streamlit.
Le client est mis en cache via @st.cache_resource afin d’être réutilisé
par toutes les pages sans recréer de connexion à chaque rafraîchissement.
'''

from __future__ import annotations
import streamlit as st
from elasticsearch import Elasticsearch
import os

@st.cache_resource(show_spinner=False)
def get_es_client() -> Elasticsearch:
    """Retourne une instance Elasticsearch configurée.
    Les paramètres peuvent être personnalisés via des variables d’environnement :
    - ELASTIC_HOST (default: http://localhost:9200)
    - ELASTIC_USERNAME / ELASTIC_PASSWORD (facultatif)
    """
    host = os.getenv('ELASTIC_HOST', 'http://localhost:9200')
    username = os.getenv('ELASTIC_USERNAME')
    password = os.getenv('ELASTIC_PASSWORD')

    if username and password:
        es = Elasticsearch(
            hosts=[host],
            http_auth=(username, password),
            # connection_class=RequestsHttpConnection,  # Not needed for elasticsearch>=8
            timeout=30,
        )
    else:
        es = Elasticsearch(hosts=[host], timeout=30)

    # Vérifier la connexion
    try:
        if not es.ping():
            raise ConnectionError
    except Exception:
        st.warning(f"⚠️ Elasticsearch non disponible à {host}. Utilisation d'un client factice.")
        class DummyES:
            def __init__(self):
                self.info = {"message": "Dummy client"}
            def ping(self):
                return False
            def count(self, *args, **kwargs):
                # Retourne un compte factice pour éviter les erreurs dans le tableau de bord
                return {"count": 0}
            def search(self, *args, **kwargs):
                # Structure minimale attendue par le dashboard
                return {
                    "hits": {"total": 0, "hits": []},
                    "aggregations": {
                        "by_categ": {"buckets": []},
                        "avg_years": {"value": 0}
                    }
                }
            def index(self, *args, **kwargs):
                return {"result": "created"}
            def delete(self, *args, **kwargs):
                return {"result": "deleted"}
            def exists(self, *args, **kwargs):
                return False
        return DummyES()
    return es
