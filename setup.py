#!/usr/bin/env python3
"""
setup.py — Script d'installation complet du projet cv-pipeline.
===============================================================
Lance UNE SEULE commande pour tout configurer :

    python setup.py

Ce script effectue dans l'ordre :
  1. Verifie les dependances (Elasticsearch, Kibana)
  2. Cree l'index Elasticsearch avec le bon mapping
  3. Injecte les CVs depuis output/cvs_data.json (si present)
  4. Cree automatiquement le dashboard Kibana complet

Apres execution :
  - Streamlit  : streamlit run app.py
  - Kibana     : http://localhost:5601/app/dashboards
"""
from __future__ import annotations

import os
import sys
import subprocess
import time
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
KIBANA_URL = os.getenv("KIBANA_URL", "http://localhost:5601")
ES_URL = os.getenv("ELASTIC_HOST", "http://localhost:9200")


def print_step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")
    print("-" * 50)


def check_service(url: str, name: str, retries: int = 10, delay: int = 6) -> bool:
    print(f"  Attente de {name} ({url}) ...", end="", flush=True)
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code < 500:
                print(" OK")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(delay)
    print(f" ECHEC")
    return False


def run(cmd: list[str], cwd: str = ROOT) -> int:
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def main() -> None:
    total = 4
    print("\n" + "=" * 60)
    print("  CV Pipeline — Installation complete")
    print("=" * 60)

    # --- Etape 1 : Verifier les services Docker ---
    print_step(1, total, "Verification des services (Elasticsearch + Kibana)")
    es_ok = check_service(f"{ES_URL}/_cluster/health", "Elasticsearch")
    kb_ok = check_service(f"{KIBANA_URL}/api/status", "Kibana")

    if not es_ok:
        print("\n  Elasticsearch n'est pas disponible.")
        print("  Lancez : docker compose up -d")
        print("  Puis relancez : python setup.py")
        sys.exit(1)

    if not kb_ok:
        print("\n  Kibana n'est pas encore pret (il peut mettre 30-60s a demarrer).")
        print("  Attendez quelques secondes puis relancez : python setup.py")
        sys.exit(1)

    print("  Elasticsearch et Kibana sont operationnels.")

    # --- Etape 2 : Creer l'index Elasticsearch ---
    print_step(2, total, "Creation de l'index Elasticsearch 'cvs'")
    ret = run([sys.executable, "create_index.py"])
    if ret != 0:
        print("  [AVERTISSEMENT] create_index.py a echoue ou l'index existait deja.")

    # --- Etape 3 : Injecter les CVs (si le fichier JSON est present) ---
    print_step(3, total, "Injection des CVs dans Elasticsearch")
    data_file = os.path.join(ROOT, "output", "cvs_data.json")
    if os.path.exists(data_file):
        ret = run([sys.executable, "cv_injector.py"])
        if ret != 0:
            print("  [AVERTISSEMENT] cv_injector.py a signale une erreur.")
        else:
            print("  CVs injectes avec succes.")
    else:
        print(f"  Fichier {data_file} absent — injection ignoree.")
        print("  Pour traiter vos CVs : python cv_extractor.py && python cv_saver.py")

    # --- Etape 4 : Creer le dashboard Kibana ---
    print_step(4, total, "Creation automatique du dashboard Kibana")
    setup_kibana = os.path.join(ROOT, "scripts", "setup_kibana.py")
    ret = run([sys.executable, setup_kibana])
    if ret != 0:
        print("  [AVERTISSEMENT] Le dashboard Kibana n'a pas pu etre cree.")

    # --- Fin ---
    print("\n" + "=" * 60)
    print("  Installation terminee !")
    print("=" * 60)
    print(f"\n  Streamlit  : streamlit run app.py")
    print(f"  Kibana     : {KIBANA_URL}/app/dashboards")
    print(f"  Elasticsearch : {ES_URL}")
    print()


if __name__ == "__main__":
    main()
