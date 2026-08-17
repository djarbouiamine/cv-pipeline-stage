#!/usr/bin/env python3
"""Import Kibana Saved Objects from the NDJSON file previously exported.

Usage:
    python scripts/import_kibana.py                          # Import kibana_dashboard.ndjson
    python scripts/import_kibana.py --file my_dashboard.ndjson
    python scripts/import_kibana.py --kibana http://host:5601
"""
from __future__ import annotations

import argparse
import os
import sys
import requests


DEFAULT_KIBANA = os.getenv("KIBANA_URL", "http://localhost:5601")
DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "..", "kibana_dashboard.ndjson")


def import_saved_objects(kibana_url: str, file_path: str) -> None:
    """Call the Kibana Saved Objects Import API."""
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        print(f"Fichier introuvable : {file_path}")
        print("Exportez d'abord le dashboard depuis PC1 :")
        print("   python scripts/export_kibana.py")
        sys.exit(1)

    url = f"{kibana_url.rstrip('/')}/api/saved_objects/_import?overwrite=true"
    headers = {"kbn-xsrf": "true"}

    print(f"Connexion a Kibana : {kibana_url}")
    print(f"Fichier source     : {file_path}")

    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                url,
                headers=headers,
                files={"file": ("kibana_dashboard.ndjson", f, "application/ndjson")},
                timeout=30,
            )
        resp.raise_for_status()
    except requests.ConnectionError:
        print(f"\nImpossible de joindre Kibana a {kibana_url}.")
        print("   Assurez-vous que Kibana est demarre (docker compose up -d).")
        sys.exit(1)
    except requests.HTTPError as exc:
        print(f"\nErreur HTTP {resp.status_code}: {resp.text}")
        raise SystemExit(1) from exc

    result = resp.json()
    n_success = result.get("successCount", 0)
    errors = result.get("errors", [])
    print(f"\nImport termine : {n_success} objets importes.")
    if errors:
        print(f"Avertissements ({len(errors)}) :")
        for e in errors:
            print(f"  - {e.get('type', '?')} [{e.get('id', '?')}]: {e.get('error', {}).get('message', '?')}")
    else:
        print("Aucune erreur. Rendez-vous sur http://localhost:5601 -> Dashboards.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Kibana Saved Objects from NDJSON.")
    parser.add_argument("--kibana", default=DEFAULT_KIBANA,
                        help=f"Kibana URL (default: {DEFAULT_KIBANA})")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help="NDJSON file to import (default: kibana_dashboard.ndjson)")
    args = parser.parse_args()

    import_saved_objects(args.kibana, args.file)


if __name__ == "__main__":
    main()
