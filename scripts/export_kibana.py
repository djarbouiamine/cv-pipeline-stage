#!/usr/bin/env python3
"""Export Kibana Saved Objects (dashboards, visualizations, data views, etc.)
to a single NDJSON file that can be committed to the Git repository.

Usage:
    python scripts/export_kibana.py                         # Export to kibana_dashboard.ndjson
    python scripts/export_kibana.py --out my_dashboard.ndjson
    python scripts/export_kibana.py --kibana http://host:5601

The generated NDJSON can be re-imported on any machine via:
    Kibana -> Stack Management -> Saved Objects -> Import
or via the companion script:
    python scripts/import_kibana.py
"""
from __future__ import annotations

import argparse
import os
import sys
import requests


DEFAULT_KIBANA = os.getenv("KIBANA_URL", "http://localhost:5601")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "kibana_dashboard.ndjson")

# Types of Saved Objects to export (add/remove as needed)
TYPES_TO_EXPORT = [
    "dashboard",
    "visualization",
    "lens",
    "index-pattern",         # "Data View" in recent Kibana versions
    "search",
    "map",
    "tag",
]


def export_saved_objects(kibana_url: str, out_path: str) -> None:
    """Call the Kibana Saved Objects Export API and write the result to a file."""
    url = f"{kibana_url.rstrip('/')}/api/saved_objects/_export"
    headers = {
        "kbn-xsrf": "true",
        "Content-Type": "application/json",
    }
    payload = {
        "type": TYPES_TO_EXPORT,
        "includeReferencesDeep": True,
    }

    print(f"Connexion a Kibana : {kibana_url}")
    print(f"Types exportes : {', '.join(TYPES_TO_EXPORT)}")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.ConnectionError:
        print(f"\nImpossible de joindre Kibana a {kibana_url}.")
        print("   Assurez-vous que Kibana est demarre (docker compose up -d).")
        sys.exit(1)
    except requests.HTTPError as exc:
        print(f"\nErreur HTTP {resp.status_code}: {resp.text}")
        raise SystemExit(1) from exc

    out_path = os.path.abspath(out_path)
    with open(out_path, "wb") as f:
        f.write(resp.content)

    n_objects = resp.content.count(b"\n")
    print(f"\nExport reussi : {n_objects} objets -> {out_path}")
    print("\nPensez a committer ce fichier dans votre depot Git :")
    print(f"   git add {os.path.basename(out_path)}")
    print("   git commit -m 'chore: export Kibana dashboard'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Kibana Saved Objects to NDJSON.")
    parser.add_argument("--kibana", default=DEFAULT_KIBANA,
                        help=f"Kibana URL (default: {DEFAULT_KIBANA})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output file path (default: kibana_dashboard.ndjson)")
    args = parser.parse_args()

    export_saved_objects(args.kibana, args.out)


if __name__ == "__main__":
    main()
